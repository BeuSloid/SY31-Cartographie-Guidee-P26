#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from turtlebot3_msgs.msg import SensorState
from geometry_msgs.msg import PoseStamped
from transforms3d.euler import euler2quat
from builtin_interfaces.msg import Time

class OdomNode(Node):
    """
    Nœud d'odométrie robuste fusionnant l'IMU (angle) et les encodeurs (distance).
    C'est le fichier 'cerveau' qui permet de situer le robot dans le labyrinthe.
    """
    # Caractéristiques physiques du Turtlebot3
    TICKS_PER_REV = 4096      # Résolution des encodeurs
    RAYON_ROUE = 0.033        # En mètres
    
    def __init__(self):
        super().__init__("odom_node")

        # --- Variables d'état (La Pose du robot) ---
        self.x = 0.0          # Position X dans le repère global (m)
        self.y = 0.0          # Position Y dans le repère global (m)
        self.theta = 0.0      # Orientation (rad) extraite du Gyroscope

        # Mémoire pour le calcul des deltas
        # None = pas encore reçu de message ; évite le faux saut au premier message
        self.last_left_ticks = None
        self.last_right_ticks = None
        self.v_lineaire = 0.0

        # --- Communication ROS 2 ---
        # On publie la pose fusionnée sur /robot_pose pour le map_transformer
        self.pose_pub = self.create_publisher(PoseStamped, "/robot_pose", 10)

        # On s'abonne aux capteurs bruts du robot
        self.imu_sub = self.create_subscription(Imu, "/imu", self.callback_gyro, 10)
        self.enco_sub = self.create_subscription(SensorState, "/sensor_state", self.callback_encoder, 10)

    def get_delta_t(self, stamp, timer_name):
        """Calcule le temps écoulé entre deux messages pour l'intégration."""
        now = stamp.sec + stamp.nanosec / 1e9
        prev = getattr(self, timer_name) if hasattr(self, timer_name) else now
        dt = now - prev
        setattr(self, timer_name, now)
        return dt

    def callback_gyro(self, msg):
        """Mise à jour de l'angle theta par intégration de la vitesse angulaire."""
        dt = self.get_delta_t(msg.header.stamp, "_t_gyro")
        if dt < 0:  # le bag a rebouclé : on repart de zéro
            self.theta = 0.0
            return
        if dt == 0:
            return
        vitesse_angulaire = msg.angular_velocity.z
        self.theta += vitesse_angulaire * dt

    def callback_encoder(self, msg):
        """Calcul du déplacement linéaire et mise à jour de la position (X, Y)."""
        dt = self.get_delta_t(msg.header.stamp, "_t_enco")

        # Premier message OU rebouclage du bag (temps négatif) :
        # on mémorise la baseline des encodeurs sans calculer de déplacement
        if self.last_left_ticks is None or dt < 0:
            self.last_left_ticks = msg.left_encoder
            self.last_right_ticks = msg.right_encoder
            if dt < 0:  # rebouclage : remise à zéro de la position
                self.x = 0.0
                self.y = 0.0
            return
        if dt == 0:
            return

        # 1. Calcul de la distance parcourue par chaque roue (en ticks)
        d_left = msg.left_encoder - self.last_left_ticks
        d_right = msg.right_encoder - self.last_right_ticks

        # Mise à jour de la mémoire pour le prochain calcul
        self.last_left_ticks = msg.left_encoder
        self.last_right_ticks = msg.right_encoder

        # 2. Conversion Ticks -> Vitesse linéaire (m/s)
        # Formule : (ticks / résolution) * (périmètre de la roue) / dt
        v_l = (d_left / self.TICKS_PER_REV) * (2 * np.pi * self.RAYON_ROUE / dt)
        v_r = (d_right / self.TICKS_PER_REV) * (2 * np.pi * self.RAYON_ROUE / dt)
        self.v_lineaire = (v_l + v_r) / 2.0

        # 3. Projection sur les axes globaux X et Y
        # On utilise l'angle theta qui vient d'être mis à jour par le Gyroscope
        self.x += self.v_lineaire * np.cos(self.theta) * dt
        self.y += self.v_lineaire * np.sin(self.theta) * dt

        # 4. Envoi du message de pose
        self.publier_pose(msg.header.stamp)

    def publier_pose(self, timestamp):
        """Crée et publie le message PoseStamped."""
        msg = PoseStamped()
        msg.header.stamp = timestamp
        msg.header.frame_id = "odom"
        
        # Position
        msg.pose.position.x = self.x
        msg.pose.position.y = self.y
        
        # Conversion Euler -> Quaternion pour l'orientation
        q = euler2quat(0, 0, self.theta)
        msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z = q
        
        self.pose_pub.publish(msg)

def main():
    rclpy.init()
    node = OdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == "__main__":
    main()
    
    
    
    
#__init__ : Cette fonction prépare la structure de données du nœud. Elle définit les caractéristiques physiques du robot (comme le rayon de la roue de 0.033 m) et initialise la "pose" du robot (x,y et θ) à zéro. Elle établit également les communications ROS 2 en créant le publisher pour la pose finale et les deux abonnements (subscribers) pour recevoir les données de l'IMU et des encodeurs.

#    get_delta_t : C'est une fonction utilitaire cruciale pour la précision. Elle calcule le temps exact (dt) écoulé entre la réception de deux messages successifs en utilisant les timestams du système. Sans ce calcul dynamique du temps, l'intégration des vitesses (angulaire et linéaire) serait imprécise.

 #   callback_gyro : Cette fonction est dédiée uniquement à l'orientation. Elle récupère la vitesse angulaire sur l'axe Z (le lacet) depuis l'IMU et l'intègre au fil du temps pour mettre à jour la variable theta. En isolant ainsi le calcul de l'angle sur le gyroscope, on évite que les erreurs de glissement des roues n'affectent la direction estimée du robot.

#    callback_encoder : C'est ici que le déplacement est calculé. La fonction mesure d'abord la variation du nombre de "ticks" pour chaque roue, puis convertit cette différence en vitesse réelle (m/s) en utilisant la résolution de 4096 ticks et le périmètre de la roue. Une fois la vitesse linéaire obtenue, elle projette ce mouvement sur les axes X et Y de la carte globale en utilisant l'angle theta (mis à jour par le gyro). Cette méthode garantit que le robot "sait" dans quelle direction il avance.

#    publier_pose : Cette dernière étape formate les résultats pour les autres nœuds. Elle convertit l'angle de rotation (Euler) en un quaternion (format 3D requis par ROS) via la fonction euler2quat. Elle publie ensuite l'ensemble des données dans un message PoseStamped, rattaché au repère fixe odom, permettant au nœud de transformation LiDAR de placer ses points correctement dans le labyrinthe.
