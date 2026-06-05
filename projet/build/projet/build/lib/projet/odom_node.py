#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from turtlebot3_msgs.msg import SensorState
from geometry_msgs.msg import PoseStamped
from transforms3d.euler import euler2quat


class OdomNode(Node):
    """Calcule la pose (x, y, theta) du robot par integration.

    On decouple les deux capteurs selon ce qu'ils mesurent le mieux :
      - theta : integre depuis la vitesse angulaire du gyroscope (IMU)
      - distance : depuis les encodeurs de roues
    On evite ainsi d'estimer l'angle par les roues, qui patinent dans
    les virages serres du labyrinthe.
    """

    # Caracteristiques du Turtlebot3
    TICKS_PER_REV = 4096      # ticks par tour de roue
    RAYON_ROUE = 0.033        # rayon de la roue (m)

    def __init__(self):
        super().__init__("odom_node")

        # Pose du robot dans le repere global
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Temps du dernier message recu (pour calculer dt du gyro)
        self.t_gyro = None

        # Position precedente des encodeurs (en ticks)
        self.last_left = None
        self.last_right = None

        # Publisher : pose du robot
        self.pose_pub = self.create_publisher(PoseStamped, "/robot_pose", 10)

        # Subscribers : capteurs bruts
        self.create_subscription(Imu, "/imu", self.callback_gyro, 10)
        self.create_subscription(SensorState, "/sensor_state", self.callback_encoder, 10)

    def callback_gyro(self, msg: Imu):
        """Met a jour l'angle theta en integrant la vitesse angulaire du gyro."""
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Premier message : on memorise juste le temps
        if self.t_gyro is None:
            self.t_gyro = t
            return

        dt = t - self.t_gyro
        self.t_gyro = t

        # On ignore les dt aberrants
        if dt <= 0.0 or dt > 1.0:
            return

        # Integration : theta = theta + omega * dt
        omega = msg.angular_velocity.z
        self.theta += omega * dt

    def callback_encoder(self, msg: SensorState):
        """Met a jour la position (x, y) a partir du deplacement des roues."""
        # Premier message : on memorise la position des roues
        if self.last_left is None:
            self.last_left = msg.left_encoder
            self.last_right = msg.right_encoder
            return

        # 1. Variation du nombre de ticks depuis le dernier message
        d_left = msg.left_encoder - self.last_left
        d_right = msg.right_encoder - self.last_right
        self.last_left = msg.left_encoder
        self.last_right = msg.right_encoder

        # 2. Conversion ticks -> distance parcourue (m)
        perimetre = 2 * np.pi * self.RAYON_ROUE
        dist_left = (d_left / self.TICKS_PER_REV) * perimetre
        dist_right = (d_right / self.TICKS_PER_REV) * perimetre

        # Distance du robot = moyenne des deux roues
        dist = (dist_left + dist_right) / 2.0

        # 3. Projection sur les axes globaux avec l'angle du gyro
        self.x += dist * np.cos(self.theta)
        self.y += dist * np.sin(self.theta)

        # 4. Publication de la pose
        self.publier_pose(msg.header.stamp)

    def publier_pose(self, stamp):
        """Construit et publie le message PoseStamped."""
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "odom"

        msg.pose.position.x = self.x
        msg.pose.position.y = self.y

        # Angle theta -> quaternion (format ROS)
        q = euler2quat(0, 0, self.theta)
        msg.pose.orientation.w = q[0]
        msg.pose.orientation.x = q[1]
        msg.pose.orientation.y = q[2]
        msg.pose.orientation.z = q[3]

        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(OdomNode())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()



"""Pour situer le robot dans le labyrinthe, nous reconstruisons sa pose (x, y, θ) par intégration des capteurs proprioceptifs,
 sans recourir à l'odométrie fournie par le constructeur. Plutôt que de fusionner les capteurs au sens d'un filtre, 
 nous avons fait le choix de les découpler selon ce que chacun mesure le mieux : l'orientation θ est obtenue en intégrant la vitesse angulaire du 
 gyromètre (IMU), tandis que la distance parcourue est calculée à partir des encodeurs de roues. La distance de chaque roue se déduit du nombre de 
 ticks via la résolution de l'encodeur (4096 ticks/tour) et le périmètre de roue (rayon 0,033 m) ; la distance du robot est la moyenne des deux roues. 
 Ce déplacement est ensuite projeté sur les axes globaux à l'aide de l'angle issu du gyromètre, ce qui constitue le point de rencontre des deux sources 
 d'information : x ← x + d·cos(θ) et y ← y + d·sin(θ).
Ce découplage est un choix délibéré. Les encodeurs permettraient certes d'estimer l'orientation par rotation différentielle des roues, mais dans les 
virages serrés du labyrinthe les roues patinent : l'angle ainsi calculé dériverait rapidement. Le gyromètre, qui mesure la rotation réelle du châssis 
indépendamment de l'adhérence, fournit une orientation nettement plus fiable. Nous avons donc confié l'angle au seul gyromètre et la distance aux seuls 
encodeurs. La pose obtenue est publiée sur le topic /robot_pose au format PoseStamped (l'angle étant converti en quaternion), où elle est consommée par 
le nœud de cartographie pour replacer les points LiDAR dans le repère global odom.

Pour l'odométrie, nous nous sommes appuyés sur les notions des TP, ne sollicitant l'IA que pour la syntaxe ROS 2 et la résolution de bugs techniques. La logique (découpler le gyromètre pour l'angle et les encodeurs pour la distance) a été décidée par nous-mêmes ; l'IA a contribué surtout à la mise en forme du code.
L'IA nous a aidés sur les points syntaxiques propres à rclpy : la structure du nœud avec ses deux abonnements (/imu et /sensor_state), la conversion de l'angle en quaternion via euler2quat, et la lecture du timestamp des messages pour calculer le pas de temps dt nécessaire à l'intégration du gyromètre.
En revanche, nous avons corrigé plusieurs choix de conception. La version initiale parlait de « fusion de capteurs » : nous avons identifié qu'il s'agissait 
en réalité d'un découplage et corrigé la formulation. L'IA proposait aussi de calculer une vitesse puis de la multiplier par dt ; nous avons simplifié en intégrant directement la distance. Enfin, nous avions envisagé de moyenner l'angle du gyromètre et celui des encodeurs, mais nous avons écarté cette idée après analyse : moyenner une estimation fiable avec une estimation faussée par le patinage des roues aurait dégradé le résultat."""