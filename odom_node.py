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

        # Horodatage du dernier message IMU (pour calculer dt)
        self.t_gyro = None

        # Position precedente des encodeurs (en ticks)
        # None = pas encore recu de message ; evite le faux saut au premier message
        self.last_left = None
        self.last_right = None

        # Publisher : pose du robot
        self.pose_pub = self.create_publisher(PoseStamped, "/robot_pose", 10)

        # Subscribers : capteurs bruts
        self.create_subscription(Imu, "/imu", self.callback_gyro, 10)
        self.create_subscription(SensorState, "/sensor_state", self.callback_encoder, 10)

    def callback_gyro(self, msg: Imu):
        """Met a jour l'angle theta en integrant la vitesse angulaire du gyro.

        Cette fonction est dediee uniquement a l'orientation. En isolant le calcul
        de l'angle sur le gyroscope, on evite que les erreurs de glissement des roues
        n'affectent la direction estimee du robot.
        """
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Premier message : on memorise juste le temps
        if self.t_gyro is None:
            self.t_gyro = t
            return

        dt = t - self.t_gyro
        self.t_gyro = t

        # On ignore les dt nuls ou aberrants (pause, saut > 1s)
        if dt == 0.0 or dt > 1.0:
            return

        # Integration : theta += omega * dt
        self.theta += msg.angular_velocity.z * dt

    def callback_encoder(self, msg: SensorState):
        """Calcul du deplacement lineaire et mise a jour de la position (X, Y).

        La distance de chaque roue se deduit du nombre de ticks via la resolution
        de l'encodeur (4096 ticks/tour) et le perimetre de roue (rayon 0.033 m).
        La distance du robot est la moyenne des deux roues. Ce deplacement est
        ensuite projete sur les axes globaux a l'aide de l'angle issu du gyrometre :
        x += d*cos(theta)  et  y += d*sin(theta).
        """
        # Premier message : on memorise la baseline sans calculer de deplacement
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
        """Construit et publie le message PoseStamped.

        L'angle theta est converti en quaternion (format 3D requis par ROS) via
        euler2quat. La pose est publiee dans le repere fixe 'odom', ce qui permet
        au noeud de cartographie de placer les points LiDAR correctement.
        """
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


"""
DECLARATION D'UTILISATION DE L'IA

Pour l'odometrie, nous nous sommes appuyes sur les notions des TP, ne sollicitant
l'IA que pour la syntaxe ROS 2 et la resolution de bugs techniques. La logique
(decoupler le gyrometre pour l'angle et les encodeurs pour la distance) a ete
decidee par nous-memes ; l'IA a contribue surtout a la mise en forme du code.

L'IA nous a aides sur les points syntaxiques propres a rclpy : la structure du
noeud avec ses deux abonnements (/imu et /sensor_state), la conversion de l'angle
en quaternion via euler2quat, et la lecture du timestamp des messages pour calculer
le pas de temps dt necessaire a l'integration du gyrometre.

En revanche, nous avons corrige plusieurs choix de conception. La version initiale
parlait de "fusion de capteurs" : nous avons identifie qu'il s'agissait en realite
d'un decouplage et corrige la formulation. Nous avons egalement ecarte l'idee de
moyenner l'angle du gyrometre et celui des encodeurs : moyenner une estimation
fiable avec une estimation faussee par le patinage des roues aurait degrade le
resultat.
"""
