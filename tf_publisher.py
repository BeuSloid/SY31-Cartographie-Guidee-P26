#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster


class TfPublisher(Node):
    """Publie la TF odom -> base_scan à partir de l'odométrie maison /robot_pose.
    
    Permet à RViz d'afficher les points LiDAR (frame base_scan) 
    dans le repère monde (odom).
    """
    def __init__(self):
        super().__init__("tf_publisher")
        self.broadcaster = TransformBroadcaster(self)
        self.sub = self.create_subscription(
            PoseStamped, "/robot_pose", self.callback, 10
        )

    def callback(self, msg: PoseStamped):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_scan"

        # Position du robot
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = 0.17  # hauteur LiDAR Turtlebot3

        # Orientation du robot
        t.transform.rotation = msg.pose.orientation

        self.broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(TfPublisher())
    except KeyboardInterrupt:
        pass


"""Le nœud tf_publisher assure la liaison entre notre odométrie maison et le système de 
transformations (TF) de ROS, indispensable pour qu'RViz affiche correctement les données. 
Le LiDAR exprime ses points dans le repère base_scan, solidaire du robot ; pour les visualiser 
dans le repère monde odom, RViz a besoin de connaître à chaque instant la position de base_scan 
relativement à odom. À chaque pose reçue sur /robot_pose, le nœud construit donc une transformation 
odom → base_scan dont la translation reprend la position (x, y) du robot et la hauteur du LiDAR, 
et dont la rotation reprend directement l'orientation du robot, puis la diffuse via un 
TransformBroadcaster. Nous avons isolé cette tâche dans un nœud dédié, à responsabilité unique, 
plutôt que de la mêler au calcul d'odométrie, afin de garder une architecture claire où chaque nœud
 a un rôle bien délimité.
Pour ce nœud, la difficulté n'était pas algorithmique mais relevait de la mécanique propre 
à tf2_ros dans ROS 2. Nous avons sollicité l'IA uniquement pour la syntaxe de construction et 
de diffusion d'un TransformStamped (remplissage des champs header.frame_id, child_frame_id, et
 emploi du TransformBroadcaster), points purement techniques que nous avons ensuite vérifiés dans 
 la documentation. Le choix d'architecture — réutiliser notre pose /robot_pose comme source unique 
 de la TF plutôt que de dépendre d'une odométrie externe — découle de notre volonté de cohérence 
 avec le reste de la chaîne, où la même pose alimente la cartographie et la localisation des 
 flèches."""