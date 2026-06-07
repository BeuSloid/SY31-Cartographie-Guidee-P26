#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Header
from transforms3d.euler import quat2euler
from geometry_msgs.msg import PoseStamped

from .utils import make_pointcloud2


class MapTransformer(Node):
    def __init__(self):
        super().__init__("map_transformer")

        self.sub_scan = self.create_subscription(LaserScan, "scan", self.scan_callback, 10)
        self.sub_pose = self.create_subscription(PoseStamped, "/robot_pose", self.pose_callback, 10)
        self.pub = self.create_publisher(PointCloud2, "global_points", 10)

        # Pose courante du robot
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.angle_robot = 0.0

        # Distances LiDAR a garder (m)
        self.dist_min = 0.10
        self.dist_max = 3.5

        # Carte accumulee
        self.acc_x = []
        self.acc_y = []
        self.acc_i = []
        self._last_stamp = None

        # On republie toute la carte une fois par seconde
        self.create_timer(1.0, self.publish_map)
        self.has_pose = False

    def pose_callback(self, msg: PoseStamped):
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        q = msg.pose.orientation
        _, _, self.angle_robot = quat2euler([q.w, q.x, q.y, q.z])
        self.has_pose = True

    def scan_callback(self, msg: LaserScan):
        if not self.has_pose:
            return
        # 1. Angles et distances en tableaux NumPy
        ranges = np.array(msg.ranges)
        n = len(ranges)
        angles = msg.angle_min + np.arange(n) * msg.angle_increment

        # 2. On garde les distances valides (pas de inf/nan, ni trop proche/loin)
        ok = np.isfinite(ranges) & (ranges > self.dist_min) & (ranges < self.dist_max)
        ranges = ranges[ok]
        angles = angles[ok]
        if len(ranges) == 0:
            return

        # 3. Polaire -> cartesien (repere robot)
        x_local = ranges * np.cos(angles)
        y_local = ranges * np.sin(angles)

        # 4. Rotation (angle robot) + translation (position robot) -> repere odom
        cos_t = np.cos(self.angle_robot)
        sin_t = np.sin(self.angle_robot)
        x_global = x_local * cos_t - y_local * sin_t + self.robot_x
        y_global = x_local * sin_t + y_local * cos_t + self.robot_y

        intens = np.array(msg.intensities)[ok] if len(msg.intensities) == n else np.zeros(len(ranges))

        # 5. Reset de la carte si le bag a reboucle (temps qui repart en arriere)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        if self._last_stamp is not None and stamp < self._last_stamp:
            self.acc_x.clear()
            self.acc_y.clear()
            self.acc_i.clear()
        self._last_stamp = stamp

        # 6. Accumulation
        self.acc_x.extend(x_global[::2])
        self.acc_y.extend(y_global[::2])
        self.acc_i.extend(intens[::2])

    def publish_map(self):
        if not self.acc_x:
            return
        header = Header()
        header.frame_id = "odom"
        header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(make_pointcloud2(header, self.acc_x, self.acc_y, self.acc_i))


def main(args=None):
    rclpy.init(args=args)
    node = MapTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

"""Une fois la pose du robot reconstruite, le rôle du nœud de cartographie est de replacer chaque
 point mesuré par le LiDAR dans un repère fixe afin de dessiner progressivement le labyrinthe. 
 À chaque balayage, les distances renvoyées par le capteur sont d'abord converties de coordonnées 
 polaires en coordonnées cartésiennes dans le repère propre du robot, selon les formules classiques 
 x = r·cos(α) et y = r·sin(α), où α est l'angle de chaque rayon. Ces points, encore exprimés 
 relativement au robot, sont ensuite transformés vers le repère global odom par une rotation 
 suivie d'une translation : la rotation utilise l'angle θ issu du gyromètre, et la translation 
 utilise la position (x, y) du robot. Cette opération correspond exactement à l'équation de 
 changement de repère fournie dans l'énoncé, où la matrice homogène 3×3 combine en une seule 
 étape la rotation et la translation appliquées à l'ensemble des points d'un scan.
Le choix d'utiliser l'angle du gyromètre plutôt qu'une orientation déduite des roues découle 
directement de la logique adoptée pour l'odométrie : dans les virages serrés, le patinage fausse 
toute estimation d'angle basée sur les encodeurs, alors que le gyromètre mesure la rotation réelle 
du châssis. Réutiliser cette orientation fiable pour la cartographie garantit que les murs se 
positionnent correctement même lorsque le robot tourne. Avant transformation, les mesures sont
 filtrées : les points trop proches (reflets sur le châssis) et les valeurs aberrantes ou infinies 
 (rayons ne touchant aucun obstacle) sont écartés, ce qui évite de polluer la carte. Contrairement 
 à un simple affichage scan par scan, où chaque balayage effacerait le précédent, nous accumulons 
 l'ensemble des points déjà observés et republions périodiquement la carte complète. C'est cette 
 accumulation, rendue possible par la connaissance de la pose à chaque instant, qui permet de voir
   le tracé du labyrinthe se construire au fil du parcours. Lorsque le fichier bag reboucle, 
   la détection d'un retour en arrière de l'horodatage réinitialise la carte pour éviter la 
   superposition de deux tours décalés.
Pour ce nœud, la logique géométrique (conversion polaire-cartésien, puis rotation et translation 
vers le repère global) provient directement des TP, notamment du TP de transformation LiDAR. 
L'IA n'a été sollicitée que sur des aspects syntaxiques de rclpy : la structure du nœud avec 
son abonnement au scan et à la pose, la construction du message PointCloud2 via les utilitaires 
fournis, et la vectorisation des calculs sous NumPy pour traiter tout un balayage d'un coup
 plutôt que rayon par rayon. Nous avons par ailleurs corrigé plusieurs points de la version 
 initiale. Celle-ci ne faisait qu'afficher le scan courant sans rien accumuler : nous avons 
 ajouté la mémoire des points et la republication périodique. Le filtrage des valeurs infinies,
   absent au départ, a été ajouté pour fiabiliser la carte. Enfin, nous avons ajusté le seuil 
   de distance maximale en fonction de la taille réelle du labyrinthe, après avoir constaté que
   des façades lointaines disparaissaient lorsque ce seuil était trop bas."""