#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from transforms3d.euler import quat2euler
from geometry_msgs.msg import PoseStamped

from .utils import make_pointcloud2

class MapTransformer(Node):
    def __init__(self):
        super().__init__("map_transformer")

        # Souscription au LiDAR et à l'odométrie officielle du bag
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.scan_callback, 10)
        self.sub_pose = self.create_subscription(PoseStamped, "/robot_pose", self.pose_callback, 10)  

        # Topic de sortie pour RViz (les points fixes dans le labyrinthe)
        self.pub = self.create_publisher(PointCloud2, "global_points", 10)

        # Variables pour stocker la position actuelle du robot
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.angle_robot = 0.0

    def pose_callback(self, msg: PoseStamped):
        """On récupère la position X, Y et l'angle du robot venant de l'odométrie."""
        self.robot_x = msg.pose.position.x        # un seul .pose
        self.robot_y = msg.pose.position.y
        
        # Conversion du format quaternion vers un angle simple (lacet)
        quaternion = msg.pose.orientation
        _, _, self.angle_robot = quat2euler([quaternion.w, quaternion.x, quaternion.y, quaternion.z])

    def scan_callback(self, msg: LaserScan):
        """Traitement de chaque balayage laser pour le mettre dans la carte."""
        points_x = []
        points_y = []
        intensites = []

        # On parcourt chaque rayon du LiDAR (Logique du TP 4)
        for i, angle_balayage in enumerate(np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)):
            dist = msg.ranges[i]

            # --- FILTRAGE (Inspiré du fichier de mon ami) ---
            # On ignore les points trop proches (le châssis) ou trop loin (bruit)
            if dist < 0.10 or dist > 3.5:
                continue

            # 1. Passage en coordonnées cartésiennes LOCALES (repère robot)
            x_robot = dist * np.cos(angle_balayage)
            y_robot = dist * np.sin(angle_balayage)

            # 2. Application de la rotation (selon l'angle du gyroscope)
            # On utilise le gyro ici car les roues patinent dans le labyrinthe
            x_tourne = x_robot * np.cos(self.angle_robot) - y_robot * np.sin(self.angle_robot)
            y_tourne = x_robot * np.sin(self.angle_robot) + y_robot * np.cos(self.angle_robot)

            # 3. Application de la translation (décalage du robot)
            x_final = x_tourne + self.robot_x
            y_final = y_tourne + self.robot_y

            # Ajout aux listes pour le message final
            points_x.append(x_final)
            points_y.append(y_final)
            intensites.append(msg.intensities[i])

        # Si on a trouvé des points, on les publie
        if len(points_x) > 0:
            # On change le frame_id en 'odom' pour que RViz comprenne que c'est du global
            msg.header.frame_id = "odom"
            nuage_points = make_pointcloud2(msg.header, points_x, points_y, intensites)
            self.pub.publish(nuage_points)

def main(args=None):
    rclpy.init(args=args)
    node = MapTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
    
    
    
    
"""
ANALYSE TECHNIQUE DU FONCTIONNEMENT - ÉTAPE B DU PROJET

Ce nœud assure le passage des données brutes du LiDAR (mesurées en local) vers 
le repère global 'odom' du labyrinthe. Voici la logique détaillée :

1. FILTRAGE DES DONNÉES :
   Nous avons défini une plage de confiance entre 0.10m et 0.65m. 
   Le seuil bas élimine les reflets sur le propre châssis du robot, tandis que 
   le seuil haut permet d'ignorer les mesures instables en fond de labyrinthe.

2. TRANSFORMATION GÉOMÉTRIQUE (POLAIRE VERS CARTÉSIEN) :
   En reprenant la structure du TP #4, chaque rayon LiDAR est d'abord converti 
   en coordonnées (x,y) relatives au robot. 
   On utilise les formules trigonométriques classiques : x = r * cos(angle) 
   et y = r * sin(angle).

3. CHANGEMENT DE REPÈRE (ROBOT -> MAP) :
   C'est l'étape cruciale pour la cartographie. Pour chaque point, nous appliquons 
   deux transformations successives basées sur la pose robuste calculée dans odom_node :
   - LA ROTATION : On fait pivoter les points locaux selon l'angle 'angle_robot' 
     calculé par le gyroscope. Nous avons privilégié le 
     gyroscope car les encodeurs des roues sont sujets au patinage lors des 
     virages serrés du labyrinthe.
   - LA TRANSLATION : On décale ensuite ces points tournés par les coordonnées 
     X et Y du robot pour les situer dans le repère global.

4. PUBLICATION ET VISUALISATION :
   Le changement du 'frame_id' de 'base_scan' vers 'odom' est indispensable. 
   Il indique à RViz que ces points ne sont plus attachés au robot mais sont 
   des éléments fixes du décor. Cela permet de voir les 
   murs se dessiner de façon stable même lorsque le robot est en mouvement.
"""