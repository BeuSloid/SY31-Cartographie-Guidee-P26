#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy

from .utils import make_pointcloud2, declare_param


class IntensityFilter(Node):
    def __init__(self):
        super().__init__("intensity_filter")

        # Seuil d'intensite reglable a chaud (ros2 param set)
        declare_param(self, "intensity_threshold", 20.0)

        self.pub = self.create_publisher(PointCloud2, "points_filtered", 10)
        self.sub = self.create_subscription(PointCloud2, "points", self.callback, 10)

    def callback(self, msg: PointCloud2):
        # Decodes the points in a numpy array of shape [[x0, y0, i0], [x1, y1, i1], ...]
        points = read_points_numpy(msg, ["x", "y", "intensity"])

        if len(points) == 0:
            return

        # On ne garde que les points dont l'intensite depasse le seuil
        points_filt = points[points[:, 2] > self.intensity_threshold]

        filt = make_pointcloud2(msg.header, points_filt[:, 0], points_filt[:, 1], points_filt[:, 2])
        self.pub.publish(filt)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(IntensityFilter())
    except KeyboardInterrupt:
        pass



"""Le nœud intensity_filter opère un seuillage sur l'intensité des points LiDAR. Le capteur renvoie, pour chaque point, une mesure de l'intensité du retour laser, plus élevée sur les surfaces réfléchissantes. En ne conservant que les points dont l'intensité dépasse un seuil réglable, on peut isoler certaines surfaces et alléger le nuage transmis à l'étape de clustering. Le seuil est exposé comme paramètre ROS modifiable à chaud via notre utilitaire declare_param, ce qui nous a permis de l'ajuster directement pendant la lecture des fichiers bag sans relancer la chaîne de nœuds. Un garde-fou ignore les nuages vides afin d'éviter toute erreur d'indexation lorsque aucun point n'est reçu.
Ce nœud est l'un des plus directs de la chaîne et reprend l'ossature fournie en TP ; le filtrage vectorisé par masque booléen NumPy relève des notions de manipulation de tableaux vues en cours. L'IA n'a pas été nécessaire sur la logique. Lors de notre relecture qualité, nous avons surtout veillé à la cohérence avec le reste du projet : nous avons unifié la déclaration du paramètre de seuil pour qu'elle passe par notre utilitaire commun declare_param (réglage à chaud), plutôt que par une lecture unique au démarrage, et ramené la profondeur de file du publisher à une valeur raisonnable adaptée à un traitement scan par scan"""