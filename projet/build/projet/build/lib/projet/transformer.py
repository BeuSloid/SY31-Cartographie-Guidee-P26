#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2

from .utils import make_pointcloud2


class Transformer(Node):
    def __init__(self):
        super().__init__("transformer")
        self.pub = self.create_publisher(PointCloud2, "points", 10)
        self.sub = self.create_subscription(LaserScan, "scan", self.callback, 10)

    def callback(self, msg: LaserScan):
        x = []
        y = []
        intensities = []

        # On parcourt chaque mesure du scan
        for i in range(len(msg.ranges)):
            r = msg.ranges[i]

            # On calcule l'angle de ce rayon a partir de son indice
            theta = msg.angle_min + i * msg.angle_increment

            # On ignore les points trop proches (chassis) ou invalides
            if r < 0.1 or np.isinf(r) or np.isnan(r):
                continue

            # Passage polaire -> cartesien
            x.append(r * np.cos(theta))
            y.append(r * np.sin(theta))
            # Certains drivers LDS ne publient pas d'intensites
            if i < len(msg.intensities):
                intensities.append(msg.intensities[i])
            else:
                intensities.append(0.0)

        self.pub.publish(make_pointcloud2(msg.header, x, y, intensities))


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(Transformer())
    except KeyboardInterrupt:
        pass




"""Le nœud transformer convertit les balayages laser bruts (LaserScan), exprimés en coordonnées polaires, en un nuage de points cartésien (PointCloud2). Pour chaque rayon d'indice i, l'angle est reconstruit par la relation θ = angle_min + i × angle_increment, puis les coordonnées sont obtenues par les formules classiques x = r·cos(θ) et y = r·sin(θ). Nous avons fait le choix de parcourir directement msg.ranges par son indice plutôt que de générer les angles avec np.arange(angle_min, angle_max, angle_increment) : cette dernière approche, à cause des erreurs d'arrondi en virgule flottante, peut produire un nombre d'angles différent du nombre de mesures et désynchroniser indices et angles, ce qui faisait perdre les derniers rayons de chaque scan et laissait des trous dans les façades cartographiées. Nous filtrons également les mesures trop proches (r < 0,1 m, correspondant au châssis du robot) ainsi que les valeurs infinies ou non définies (inf/nan) renvoyées par le capteur lorsqu'aucun obstacle n'est détecté, afin de ne conserver que des points exploitables."""


"""L'IA nous a aidés à diagnostiquer un bug que nous n'arrivions pas à localiser : certaines façades du labyrinthe disparaissaient de façon intermittente. Nous avions d'abord soupçonné un problème de filtrage. En soumettant notre callback, l'IA a identifié que la génération des angles par np.arange(angle_min, angle_max, angle_increment) produisait, à cause des arrondis flottants, un tableau d'angles plus court que msg.ranges, désynchronisant indices et angles et tronquant la fin de chaque scan. Nous avons retenu sa suggestion de reconstruire l'angle directement depuis l'indice (θ = angle_min + i × angle_increment), puis ajouté nous-mêmes le filtrage des valeurs inf/nan du LiDAR."""