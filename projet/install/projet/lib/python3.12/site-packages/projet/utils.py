#!/usr/bin/env python3

from typing import Any, Iterable

import numpy as np
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.parameter_service import Parameter, SetParametersResult
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


# Structure d'un point publie sur les topics PointCloud2.
# clusterId sert a colorer les points par cluster dans RViz.
PC2FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="clusterId", offset=16, datatype=PointField.FLOAT32, count=1),
]


def declare_param(node: Node, name: str, default_value: Any) -> None:
    """Declare un parametre ROS modifiable a chaud via ros2 param set.

    L'attribut du noeud est mis a jour a chaque changement, ce qui evite
    de relancer le noeud pour tester une nouvelle valeur.
    """
    # On enregistre le callback une seule fois par noeud
    if not hasattr(node, "_param_callback_set"):
        def callback(params: Iterable[Parameter]) -> SetParametersResult:
            for param in params:
                setattr(node, param.name, param.value)
            return SetParametersResult(successful=True)

        node.add_on_set_parameters_callback(callback)
        node._param_callback_set = True

    node.declare_parameter(name, default_value)


def make_pointcloud2(header, x, y, i, c=None):
    """Construit un PointCloud2 a partir des tableaux x, y, intensite, clusterId.

    Si clusterId n'est pas fourni il est mis a 0.
    La coordonnee z est forcee a 0 (robot en 2D).
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    i = np.asarray(i, dtype=np.float32)
    zeros = np.zeros(len(x), dtype=np.float32)
    if c is None:
        c = zeros

    points = np.vstack((x, y, zeros, i, c)).T
    return create_cloud(header, PC2FIELDS, points)


def make_markers(header, shapes, width=0.05):
    """Construit un MarkerArray pour afficher les fleches dans RViz.

    Chaque forme (x, y, rayon) est representee par un cylindre.
    """
    markers = MarkerArray()
    # On efface les anciens markers avant d'en publier de nouveaux
    markers.markers.append(Marker(header=header, action=Marker.DELETEALL))

    n = len(shapes) if hasattr(shapes, "__len__") else 0

    for c, shape in enumerate(shapes):
        # Couleur du rouge au vert pour distinguer les clusters
        rainbow = c / n if n > 0 else 0.0
        marker = Marker(header=header, action=Marker.ADD, id=int(c + 1))
        marker.color.r = 1.0 - rainbow
        marker.color.g = rainbow
        marker.color.b = 0.0
        marker.color.a = 0.8

        x, y, radius = shape
        marker.type = Marker.CYLINDER
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.scale.x = marker.scale.y = 2.0 * radius
        marker.scale.z = 0.3

        markers.markers.append(marker)

    return markers



"""Le module utils.py regroupe les outils partagés par l'ensemble des nœuds. La fonction make_pointcloud2 assemble les tableaux de positions, d'intensités et d'identifiants de cluster en un message PointCloud2 ; la coordonnée z est forcée à zéro puisque le robot évolue en 2D, et les entrées sont converties en tableaux float32 afin d'accepter indifféremment des listes Python ou des tableaux NumPy. La fonction declare_param permet de déclarer des paramètres ROS modifiables à chaud (via ros2 param set), ce qui nous a évité de relancer les nœuds à chaque ajustement de seuil lors des tests ; le callback de mise à jour n'est enregistré qu'une seule fois par nœud à l'aide d'un drapeau, pour éviter qu'il ne soit déclenché plusieurs fois lorsqu'un nœud déclare plusieurs paramètres. Enfin, make_markers génère les marqueurs RViz représentant les flèches détectées, chacune affichée sous forme de cylindre coloré à la position de son centroïde."""


"""Sur ce module utilitaire, l'IA a surtout joué un rôle de relecture qualité. Elle a repéré que notre fonction declare_param accédait à un attribut interne de ROS (_on_set_parameters_callbacks) pour éviter d'enregistrer deux fois le callback de mise à jour des paramètres, solution fragile et non documentée. Nous avons remplacé cette astuce par un mécanisme explicite à base de drapeau (hasattr), que nous maîtrisons et pouvons justifier. L'IA nous a également suggéré de convertir les entrées de make_pointcloud2 en float32 via np.asarray pour accepter indifféremment listes et tableaux NumPy, ce qui a supprimé une source d'erreurs de type rencontrée entre nos différents nœuds."""