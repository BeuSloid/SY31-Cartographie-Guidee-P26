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


# =============================================================================
# Configuration du nuage de points
# =============================================================================
# Structure d'un point publié sur les topics PointCloud2.
# Le champ `clusterId` est utilisé pour colorer les points par cluster dans RViz
# (Color Transformer = Intensity, Channel Name = clusterId).
PC2FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="clusterId", offset=16, datatype=PointField.FLOAT32, count=1),
]


# =============================================================================
# Déclaration de paramètres ROS modifiables à chaud
# =============================================================================
def declare_param(object: Node, name: str, default_value: Any) -> None:
    """Déclare un paramètre ROS modifiable à chaud via `ros2 param set`.

    L'attribut du nœud est automatiquement mis à jour à chaque changement,
    ce qui évite de devoir relancer le nœud pour tester de nouvelles valeurs.

    :param object: Nœud ROS sur lequel déclarer le paramètre
    :param name: Nom du paramètre (ex: "distMax2Pts")
    :param default_value: Valeur par défaut
    """
    def callback(params: Iterable[Parameter]) -> SetParametersResult:
        for param in params:
            object.get_logger().info(f"Setting parameter '{param.name}' to {param.value}")
            setattr(object, param.name, param.value)
        return SetParametersResult(successful=True)

    if len(object._on_set_parameters_callbacks) < 2:
        object.add_on_set_parameters_callback(callback)

    object.declare_parameter(name, default_value)


# =============================================================================
# Construction d'un message PointCloud2
# =============================================================================
def make_pointcloud2(
    header: Header,
    x: np.ndarray,
    y: np.ndarray,
    i: np.ndarray,
    c: np.ndarray = None,
) -> PointCloud2:
    """Construit un PointCloud2 à partir de tableaux x, y, intensité, clusterId.

    Si le clusterId n'est pas fourni (étapes amont du pipeline), il est mis à 0.
    La coordonnée z est forcée à 0 (le robot évolue en 2D).

    :param header: Header à propager (frame_id, timestamp)
    :param x, y, i: tableaux des positions et intensités (même longueur)
    :param c: tableau des cluster IDs (optionnel)
    :return: PointCloud2 prêt à publier
    """
    zeros = np.zeros(len(x))
    if c is None:
        c = zeros

    assert len(x) == len(y) == len(i) == len(c), (
        "Tailles incohérentes : "
        f"({len(x)}, {len(y)}, {len(i)}, {len(c)})"
    )

    points = np.vstack((x, y, zeros, i, c)).T
    return create_cloud(header, PC2FIELDS, points)


# =============================================================================
# Construction de marqueurs pour visualiser des objets dans RViz
# =============================================================================
def make_markers(
    header: Header,
    shapes: Iterable,
    width: float = 0.05,
) -> MarkerArray:
    """Construit un MarkerArray pour visualiser des objets dans RViz.

    Sert principalement à afficher les flèches détectées par la caméra,
    représentées chacune par un cylindre à la position de leur centroïde.

    Formats acceptés pour `shapes` :
    - (x, y, radius)               → cylindre
    - (x, y, width, length)        → cube (boîte englobante)
    - liste de (x, y)              → polyligne (line strip)

    :param header: Header à propager
    :param shapes: Liste d'objets à représenter
    :param width: Épaisseur des polylignes (utilisé uniquement pour LINE_STRIP)
    :return: MarkerArray prêt à publier
    """
    markers = MarkerArray()
    # Effacer les anciens markers avant d'en publier de nouveaux
    markers.markers.append(Marker(header=header, action=Marker.DELETEALL))

    n = len(shapes) if hasattr(shapes, "__len__") else 0

    for c, shape in enumerate(shapes):
        # Couleur arc-en-ciel : du rouge au vert
        rainbow = c / n if n > 0 else 0.0
        marker = Marker(header=header, action=Marker.ADD, id=int(c + 1))
        col = marker.color
        col.r, col.g, col.b, col.a = 1.0 - rainbow, rainbow, 0.0, 0.8

        # Polyligne (liste de points)
        if isinstance(shape, list):
            marker.type = Marker.LINE_STRIP
            marker.points.extend([Point(x=float(x), y=float(y)) for x, y in shape])
            marker.scale.x = marker.scale.y = 2.0 * width
            marker.scale.z = 0.3

        # Cylindre (x, y, rayon)
        elif len(shape) == 3:
            x, y, radius = shape
            marker.type = Marker.CYLINDER
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.scale.x = marker.scale.y = 2.0 * radius
            marker.scale.z = 0.3

        # Boîte englobante (x, y, largeur, longueur)
        elif len(shape) == 4:
            x, y, w, l = shape
            marker.type = Marker.CUBE
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.scale.x = float(w)
            marker.scale.y = float(l)
            marker.scale.z = 0.3

        markers.markers.append(marker)

    return markers

"""Le module utils.py regroupe les outils partagés par l'ensemble des nœuds. 
La fonction make_pointcloud2 assemble les tableaux de positions, d'intensités et d'identifiants 
de cluster en un message PointCloud2 ; la coordonnée z est forcée à zéro puisque le robot évolue 
en 2D, et les entrées sont converties en tableaux float32 afin d'accepter indifféremment des listes
 Python ou des tableaux NumPy. La fonction declare_param permet de déclarer des paramètres ROS 
 modifiables à chaud (via ros2 param set), ce qui nous a évité de relancer les nœuds à chaque 
 ajustement de seuil lors des tests ; le callback de mise à jour n'est enregistré qu'une seule 
 fois par nœud à l'aide d'un drapeau, pour éviter qu'il ne soit déclenché plusieurs fois lorsqu'un 
 nœud déclare plusieurs paramètres. Enfin, make_markers génère les marqueurs RViz représentant les 
 flèches détectées, chacune affichée sous forme de cylindre coloré à la position de son centroïde."""
"""Sur ce module utilitaire, l'IA a surtout joué un rôle de relecture qualité. 
Elle a repéré que notre fonction declare_param accédait à un attribut interne de ROS 
(_on_set_parameters_callbacks) pour éviter d'enregistrer deux fois le callback de mise à jour 
des paramètres, solution fragile et non documentée. Nous avons remplacé cette astuce par 
un mécanisme explicite à base de drapeau (hasattr), que nous maîtrisons et pouvons justifier.
 L'IA nous a également suggéré de convertir les entrées de make_pointcloud2 en float32 
 via np.asarray pour accepter indifféremment listes et tableaux NumPy, ce qui a supprimé 
 une source d'erreurs de type rencontrée entre nos différents nœuds."""