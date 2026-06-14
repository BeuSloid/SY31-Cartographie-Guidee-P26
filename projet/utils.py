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


# structure d'un point PointCloud2, clusterId permet de colorer par cluster dans RViz.
PC2FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="clusterId", offset=16, datatype=PointField.FLOAT32, count=1),
]


def declare_param(object: Node, name: str, default_value: Any) -> None:
    # déclare un paramètre ROS et met à jour l'attribut du nœud à chaque ros2 param set
    def callback(params: Iterable[Parameter]) -> SetParametersResult:
        for param in params:
            object.get_logger().info(f"Setting parameter '{param.name}' to {param.value}")
            setattr(object, param.name, param.value)
        return SetParametersResult(successful=True)

    if len(object._on_set_parameters_callbacks) < 2:
        object.add_on_set_parameters_callback(callback)

    object.declare_parameter(name, default_value)


def make_pointcloud2(header, x, y, i, c=None):
    zeros = np.zeros(len(x))
    if c is None:
        c = zeros

    assert len(x) == len(y) == len(i) == len(c), (
        "Tailles incohérentes : "
        f"({len(x)}, {len(y)}, {len(i)}, {len(c)})"
    )

    points = np.vstack((x, y, zeros, i, c)).T
    return create_cloud(header, PC2FIELDS, points)


def make_markers(header, shapes, width=0.05):
    markers = MarkerArray()
    # effacer les anciens markers avant d'en publier de nouveaux
    markers.markers.append(Marker(header=header, action=Marker.DELETEALL))

    n = len(shapes) if hasattr(shapes, "__len__") else 0

    for c, shape in enumerate(shapes):
        # couleur arc en ciel : du rouge au vert
        rainbow = c / n if n > 0 else 0.0
        marker = Marker(header=header, action=Marker.ADD, id=int(c + 1))
        col = marker.color
        col.r, col.g, col.b, col.a = 1.0 - rainbow, rainbow, 0.0, 0.8

        # polyligne (liste de points)
        if isinstance(shape, list):
            marker.type = Marker.LINE_STRIP
            marker.points.extend([Point(x=float(x), y=float(y)) for x, y in shape])
            marker.scale.x = marker.scale.y = 2.0 * width
            marker.scale.z = 0.3

        # cylindre (x, y, rayon)
        elif len(shape) == 3:
            x, y, radius = shape
            marker.type = Marker.CYLINDER
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.scale.x = marker.scale.y = 2.0 * radius
            marker.scale.z = 0.3

        # boite englobante (x, y, largeur, longueur)
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
