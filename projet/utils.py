#!/usr/bin/env python3

from typing import Any, Iterable

import numpy as np
from rclpy.node import Node
from rclpy.parameter_service import Parameter, SetParametersResult
from sensor_msgs.msg import PointField
from sensor_msgs_py.point_cloud2 import create_cloud


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
