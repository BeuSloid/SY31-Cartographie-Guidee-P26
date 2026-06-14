#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy

from .utils import make_pointcloud2, declare_param


class IntensityFilter(Node):
    def __init__(self):
        super().__init__("intensity_filter")

        declare_param(self, "intensity_threshold", 20.0)
        self.intensity_threshold = self.get_parameter("intensity_threshold").value

        self.pub = self.create_publisher(PointCloud2, "points_filtered", 2000)
        self.sub = self.create_subscription(PointCloud2, "points", self.callback, 10)

    def callback(self, msg: PointCloud2):
        # [[x0, y0, i0], [x1, y1, i1], ...]
        points = read_points_numpy(msg, ["x", "y", "intensity"])
    
        points_filt = points[points[:, 2] > self.intensity_threshold]

        filt = make_pointcloud2(msg.header, points_filt[:, 0], points_filt[:, 1], points_filt[:, 2])
        self.pub.publish(filt)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(IntensityFilter())
    except KeyboardInterrupt:
        pass

