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

        for i, theta in enumerate(np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)):

            if(msg.ranges[i]<0.1):
                continue

            x.append(msg.ranges[i] * np.cos(theta))
            y.append(msg.ranges[i] * np.sin(theta))

            intensities.append(msg.intensities[i])
        self.pub.publish(make_pointcloud2(msg.header, x, y, intensities))


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(Transformer())
    except KeyboardInterrupt:
        pass
