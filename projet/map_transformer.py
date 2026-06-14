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

        # pose courante du robot
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.angle_robot = 0.0

        # distances lidar a garder (m)
        self.dist_min = 0.10
        self.dist_max = 3.5

        # voxel grid : resolutions et seuil de qualite
        # chaque cellule (ix, iy) accumule les points qui tombent dedans.
        # On ne publie une cellule que si elle a ete touchee >= MIN_HITS fois
        # ce qui elimine le bruit et les points aberrants isoles
        self.GRID_RES = 0.05 # taille d'une cellule (m)
        self.MIN_HITS = 2 # passages minimum pour valider un point

        # carte accumulee : dict (ix, iy) -> [sum_x, sum_y, sum_i, count]
        self.grid = {}

        # on republie toute la carte une fois par seconde
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
        # angles et distances en tableaux NumPy
        ranges = np.array(msg.ranges)
        n = len(ranges)
        angles = msg.angle_min + np.arange(n) * msg.angle_increment

        # on garde les distances valides
        ok = np.isfinite(ranges) & (ranges > self.dist_min) & (ranges < self.dist_max)
        ranges = ranges[ok]
        angles = angles[ok]
        if len(ranges) == 0:
            return

        # polaire -> cartesien (repere robot)
        x_local = ranges * np.cos(angles)
        y_local = ranges * np.sin(angles)

        # rotation (angle robot) + translation (position robot) -> repere odom
        cos_t = np.cos(self.angle_robot)
        sin_t = np.sin(self.angle_robot)
        x_global = x_local * cos_t - y_local * sin_t + self.robot_x
        y_global = x_local * sin_t + y_local * cos_t + self.robot_y

        intens = np.array(msg.intensities)[ok] if len(msg.intensities) == n else np.zeros(len(ranges))

        # accumulation dans la grille voxel
        ix = np.floor(x_global / self.GRID_RES).astype(int)
        iy = np.floor(y_global / self.GRID_RES).astype(int)
        for k, xg, yg, ig in zip(zip(ix, iy), x_global, y_global, intens):
            if k in self.grid:
                self.grid[k][0] += xg
                self.grid[k][1] += yg
                self.grid[k][2] += ig
                self.grid[k][3] += 1
            else:
                self.grid[k] = [xg, yg, ig, 1]

    def publish_map(self):
        # on ne publie que les cellules ayant ete touchees >= MIN_HITS fois
        cells = [v for v in self.grid.values() if v[3] >= self.MIN_HITS]
        if not cells:
            return
        xs = [v[0] / v[3] for v in cells]
        ys = [v[1] / v[3] for v in cells]
        ins = [v[2] / v[3] for v in cells]
        header = Header()
        header.frame_id = "odom"
        header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(make_pointcloud2(header, xs, ys, ins))


def main(args=None):
    rclpy.init(args=args)
    node = MapTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
