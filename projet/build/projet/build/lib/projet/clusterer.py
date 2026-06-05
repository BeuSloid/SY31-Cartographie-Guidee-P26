#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
from geometry_msgs.msg import PoseStamped
from transforms3d.euler import quat2euler

from .utils import make_pointcloud2, declare_param


class Clusterer(Node):
    def __init__(self):
        super().__init__("clusterer")

        declare_param(self, "distMax2Pts", 0.1)
        declare_param(self, "min_cluster_size", 5)
        declare_param(self, "max_cluster_size", 1000)

        # Pose courante du robot (mise à jour par /robot_pose)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.angle_robot = 0.0

        self.pub = self.create_publisher(PointCloud2, "clusters", 10)
        self.sub = self.create_subscription(PointCloud2, "points_filtered", self.callback, 10)
        self.sub_pose = self.create_subscription(PoseStamped, "/robot_pose", self.pose_callback, 10)

    def pose_callback(self, msg: PoseStamped):
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        q = msg.pose.orientation
        _, _, self.angle_robot = quat2euler([q.w, q.x, q.y, q.z])

    def dist(self, p1, p2):
        """Distance euclidienne entre deux points (x, y)."""
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def callback(self, msg: PointCloud2):
        points = read_points_numpy(msg, ["x", "y", "intensity", "clusterId"])

        distMax2Pts = self.get_parameter("distMax2Pts").value
        min_size = self.get_parameter("min_cluster_size").value
        max_size = self.get_parameter("max_cluster_size").value

        if len(points) < 3:
            msg.header.frame_id = "odom"
            self.pub.publish(make_pointcloud2(msg.header, *points.T))
            return

        """
        Phase 1 : suppression du bruit (algo du td)
        Un point i est aberrant si :
            - ses 2 voisins (i-1 et i+1) sont proches entre eux
            - MAIS i est loin de ces 2 voisins
        """
        keep = np.ones(len(points), dtype=bool)
        for i in range(1, len(points) - 1):
            if self.dist(points[i - 1], points[i + 1]) < distMax2Pts:
                if (self.dist(points[i - 1], points[i]) > distMax2Pts and
                    self.dist(points[i], points[i + 1]) > distMax2Pts):
                    keep[i] = False

        points = points[keep]

        if len(points) == 0:
            msg.header.frame_id = "odom"
            self.pub.publish(make_pointcloud2(msg.header, *points.T))
            return

        """
        Phase 2 : clustering
        Deux points consécutifs à distance < distMax2Pts -> même cluster
        Sinon -> nouveau cluster
        """
        C = np.zeros(len(points), dtype=np.int32)
        current_cluster_id = 1
        C[0] = current_cluster_id

        for i in range(1, len(points)):
            if self.dist(points[i - 1], points[i]) < distMax2Pts:
                C[i] = current_cluster_id           # même objet
            else:
                current_cluster_id += 1             # nouvel objet
                C[i] = current_cluster_id

        """
        Phase 3 : gestion de la discontinuité (premier et dernier point)
        Si le premier point et le dernier sont proches, on fusionne leurs clusters
        """
        if self.dist(points[0], points[-1]) < distMax2Pts:
            c_last = C[-1]
            c_first = C[0]
            if c_last != c_first:
                C[C == c_last] = c_first

        """
        Phase 4 : filtrage par taille de cluster
        """
        cluster_sizes = np.bincount(C)
        valid_clusters = np.where(
            (cluster_sizes >= min_size) & (cluster_sizes <= max_size)
        )[0]
        valid_clusters = valid_clusters[valid_clusters != 0]

        # Renumérotation des clusters valides
        new_C = np.zeros_like(C)
        for new_id, old_id in enumerate(valid_clusters, start=1):
            new_C[C == old_id] = new_id

        points[:, 3] = new_C

        # Transformation base_scan -> odom (même logique que map_transformer)
        cos_a = np.cos(self.angle_robot)
        sin_a = np.sin(self.angle_robot)
        x_local = points[:, 0].copy()
        y_local = points[:, 1].copy()
        points[:, 0] = x_local * cos_a - y_local * sin_a + self.robot_x
        points[:, 1] = x_local * sin_a + y_local * cos_a + self.robot_y

        msg.header.frame_id = "odom"
        self.pub.publish(make_pointcloud2(msg.header, *points.T))


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(Clusterer())
    except KeyboardInterrupt:
        pass