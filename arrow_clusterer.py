#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

from .utils import declare_param


class ArrowClusterer(Node):
    """Clustering incrémental par centroïde des détections de flèches.
    
    Quand une flèche est détectée plusieurs fois (en passant devant),
    on fusionne les détections proches en un seul cluster représenté
    par son centroïde. Filtre les faux positifs (1-2 détections).
    """

    def __init__(self):
        super().__init__("arrow_clusterer")

        declare_param(self, "D", 0.3)             # dist max pour fusionner (m)
        declare_param(self, "min_detections", 3)  # filtre des faux positifs

        # Clusters : liste de dicts {centroid, count}
        self.clusters_red = []
        self.clusters_blue = []

        # Subscribers
        self.sub_red = self.create_subscription(
            PointStamped, "arrow_red", self.cb_red, 10
        )
        self.sub_blue = self.create_subscription(
            PointStamped, "arrow_blue", self.cb_blue, 10
        )

        # Publisher : markers pour RViz
        self.pub = self.create_publisher(MarkerArray, "arrow_clusters", 10)

        # Publication régulière
        self.create_timer(0.5, self.publish_markers)

    def cb_red(self, msg):
        self._add(msg, self.clusters_red)

    def cb_blue(self, msg):
        self._add(msg, self.clusters_blue)

    def _add(self, msg, clusters):
        D = self.get_parameter("D").value
        pt = np.array([msg.point.x, msg.point.y])

        # Cluster le plus proche
        best, best_d = -1, np.inf
        for i, c in enumerate(clusters):
            d = np.linalg.norm(pt - c["centroid"])
            if d < best_d:
                best_d, best = d, i

        if best_d < D:
            # Mise à jour incrémentale du centroïde
            c = clusters[best]
            c["count"] += 1
            c["centroid"] += (pt - c["centroid"]) / c["count"]
        else:
            # Nouveau cluster
            clusters.append({"centroid": pt.copy(), "count": 1})

    def publish_markers(self):
        min_det = self.get_parameter("min_detections").value
        ma = MarkerArray()
        ma.markers.append(Marker(action=Marker.DELETEALL))

        mid = 0
        for clusters, color, label in [
            (self.clusters_red, (1.0, 0.0, 0.0), "red"),
            (self.clusters_blue, (0.0, 0.0, 1.0), "blue"),
        ]:
            for c in clusters:
                if c["count"] < min_det:
                    continue
                m = Marker()
                m.header.frame_id = "odom"
                m.header.stamp = self.get_clock().now().to_msg()
                m.id = mid
                mid += 1
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(c["centroid"][0])
                m.pose.position.y = float(c["centroid"][1])
                m.pose.position.z = 0.2
                m.scale.x = m.scale.y = m.scale.z = 0.2
                m.color.r, m.color.g, m.color.b = color
                m.color.a = 0.9
                ma.markers.append(m)

        self.pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(ArrowClusterer())
    except KeyboardInterrupt:
        pass