#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from .utils import declare_param


class ArrowClusterer(Node):
    """clustering incremental par centroide des detections de fleches
    quand une fleche est detectee plusieurs fois (en passant devant),
    on fusionne les detections proches en un seul cluster representé
    par son centroide. filtre les faux positifs (1-2 detections).
    """

    def __init__(self):
        super().__init__("arrow_clusterer")

        declare_param(self, "D", 0.3) # dist max pour fusionner en mètre
        declare_param(self, "min_detections", 3) # filtre des faux positifs

        # clusters : liste de dicts (centroid, count)
        self.clusters_red = []
        self.clusters_blue = []

        # subscribers
        self.sub_red = self.create_subscription(
            PointStamped, "arrow_red", self.cb_red, 10
        )
        self.sub_blue = self.create_subscription(
            PointStamped, "arrow_blue", self.cb_blue, 10
        )

        # publisher : markers pour RViz
        self.pub = self.create_publisher(MarkerArray, "arrow_clusters", 10)

        # publication reguliere
        self.create_timer(0.5, self.publish_markers)

    def cb_red(self, msg):
        self._add(msg, self.clusters_red)

    def cb_blue(self, msg):
        self._add(msg, self.clusters_blue)

    def _add(self, msg, clusters):
        pt = np.array([msg.point.x, msg.point.y])

        # recherche du cluster existant le plus proche
        best, best_d = -1, np.inf
        for i, c in enumerate(clusters):
            d = np.linalg.norm(pt - c["centroid"])
            if d < best_d:
                best_d, best = d, i

        if best_d < self.D:
            # mise a jour incrementale du centroide (moyenne glissante)
            c = clusters[best]
            c["count"] += 1
            c["centroid"] += (pt - c["centroid"]) / c["count"]
        else:
            # nouveau cluster
            clusters.append({"centroid": pt.copy(), "count": 1})

    def publish_markers(self):
        ma = MarkerArray()
        ma.markers.append(Marker(action=Marker.DELETEALL))

        mid = 0
        for clusters, ns, color in [
            (self.clusters_red, "arrow_red", (1.0, 0.0, 0.0)),
            (self.clusters_blue, "arrow_blue", (0.0, 0.0, 1.0)),
        ]:
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = ns
            m.id = mid
            mid += 1
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = m.scale.y = 0.04   # taille des points en m
            m.color.r, m.color.g, m.color.b = color
            m.color.a = 1.0
            for c in clusters:
                if c["count"] < self.min_detections:
                    continue
                m.points.append(Point(x=float(c["centroid"][0]),
                                    y=float(c["centroid"][1]), z=0.05))
            ma.markers.append(m)

        self.pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(ArrowClusterer())
    except KeyboardInterrupt:
        pass
