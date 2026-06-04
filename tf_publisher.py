#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster


class TfPublisher(Node):
    """Publie la TF odom -> base_scan à partir de l'odométrie maison /robot_pose.
    
    Permet à RViz d'afficher les points LiDAR (frame base_scan) 
    dans le repère monde (odom).
    """
    def __init__(self):
        super().__init__("tf_publisher")
        self.broadcaster = TransformBroadcaster(self)
        self.sub = self.create_subscription(
            PoseStamped, "/robot_pose", self.callback, 10
        )

    def callback(self, msg: PoseStamped):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_scan"

        # Position du robot
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = 0.17  # hauteur LiDAR Turtlebot3

        # Orientation du robot
        t.transform.rotation = msg.pose.orientation

        self.broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(TfPublisher())
    except KeyboardInterrupt:
        pass