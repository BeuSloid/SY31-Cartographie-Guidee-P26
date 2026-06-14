#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from turtlebot3_msgs.msg import SensorState
from geometry_msgs.msg import PoseStamped
from transforms3d.euler import euler2quat


class OdomNode(Node):
    """calcule la pose (x, y, theta) du robot par integration
    on decouple les deux capteurs:
    distance : depuis les encodeurs de roueslon ce qu'ils mesurent le mieux
    theta : integre depuis la vitesse angulaire du gyroscope (IMU)
    on evite ainsi d'estimer l'angle par les roues, qui patinent dans
    les virages serres du labyrinthe comme expliqué dans le rapport
    """

    # caracteristiques du turtlebot3
    TICKS_PER_REV = 4096      # ticks par tour de roue
    RAYON_ROUE = 0.033        # rayon de la roue (m)

    def __init__(self):
        super().__init__("odom_node")

        # pose du robot dans le repere global
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # horodatage du dernier message IMU (pour calculer dt)
        self.t_gyro = None

        # position precedente des encodeurs (en ticks)
        # none = pas encore recu de message  evite le faux saut au premier message
        self.last_left = None
        self.last_right = None

        # publisher pose du robot
        self.pose_pub = self.create_publisher(PoseStamped, "/robot_pose", 10)

        # subscribers capteurs bruts
        self.create_subscription(Imu, "/imu", self.callback_gyro, 10)
        self.create_subscription(SensorState, "/sensor_state", self.callback_encoder, 10)

    def callback_gyro(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # premier message on memorise juste le temps
        if self.t_gyro is None:
            self.t_gyro = t
            return

        dt = t - self.t_gyro
        self.t_gyro = t

        # on ignore les dt nuls ou aberrants
        if dt == 0.0 or dt > 1.0:
            return

        # integration theta += omega * dt
        self.theta += msg.angular_velocity.z * dt

    def callback_encoder(self, msg: SensorState):
        # premier message on memorise la baseline sans calculer de deplacement
        if self.last_left is None:
            self.last_left = msg.left_encoder
            self.last_right = msg.right_encoder
            return

        # variation du nombre de ticks depuis le dernier message
        d_left = msg.left_encoder - self.last_left
        d_right = msg.right_encoder - self.last_right
        self.last_left = msg.left_encoder
        self.last_right = msg.right_encoder

        # conversion ticks -> distance parcourue (m)
        perimetre = 2 * np.pi * self.RAYON_ROUE
        dist_left = (d_left / self.TICKS_PER_REV) * perimetre
        dist_right = (d_right / self.TICKS_PER_REV) * perimetre

        # distance du robot = moyenne des deux roues
        dist = (dist_left + dist_right) / 2.0

        # projection sur les axes globaux avec l'angle du gyro
        self.x += dist * np.cos(self.theta)
        self.y += dist * np.sin(self.theta)

        # publication de la pose
        self.publier_pose(msg.header.stamp)

    def publier_pose(self, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "odom"

        msg.pose.position.x = self.x
        msg.pose.position.y = self.y

        # angle theta -> quaternion 
        q = euler2quat(0, 0, self.theta)
        msg.pose.orientation.w = q[0]
        msg.pose.orientation.x = q[1]
        msg.pose.orientation.y = q[2]
        msg.pose.orientation.z = q[3]

        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(OdomNode())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


