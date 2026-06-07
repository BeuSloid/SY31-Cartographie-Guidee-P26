#!/usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, PoseStamped
from transforms3d.euler import quat2euler

from .utils import declare_param


class ArrowDetector(Node):
    """Detecte les fleches rouges et bleues dans l'image camera.

    Une fleche est validee si elle est :
    1. de la bonne couleur (masque HSV)
    2. assez grosse (proche du robot)
    3. au centre de l'image (droit devant le robot)
    """

    # Rouge : la teinte rouge est a la fois vers H=0 et vers H=180,
    # on couvre donc les deux intervalles.
    RED_HSV_MIN_1 = np.array([0, 100, 60], dtype=np.uint8)
    RED_HSV_MAX_1 = np.array([10, 255, 255], dtype=np.uint8)
    RED_HSV_MIN_2 = np.array([160, 100, 60], dtype=np.uint8)
    RED_HSV_MAX_2 = np.array([180, 255, 255], dtype=np.uint8)

    # Bleu : saturation et valeur minimales élevées pour rejeter les bleus pâles
    BLUE_HSV_MIN = np.array([95, 180, 80], dtype=np.uint8)
    BLUE_HSV_MAX = np.array([130, 255, 255], dtype=np.uint8)

    # Morphologie : noyaux séparés pour mieux contrôler le nettoyage
    OPEN_KERNEL_SIZE = 15
    CLOSE_KERNEL_SIZE = 10

    # Filtres de validation
    AREA_SEUIL = 6000           # aire min (px) pour valider une fleche
    CENTER_RADIUS_SEUIL = 150   # px, distance max au centre de l'image
    ARROW_FORWARD_OFFSET = 0.5  # distance estimee robot -> fleche (m)

    def __init__(self):
        super().__init__("arrow_detector")
        self.bridge = CvBridge()

        declare_param(self, "cooldown", 1.0)  # secondes min entre deux publications

        # Position et cap courants du robot (mis a jour par /robot_pose)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.has_odom = False

        # Centre de l'image (calcule au premier message)
        self.camera_center = None

        # Anti-spam : horodatage de la derniere publication par couleur
        self._last_red_pub = 0.0
        self._last_blue_pub = 0.0

        # Subscribers
        self.sub_img = self.create_subscription(
            CompressedImage, "/turtlecam/image_raw/compressed", self.callback_img, 10
        )
        self.sub_odom = self.create_subscription(
            PoseStamped, "/robot_pose", self.callback_odom, 10
        )

        # Publishers
        self.pub_debug_img = self.create_publisher(CompressedImage, "arrow_detections/compressed", 10)
        self.pub_mask_red = self.create_publisher(CompressedImage, "arrow_mask_red/compressed", 10)
        self.pub_mask_blue = self.create_publisher(CompressedImage, "arrow_mask_blue/compressed", 10)
        self.pub_direction = self.create_publisher(String, "detection_fleche", 10)
        self.pub_red = self.create_publisher(PointStamped, "arrow_red", 10)
        self.pub_blue = self.create_publisher(PointStamped, "arrow_blue", 10)

    def callback_odom(self, msg: PoseStamped):
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        q = msg.pose.orientation
        _, _, self.robot_theta = quat2euler([q.w, q.x, q.y, q.z])
        self.has_odom = True

    def callback_img(self, msg: CompressedImage):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            return

        if self.camera_center is None:
            h, w = img.shape[:2]
            self.camera_center = np.array([w / 2, h / 2])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Masque rouge (double intervalle) + masque bleu
        mask_red_1 = cv2.inRange(hsv, self.RED_HSV_MIN_1, self.RED_HSV_MAX_1)
        mask_red_2 = cv2.inRange(hsv, self.RED_HSV_MIN_2, self.RED_HSV_MAX_2)
        mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
        mask_blue = cv2.inRange(hsv, self.BLUE_HSV_MIN, self.BLUE_HSV_MAX)

        # Nettoyage morphologique
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.OPEN_KERNEL_SIZE, self.OPEN_KERNEL_SIZE)
        )
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.CLOSE_KERNEL_SIZE, self.CLOSE_KERNEL_SIZE)
        )
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel_open)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel_close)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel_open)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel_close)

        img_debug = img.copy()
        text = ""

        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        cooldown = self.get_parameter("cooldown").value

        # Rouge → tourner a gauche
        area_red, centre_red = self.draw_largest_contour(img_debug, mask_red, (0, 0, 255))
        if self.is_arrow_valid(area_red, centre_red):
            text = "Tourner a gauche"
            if stamp_sec - self._last_red_pub >= cooldown:
                self.publish_arrow("rouge", self.pub_red, msg.header.stamp)
                self._last_red_pub = stamp_sec

        # Bleu → tourner a droite
        area_blue, centre_blue = self.draw_largest_contour(img_debug, mask_blue, (255, 0, 0))
        if self.is_arrow_valid(area_blue, centre_blue):
            text = "Tourner a droite"
            if stamp_sec - self._last_blue_pub >= cooldown:
                self.publish_arrow("bleue", self.pub_blue, msg.header.stamp)
                self._last_blue_pub = stamp_sec

        cv2.putText(img_debug, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Masques colorés : couleur détectée sur fond noir
        img_mask_red = np.zeros_like(img)
        img_mask_red[mask_red > 0] = (0, 0, 255)
        img_mask_blue = np.zeros_like(img)
        img_mask_blue[mask_blue > 0] = (255, 0, 0)

        try:
            self.pub_debug_img.publish(self.bridge.cv2_to_compressed_imgmsg(img_debug))
            self.pub_mask_red.publish(self.bridge.cv2_to_compressed_imgmsg(img_mask_red))
            self.pub_mask_blue.publish(self.bridge.cv2_to_compressed_imgmsg(img_mask_blue))
        except CvBridgeError:
            pass

    def is_arrow_valid(self, area, centre):
        if area is None or centre is None:
            return False
        if area < self.AREA_SEUIL:
            return False
        dist_centre = np.linalg.norm(self.camera_center - centre)
        if dist_centre > self.CENTER_RADIUS_SEUIL:
            return False
        return True

    def publish_arrow(self, color_name, publisher, stamp):
        msg_dir = String()
        msg_dir.data = color_name
        self.pub_direction.publish(msg_dir)

        if not self.has_odom:
            return
        point = PointStamped()
        point.header.stamp = stamp
        point.header.frame_id = "odom"
        point.point.x = self.robot_x + self.ARROW_FORWARD_OFFSET * np.cos(self.robot_theta)
        point.point.y = self.robot_y + self.ARROW_FORWARD_OFFSET * np.sin(self.robot_theta)
        publisher.publish(point)

    def draw_largest_contour(self, img, mask, color):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        max_contour = max(contours, key=cv2.contourArea)
        max_area = cv2.contourArea(max_contour)
        cv2.drawContours(img, [max_contour], -1, color, 3)
        centre = np.mean(max_contour, axis=0).astype(int).flatten()
        return max_area, centre


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    try:
        rclpy.spin(ArrowDetector())
    except KeyboardInterrupt:
        pass
