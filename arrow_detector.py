#!/usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, PoseStamped

from .utils import declare_param


class ArrowDetector(Node):
    """Détecte les flèches rouges et bleues dans l'image caméra.
    
    Valide une flèche selon trois critères :
    1. Aire suffisante (assez visible)
    2. Position au centre de l'image (devant le robot)
    3. Forme géométrique compatible avec une flèche
    """

    # Rouge : on couvre les deux intervalles H=0-10 ET H=160-180
    RED_HSV_MIN_1 = np.array([0, 100, 60], dtype=np.uint8)
    RED_HSV_MAX_1 = np.array([10, 255, 255], dtype=np.uint8)
    RED_HSV_MIN_2 = np.array([160, 100, 60], dtype=np.uint8)
    RED_HSV_MAX_2 = np.array([180, 255, 255], dtype=np.uint8)

    # Bleu : saturation min haute pour ignorer les vêtements/décors bleu pâle
    BLUE_HSV_MIN = np.array([95, 180, 80], dtype=np.uint8)
    BLUE_HSV_MAX = np.array([130, 255, 255], dtype=np.uint8)

    # Morphologie
    OPEN_KERNEL_SIZE = 15
    CLOSE_KERNEL_SIZE = 10

    # Filtres de validation
    AREA_SEUIL = 6000           # Aire min pour valider une flèche
    CENTER_RADIUS_SEUIL = 150   # px, distance max au centre de l'image

    # Filtres de forme (flèche)
    MIN_VERTICES = 5            # nb min de sommets après approxPolyDP
    MAX_VERTICES = 12           # nb max de sommets
    MIN_ASPECT_RATIO = 1.3      # forme allongée (longueur / largeur)
    MAX_ASPECT_RATIO = 3.5
    MIN_SOLIDITY = 0.5          # ratio aire / aire enveloppe convexe
    MAX_SOLIDITY = 0.95         # une flèche a une concavité (≠ rectangle plein)

    def __init__(self):
        super().__init__("arrow_detector")
        self.bridge = CvBridge()

        declare_param(self, "debug", False)

        # Position courante du robot
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.has_odom = False

        # Subscribers
        self.sub_img = self.create_subscription(
            Image, "/turtlecam/image_raw", self.callback_img, 10
)
        self.sub_odom = self.create_subscription(
            PoseStamped, "/robot_pose", self.callback_odom, 10
        )

        # Publishers
        self.pub_debug_img = self.create_publisher(Image, "arrow_detections/image", 10)
        self.pub_direction = self.create_publisher(String, "detection_fleche", 10)
        self.pub_red = self.create_publisher(PointStamped, "arrow_red", 10)
        self.pub_blue = self.create_publisher(PointStamped, "arrow_blue", 10)

        # Centre de l'image (calculé au premier message)
        self.camera_center = None

    """
    Callback odométrie : mémorise la position courante du robot
    """
    def callback_odom(self, msg: PoseStamped):
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        self.has_odom = True

    """
    Callback image : détection des flèches
    """
    def callback_img(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            return

        # Centre de l'image (calculé une seule fois)
        if self.camera_center is None:
            h, w = img.shape[:2]
            self.camera_center = np.array([w / 2, h / 2])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Masque ROUGE (double intervalle car H=0 et H=180 sont rouges)

        mask_red_1 = cv2.inRange(hsv, self.RED_HSV_MIN_1, self.RED_HSV_MAX_1)
        mask_red_2 = cv2.inRange(hsv, self.RED_HSV_MIN_2, self.RED_HSV_MAX_2)
        mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

        # Masque BLEU

        mask_blue = cv2.inRange(hsv, self.BLUE_HSV_MIN, self.BLUE_HSV_MAX)

        # Morphologie : nettoyer les masques

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

        # Détection et publication

        img_debug = img.copy()
        text = ""

        # Rouge faut tourner à gauche
        img_debug, area_red, centre_red, contour_red = self.draw_largest_contour(
            img_debug, mask_red, (0, 0, 255)
        )
        if self.is_arrow_valid(contour_red, area_red, centre_red):
            text = "Tourner a gauche"
            self.publish_arrow("rouge", self.pub_red, msg.header.stamp)

        # Bleu faut tourner à droite
        img_debug, area_blue, centre_blue, contour_blue = self.draw_largest_contour(
            img_debug, mask_blue, (255, 0, 0)
        )
        if self.is_arrow_valid(contour_blue, area_blue, centre_blue):
            text = "Tourner a droite"
            self.publish_arrow("bleue", self.pub_blue, msg.header.stamp)

        # Affichage du texte sur l'image
        cv2.putText(img_debug, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Publication de l'image annotée
        try:
            msg_debug = self.bridge.cv2_to_imgmsg(img_debug, "bgr8")
            self.pub_debug_img.publish(msg_debug)
        except CvBridgeError:
            pass

    # Validation d'une flèche (aire + position + forme)

    def is_arrow_valid(self, contour, area, centre):
        if area is None or centre is None or contour is None:
            return False

        # 1. Aire suffisante
        if area < self.AREA_SEUIL:
            return False

        # 2. Au centre de l'image
        dist_centre = np.linalg.norm(self.camera_center - centre)
        if dist_centre > self.CENTER_RADIUS_SEUIL:
            return False

        # 3. Forme de flèche
        if not self.is_arrow_shape(contour):
            return False

        return True

    # Filtre de forme : vérifie que le contour ressemble à une flèche

    def is_arrow_shape(self, contour):
        """Vérifie que le contour a une géométrie compatible avec une flèche.
        
        Critères :
        - Approximation polygonale : entre MIN_VERTICES et MAX_VERTICES sommets
        - Rapport d'aspect : forme allongée (longueur/largeur entre 1.3 et 3.5)
        - Solidité : présence d'une concavité (entre 0.5 et 0.95)
        """
        # Approximation polygonale
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        nb_sommets = len(approx)
        if nb_sommets < self.MIN_VERTICES or nb_sommets > self.MAX_VERTICES:
            return False

        # Rapport d'aspect (forme allongée)
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            return False
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio < self.MIN_ASPECT_RATIO or aspect_ratio > self.MAX_ASPECT_RATIO:
            return False

        # Solidité (concavité de la flèche)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return False
        solidity = cv2.contourArea(contour) / hull_area
        if solidity < self.MIN_SOLIDITY or solidity > self.MAX_SOLIDITY:
            return False

        return True

    # Publication d'une flèche détectée

    def publish_arrow(self, color_name, publisher, stamp):
        # Direction textuelle
        msg_dir = String()
        msg_dir.data = color_name
        self.pub_direction.publish(msg_dir)

        # Position spatiale (position du robot au moment de la détection)
        if not self.has_odom:
            return
        point = PointStamped()
        point.header.stamp = stamp
        point.header.frame_id = "odom"
        point.point.x = self.robot_x
        point.point.y = self.robot_y
        publisher.publish(point)

    # Détection du plus gros contour dans un masque
    
    def draw_largest_contour(self, img, mask, color):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img, None, None, None

        max_contour = max(contours, key=cv2.contourArea)
        max_area = cv2.contourArea(max_contour)
        cv2.drawContours(img, [max_contour], -1, color, 3)
        centre = np.mean(max_contour, axis=0).astype(int).flatten()
        cv2.circle(img, (centre[0], centre[1]), 5, color, -1)
        return img, max_area, centre, max_contour


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    try:
        rclpy.spin(ArrowDetector())
    except KeyboardInterrupt:
        pass