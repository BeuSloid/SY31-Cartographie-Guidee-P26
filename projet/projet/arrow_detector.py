#!/usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, PoseStamped

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

    # Bleu
    BLUE_HSV_MIN = np.array([95, 150, 60], dtype=np.uint8)
    BLUE_HSV_MAX = np.array([130, 255, 255], dtype=np.uint8)

    # Taille du noyau de nettoyage morphologique
    KERNEL_SIZE = 5

    # Filtres de validation
    AREA_SEUIL = 1500           # aire min (px) pour valider une fleche
    CENTER_RADIUS_SEUIL = 200   # px, distance max au centre de l'image

    def __init__(self):
        super().__init__("arrow_detector")
        self.bridge = CvBridge()

        # Position courante du robot (mise a jour par /robot_pose)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.has_odom = False

        # Centre de l'image (calcule au premier message)
        self.camera_center = None

        # Subscribers
        self.sub_img = self.create_subscription(
            CompressedImage, "/turtlecam/image_raw/compressed", self.callback_img, 10
        )
        self.sub_odom = self.create_subscription(
            PoseStamped, "/robot_pose", self.callback_odom, 10
        )

        # Publishers
        self.pub_debug_img = self.create_publisher(CompressedImage, "arrow_detections/compressed", 10)
        self.pub_direction = self.create_publisher(String, "detection_fleche", 10)
        self.pub_red = self.create_publisher(PointStamped, "arrow_red", 10)
        self.pub_blue = self.create_publisher(PointStamped, "arrow_blue", 10)

    def callback_odom(self, msg: PoseStamped):
        # Memorise la position courante du robot
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        self.has_odom = True

    def callback_img(self, msg: CompressedImage):
        # Decompression de l'image
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            return

        # Centre de l'image (calcule une seule fois)
        if self.camera_center is None:
            h, w = img.shape[:2]
            self.camera_center = np.array([w / 2, h / 2])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Masque rouge (double intervalle) + masque bleu
        mask_red_1 = cv2.inRange(hsv, self.RED_HSV_MIN_1, self.RED_HSV_MAX_1)
        mask_red_2 = cv2.inRange(hsv, self.RED_HSV_MIN_2, self.RED_HSV_MAX_2)
        mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
        mask_blue = cv2.inRange(hsv, self.BLUE_HSV_MIN, self.BLUE_HSV_MAX)

        # Nettoyage : on enleve le bruit (open) et on bouche les trous (close)
        kernel = np.ones((self.KERNEL_SIZE, self.KERNEL_SIZE), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

        img_debug = img.copy()
        text = ""

        # Rouge -> tourner a gauche
        area_red, centre_red = self.draw_largest_contour(img_debug, mask_red, (0, 0, 255))
        if self.is_arrow_valid(area_red, centre_red):
            text = "Tourner a gauche"
            self.publish_arrow("rouge", self.pub_red, msg.header.stamp)

        # Bleu -> tourner a droite
        area_blue, centre_blue = self.draw_largest_contour(img_debug, mask_blue, (255, 0, 0))
        if self.is_arrow_valid(area_blue, centre_blue):
            text = "Tourner a droite"
            self.publish_arrow("bleue", self.pub_blue, msg.header.stamp)

        # Texte sur l'image de debug
        cv2.putText(img_debug, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        try:
            msg_debug = self.bridge.cv2_to_compressed_imgmsg(img_debug)
            self.pub_debug_img.publish(msg_debug)
        except CvBridgeError:
            pass

    def is_arrow_valid(self, area, centre):
        # Aucun contour trouve
        if area is None or centre is None:
            return False

        # 1. Aire suffisante (fleche assez proche)
        if area < self.AREA_SEUIL:
            return False

        # 2. Au centre de l'image (fleche droit devant le robot)
        dist_centre = np.linalg.norm(self.camera_center - centre)
        if dist_centre > self.CENTER_RADIUS_SEUIL:
            return False

        return True

    def publish_arrow(self, color_name, publisher, stamp):
        # Direction textuelle
        msg_dir = String()
        msg_dir.data = color_name
        self.pub_direction.publish(msg_dir)

        # Position spatiale = position du robot au moment de la detection
        if not self.has_odom:
            return
        point = PointStamped()
        point.header.stamp = stamp
        point.header.frame_id = "odom"
        point.point.x = self.robot_x
        point.point.y = self.robot_y
        publisher.publish(point)

    def draw_largest_contour(self, img, mask, color):
        # Trouve le plus gros contour du masque (la fleche)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        max_contour = max(contours, key=cv2.contourArea)
        max_area = cv2.contourArea(max_contour)

        # Dessin sur l'image de debug
        cv2.drawContours(img, [max_contour], -1, color, 3)
        centre = np.mean(max_contour, axis=0).astype(int).flatten()
        cv2.circle(img, (centre[0], centre[1]), 5, color, -1)

        return max_area, centre


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    try:
        rclpy.spin(ArrowDetector())
    except KeyboardInterrupt:
        pass



"""Le nœud arrow_detector repère les flèches de couleur dans le flux caméra. Nous travaillons dans l'espace colorimétrique HSV plutôt que BGR, car il sépare la teinte de la luminosité et rend la détection plus robuste aux variations d'éclairage du labyrinthe. La couleur rouge occupant les deux extrémités du cercle des teintes (autour de H=0 et de H=180), nous construisons son masque en réunissant deux intervalles, tandis que le bleu n'en demande qu'un. Les masques sont nettoyés par ouverture morphologique (suppression du bruit ponctuel) puis fermeture (rebouchage des trous). Pour chaque couleur, nous extrayons le plus grand contour et validons une flèche selon trois critères simples et explicables : une aire minimale (la flèche doit être assez proche), une position proche du centre de l'image (la flèche est droit devant le robot) et bien sûr la correspondance de couleur. Une flèche rouge commande de tourner à gauche, une bleue à droite. La position publiée est celle du robot au moment de la détection, lue sur /robot_pose ; c'est cette information de pose qui permettra ensuite de regrouper spatialement les détections. Nous avons délibérément écarté un filtrage géométrique plus fin de la forme de la flèche (nombre de sommets, solidité, rapport d'aspect) : testé sur les fichiers bag, il rejetait trop souvent des flèches réelles à cause du bruit de compression de l'image, et ajoutait une fragilité de réglage sans gain de fiabilité.
Concernant l'usage de l'IA sur ce nœud, nous nous sommes appuyés sur les notions de traitement d'image vues en TP (seuillage HSV, opérations morphologiques, détection de contours OpenCV) et n'avons sollicité l'assistance que sur des points syntaxiques de cv_bridge : en particulier le passage du décodage d'images brutes (imgmsg_to_cv2) au décodage d'images compressées (compressed_imgmsg_to_cv2), nécessaire car le fichier bag ne fournit que le topic /turtlecam/image_raw/compressed. La logique de validation (les trois critères couleur/aire/centrage) et la convention rouge-gauche/bleu-droite relèvent de nos propres choix. L'analyse critique nous a par ailleurs conduits à supprimer un filtre de forme initialement envisagé, après avoir constaté quantitativement sur les bags qu'il dégradait le taux de bonne détection."""
