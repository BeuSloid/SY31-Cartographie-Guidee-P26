#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from .utils import declare_param


class ArrowClusterer(Node):
    """Clustering incremental par centroide des detections de fleches.

    Quand une fleche est detectee plusieurs fois (en passant devant),
    on fusionne les detections proches en un seul cluster represente
    par son centroide. Filtre les faux positifs (1-2 detections).
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

        # Publication reguliere
        self.create_timer(0.5, self.publish_markers)

    def cb_red(self, msg):
        self._add(msg, self.clusters_red)

    def cb_blue(self, msg):
        self._add(msg, self.clusters_blue)

    def _add(self, msg, clusters):
        pt = np.array([msg.point.x, msg.point.y])

        # Recherche du cluster existant le plus proche
        best, best_d = -1, np.inf
        for i, c in enumerate(clusters):
            d = np.linalg.norm(pt - c["centroid"])
            if d < best_d:
                best_d, best = d, i

        if best_d < self.D:
            # Mise a jour incrementale du centroide (moyenne glissante)
            c = clusters[best]
            c["count"] += 1
            c["centroid"] += (pt - c["centroid"]) / c["count"]
        else:
            # Nouveau cluster
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

"""Le nœud arrow_clusterer répond directement à la demande de l'énoncé : regrouper les flèches détectées en clusters 
et représenter chacune par un point unique. Comme le robot voit la même flèche sur plusieurs images successives en passant 
devant, arrow_detector émet de nombreux points spatialement très proches pour une seule flèche réelle. Nous regroupons ces 
détections par un clustering incrémental par centroïde : pour chaque nouveau point, nous cherchons le cluster existant le 
plus proche et, si la distance est inférieure à un seuil D, nous y rattachons le point en mettant à jour le centroïde du 
cluster par moyenne glissante (centroïde ← centroïde + (point − centroïde) / n) ; sinon nous créons un nouveau cluster. 
Cette formulation incrémentale évite de stocker l'historique complet des points tout en maintenant exactement la moyenne. 
Les flèches rouges et bleues sont traitées dans deux listes distinctes afin de ne jamais fusionner deux flèches de directions 
opposées. Enfin, un cluster n'est affiché (sous forme de sphère dans RViz, dans le repère odom) que s'il a accumulé 
un nombre minimal de détections, ce qui élimine les faux positifs vus une ou deux fois seulement. Le seuil de fusion D et 
le nombre minimal de détections sont des paramètres ROS réglables à chaud, ce qui nous a permis de les ajuster sans relancer 
le nœud lors des tests sur les fichiers bag.
Sur ce nœud, la logique de clustering (choix d'un regroupement incrémental par centroïde plutôt qu'un algorithme global 
type DBSCAN, séparation des deux couleurs, filtrage par nombre de détections) découle de notre propre analyse du besoin 
et des notions de clustering vues en TP. Nous n'avons sollicité l'IA que sur des aspects syntaxiques propres à ROS 2 : 
la construction des messages MarkerArray pour l'affichage RViz et l'emploi du marqueur DELETEALL pour effacer les anciennes 
sphères avant chaque republication. Lors de la relecture, nous avons harmonisé la lecture des paramètres pour qu'elle passe 
systématiquement par notre utilitaire declare_param (qui expose chaque paramètre comme un attribut du nœud mis à jour à chaud),
 plutôt que par un mélange d'accès directs qui nuisait à la cohérence du code entre nos différents nœuds."""