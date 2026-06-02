# SY31-Cartographie-Guidee-P26

Projet ROS 2 de cartographie guidée d'un labyrinthe à l'aide d'un TurtleBot3.
Le robot maintient une odométrie fusionnée (encodeurs + gyromètre) et accumule les
points LiDAR dans un repère fixe pour construire une carte 2D. La caméra détecte des
flèches de couleur indiquant au robot de tourner à gauche ou à droite ; ces flèches sont
regroupées en clusters et affichées sur la carte.

---

## Contenu du package

```
projet/
├── launch/
│   └── labyrinthe.launch.xml   # Fichier de lancement principal
├── projet/
│   ├── odom_node.py            # Odométrie (encodeurs + gyromètre)
│   ├── tf_publisher.py         # Publication TF dynamique odom → base_scan
│   ├── transformer.py          # LaserScan → PointCloud2 (repère local)
│   ├── intensity_filter.py     # Filtre d'intensité LiDAR
│   ├── clusterer.py            # Clustering LiDAR (repère odom)
│   ├── map_transformer.py      # Accumulation LiDAR en repère global (odom)
│   ├── arrow_detector.py       # Détection flèches rouges/bleues (caméra)
│   ├── arrow_clusterer.py      # Clustering incrémental des flèches détectées
│   └── utils.py                # Fonctions partagées (PointCloud2, Markers, params)
├── rviz/
│   └── projet.rviz             # Configuration RViz2 prête à l'emploi
└── README.md
```

---

## Dépendances

| Paquet | Usage |
|---|---|
| `rclpy` | Framework ROS 2 Python |
| `opencv-python` (`cv2`) | Traitement d'image pour la détection de flèches |
| `cv_bridge` | Conversion messages ROS ↔ images OpenCV |
| `numpy` | Calculs numériques |
| `transforms3d` | Conversion quaternion ↔ angles d'Euler |
| `sensor_msgs_py` | Lecture/écriture de PointCloud2 en numpy |
| `turtlebot3_msgs` | Lecture des encodeurs (`SensorState`) |
| `image_transport` | Décompression du flux caméra compressé |

---

## Architecture des nœuds

```
/imu            ──┐
                  ├──► odom_node ──► /robot_pose ──► tf_publisher (TF odom→base_scan)
/sensor_state   ──┘                      │
                                         │
/scan ──► transformer ──► intensity_filter ──► clusterer ──► /clusters
/scan ──────────────────────────────────────────────────────► /global_points
                                    (map_transformer, utilise /robot_pose)

/turtlecam/image_raw/compressed ──► [decompresseur] ──► /turtlecam/image_raw
                                                              │
                                                    arrow_detector ──► /arrow_red
                                                         │         ──► /arrow_blue
                                                         └──► /detection_fleche
                                                    arrow_clusterer ──► /arrow_clusters
```

**Topics publiés utiles dans RViz2 :**

| Topic | Type | Description |
|---|---|---|
| `/global_points` | `PointCloud2` | Carte LiDAR accumulée (repère `odom`) |
| `/clusters` | `PointCloud2` | Objets détectés, colorés par cluster (repère `odom`) |
| `/arrow_clusters` | `MarkerArray` | Position des flèches sur la carte (repère `odom`) |
| `/robot_pose` | `PoseStamped` | Position estimée du robot |
| `/arrow_detections/image` | `Image` | Flux caméra annoté (debug) |
| `/detection_fleche` | `String` | Direction courante : `"rouge"` ou `"bleue"` |

---

## Instructions de lancement

### 1. Compiler le package

```bash
cd ~/SY31/sy31_ws
colcon build --packages-select projet
source install/setup.bash
```

### 2. Lancer les nœuds

```bash
ros2 launch projet labyrinthe.launch.xml
```

### 3. Rejouer le fichier bag

Dans un autre terminal (sourcer l'environnement au préalable) :

```bash
ros2 bag play <chemin/vers/le/bag> --loop
```

L'option `--loop` rejoue le bag en boucle pour faciliter les tests.

### 4. Ouvrir RViz2

```bash
rviz2 -d install/projet/share/projet/rviz/projet.rviz
```

La configuration RViz2 fournie affiche automatiquement `global_points`, `clusters` et
`arrow_clusters` avec le repère fixe `odom`.

> **Note :** Ne pas utiliser `base_scan` comme Fixed Frame dans RViz2. Notre
> `tf_publisher` publie la TF `odom → base_scan` dynamiquement, ce qui permet
> d'utiliser `odom` et d'obtenir une carte globale stable.

---

## Réglage des paramètres à chaud

Les paramètres des nœuds sont modifiables sans relancement via `ros2 param set` :

```bash
# Distance max entre 2 points LiDAR pour appartenir au même cluster (défaut : 0.1 m)
ros2 param set /clusterer distMax2Pts 0.15

# Nombre minimum de détections pour valider une flèche (défaut : 3)
ros2 param set /arrow_clusterer min_detections 5

# Seuil d'intensité LiDAR (défaut : 20.0)
ros2 param set /intensity_filter intensity_threshold 25.0
```

---

## Utilisation des LLM

L'assistant Claude Code (Anthropic) a été utilisé pour :
- Corriger deux bugs d'odométrie dans `odom_node.py` (initialisation des encodeurs et
  réinitialisation lors du rebouclage du bag)
- Corriger le décalage entre `clusters` et `global_points` dans RViz2 en ajoutant la
  transformation `base_scan → odom` directement dans `clusterer.py`
- Rédiger ce README

Toutes les modifications ont été relues et comprises par les membres du binôme.
