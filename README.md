# SY31-Cartographie-Guidee-P26

Ce projet s'inscrit dans le cadre de l'UV SY31 Capteurs pour les systèmes intelligents, et a pour objectif de mettre en œuvre une solution robotique combinant perception et localisation pour un robot mobile évoluant dans un labyrinthe. Le sujet traité est celui de la cartographie guidée : un TurtleBot3 Burger est guidé manuellement à l'intérieur d'un labyrinthe à l'aide de commandes clavier, tout en construisant progressivement une carte de son environnement.

Le robot utilise une fusion d'informations provenant de ses capteurs proprioceptifs (encodeurs de roue et gyroscope) pour estimer sa position, et de son LiDAR pour générer une carte basée sur l'accumulation de points dans un repère fixe. En parallèle, une caméra embarquée permet de détecter des flèches colorées disposées dans le labyrinthe. Cette reconnaissance visuelle assiste l'opérateur dans ses choix directionnels pendant la commande.

Le projet a été implémenté sous ROS2 (Robot Operating System 2) en Python, en s'appuyant sur une architecture modulaire distribuée en plusieurs nœuds spécialisés. Nous avons calculé nous-mêmes l'odométrie à partir des capteurs proprioceptifs.

---
Voici le contenu du package après avoir dézippé le projet :
## Contenu du package

```
[dossier] projet/
    [dossier] launch/
        [fichier] labyrinthe.launch.xml   # Fichier de lancement principal
    [dossier] projet/
        [fichier] odom_node.py            # Odométrie (encodeurs + gyromètre)
        [fichier] tf_publisher.py         # Publication TF dynamique odom → base_scan
        [fichier] transformer.py          # LaserScan → PointCloud2 (repère local)
        [fichier] intensity_filter.py     # Filtre d'intensité LiDAR
        [fichier] clusterer.py            # Clustering LiDAR (repère odom)
        [fichier] map_transformer.py      # Accumulation LiDAR en repère global (odom)
        [fichier] arrow_detector.py       # Détection flèches rouges/bleues (caméra)
        [fichier] arrow_clusterer.py      # Clustering incrémental des flèches détectées
        [fichier] utils.py                # Fonctions partagées (PointCloud2, Markers, params)
        [fichier] README.md
        [dossier] labyrinthe/
            [fichier] labyrinthe_0.mcap   # Fichier bag fourni par Mr.Lima
    [dossier] rviz/
        [fichier] projet.rviz             # Configuration RViz2 prête à l'emploi
    [dossier] ressource/
    [dossier] test/
    [fichier] setup.py
    [fichier] package.xml
```


---

## Architecture des nœuds

```
![Architecture du projet](docs/architecture.png)
```

---

## Instructions de lancement

### 1. Zenohd
Lancer dans le premier terminal:
```bash
zenohd
```

### 2. Jouer le fichier bag
Dans un deuxième terminal, lancer le fichier bag du projet (nous avons également nos propres fichier bag, mais par soucis de simplicité nous prenons les fichiers bag fournis par Mr.Lima).

```bash
ros2 bag play <chemin/vers/le/bag> --loop #par ex ici : ros2 bag play /projet/labyrinthe/
```

### 3. Compiler le package
Dans un nouveau terminal (dans le même dossier qu'on a dezippé):
```bash
colcon build --packages-select projet # ou: colcon build
source install/setup.bash
```

### 4. Lancer les nœuds
Dans le même terminal:
```bash
ros2 launch projet labyrinthe.launch.xml # pour lancer : ros2 launch (nom du package) (nom du fichier launch)
```


### 5. Ouvrir RViz2

```bash
rviz2 -d install/projet/share/projet/rviz/projet.rviz
```

La configuration RViz2 fournie affiche automatiquement `global_points`, `clusters` et
`arrow_clusters` avec le repère fixe `odom`.

> **Note :** Ne pas utiliser `base_scan` comme Fixed Frame dans RViz2. Notre
> `tf_publisher` publie la TF `odom → base_scan` dynamiquement, ce qui permet
> d'utiliser `odom` et d'obtenir une carte globale stable.

---

### 6. Ouvrir rqt
Dans un nouveau terminal:
```bash
rqt
```
---
