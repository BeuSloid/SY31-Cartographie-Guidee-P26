import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'projet'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml') + glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='benoit',
    maintainer_email='caibenenoit12@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
          'trans = projet.transformer:main',
          'inten = projet.intensity_filter:main',
          'cluster = projet.clusterer:main',
          'odom_node = projet.odom_node:main',
          'map_transformer = projet.map_transformer:main',
          'tf_publisher = projet.tf_publisher:main',
          'arrow_detector = projet.arrow_detector:main',
          'arrow_clusterer = projet.arrow_clusterer:main',
        ],
    },
)
