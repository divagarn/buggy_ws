from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ground_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'rviz'),
            glob(os.path.join('rviz', '*.rviz'))),
    ],
    install_requires=['setuptools', 'numpy', 'scikit-learn', 'scipy'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='hariharan.saravanan@srmtech.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segment_ground = ground_segmentation.segment_ground:main',
            'front_view_occupancy_map = ground_segmentation.front_view_occupancy_map:main',
            'obstacle_detector = ground_segmentation.obstacle_detector:main',
            'velodyne_static_tf = ground_segmentation.velodyne_static_tf:main',
            'steering_calculator_node = ground_segmentation.steering_calculator_node:main',
        ],
    },
)
