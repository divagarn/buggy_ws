import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'local_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='divagar',
    maintainer_email='n.divagar18@gmail.com',
    description='Map-free local costmap + MPPI (Ackermann) obstacle avoidance',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'carrot_path_publisher = local_nav.carrot_path_publisher:main',
            'tf_odom_relay = local_nav.tf_odom_relay:main',
            'self_hit_filter = local_nav.self_hit_filter:main',
            'steering_uart_bridge = local_nav.steering_uart_bridge:main',
            'wheel_odometry = local_nav.wheel_odometry:main',
            'radar_sim = local_nav.radar_sim:main',
            'localization_creep = local_nav.localization_creep:main',
            'pause_resume_console = local_nav.pause_resume_console:main',
            'posegraph_align_creep = local_nav.posegraph_align_creep:main',
            'initialpose_to_slam_toolbox = local_nav.initialpose_to_slam_toolbox:main',
            'planner_tuning_client = local_nav.planner_tuning_client:main',
        ],
    },
)
