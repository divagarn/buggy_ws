from glob import glob

from setuptools import find_packages, setup

package_name = 'buggy_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='divagar',
    maintainer_email='n.divagar18@gmail.com',
    description='Ground segmentation + self-hit filtering + tf_odom_relay + real-hardware TF/odometry/actuation bridges',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segment_ground = buggy_perception.segment_ground:main',
            'self_hit_filter = buggy_perception.self_hit_filter:main',
            'tf_odom_relay = buggy_perception.tf_odom_relay:main',
            'initialpose_to_slam_toolbox = buggy_perception.initialpose_to_slam_toolbox:main',
            'velodyne_static_tf = buggy_perception.velodyne_static_tf:main',
            'chcnav_static_tf = buggy_perception.chcnav_static_tf:main',
            'wheel_odometry = buggy_perception.wheel_odometry:main',
            'steering_uart_bridge = buggy_perception.steering_uart_bridge:main',
        ],
    },
)
