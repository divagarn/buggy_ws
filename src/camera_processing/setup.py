from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'camera_processing'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orin2',
    maintainer_email='orin2@todo.todo',
    description='ROS2 H265 Camera Processing Node - High Quality Video Capture for AGX Orin',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_h265_node = camera_processing.camera_h265_node:main',
            'camera_h265_decoder_node = camera_processing.camera_h265_decoder_node:main',
        ],
    },
)
