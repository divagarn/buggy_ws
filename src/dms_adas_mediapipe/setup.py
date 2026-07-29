#!/usr/bin/env python3
from glob import glob
from setuptools import setup

package_name = 'dms_adas_mediapipe'

setup(
    name=package_name,
    version='0.1.0',
    package_dir={package_name: '.'},
    packages=[
        package_name,
        f'{package_name}.scripts',
        f'{package_name}.scripts.utils',
        f'{package_name}.utils',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        (f'share/{package_name}/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orin2',
    maintainer_email='orin2@todo.todo',
    description='Driver Monitoring System package migrated to ROS2 Python.',
    license='TODO',
    entry_points={
        'console_scripts': [
            'dms_node = dms_adas_mediapipe.scripts.dms_node:main',
            'dms_trial_node = dms_adas_mediapipe.scripts.dms_trial_node:main',
            'gstreamer_camera_node = dms_adas_mediapipe.scripts.gstreamer_camera_node:main',
        ],
    },
)
