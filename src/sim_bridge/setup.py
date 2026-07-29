import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'sim_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='divagar',
    maintainer_email='n.divagar18@gmail.com',
    description='Simulation stand-in for uart_sender_node, driving Gazebo instead of a serial port',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim_actuation_bridge = sim_bridge.sim_actuation_bridge:main',
        ],
    },
)
