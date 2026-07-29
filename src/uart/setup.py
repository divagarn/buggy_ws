from setuptools import find_packages, setup

package_name = 'uart'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='hariharan.saravanan@srmtech.com',
    description='UART communication nodes for golf cart',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'uart_bridge = uart.uart_bridge:main',
            'uart_sender_node = uart.uart_sender_node:main',
            'topic_relay_node = uart.topic_relay_node:main',
        ],
    },
)
