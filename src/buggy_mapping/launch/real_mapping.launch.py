"""Real-hardware counterpart of mapping.launch.py: real Velodyne + real
wheel feedback instead of Gazebo, otherwise the same perception -> 2D scan
-> slam_toolbox (mapping mode) pipeline, plus the real UART actuation path
so you can drive it with teleop_twist_keyboard over /cmd_vel (same
contract real_navigate.launch.py's steering_uart_bridge/uart_sender_node
use) instead of physical RC control.

Drive it yourself, in a SEPARATE terminal:

    ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel

loopback:=true by default, same safety reasoning as real_navigate.launch.py:
exercises the full UART code path without writing to the real serial port
until you've confirmed /steering_angle and /odom look sane, then set
loopback:=false.

Drive the full loop, then save via slam_toolbox's own serialization
service (NOT nav2_map_server's map_saver_cli - see mapping.launch.py's
docstring / maps/README.md for why, and the exact command).

SAFETY: verify sender_port/feedback_port match the actual physical wiring
before running with loopback:=false.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_buggy_mapping = get_package_share_directory('buggy_mapping')
    pkg_velodyne_pointcloud = get_package_share_directory('velodyne_pointcloud')

    slam_params = os.path.join(pkg_buggy_mapping, 'config', 'slam_toolbox_mapping.yaml')
    p2l_params = os.path.join(pkg_buggy_mapping, 'config', 'pointcloud_to_laserscan.yaml')
    rviz_config = os.path.join(pkg_buggy_mapping, 'rviz', 'mapping.rviz')

    loopback_arg = DeclareLaunchArgument(
        'loopback', default_value='true',
        description='uart_sender_node loopback mode - true exercises the full code path '
                    'without writing to the real serial port. Set false only once '
                    '/steering_angle and /odom have been watched and look correct.')
    sender_port_arg = DeclareLaunchArgument('sender_port', default_value='/dev/ttyUSB0')
    feedback_port_arg = DeclareLaunchArgument('feedback_port', default_value='/dev/ttyUSB0')
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')

    loopback = ParameterValue(LaunchConfiguration('loopback'), value_type=bool)

    # Same VLP16 Hi-Res calibration/device_ip/min_range as
    # real_navigate.launch.py - see that file's comment for the full trail.
    velodyne_calibration = os.path.join(pkg_velodyne_pointcloud, 'params', 'VLP16_hires_db.yaml')
    velodyne_driver_node = Node(
        package='velodyne_driver',
        executable='velodyne_driver_node',
        name='velodyne_driver_node',
        output='screen',
        parameters=[{
            'device_ip': '',
            'gps_time': False,
            'time_offset': 0.0,
            'enabled': True,
            'read_once': False,
            'read_fast': False,
            'repeat_delay': 0.0,
            'frame_id': 'velodyne',
            'model': 'VLP16',
            'rpm': 1200.0,
            'port': 2368,
            'timestamp_first_packet': False,
            'use_sim_time': False,
        }],
    )
    velodyne_transform_node = Node(
        package='velodyne_pointcloud',
        executable='velodyne_transform_node',
        name='velodyne_transform_node',
        output='screen',
        parameters=[{
            'calibration': velodyne_calibration,
            'model': 'VLP16',
            'min_range': 0.4,
            'max_range': 130.0,
            'view_direction': 0.0,
            'fixed_frame': '',
            'target_frame': '',
            'organize_cloud': True,
            'use_sim_time': False,
        }],
    )

    velodyne_static_tf = Node(
        package='ground_segmentation',
        executable='velodyne_static_tf',
        name='velodyne_static_tf',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    segment_ground = Node(
        package='ground_segmentation',
        executable='segment_ground',
        name='segment_ground',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    self_hit_filter = Node(
        package='local_nav',
        executable='self_hit_filter',
        name='self_hit_filter',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    uart_bridge = Node(
        package='uart',
        executable='uart_bridge',
        name='uart_bridge',
        output='screen',
        parameters=[{
            'port': LaunchConfiguration('feedback_port'),
            'baud': 115200,
            'use_sim_time': False,
        }],
    )

    # odom->base_link TF, same integrator real_navigate.launch.py uses -
    # slam_toolbox needs this exactly like any other TF consumer here.
    wheel_odometry = Node(
        package='local_nav',
        executable='wheel_odometry',
        name='wheel_odometry',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    pointcloud_to_laserscan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        output='screen',
        parameters=[p2l_params, {'use_sim_time': False}],
        remappings=[
            ('cloud_in', '/non_ground_points_filtered'),
            ('scan', '/scan'),
        ],
    )

    slam_toolbox = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params, {'use_sim_time': False}],
    )

    steering_uart_bridge = Node(
        package='local_nav',
        executable='steering_uart_bridge',
        name='steering_uart_bridge',
        output='screen',
        parameters=[{
            'cmd_vel_topic': '/cmd_vel',
            'use_sim_time': False,
        }],
    )

    uart_sender = Node(
        package='uart',
        executable='uart_sender_node',
        name='uart_sender_node',
        output='screen',
        parameters=[{
            'port': LaunchConfiguration('sender_port'),
            'baudrate': 115200,
            'loopback': loopback,
            'use_sim_time': False,
        }],
    )

    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        parameters=[{'use_sim_time': False}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        loopback_arg,
        sender_port_arg,
        feedback_port_arg,
        rviz_arg,
        velodyne_driver_node,
        velodyne_transform_node,
        velodyne_static_tf,
        segment_ground,
        self_hit_filter,
        uart_bridge,
        wheel_odometry,
        pointcloud_to_laserscan,
        slam_toolbox,
        steering_uart_bridge,
        uart_sender,
        rviz2,
    ])
