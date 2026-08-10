"""Real-hardware counterpart of localize.launch.py: real Velodyne + real
wheel feedback (buggy_sensors' real_sensors_bringup.launch.py) instead of
Gazebo, otherwise the same perception -> 2D scan -> slam_toolbox
(LOCALIZATION mode, loading a map saved by real_mapping_2d.launch.py or
mapping_2d.launch.py) pipeline as localize.launch.py.

"sync with the real world" here means physically placing the vehicle at
(or very near) wherever real_mapping_2d.launch.py / mapping_2d.launch.py
was actually started from when the loaded map was built - map_start_pose
in slam_toolbox_localization.yaml is [0,0,0], so the map's own origin IS
wherever the odom frame started during that mapping run (see that yaml's
own comment). There's no spawn_x/y/yaw to set here like the sim version -
you physically position the vehicle instead.

No controller_server/planner_server/carrot_path_publisher here at all -
pure perception + localization, nothing drives on its own, and no
actuation node either (no steering_uart_bridge/uart_sender) - this launch
cannot move the vehicle even if something tried to.

executable MUST be localization_slam_toolbox_node, not
async_slam_toolbox_node - same confirmed behavior difference as the sim
version (see that file's docstring) - this only matters for the manual
relocalization below, not ordinary scan-matching.

MANUAL POSE CORRECTION: same workflow as localize.launch.py - if the live
scan doesn't line up with the map in RViz, use RViz's "2D Pose Estimate"
tool. initialpose_to_slam_toolbox bridges that click into slam_toolbox's
real LOCALIZE_AT_POSE mechanism - the TF (map->odom) jumps immediately,
the vehicle does not need to move for the pose itself to take effect.

SAFETY: verify feedback_port matches the actual physical wiring before
running.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_buggy_sensors = get_package_share_directory('buggy_sensors')
    pkg_buggy_mapping = get_package_share_directory('buggy_mapping')

    slam_params = os.path.join(pkg_buggy_mapping, 'config', 'slam_toolbox_localization.yaml')
    p2l_params = os.path.join(pkg_buggy_mapping, 'config', 'pointcloud_to_laserscan.yaml')
    rviz_config = os.path.join(pkg_buggy_mapping, 'rviz', 'closed_loop_navigate.rviz')
    maps_dir = os.path.join(pkg_buggy_mapping, 'maps')

    feedback_port_arg = DeclareLaunchArgument('feedback_port', default_value='/dev/ttyUSB0')
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')
    ground_filter_arg = DeclareLaunchArgument(
        'ground_filter', default_value='true',
        description='false: skip segment_ground - self_hit_filter runs on raw '
                    '/velodyne_points instead. Recommended true for reliable localization.')
    map_name_arg = DeclareLaunchArgument(
        'map_name', default_value='my_map2',
        description='Basename (no extension) under buggy_mapping/maps/ saved via '
                    'real_mapping_2d.launch.py + /slam_toolbox/serialize_map - loads '
                    '<map_name>.data/.posegraph.')

    map_file_name = PathJoinSubstitution([maps_dir, LaunchConfiguration('map_name')])

    real_sensors_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_buggy_sensors, 'launch', 'real_sensors_bringup.launch.py')
        ),
        launch_arguments={
            'feedback_port': LaunchConfiguration('feedback_port'),
            'ground_filter': LaunchConfiguration('ground_filter'),
        }.items(),
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
        executable='localization_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params, {
            'use_sim_time': False,
            'map_file_name': map_file_name,
        }],
    )

    initialpose_to_slam_toolbox = Node(
        package='buggy_perception',
        executable='initialpose_to_slam_toolbox',
        name='initialpose_to_slam_toolbox',
        output='screen',
        parameters=[{
            'map_file_name': map_file_name,
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
        feedback_port_arg,
        rviz_arg,
        ground_filter_arg,
        map_name_arg,
        real_sensors_bringup,
        pointcloud_to_laserscan,
        slam_toolbox,
        initialpose_to_slam_toolbox,
        rviz2,
    ])
