"""Standalone SmacPlannerHybrid tuning tool - see planner_tuning_client.py's
own docstring for the full workflow. Brings up ONLY map_server +
planner_server (+ this package's global_costmap_mapped.yaml,
local_nav's smac_planner_server.yaml) + RViz - no Gazebo, no robot, no
slam_toolbox/AMCL, no controller_server. Nothing here simulates physics or
sensors, so none of the CPU/real-time-factor overhead the full closed-loop
launches hit in a constrained environment applies - this is for fast
iteration on planner parameters specifically, not for testing driving
behavior.

use_sim_time is False throughout (no /clock source exists here - nothing
publishes it without Gazebo running).

Workflow: edit local_nav/config/smac_planner_server.yaml, restart this
launch, click "2D Pose Estimate" for a start point, click "2D Goal Pose"
for a goal - the console immediately prints success/failure, planning
time, and path length, and RViz's Path display shows the result. Repeat
goal clicks reuse the same start; a new "2D Pose Estimate" click changes
it.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_local_nav = get_package_share_directory('local_nav')
    pkg_buggy_mapping = get_package_share_directory('buggy_mapping')

    global_costmap_mapped_params = os.path.join(
        pkg_buggy_mapping, 'config', 'global_costmap_mapped.yaml')
    planner_server_params = os.path.join(pkg_local_nav, 'config', 'smac_planner_server.yaml')
    rviz_config = os.path.join(pkg_buggy_mapping, 'rviz', 'tune_planner.rviz')
    maps_dir = os.path.join(pkg_buggy_mapping, 'maps')

    map_name_arg = DeclareLaunchArgument(
        'map_name', default_value='track',
        description='Basename (no extension) under buggy_mapping/maps/ you map_saver_cli\'d - '
                    'loads <map_name>_map.yaml/.pgm.')
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')

    map_yaml_file = PathJoinSubstitution(
        [maps_dir, [LaunchConfiguration('map_name'), '_map.yaml']])

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'yaml_filename': map_yaml_file,
        }],
    )

    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[global_costmap_mapped_params, planner_server_params, {'use_sim_time': False}],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_tuning',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['map_server', 'planner_server'],
        }],
    )

    planner_tuning_client = Node(
        package='local_nav',
        executable='planner_tuning_client',
        name='planner_tuning_client',
        output='screen',
        parameters=[{'use_sim_time': False}],
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
        map_name_arg,
        rviz_arg,
        map_server,
        planner_server,
        lifecycle_manager,
        planner_tuning_client,
        rviz2,
    ])
