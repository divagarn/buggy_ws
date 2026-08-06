"""Standalone GLOBAL (SmacPlannerHybrid) + LOCAL (TEB/controller_server)
planner tuning tool - no Gazebo, no robot, no slam_toolbox/AMCL. Brings up
map_server + planner_server + controller_server + kinematic_sim (a plain
unicycle integrator standing in for a real/simulated robot - see that
file's own docstring for why it's deliberately NOT a physics simulator)
+ RViz. None of Gazebo's per-test startup time or this project's observed
CPU/real-time-factor overhead in a constrained environment applies here,
since nothing is actually simulating physics or sensors - this is for
fast iteration on planner/controller parameters, not for testing driving
behavior against real obstacles (the local costmap's obstacle_layer has
no live sensor data feeding it here - there is no LiDAR/radar - so it
will only ever show inflation around nothing; this exercises TEB's
trajectory tracking against a fake robot, not its obstacle avoidance).

use_sim_time is False throughout (no /clock source exists here - nothing
publishes it without Gazebo running) - both local_costmap.yaml and
teb_controller.yaml default use_sim_time: true, overridden back to False
per-node below the same way real_closed_loop_navigate.launch.py does for
real hardware, and for the same reason (no /clock means a use_sim_time:
true node's clock never advances).

COMBINED WORKFLOW: click "2D Pose Estimate" for a start (this ALSO resets
kinematic_sim's fake robot to that pose - see that node's docstring),
click "2D Goal Pose" for a goal. planner_tuning_client computes the
global plan (as it always did) and publishes it on /tuning_plan;
follow_path_trigger immediately forwards that same path to
controller_server's FollowPath action, which drives kinematic_sim's fake
robot along it - watch the green arrow marker move in RViz and the local
costmap roll with it. Edit smac_planner_server.yaml and/or
teb_controller.yaml, restart this launch, repeat. Console logs cover both
stages: planner_tuning_client's success/failure/planning_time for the
global plan, follow_path_trigger's periodic distance_to_goal/speed
feedback and final result for the local controller.

A static identity map->odom transform is published below - REQUIRED, not
optional decoration. planner_server transforms the given start/goal poses
(frame_id "map") into the costmap's own operating frame before invoking
the planner, regardless of use_start/use_sim_time - and
global_costmap_mapped.yaml sets global_frame: odom (see that file's own
comment for why), so without SOME map->odom transform already existing,
that lookup fails outright ("Could not transform the start or goal pose
in the costmap frame") and every single plan is aborted before
SmacPlannerHybrid ever runs. Identity is fine here - nothing in this
standalone tool cares about real odometry, the frame just needs to exist.
odom->base_link, unlike map->odom, is NOT static - kinematic_sim
broadcasts that dynamically as the fake robot actually moves.
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
    local_costmap_params = os.path.join(pkg_local_nav, 'config', 'local_costmap.yaml')
    teb_params = os.path.join(pkg_local_nav, 'config', 'teb_controller.yaml')
    rviz_config = os.path.join(pkg_buggy_mapping, 'rviz', 'tune_planner.rviz')
    maps_dir = os.path.join(pkg_buggy_mapping, 'maps')

    map_name_arg = DeclareLaunchArgument(
        'map_name', default_value='track',
        description='Basename (no extension) under buggy_mapping/maps/ you map_saver_cli\'d - '
                    'loads <map_name>_map.yaml/.pgm.')
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')

    map_yaml_file = PathJoinSubstitution(
        [maps_dir, [LaunchConfiguration('map_name'), '_map.yaml']])

    # REQUIRED, not decoration - see this file's own docstring for why
    # planner_server needs a map->odom transform to exist at all here.
    map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_identity',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    )

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

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[local_costmap_params, teb_params, {'use_sim_time': False}],
        remappings=[
            ('cmd_vel', '/cmd_vel'),
        ],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_tuning',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['map_server', 'planner_server', 'controller_server'],
        }],
    )

    kinematic_sim = Node(
        package='local_nav',
        executable='kinematic_sim',
        name='kinematic_sim',
        output='screen',
        parameters=[{
            'cmd_vel_topic': '/cmd_vel',
            'odom_topic': '/odom',
            'use_sim_time': False,
        }],
    )

    planner_tuning_client = Node(
        package='local_nav',
        executable='planner_tuning_client',
        name='planner_tuning_client',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    follow_path_trigger = Node(
        package='local_nav',
        executable='follow_path_trigger',
        name='follow_path_trigger',
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
        map_to_odom,
        map_server,
        planner_server,
        controller_server,
        lifecycle_manager,
        kinematic_sim,
        planner_tuning_client,
        follow_path_trigger,
        rviz2,
    ])
