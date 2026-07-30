"""Map-free local navigation: Gazebo -> segment_ground -> local costmap
-> planner_server (NavfnPlanner, real path planning ~5m ahead)
-> DWB (dwb_core::DWBLocalPlanner) -> ackermann_steering_controller.

Two fixes layered on top of each other here:
1. MPPI -> DWB: MPPI's CostCritic is a soft weighted penalty, and
   PathAlignCritic's pull toward the reference path (which could point
   straight through a wall) was winning that tug-of-war often enough to
   still hit walls even after the self-hit-filter fix. DWB's BaseObstacle
   critic hard-rejects any candidate trajectory whose footprint overlaps a
   lethal/inscribed costmap cell outright, so obstacle avoidance can't be
   out-voted by path-following desire. dwb_controller.yaml has the
   details; mppi_controller.yaml is kept for reference/rollback.
2. Naive straight-line reference path -> real planning: carrot_path_publisher
   used to just extend a straight line ahead of the vehicle and hand it to
   the controller as-is, trusting the controller to deviate around
   whatever was in the way. That's not path *planning*, and a straight
   line into a wall gave the controller a reference it fundamentally
   couldn't reconcile with obstacle avoidance. planner_server
   (nav2_navfn_planner) now computes a real, obstacle-avoiding path
   through the rolling local costmap out to ~5m ahead, and that computed
   path is what gets hand to DWB - not a straight line.

Unlike navigate.launch.py / gazebo_navigate.launch.py, this path does NOT
use obstacle_detector, steering_calculator_node, or topic_relay_node - the
local planner + controller handle obstacle avoidance directly from the
costmap (built from /non_ground_points_filtered), so the reactive
lateral-histogram chain isn't in the loop at all here.

Three tunable distances, all overridable from the command line
(`ros2 launch local_nav mppi_navigate.launch.py carrot_distance:=8.0 ...`):
  - carrot_distance: how far ahead the /plan goal (sent to NavfnPlanner)
    is placed.
  - scan_distance: how far ahead carrot_path_publisher looks when scanning
    for the clearest turn direction - kept larger than carrot_distance so
    a corner is noticed with enough room left to actually turn into it.
  - dwb_sim_time: DWB's own local trajectory rollout horizon (seconds);
    this is what /local_plan reflects each cycle - actual distance is
    roughly dwb_sim_time * current speed.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_buggy_description = get_package_share_directory('buggy_description')
    pkg_local_nav = get_package_share_directory('local_nav')

    costmap_params = os.path.join(pkg_local_nav, 'config', 'local_costmap.yaml')
    dwb_params = os.path.join(pkg_local_nav, 'config', 'dwb_controller.yaml')
    planner_costmap_params = os.path.join(pkg_local_nav, 'config', 'planner_costmap.yaml')
    planner_server_params = os.path.join(pkg_local_nav, 'config', 'planner_server.yaml')

    carrot_distance_arg = DeclareLaunchArgument('carrot_distance', default_value='9.0')
    scan_distance_arg = DeclareLaunchArgument('scan_distance', default_value='9.0')
    dwb_sim_time_arg = DeclareLaunchArgument('dwb_sim_time', default_value='2.5')

    # ParameterValue(..., value_type=float) coerces the launch argument's
    # string substitution to an actual float parameter - without it, DWB/
    # carrot_path_publisher would get a string and reject it with a
    # "Wrong parameter type" error at configure time.
    carrot_distance = ParameterValue(LaunchConfiguration('carrot_distance'), value_type=float)
    scan_distance = ParameterValue(LaunchConfiguration('scan_distance'), value_type=float)
    dwb_sim_time = ParameterValue(LaunchConfiguration('dwb_sim_time'), value_type=float)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_buggy_description, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': os.path.join(
                pkg_buggy_description, 'worlds', 'closed_loop_track2.world'),
            'spawn_x': '5.0',
            'spawn_y': '5.0',
        }.items(),
    )

    segment_ground = Node(
        package='ground_segmentation',
        executable='segment_ground',
        name='segment_ground',
        output='screen',
    )

    # ground_segmentation is the team's unmodified real-hardware code; it
    # does no self-hit filtering because the real Velodyne mount doesn't
    # clip the chassis the way it does in Gazebo. Interposing this filter
    # downstream keeps their node untouched while stopping the self-hit
    # blob from reaching the costmap.
    self_hit_filter = Node(
        package='local_nav',
        executable='self_hit_filter',
        name='self_hit_filter',
        output='screen',
    )

    # ackermann_steering_controller publishes its odom transform on
    # /ackermann_steering_controller/tf_odometry (tf2_msgs/TFMessage)
    # instead of the standard /tf topic, so tf2_ros buffer clients
    # (including the costmap above) can't see the odom->base_link
    # transform without this relay.
    tf_odom_relay = Node(
        package='local_nav',
        executable='tf_odom_relay',
        name='tf_odom_relay',
        output='screen',
    )

    # controller_server creates and owns its own "local_costmap" internally
    # by default (confirmed via its startup log: "[local_costmap.local_costmap]:
    # Creating Costmap" printed from within the controller_server process) -
    # a separate nav2_costmap_2d node is redundant, and that standalone
    # executable hardcodes its own internal node name ("costmap") regardless
    # of launch-time remapping, which doesn't match this config's
    # "local_costmap" naming at all. Passing both yaml files to
    # controller_server directly is the correct, simpler wiring.
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        # dwb_params first, then the dotted override - later entries in
        # this list win for any parameter both provide, which is how a
        # single nested value (FollowPath.sim_time) gets overridden
        # without needing a whole separate yaml file.
        parameters=[costmap_params, dwb_params, {'FollowPath.sim_time': dwb_sim_time}],
        remappings=[
            ('cmd_vel', '/ackermann_steering_controller/reference_unstamped'),
        ],
    )

    # planner_server creates and owns its own "global_costmap" internally,
    # same as controller_server does for "local_costmap" - a separate
    # rolling window fed by the same /non_ground_points_filtered, used only
    # to plan the ~5m-ahead reference path (see planner_costmap.yaml).
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[planner_costmap_params, planner_server_params],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_local_nav',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['planner_server', 'controller_server'],
        }],
    )

    carrot_path_publisher = Node(
        package='local_nav',
        executable='carrot_path_publisher',
        name='carrot_path_publisher',
        output='screen',
        parameters=[{
            'carrot_distance': carrot_distance,
            'scan_distance': scan_distance,
            'resend_period_sec': 1.0,
            'controller_id': 'FollowPath',
            'planner_id': 'GridBased',
        }],
    )

    # Real-hardware dry run: converts the exact same DWB Twist into the
    # real vehicle's actual contract (/steering_angle degrees,
    # /final_red_detected, /final_yellow_detected - what uart_sender_node.py
    # subscribes to). Harmless alongside Gazebo - it's a separate topic
    # pair that doesn't affect ackermann_steering_controller - and lets us
    # watch/validate what would actually be sent to the real cart before
    # ever running this on it.
    steering_uart_bridge = Node(
        package='local_nav',
        executable='steering_uart_bridge',
        name='steering_uart_bridge',
        output='screen',
        parameters=[{
            'cmd_vel_topic': '/ackermann_steering_controller/reference_unstamped',
        }],
    )

    return LaunchDescription([
        carrot_distance_arg,
        scan_distance_arg,
        dwb_sim_time_arg,
        gazebo,
        segment_ground,
        self_hit_filter,
        tf_odom_relay,
        controller_server,
        planner_server,
        lifecycle_manager,
        carrot_path_publisher,
        steering_uart_bridge,
    ])
