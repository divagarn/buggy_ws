"""RRT* counterpart of navigate_algo_theta_star.launch.py - same split
workflow: launch buggy_mapping's localize.launch.py FIRST (its own
terminal), verify/correct pose in RViz, THEN launch this file separately.

Adds controller_server (TEB, unchanged) + planner_server (kept running,
but ONLY for the /global_costmap/costmap it builds/publishes from
global_costmap_mapped.yaml - its own compute_path_to_pose action is never
targeted, see rrt_star_planner_node.py's own docstring for exactly why it
has to stay running anyway rather than being replaced outright) +
rrt_star_planner_node (this file's actual planner - a standalone action
server, NOT a planner_server plugin) + carrot_path_publisher.

Both nodes' action name is set via a real ROS PARAMETER
(rrt_star_planner's action_name, carrot_path_publisher's
planner_action_name), NOT a launch-time `remappings=` entry - confirmed
live that plain Node remaps do not rename an rclpy ActionServer OR
ActionClient (ros2 node info kept showing the un-remapped default name on
both ends after trying that first). Parametrizing the actual name on both
nodes is what actually separates this from planner_server's own
ALSO-running action of the same default name.

auto_return_to_start:=false (default here, same as the Theta* version) -
manual "2D Goal Pose" clicks, one leg at a time.

costmap_topic defaults to /local_costmap/costmap (controller_server's
rolling 30x30m window, live sensor data only - NO static map layer, see
local_costmap.yaml's own comment), not planner_server's /global_costmap/
costmap - both rrt_star_planner_node AND carrot_path_publisher's own
heading-picker read from this SAME topic, so what the planner searches
against and what the picker's raycasts see always agree. Trades away
awareness of walls outside current sensor range (no static layer to fall
back on) for costmap updates as fast as the sensor pipeline itself runs
(15Hz) instead of only at replan time - paired with carrot_path_publisher's
path-safety monitor (cancels + replans immediately, not just at the next
1Hz cycle, the moment the currently-executing path crosses an inflated/
lethal cell in a fresh costmap) and the pose-graph bias below, which
pulls candidate selection back toward where the vehicle was actually
driven during mapping - compensating for the lost static-layer awareness
by preferring known-good ground over unexplored-by-driving open space
instead.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_buggy_nav = get_package_share_directory('buggy_nav')

    costmap_params = os.path.join(pkg_buggy_nav, 'config', 'local_costmap.yaml')
    teb_params = os.path.join(pkg_buggy_nav, 'config', 'teb_controller.yaml')
    # Reused purely to give planner_server a valid GridBased plugin to load -
    # its action is never called in this launch (see module docstring).
    planner_server_params = os.path.join(pkg_buggy_nav, 'config', 'theta_star_planner_server.yaml')
    global_costmap_mapped_params = os.path.join(pkg_buggy_nav, 'config', 'global_costmap_mapped.yaml')
    rrt_star_params = os.path.join(pkg_buggy_nav, 'config', 'rrt_star_planner.yaml')

    carrot_distance_arg = DeclareLaunchArgument('carrot_distance', default_value='13.0')
    scan_distance_arg = DeclareLaunchArgument('scan_distance', default_value='13.0')
    use_global_planner_arg = DeclareLaunchArgument(
        'use_global_planner', default_value='true',
        description='true (default here): RRT* plans against the full saved map (via its '
                    'own action, remapped in for carrot_path_publisher - see this file\'s '
                    'docstring). false: fall back to the straight-segment technique instead.')
    centering_weight_arg = DeclareLaunchArgument('centering_weight', default_value='2.0')
    corridor_check_distance_arg = DeclareLaunchArgument('corridor_check_distance', default_value='3.0')
    goal_safety_margin_arg = DeclareLaunchArgument(
        'goal_safety_margin', default_value='1.0',
        description='Placed goals stop this many meters short of the first unsafe '
                    '(lethal/inflated/unknown) costmap cell, instead of right at its edge - '
                    'see carrot_path_publisher.py\'s own comment for why this was needed '
                    '(confirmed live via /carrot_marker sitting inside the inflated zone).')
    costmap_topic_arg = DeclareLaunchArgument(
        'costmap_topic', default_value='/local_costmap/costmap',
        description='Costmap both rrt_star_planner_node and carrot_path_publisher\'s '
                    'heading-picker read from. Default is the LOCAL costmap (live sensor '
                    'data, rolling window, no static map layer) - see this file\'s own '
                    'docstring for the trade-off. Set to /global_costmap/costmap to go '
                    'back to full-map-aware planning.')
    path_safety_check_arg = DeclareLaunchArgument(
        'path_safety_check', default_value='true',
        description='true (default): carrot_path_publisher cancels the current FollowPath '
                    'goal and replans immediately (not just at the next 1Hz cycle) the '
                    'moment a fresh costmap shows the executing path crossing an inflated/ '
                    'lethal cell.')
    pose_graph_weight_arg = DeclareLaunchArgument(
        'pose_graph_weight', default_value='1.5',
        description='Soft bias toward slam_toolbox\'s recorded pose graph (where the '
                    'vehicle was actually driven during mapping) - same weighted-bonus '
                    'pattern as centering_weight, applied in both reactive and goal-seeking '
                    'modes. 0 disables it entirely. Compensates for costmap_topic defaulting '
                    'to the local costmap (no static layer, no map-wide wall awareness).')
    pose_graph_max_influence_arg = DeclareLaunchArgument(
        'pose_graph_max_influence', default_value='5.0',
        description='Beyond this distance (m) from the nearest pose-graph vertex, the bias '
                    'contributes nothing (capped at 0, never a penalty - a soft preference, '
                    'not a hard exclusion).')
    waypoint_step_m_arg = DeclareLaunchArgument('waypoint_step_m', default_value='1.5')
    max_waypoint_steps_arg = DeclareLaunchArgument('max_waypoint_steps', default_value='50')
    goal_xy_tolerance_arg = DeclareLaunchArgument('goal_xy_tolerance', default_value='0.5')
    use_lane_following_arg = DeclareLaunchArgument('use_lane_following', default_value='false')
    lane_simplify_epsilon_arg = DeclareLaunchArgument('lane_simplify_epsilon', default_value='0.3')
    auto_return_to_start_arg = DeclareLaunchArgument(
        'auto_return_to_start', default_value='false',
        description='false (default here): plain single-goal behavior, one "2D Goal '
                    'Pose" click per leg. true: first goal reached auto-chains a '
                    'return-to-start leg (full closed loop).')
    target_speed_kmph_arg = DeclareLaunchArgument(
        'target_speed_kmph', default_value='4.0',
        description='Fixed speed the real buggy actually drives at (2 or 4 km/h are the '
                    'two real presets) - speed_governor forces the sim vehicle to this '
                    'exact speed whenever moving, since the real UART protocol has no '
                    'speed command at all (steering angle + stop/go flags only).')

    carrot_distance = ParameterValue(LaunchConfiguration('carrot_distance'), value_type=float)
    scan_distance = ParameterValue(LaunchConfiguration('scan_distance'), value_type=float)
    use_global_planner = ParameterValue(LaunchConfiguration('use_global_planner'), value_type=bool)
    centering_weight = ParameterValue(LaunchConfiguration('centering_weight'), value_type=float)
    corridor_check_distance = ParameterValue(LaunchConfiguration('corridor_check_distance'), value_type=float)
    goal_safety_margin = ParameterValue(LaunchConfiguration('goal_safety_margin'), value_type=float)
    pose_graph_weight = ParameterValue(LaunchConfiguration('pose_graph_weight'), value_type=float)
    pose_graph_max_influence = ParameterValue(
        LaunchConfiguration('pose_graph_max_influence'), value_type=float)
    waypoint_step_m = ParameterValue(LaunchConfiguration('waypoint_step_m'), value_type=float)
    max_waypoint_steps = ParameterValue(LaunchConfiguration('max_waypoint_steps'), value_type=int)
    goal_xy_tolerance = ParameterValue(LaunchConfiguration('goal_xy_tolerance'), value_type=float)
    use_lane_following = ParameterValue(LaunchConfiguration('use_lane_following'), value_type=bool)
    lane_simplify_epsilon = ParameterValue(LaunchConfiguration('lane_simplify_epsilon'), value_type=float)
    auto_return_to_start = ParameterValue(LaunchConfiguration('auto_return_to_start'), value_type=bool)

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[costmap_params, teb_params, {'use_sim_time': True}],
        remappings=[
            # NOT the final Gazebo topic anymore - speed_governor sits in
            # between and republishes the actual driven command onto
            # /ackermann_steering_controller/reference_unstamped. See that
            # node's own docstring for why (TEB's freely-varying speed has
            # no real-hardware equivalent - the real buggy only accepts a
            # steering angle, speed is fixed in the vehicle itself).
            ('cmd_vel', '/cmd_vel_raw'),
        ],
    )

    speed_governor = Node(
        package='buggy_nav',
        executable='speed_governor',
        name='speed_governor',
        output='screen',
        parameters=[{
            'target_speed_kmph': LaunchConfiguration('target_speed_kmph'),
            'wheelbase': 1.6,
            'max_steering_deg': 20.0,
            'use_sim_time': True,
        }],
        remappings=[
            ('cmd_vel_in', '/cmd_vel_raw'),
        ],
    )

    # Kept alive purely for its /global_costmap/costmap - see module
    # docstring. Its own compute_path_to_pose action is never targeted.
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[global_costmap_mapped_params, planner_server_params, {'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('use_global_planner')),
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
        condition=IfCondition(LaunchConfiguration('use_global_planner')),
    )
    lifecycle_manager_no_planner = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_local_nav',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['controller_server'],
        }],
        condition=UnlessCondition(LaunchConfiguration('use_global_planner')),
    )

    rrt_star_planner = Node(
        package='buggy_nav',
        executable='rrt_star_planner_node',
        name='rrt_star_planner',
        output='screen',
        parameters=[rrt_star_params, {
            # A plain Node-level `remappings=` entry does NOT rename this
            # node's ActionServer - confirmed live (`ros2 node info` still
            # showed the plain /compute_path_to_pose after trying that,
            # unlike a topic/service remap on the same node, which DOES
            # take effect for rclpy). Setting the actual action name via
            # parameter instead (see rrt_star_planner_node.py's own
            # action_name parameter) is what actually separates this from
            # planner_server's own ALSO-running action of the same
            # default name (see module docstring for why planner_server
            # has to stay running anyway).
            'action_name': 'rrt_star_planner/compute_path_to_pose',
            'costmap_topic': LaunchConfiguration('costmap_topic'),
        }],
        condition=IfCondition(LaunchConfiguration('use_global_planner')),
    )

    carrot_path_publisher = Node(
        package='buggy_nav',
        executable='carrot_path_publisher',
        name='carrot_path_publisher',
        output='screen',
        parameters=[{
            'carrot_distance': carrot_distance,
            'scan_distance': scan_distance,
            'resend_period_sec': 1.0,
            'controller_id': 'FollowPath',
            'planner_id': 'GridBased',
            'global_planner_name': 'RRTStar',
            'planner_action_name': 'rrt_star_planner/compute_path_to_pose',
            'use_global_planner': use_global_planner,
            'costmap_topic': LaunchConfiguration('costmap_topic'),
            'path_safety_check': LaunchConfiguration('path_safety_check'),
            'centering_weight': centering_weight,
            'corridor_check_distance': corridor_check_distance,
            'goal_safety_margin': goal_safety_margin,
            'pose_graph_weight': pose_graph_weight,
            'pose_graph_max_influence': pose_graph_max_influence,
            'waypoint_step_m': waypoint_step_m,
            'max_waypoint_steps': max_waypoint_steps,
            'goal_xy_tolerance': goal_xy_tolerance,
            'use_lane_following': use_lane_following,
            'lane_simplify_epsilon': lane_simplify_epsilon,
            'auto_return_to_start': auto_return_to_start,
            'use_sim_time': True,
        }],
    )

    return LaunchDescription([
        carrot_distance_arg,
        scan_distance_arg,
        use_global_planner_arg,
        centering_weight_arg,
        corridor_check_distance_arg,
        goal_safety_margin_arg,
        costmap_topic_arg,
        path_safety_check_arg,
        pose_graph_weight_arg,
        pose_graph_max_influence_arg,
        waypoint_step_m_arg,
        max_waypoint_steps_arg,
        goal_xy_tolerance_arg,
        use_lane_following_arg,
        lane_simplify_epsilon_arg,
        auto_return_to_start_arg,
        target_speed_kmph_arg,
        controller_server,
        speed_governor,
        planner_server,
        lifecycle_manager,
        lifecycle_manager_no_planner,
        rrt_star_planner,
        carrot_path_publisher,
    ])
