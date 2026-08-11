"""Real-hardware counterpart of navigate_algo_rrt_star.launch.py - same
relationship real_navigate_algo_theta_star.launch.py has to
navigate_algo_theta_star.launch.py. Launch buggy_mapping's
real_localize.launch.py FIRST (its own terminal), verify/correct pose in
RViz, THEN launch this file separately.

Adds controller_server (TEB) + planner_server (kept running, but ONLY for
its own lifecycle bring-up - see below) + rrt_star_planner_node (the
actual planner - a standalone action server, NOT a planner_server plugin,
see that file's own docstring) + carrot_path_publisher, plus the real
actuation path: steering_uart_bridge (Twist -> /steering_angle +
/final_red_detected, also now shows target distance in its telemetry
text - see that file's own recent change) -> uart_sender_node (writes the
actual serial packet).

Same action-name-separation as the sim version: both rrt_star_planner's
action_name and carrot_path_publisher's planner_action_name are set to
'rrt_star_planner/compute_path_to_pose' via real ROS PARAMETERS, not
launch-time remappings= (confirmed live in sim: plain remaps do not
rename an rclpy ActionServer/ActionClient).

costmap_topic defaults to /local_costmap/costmap here too (live sensor
data, rolling window, no static map layer - see navigate_algo_rrt_star
.launch.py's own docstring for the full trade-off/rationale, unchanged
for real hardware) - both rrt_star_planner_node and carrot_path_publisher
read from it. path_safety_check (default true) cancels + replans
immediately the moment the executing path crosses an inflated/lethal
cell in a fresh costmap. pose_graph_weight softly biases candidate
selection toward slam_toolbox's recorded pose graph (compensates for the
local costmap's missing static-layer awareness) - same three features as
the sim file, unchanged logic (all three live in carrot_path_publisher.py,
shared by every launch in this package).

target_speed_kmph (default 4.0, real presets are 2 or 4): the real UART
protocol has no continuous speed field (confirmed: uart_sender_node.py
only sends yellow/red/degree) - the vehicle instead runs one of two
FIXED firmware speed presets, selected by the yellow byte: yellow=1 ->
2 km/h preset, yellow=0 -> 4 km/h preset (red=1 overrides both and stops
the vehicle). steering_uart_bridge.py's target_speed_kmph parameter
(wired below from this same launch arg) picks which preset byte to send,
so target_speed_kmph here doubles as "which real speed preset to use."
Separately, TEB's own internal steering-angle optimization must be
computed assuming that SAME real speed, not its default 2.5 m/s (9 km/h)
tuned for a vehicle that doesn't exist - a mismatch there means the
atan2-derived steering angle steering_uart_bridge.py computes from TEB's
angular.z won't correspond to the curvature actually achieved at the
real vehicle's real (much slower, fixed) speed. So max_vel_x is also
overridden here to target_speed_kmph converted to m/s. No separate
"speed governor" NODE like the sim version has (nothing continuous to
govern - UART only ever receives a two-way preset flag, unlike Gazebo's
ackermann_steering_controller, which DOES take a commanded linear.x and
would otherwise drive at whatever TEB felt like).

auto_return_to_start:=false (default here, same as every other launch in
this package) - manual "2D Goal Pose" clicks, one leg at a time.

loopback:=true (default here) - uart_sender_node/uart_interface.py's
loopback gate: true exercises the full carrot_path_publisher -> TEB ->
steering_uart_bridge -> uart_sender_node path without ever writing to the
serial port. Watch /steering_angle, /steering_visualization (now
including target distance), and uart_sender_node's own log output,
confirm they look sane, THEN set loopback:=false only once ready for the
real vehicle to actually move.

SAFETY: verify sender_port matches the actual physical wiring before
setting loopback:=false. Confirm the planned path actually looks centered
and drivable in RViz first. This file has not been run against real
hardware yet - test with loopback:=true first, same as every other real-
hardware launch in this project.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_buggy_nav = get_package_share_directory('buggy_nav')

    costmap_params = os.path.join(pkg_buggy_nav, 'config', 'local_costmap.yaml')
    teb_params = os.path.join(pkg_buggy_nav, 'config', 'teb_controller.yaml')
    # Reused purely to give planner_server a valid GridBased plugin to load
    # during its own lifecycle bring-up - its action is never called and,
    # with costmap_topic defaulting to local, nothing reads its costmap
    # either anymore. Kept running anyway for now, matching the sim RRT*
    # file's current structure (see that file's own module docstring) -
    # dropping it entirely is a reasonable future cleanup, not done here to
    # avoid changing lifecycle-management behavior in the same change that
    # first ports these updates to real hardware.
    planner_server_params = os.path.join(pkg_buggy_nav, 'config', 'theta_star_planner_server.yaml')
    global_costmap_mapped_params = os.path.join(pkg_buggy_nav, 'config', 'global_costmap_mapped.yaml')
    rrt_star_params = os.path.join(pkg_buggy_nav, 'config', 'rrt_star_planner.yaml')

    carrot_distance_arg = DeclareLaunchArgument('carrot_distance', default_value='13.0')
    scan_distance_arg = DeclareLaunchArgument('scan_distance', default_value='13.0')
    loopback_arg = DeclareLaunchArgument(
        'loopback', default_value='true',
        description='uart_sender_node loopback mode - true exercises the full code path '
                    'without writing to the real serial port. Set false only once '
                    '/steering_angle and /odom have been watched and look correct.')
    sender_port_arg = DeclareLaunchArgument('sender_port', default_value='/dev/ttyUSB0')
    use_global_planner_arg = DeclareLaunchArgument(
        'use_global_planner', default_value='true',
        description='true (default here): RRT* plans against the local costmap (via its '
                    'own action, wired in for carrot_path_publisher - see this file\'s '
                    'docstring). false: fall back to the straight-segment technique instead.')
    centering_weight_arg = DeclareLaunchArgument('centering_weight', default_value='2.0')
    corridor_check_distance_arg = DeclareLaunchArgument('corridor_check_distance', default_value='3.0')
    goal_safety_margin_arg = DeclareLaunchArgument(
        'goal_safety_margin', default_value='1.0',
        description='Placed goals stop this many meters short of the first unsafe '
                    '(lethal/inflated/unknown) costmap cell, instead of right at its edge - '
                    'see carrot_path_publisher.py\'s own comment for why this was needed.')
    costmap_topic_arg = DeclareLaunchArgument(
        'costmap_topic', default_value='/local_costmap/costmap',
        description='Costmap both rrt_star_planner_node and carrot_path_publisher\'s '
                    'heading-picker read from - see this file\'s own docstring for the '
                    'trade-off. Set to /global_costmap/costmap to go back to full-map-aware '
                    'planning.')
    path_safety_check_arg = DeclareLaunchArgument(
        'path_safety_check', default_value='true',
        description='true (default): carrot_path_publisher cancels the current FollowPath '
                    'goal and replans immediately the moment a fresh costmap shows the '
                    'executing path crossing an inflated/lethal cell.')
    pose_graph_weight_arg = DeclareLaunchArgument(
        'pose_graph_weight', default_value='1.5',
        description='Soft bias toward slam_toolbox\'s recorded pose graph. 0 disables it.')
    pose_graph_max_influence_arg = DeclareLaunchArgument(
        'pose_graph_max_influence', default_value='5.0',
        description='Beyond this distance (m) from the nearest pose-graph vertex, the bias '
                    'contributes nothing.')
    target_speed_kmph_arg = DeclareLaunchArgument(
        'target_speed_kmph', default_value='4.0',
        description='The real buggy\'s actual fixed speed (2 or 4 km/h are the two real '
                    'firmware presets). Selects the UART yellow preset byte (<=3.0 -> '
                    'yellow=1/2kmph, else yellow=0/4kmph, via steering_uart_bridge) AND '
                    'overrides teb_controller.yaml\'s max_vel_x so TEB\'s own steering-angle '
                    'optimization is computed assuming that same real speed - see this '
                    'file\'s own docstring for the full explanation.')
    max_steering_rate_deg_s_arg = DeclareLaunchArgument(
        'max_steering_rate_deg_s', default_value='30.0',
        description='Slew-rate limit on steering_uart_bridge\'s output steering angle - caps '
                    'how many degrees/sec it is allowed to change by, so even a legitimate '
                    'sharp curvature change (new carrot point, replan) ramps in instead of '
                    'jumping instantly. Same fix/param as the sim launch\'s speed_governor.')
    lookahead_distance_arg = DeclareLaunchArgument(
        'lookahead_distance', default_value='4.0',
        description='How far ahead along /plan steering_uart_bridge looks for its '
                    'pure-pursuit steering target. Shorter = tighter path tracking but '
                    'jerkier, longer = smoother but cuts corners more. Same meaning/default '
                    'as the sim launch\'s speed_governor.')
    waypoint_step_m_arg = DeclareLaunchArgument('waypoint_step_m', default_value='1.5')
    max_waypoint_steps_arg = DeclareLaunchArgument('max_waypoint_steps', default_value='50')
    goal_xy_tolerance_arg = DeclareLaunchArgument('goal_xy_tolerance', default_value='0.5')
    use_lane_following_arg = DeclareLaunchArgument('use_lane_following', default_value='false')
    lane_simplify_epsilon_arg = DeclareLaunchArgument('lane_simplify_epsilon', default_value='0.3')
    auto_return_to_start_arg = DeclareLaunchArgument(
        'auto_return_to_start', default_value='false',
        description='false (default here): plain single-goal behavior, one "2D Goal '
                    'Pose" click per leg - manual goal-setting workflow. true: first '
                    'goal reached auto-chains a return-to-start leg (full closed loop).')

    carrot_distance = ParameterValue(LaunchConfiguration('carrot_distance'), value_type=float)
    scan_distance = ParameterValue(LaunchConfiguration('scan_distance'), value_type=float)
    loopback = ParameterValue(LaunchConfiguration('loopback'), value_type=bool)
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
    # km/h -> m/s at launch-substitution time (LaunchConfiguration values
    # are plain strings like "4.0" - PythonExpression evaluates the
    # concatenated string as a Python expression, e.g. "4.0 / 3.6").
    target_speed_ms = PythonExpression([LaunchConfiguration('target_speed_kmph'), ' / 3.6'])

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[costmap_params, teb_params, {
            'max_vel_x': ParameterValue(target_speed_ms, value_type=float),
            'use_sim_time': False,
        }],
        remappings=[
            ('cmd_vel', '/cmd_vel'),
        ],
    )

    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[global_costmap_mapped_params, planner_server_params, {'use_sim_time': False}],
        condition=IfCondition(LaunchConfiguration('use_global_planner')),
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_local_nav',
        output='screen',
        parameters=[{
            'use_sim_time': False,
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
            'use_sim_time': False,
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
            'action_name': 'rrt_star_planner/compute_path_to_pose',
            'costmap_topic': LaunchConfiguration('costmap_topic'),
            'use_sim_time': False,
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
            'odom_topic': '/odom',
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
            'use_sim_time': False,
        }],
    )

    steering_uart_bridge = Node(
        package='buggy_perception',
        executable='steering_uart_bridge',
        name='steering_uart_bridge',
        output='screen',
        parameters=[{
            'cmd_vel_topic': '/cmd_vel',
            'odom_topic': '/odom',
            # Same single source of truth TEB's max_vel_x uses - selects
            # the matching real speed preset via the yellow flag (see
            # steering_uart_bridge.py's own comment: yellow is NOT a
            # caution indicator, it's the real vehicle's speed selector).
            'target_speed_kmph': LaunchConfiguration('target_speed_kmph'),
            'max_steering_rate_deg_s': LaunchConfiguration('max_steering_rate_deg_s'),
            'lookahead_distance': LaunchConfiguration('lookahead_distance'),
            'use_sim_time': False,
        }],
    )

    uart_sender = Node(
        package='buggy_uart',
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

    return LaunchDescription([
        carrot_distance_arg,
        scan_distance_arg,
        loopback_arg,
        sender_port_arg,
        use_global_planner_arg,
        centering_weight_arg,
        corridor_check_distance_arg,
        goal_safety_margin_arg,
        costmap_topic_arg,
        path_safety_check_arg,
        pose_graph_weight_arg,
        pose_graph_max_influence_arg,
        target_speed_kmph_arg,
        max_steering_rate_deg_s_arg,
        lookahead_distance_arg,
        waypoint_step_m_arg,
        max_waypoint_steps_arg,
        goal_xy_tolerance_arg,
        use_lane_following_arg,
        lane_simplify_epsilon_arg,
        auto_return_to_start_arg,
        controller_server,
        planner_server,
        lifecycle_manager,
        lifecycle_manager_no_planner,
        rrt_star_planner,
        carrot_path_publisher,
        steering_uart_bridge,
        uart_sender,
    ])
