"""Phase 2 of map-based navigation (see mapping.launch.py for phase 1):
Gazebo -> segment_ground -> self_hit_filter -> pointcloud_to_laserscan ->
nav2_map_server + AMCL (GLOBAL localization against the occupancy grid you
map_saver_cli'd from phase 1's saved pose graph) -> local_nav's usual
controller_server (local_costmap.yaml/teb_controller.yaml, unchanged) +
planner_server, but planner_server's global costmap now uses this
package's global_costmap_mapped.yaml (adds a static_layer reading AMCL's
localization map on top of the same live obstacle_layer/inflation_layer
local_nav always used) instead of local_nav's own planner_costmap.yaml -
see that config file's own comment for why (short version: SmacPlannerHybrid
can now see the WHOLE saved map, not just the live rolling sensor window,
which is what actually makes "set a goal anywhere in the loop" work) and
its documented limitation (costmap stays in odom frame, not map frame - see
that same comment for what that does and doesn't cost you).

Why AMCL instead of slam_toolbox's localization mode (which is what this
launch used to run): slam_toolbox only scan-matches within a small window
around a pose you hand it up front (map_start_pose in
slam_toolbox_localization.yaml) - fine if the vehicle always re-enters at
the exact same known point, useless if you want to place it ANYWHERE in
the map and have it figure out where it is, which is the actual goal here.
AMCL is a particle filter: reinitialize_global_localization (called
automatically below, global_localization_delay seconds after launch)
scatters particles across the WHOLE map's free space, and they converge
via odometry + scan matches as the vehicle moves - no known starting pose
required. Trade-off: nav2_amcl has no dedicated Ackermann/bicycle motion
model (see amcl.yaml's own comment), and convergence takes real motion,
not instant like a hardcoded start pose.

"Real motion" is doing a lot of work in that last sentence - on a
geometrically repetitive track (a near-regular hexagonal loop, say),
several spots can produce a nearly identical STATIONARY scan, so a
vehicle that sits still can converge confidently onto the wrong one.
localization_creep (creep_start_delay seconds after launch, so it starts
after particles have been scattered) drives the vehicle slowly forward -
obstacle-checked against live /scan - specifically to accumulate real,
distinct viewpoints that disambiguate this, until it's covered
min_creep_distance AND AMCL's covariance has actually dropped, or
max_creep_time_sec elapses as a safety cap. See localization_creep.py's
own docstring for the full reasoning and its known limitations.

planner_server/controller_server (and therefore the global costmap's
static_layer, which snapshots the map->odom transform ONCE when it first
subscribes to /map, not continuously) come up via an OnProcessExit handler
keyed to localization_creep's own process exit, not a guessed wall-clock
delay. This matters because localization_creep's internal timing runs on
the ROS/sim clock (use_sim_time), which advances at whatever rate
Gazebo's real-time factor allows - a fixed wall-clock TimerAction here
raced against that and could fire before creep actually finished on a
machine where Gazebo runs below real-time (confirmed happening on this
one). localization_creep always eventually exits either way - converged,
or after max_creep_time_sec (its own sim-time safety cap) - so
planner_server reliably comes up once, right after, whichever it was.
Watch the "Particle Cloud" display in RViz - once the arrows collapse
into a tight cluster instead of spreading across the map, it's converged.
If planner_server comes up while it's still spread out (creep timed out
without converging), the static map will latch onto the vehicle's WRONG
odom offset (permanently, until relaunched) - increase max_creep_time_sec
and/or min_creep_distance rather than clicking a goal early in that case.

use_global_planner defaults true here (unlike goal_navigate.launch.py) -
the whole point of this launch is exercising SmacPlannerHybrid against the
full mapped area; use_lane_following/use_global_planner:=false still work
if you want to compare, same as every other launch in this project.

world_file/spawn_x/spawn_y/spawn_yaw no longer need to match wherever you
originally mapped from - that was slam_toolbox's fixed-start-pose
requirement. AMCL will localize from wherever Gazebo actually spawns the
vehicle, as long as that's somewhere within the mapped free space. map_name
must match whatever you serialize_map'd it as AND map_saver_cli'd (this
loads <map_name>_map.yaml/.pgm, not the .posegraph/.data slam_toolbox used)
- see mapping.launch.py's docstring / maps/README.md.

RViz (rviz/closed_loop_navigate.rviz) shows the full saved Map alongside
the live Local/Global Costmap and the AMCL Particle Cloud, same Plan/Local
Plan/Goal Marker/Lane Marker displays local_nav's own bag_navigate.rviz
has, Fixed Frame set to map (not odom) so the whole loop is visible at
once - click "2D Goal Pose" anywhere in the loop, not just within the
immediate sensor window (only after the particle cloud has converged).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_buggy_description = get_package_share_directory('buggy_description')
    pkg_local_nav = get_package_share_directory('local_nav')
    pkg_buggy_mapping = get_package_share_directory('buggy_mapping')

    costmap_params = os.path.join(pkg_local_nav, 'config', 'local_costmap.yaml')
    teb_params = os.path.join(pkg_local_nav, 'config', 'teb_controller.yaml')
    planner_server_params = os.path.join(pkg_local_nav, 'config', 'smac_planner_server.yaml')
    global_costmap_mapped_params = os.path.join(
        pkg_buggy_mapping, 'config', 'global_costmap_mapped.yaml')
    amcl_params = os.path.join(pkg_buggy_mapping, 'config', 'amcl.yaml')
    p2l_params = os.path.join(pkg_buggy_mapping, 'config', 'pointcloud_to_laserscan.yaml')
    rviz_config = os.path.join(pkg_buggy_mapping, 'rviz', 'closed_loop_navigate.rviz')
    maps_dir = os.path.join(pkg_buggy_mapping, 'maps')

    world_file_arg = DeclareLaunchArgument(
        'world_file', default_value='rect_loop_track.world',
        description='Must match whatever world mapping.launch.py actually mapped.')
    spawn_x_arg = DeclareLaunchArgument('spawn_x', default_value='0.0')
    spawn_y_arg = DeclareLaunchArgument('spawn_y', default_value='-9.5')
    spawn_yaw_arg = DeclareLaunchArgument('spawn_yaw', default_value='0.0')
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')
    map_name_arg = DeclareLaunchArgument(
        'map_name', default_value='track',
        description='Basename (no extension) under buggy_mapping/maps/ you serialize_map\'d '
                    'AND map_saver_cli\'d - loads <map_name>_map.yaml/.pgm for AMCL/map_server '
                    '(not the .posegraph/.data slam_toolbox used).')
    global_localization_delay_arg = DeclareLaunchArgument(
        'global_localization_delay', default_value='3.0',
        description='Seconds after launch before calling reinitialize_global_localization - '
                    'must be long enough for map_server/amcl to finish their lifecycle '
                    'configure/activate (autostart by lifecycle_manager_localization).')
    creep_start_delay_arg = DeclareLaunchArgument(
        'creep_start_delay', default_value='4.0',
        description='Seconds after launch before localization_creep starts driving forward - '
                    'must be after global_localization_delay so it drives AFTER particles have '
                    'been scattered, not before.')
    creep_speed_arg = DeclareLaunchArgument('creep_speed', default_value='0.4')
    min_forward_clearance_arg = DeclareLaunchArgument('min_forward_clearance', default_value='3.0')
    min_creep_distance_arg = DeclareLaunchArgument(
        'min_creep_distance', default_value='6.0',
        description='Minimum distance localization_creep must drive before trusting AMCL\'s '
                    'convergence - should exceed the length of the most similar-looking '
                    'repeated segment of your track (see this file\'s docstring on symmetric-'
                    'loop mislocalization) or it won\'t actually disambiguate.')
    max_creep_time_sec_arg = DeclareLaunchArgument(
        'max_creep_time_sec', default_value='20.0',
        description='Sim-time safety cap on localization_creep - planner_server/'
                    'controller_server come up right after it exits (converged or this cap '
                    'hit), via an OnProcessExit handler, not a separate wall-clock timer - '
                    'see this file\'s docstring for why.')
    carrot_distance_arg = DeclareLaunchArgument('carrot_distance', default_value='13.0')
    scan_distance_arg = DeclareLaunchArgument('scan_distance', default_value='13.0')
    use_global_planner_arg = DeclareLaunchArgument(
        'use_global_planner', default_value='true',
        description='true (default here): SmacPlannerHybrid plans against the full saved '
                    'map. false: fall back to the straight-segment technique instead.')
    centering_weight_arg = DeclareLaunchArgument('centering_weight', default_value='2.0')
    corridor_check_distance_arg = DeclareLaunchArgument('corridor_check_distance', default_value='3.0')
    waypoint_step_m_arg = DeclareLaunchArgument('waypoint_step_m', default_value='1.5')
    max_waypoint_steps_arg = DeclareLaunchArgument('max_waypoint_steps', default_value='50')
    goal_xy_tolerance_arg = DeclareLaunchArgument('goal_xy_tolerance', default_value='0.5')
    use_lane_following_arg = DeclareLaunchArgument('use_lane_following', default_value='false')
    lane_simplify_epsilon_arg = DeclareLaunchArgument('lane_simplify_epsilon', default_value='0.3')
    auto_return_to_start_arg = DeclareLaunchArgument(
        'auto_return_to_start', default_value='true',
        description='true (default here - this is what "closed loop" means): the first '
                    '"2D Goal Pose" click drives one full lap and back, auto-chaining a '
                    'return-to-start leg once the outbound goal is reached, then holds for '
                    'good. false: plain single-goal behavior, one click per leg.')

    carrot_distance = ParameterValue(LaunchConfiguration('carrot_distance'), value_type=float)
    scan_distance = ParameterValue(LaunchConfiguration('scan_distance'), value_type=float)
    use_global_planner = ParameterValue(LaunchConfiguration('use_global_planner'), value_type=bool)
    centering_weight = ParameterValue(LaunchConfiguration('centering_weight'), value_type=float)
    corridor_check_distance = ParameterValue(LaunchConfiguration('corridor_check_distance'), value_type=float)
    waypoint_step_m = ParameterValue(LaunchConfiguration('waypoint_step_m'), value_type=float)
    max_waypoint_steps = ParameterValue(LaunchConfiguration('max_waypoint_steps'), value_type=int)
    goal_xy_tolerance = ParameterValue(LaunchConfiguration('goal_xy_tolerance'), value_type=float)
    use_lane_following = ParameterValue(LaunchConfiguration('use_lane_following'), value_type=bool)
    lane_simplify_epsilon = ParameterValue(LaunchConfiguration('lane_simplify_epsilon'), value_type=float)
    auto_return_to_start = ParameterValue(LaunchConfiguration('auto_return_to_start'), value_type=bool)
    map_yaml_file = PathJoinSubstitution(
        [maps_dir, [LaunchConfiguration('map_name'), '_map.yaml']])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_buggy_description, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': PathJoinSubstitution(
                [pkg_buggy_description, 'worlds', LaunchConfiguration('world_file')]),
            'spawn_x': LaunchConfiguration('spawn_x'),
            'spawn_y': LaunchConfiguration('spawn_y'),
            'spawn_yaw': LaunchConfiguration('spawn_yaw'),
        }.items(),
    )

    segment_ground = Node(
        package='ground_segmentation',
        executable='segment_ground',
        name='segment_ground',
        output='screen',
    )

    self_hit_filter = Node(
        package='local_nav',
        executable='self_hit_filter',
        name='self_hit_filter',
        output='screen',
    )

    tf_odom_relay = Node(
        package='local_nav',
        executable='tf_odom_relay',
        name='tf_odom_relay',
        output='screen',
    )

    radar_sim = Node(
        package='local_nav',
        executable='radar_sim',
        name='radar_sim',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    pointcloud_to_laserscan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        output='screen',
        parameters=[p2l_params, {'use_sim_time': True}],
        remappings=[
            ('cloud_in', '/non_ground_points_filtered'),
            ('scan', '/scan'),
        ],
    )

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'yaml_filename': map_yaml_file,
        }],
    )

    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[amcl_params, {'use_sim_time': True}],
    )

    lifecycle_manager_localization = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['map_server', 'amcl'],
        }],
    )

    # Scatters AMCL's particles across the WHOLE map's free space instead of
    # around a single guessed pose - see this file's docstring for why.
    # Delay must clear map_server/amcl's lifecycle configure+activate first.
    trigger_global_localization = TimerAction(
        period=LaunchConfiguration('global_localization_delay'),
        actions=[ExecuteProcess(
            cmd=['ros2', 'service', 'call', '/reinitialize_global_localization',
                 'std_srvs/srv/Empty', '{}'],
            output='screen',
        )],
    )

    # Drives the vehicle slowly forward (obstacle-checked against live
    # /scan) so AMCL has real motion across distinct viewpoints to
    # disambiguate with, instead of confidently converging on a single
    # stationary scan that may look identical from several spots on a
    # symmetric track - see localization_creep.py's own docstring.
    localization_creep_node = Node(
        package='local_nav',
        executable='localization_creep',
        name='localization_creep',
        output='screen',
        parameters=[{
            'cmd_vel_topic': '/ackermann_steering_controller/reference_unstamped',
            'odom_topic': '/ackermann_steering_controller/odometry',
            'scan_topic': '/scan',
            'amcl_pose_topic': '/amcl_pose',
            'creep_speed': ParameterValue(LaunchConfiguration('creep_speed'), value_type=float),
            'min_forward_clearance': ParameterValue(
                LaunchConfiguration('min_forward_clearance'), value_type=float),
            'min_creep_distance': ParameterValue(
                LaunchConfiguration('min_creep_distance'), value_type=float),
            'max_creep_time_sec': ParameterValue(
                LaunchConfiguration('max_creep_time_sec'), value_type=float),
            'use_sim_time': True,
        }],
    )
    localization_creep = TimerAction(
        period=LaunchConfiguration('creep_start_delay'),
        actions=[localization_creep_node],
    )

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[costmap_params, teb_params],
        remappings=[
            ('cmd_vel', '/ackermann_steering_controller/reference_unstamped'),
        ],
    )

    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[global_costmap_mapped_params, planner_server_params],
        condition=IfCondition(LaunchConfiguration('use_global_planner')),
    )

    # Triggered by localization_creep's process exit (RegisterEventHandler
    # below), not started immediately - see this file's docstring for why
    # bringing up planner_server (and therefore the global costmap's
    # static_layer) before AMCL converges permanently bakes in the wrong
    # map->odom offset.
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
    delayed_lifecycle_managers = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=localization_creep_node,
            on_exit=[lifecycle_manager, lifecycle_manager_no_planner],
        )
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
            'use_global_planner': use_global_planner,
            'centering_weight': centering_weight,
            'corridor_check_distance': corridor_check_distance,
            'waypoint_step_m': waypoint_step_m,
            'max_waypoint_steps': max_waypoint_steps,
            'goal_xy_tolerance': goal_xy_tolerance,
            'use_lane_following': use_lane_following,
            'lane_simplify_epsilon': lane_simplify_epsilon,
            'auto_return_to_start': auto_return_to_start,
        }],
    )

    steering_uart_bridge = Node(
        package='local_nav',
        executable='steering_uart_bridge',
        name='steering_uart_bridge',
        output='screen',
        parameters=[{
            'cmd_vel_topic': '/ackermann_steering_controller/reference_unstamped',
        }],
    )

    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        world_file_arg,
        spawn_x_arg,
        spawn_y_arg,
        spawn_yaw_arg,
        rviz_arg,
        map_name_arg,
        global_localization_delay_arg,
        creep_start_delay_arg,
        creep_speed_arg,
        min_forward_clearance_arg,
        min_creep_distance_arg,
        max_creep_time_sec_arg,
        carrot_distance_arg,
        scan_distance_arg,
        use_global_planner_arg,
        centering_weight_arg,
        corridor_check_distance_arg,
        waypoint_step_m_arg,
        max_waypoint_steps_arg,
        goal_xy_tolerance_arg,
        use_lane_following_arg,
        lane_simplify_epsilon_arg,
        auto_return_to_start_arg,
        gazebo,
        segment_ground,
        self_hit_filter,
        tf_odom_relay,
        radar_sim,
        pointcloud_to_laserscan,
        map_server,
        amcl,
        lifecycle_manager_localization,
        trigger_global_localization,
        localization_creep,
        controller_server,
        planner_server,
        delayed_lifecycle_managers,
        carrot_path_publisher,
        steering_uart_bridge,
        rviz2,
    ])
