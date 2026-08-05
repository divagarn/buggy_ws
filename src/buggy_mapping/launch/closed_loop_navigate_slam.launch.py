"""slam_toolbox counterpart of closed_loop_navigate.launch.py (which uses
AMCL - see that file's docstring for why AMCL exists: place-anywhere global
localization on a symmetric track without a known start pose). This one
goes back to slam_toolbox's LOCALIZATION mode (fixed map_start_pose, same
as this project originally used) but adds a manual correction workflow for
when that fixed-pose assumption is wrong, instead of AMCL's automatic (but
track-symmetry-sensitive) global localization:

Gazebo -> segment_ground -> self_hit_filter -> pointcloud_to_laserscan ->
slam_toolbox (LOCALIZATION mode, loading the map you saved in phase 1) ->
local_nav's usual controller_server/planner_server (planner_server using
this package's global_costmap_mapped.yaml - see that file's own comment).

PAUSE / MANUAL POSE CORRECTION WORKFLOW: type ":pause_n_l" + Enter in THIS
launch's own terminal at any time to immediately stop navigation
(planner_server/controller_server deactivated via nav2_lifecycle_manager's
manage_nodes PAUSE, plus an immediate zero-velocity publish) - do this as
soon as you notice the vehicle is NOT where slam_toolbox thinks it is
(map/scan misaligned in RViz). With navigation paused, use RViz's "2D Pose
Estimate" tool to click-and-drag the vehicle's actual pose on the map.

That tool always publishes to /initialpose - but this slam_toolbox build's
LOCALIZATION mode does NOT subscribe to /initialpose at all (confirmed via
`ros2 node info /slam_toolbox`; an earlier assumption here that it did,
"like AMCL," was wrong). initialpose_to_slam_toolbox bridges the two: it
subscribes to /initialpose and translates each message into a call to
/slam_toolbox/deserialize_map with match_type: LOCALIZE_AT_POSE - the
same service slam_toolbox itself uses to load the map at startup
(map_start_pose), just invoked again with your clicked pose instead. That
directly resets the localization anchor, so the TF (map->odom) jumps
immediately - the vehicle does NOT need to move for the pose itself to
take effect. (Ordinary scan-matching updates after that point are still
gated by minimum_travel_distance/minimum_travel_heading, same as always -
that governs ongoing tracking, not this manual reset.)
posegraph_align_creep (an alternative that drives automatically instead)
still exists in local_nav but is commented out below by request. Once the
live scan lines up with the map in RViz, type ":resume" + Enter to
reactivate planner_server/controller_server and carry on. See
pause_resume_console.py's own docstring for exactly what pause/resume
does and doesn't touch (it does NOT pause slam_toolbox itself - only the
planner_server/controller_server pair - see that file for why).

ground_filter:=false skips segment_ground entirely (its torch-based ground
removal is the single slowest node in this pipeline) - self_hit_filter
still runs (still excludes the vehicle's own body from the scan), just
fed raw /velodyne_points instead of segment_ground's output. Trade-off:
without ground removal, the ground plane itself can show up as bogus
obstacles in the local costmap/laserscan on non-flat terrain - fine on
flat ground (Gazebo's default ground_plane, most real pavement), worth
turning back on (the default) if you see phantom obstacles.

use_global_planner defaults true here (unlike goal_navigate.launch.py) -
the whole point of this launch is exercising SmacPlannerHybrid against the
full mapped area; use_lane_following/use_global_planner:=false still work
if you want to compare, same as every other launch in this project.

world_file/spawn_x/spawn_y/spawn_yaw MUST match whatever you actually
mapped with mapping.launch.py - this launches a fresh Gazebo instance, it
doesn't remember what world you mapped. If they don't match exactly,
that's exactly the ":pause_n_l" -> 2D Pose Estimate -> ":resume" workflow
above is for. map_name must match whatever you serialize_map'd it as - see
mapping.launch.py's docstring / maps/README.md.

RViz (rviz/closed_loop_navigate.rviz) shows the full saved Map alongside
the live Local/Global Costmap, same Plan/Local Plan/Goal Marker/Lane
Marker displays local_nav's own bag_navigate.rviz has, Fixed Frame set to
map (not odom) so the whole loop is visible at once - click "2D Goal
Pose" anywhere in the loop, not just within the immediate sensor window.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
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
    slam_params = os.path.join(pkg_buggy_mapping, 'config', 'slam_toolbox_localization.yaml')
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
                    'in mapping.launch.py - loads <map_name>.posegraph/.data.')
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
    ground_filter_arg = DeclareLaunchArgument(
        'ground_filter', default_value='true',
        description='false: skip segment_ground (the slow torch-based node) entirely - '
                    'self_hit_filter runs on raw /velodyne_points instead. See this file\'s '
                    'docstring for the trade-off.')
    align_num_waypoints_arg = DeclareLaunchArgument('align_num_waypoints', default_value='15')
    align_max_creep_time_sec_arg = DeclareLaunchArgument('align_max_creep_time_sec', default_value='30.0')

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
    map_file_name = PathJoinSubstitution([maps_dir, LaunchConfiguration('map_name')])
    # segment_ground's own output topic when enabled, raw Velodyne points
    # straight into self_hit_filter when it's skipped - see ground_filter_arg.
    non_ground_points_source = PythonExpression([
        "'/non_ground_points' if '", LaunchConfiguration('ground_filter'),
        "' == 'true' else '/velodyne_points'"])

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
        condition=IfCondition(LaunchConfiguration('ground_filter')),
    )

    self_hit_filter = Node(
        package='local_nav',
        executable='self_hit_filter',
        name='self_hit_filter',
        output='screen',
        remappings=[
            ('/non_ground_points', non_ground_points_source),
        ],
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

    # executable MUST be localization_slam_toolbox_node, not
    # async_slam_toolbox_node (even with mode: localization set in the
    # yaml) - confirmed live: calling /slam_toolbox/deserialize_map with
    # match_type: LOCALIZE_AT_POSE against a running async_slam_toolbox_node
    # logs "Requested a localization deserialization in non-localization
    # mode" and silently does not update map->odom, while the exact same
    # call against localization_slam_toolbox_node updates it immediately.
    # This only matters for the mid-run manual relocalization
    # (initialpose_to_slam_toolbox below) - both executables otherwise
    # behave the same for continuous scan-matching localization.
    slam_toolbox = Node(
        package='slam_toolbox',
        executable='localization_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params, {
            'use_sim_time': True,
            'map_file_name': map_file_name,
        }],
    )

    # Bridges RViz's "2D Pose Estimate" (always publishes to /initialpose,
    # not configurable) to slam_toolbox's ACTUAL manual relocalization
    # mechanism - see initialpose_to_slam_toolbox.py's own docstring for
    # why this is needed (this slam_toolbox build does not subscribe to
    # /initialpose at all, confirmed via `ros2 node info /slam_toolbox`).
    initialpose_to_slam_toolbox = Node(
        package='local_nav',
        executable='initialpose_to_slam_toolbox',
        name='initialpose_to_slam_toolbox',
        output='screen',
        parameters=[{
            'map_file_name': map_file_name,
            'use_sim_time': True,
        }],
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

    pause_resume_console = Node(
        package='local_nav',
        executable='pause_resume_console',
        name='pause_resume_console',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'lifecycle_manager_name': 'lifecycle_manager_local_nav',
            'cmd_vel_topic': '/ackermann_steering_controller/reference_unstamped',
            'use_sim_time': True,
        }],
    )

    # Commented out by request - alignment motion after "2D Pose Estimate"
    # is done manually (teleop) instead of automatically. Node still exists
    # in local_nav/posegraph_align_creep.py if this needs to come back.
    # posegraph_align_creep = Node(
    #     package='local_nav',
    #     executable='posegraph_align_creep',
    #     name='posegraph_align_creep',
    #     output='screen',
    #     parameters=[{
    #         'cmd_vel_topic': '/ackermann_steering_controller/reference_unstamped',
    #         'scan_topic': '/scan',
    #         'num_waypoints_to_traverse': ParameterValue(
    #             LaunchConfiguration('align_num_waypoints'), value_type=int),
    #         'max_creep_time_sec': ParameterValue(
    #             LaunchConfiguration('align_max_creep_time_sec'), value_type=float),
    #         'use_sim_time': True,
    #     }],
    # )

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
        ground_filter_arg,
        align_num_waypoints_arg,
        align_max_creep_time_sec_arg,
        gazebo,
        segment_ground,
        self_hit_filter,
        tf_odom_relay,
        radar_sim,
        pointcloud_to_laserscan,
        slam_toolbox,
        initialpose_to_slam_toolbox,
        controller_server,
        planner_server,
        lifecycle_manager,
        lifecycle_manager_no_planner,
        carrot_path_publisher,
        steering_uart_bridge,
        pause_resume_console,
        # posegraph_align_creep,  # commented out - alignment driven manually via teleop instead
        rviz2,
    ])
