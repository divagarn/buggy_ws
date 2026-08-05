"""slam_toolbox counterpart of real_closed_loop_navigate.launch.py (which
uses AMCL) - real Velodyne + real wheel feedback instead of Gazebo,
otherwise the same slam_toolbox LOCALIZATION mode + manual pause/pose-
correction workflow as closed_loop_navigate_slam.launch.py. See that
file's docstring for the full PAUSE / MANUAL POSE CORRECTION workflow
(":pause_n_l" + Enter in this launch's terminal, RViz "2D Pose Estimate",
":resume" + Enter) and pause_resume_console.py's docstring for exactly
what it does and doesn't touch.

perception -> 2D scan -> slam_toolbox (localization mode, loading a map
saved by real_mapping.launch.py) -> local_nav's controller_server/
planner_server (planner_server using this package's global_costmap_mapped.
yaml, same as the sim version) -> real UART actuation, same
real_navigate.launch.py contract (steering_uart_bridge -> uart_sender_node,
loopback:=true by default).

The vehicle itself does NOT need to move for a pose correction to take
effect. RViz's "2D Pose Estimate" always publishes to /initialpose, but
this slam_toolbox build does not subscribe to that topic at all -
initialpose_to_slam_toolbox bridges it to the actual mechanism
(/slam_toolbox/deserialize_map, match_type: LOCALIZE_AT_POSE - see that
node's own docstring), which resets the localization anchor directly, so
the TF (map->odom) jumps immediately regardless of vehicle motion.
posegraph_align_creep (an alternative that drives automatically instead)
still exists in local_nav but is commented out below by request.

ground_filter:=false skips segment_ground entirely (its torch-based ground
removal is the single slowest node in this pipeline) - self_hit_filter
still runs, just fed raw Velodyne points instead of segment_ground's
output. See closed_loop_navigate_slam.launch.py's docstring for the
trade-off (ground plane can show up as bogus obstacles without it).

map_name must match whatever real_mapping.launch.py actually saved via
serialize_map - see that file's docstring / maps/README.md.

SAFETY: verify sender_port/feedback_port match the actual physical wiring
before running with loopback:=false. The pause/resume workflow above is
exactly for the case where slam_toolbox's assumed start pose turns out
wrong on real hardware - use it (and confirm the live scan lines up with
the map in RViz) before setting loopback:=false.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_local_nav = get_package_share_directory('local_nav')
    pkg_buggy_mapping = get_package_share_directory('buggy_mapping')
    pkg_velodyne_pointcloud = get_package_share_directory('velodyne_pointcloud')

    costmap_params = os.path.join(pkg_local_nav, 'config', 'local_costmap.yaml')
    teb_params = os.path.join(pkg_local_nav, 'config', 'teb_controller.yaml')
    planner_server_params = os.path.join(pkg_local_nav, 'config', 'smac_planner_server.yaml')
    global_costmap_mapped_params = os.path.join(
        pkg_buggy_mapping, 'config', 'global_costmap_mapped.yaml')
    slam_params = os.path.join(pkg_buggy_mapping, 'config', 'slam_toolbox_localization.yaml')
    p2l_params = os.path.join(pkg_buggy_mapping, 'config', 'pointcloud_to_laserscan.yaml')
    rviz_config = os.path.join(pkg_buggy_mapping, 'rviz', 'closed_loop_navigate.rviz')
    maps_dir = os.path.join(pkg_buggy_mapping, 'maps')

    map_name_arg = DeclareLaunchArgument(
        'map_name', default_value='track',
        description='Basename (no extension) under buggy_mapping/maps/ you serialize_map\'d '
                    'in real_mapping.launch.py - loads <map_name>.posegraph/.data.')
    carrot_distance_arg = DeclareLaunchArgument('carrot_distance', default_value='13.0')
    scan_distance_arg = DeclareLaunchArgument('scan_distance', default_value='13.0')
    loopback_arg = DeclareLaunchArgument(
        'loopback', default_value='true',
        description='uart_sender_node loopback mode - true exercises the full code path '
                    'without writing to the real serial port. Set false only once '
                    '/steering_angle, /odom, and the costmap have all been watched and look '
                    'correct.')
    sender_port_arg = DeclareLaunchArgument('sender_port', default_value='/dev/ttyUSB0')
    feedback_port_arg = DeclareLaunchArgument('feedback_port', default_value='/dev/ttyUSB0')
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')
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
                    'self_hit_filter runs on raw /velodyne_points instead. See '
                    'closed_loop_navigate_slam.launch.py\'s docstring for the trade-off.')
    align_num_waypoints_arg = DeclareLaunchArgument('align_num_waypoints', default_value='15')
    align_max_creep_time_sec_arg = DeclareLaunchArgument('align_max_creep_time_sec', default_value='30.0')

    carrot_distance = ParameterValue(LaunchConfiguration('carrot_distance'), value_type=float)
    scan_distance = ParameterValue(LaunchConfiguration('scan_distance'), value_type=float)
    loopback = ParameterValue(LaunchConfiguration('loopback'), value_type=bool)
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
    non_ground_points_source = PythonExpression([
        "'/non_ground_points' if '", LaunchConfiguration('ground_filter'),
        "' == 'true' else '/velodyne_points'"])

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
        condition=IfCondition(LaunchConfiguration('ground_filter')),
    )

    self_hit_filter = Node(
        package='local_nav',
        executable='self_hit_filter',
        name='self_hit_filter',
        output='screen',
        parameters=[{'use_sim_time': False}],
        remappings=[
            ('/non_ground_points', non_ground_points_source),
        ],
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

    # executable MUST be localization_slam_toolbox_node, not
    # async_slam_toolbox_node (even with mode: localization set in the
    # yaml) - see closed_loop_navigate_slam.launch.py's comment for the
    # confirmed live behavior difference this causes for the mid-run
    # manual relocalization (initialpose_to_slam_toolbox below).
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
            'use_sim_time': False,
        }],
    )

    # use_sim_time: False - see real_navigate.launch.py's own comment for
    # why this matters (no /clock on real hardware, a use_sim_time:true
    # node's clock never advances, every observation looks stale forever).
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[costmap_params, teb_params, {'use_sim_time': False}],
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
            'odom_topic': '/odom',
            'use_global_planner': use_global_planner,
            'centering_weight': centering_weight,
            'corridor_check_distance': corridor_check_distance,
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

    pause_resume_console = Node(
        package='local_nav',
        executable='pause_resume_console',
        name='pause_resume_console',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'lifecycle_manager_name': 'lifecycle_manager_local_nav',
            'cmd_vel_topic': '/cmd_vel',
            'use_sim_time': False,
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
    #         'cmd_vel_topic': '/cmd_vel',
    #         'scan_topic': '/scan',
    #         'num_waypoints_to_traverse': ParameterValue(
    #             LaunchConfiguration('align_num_waypoints'), value_type=int),
    #         'max_creep_time_sec': ParameterValue(
    #             LaunchConfiguration('align_max_creep_time_sec'), value_type=float),
    #         'use_sim_time': False,
    #     }],
    # )

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
        carrot_distance_arg,
        scan_distance_arg,
        loopback_arg,
        sender_port_arg,
        feedback_port_arg,
        rviz_arg,
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
        velodyne_driver_node,
        velodyne_transform_node,
        velodyne_static_tf,
        segment_ground,
        self_hit_filter,
        uart_bridge,
        wheel_odometry,
        pointcloud_to_laserscan,
        slam_toolbox,
        initialpose_to_slam_toolbox,
        controller_server,
        planner_server,
        lifecycle_manager,
        lifecycle_manager_no_planner,
        carrot_path_publisher,
        steering_uart_bridge,
        uart_sender,
        pause_resume_console,
        # posegraph_align_creep,  # commented out - alignment driven manually via teleop instead
        rviz2,
    ])
