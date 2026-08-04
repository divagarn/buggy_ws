"""Real-hardware counterpart of goal_navigate.launch.py: same real-Velodyne
-> wheel_odometry -> planner_server (SmacPlannerHybrid) -> TEB -> real UART
stack as real_navigate.launch.py, but with RViz launched automatically so
you can click "2D Goal Pose" and watch the vehicle actually drive there.

use_global_planner defaults true here (unlike goal_navigate.launch.py's
sim default of false) - planner_server/SmacPlannerHybrid still does the
actual path search, same as real_navigate.launch.py's own default. What
makes a manually-clicked RViz goal work at all is carrot_path_publisher's
/goal_pose subscription: _pick_goal_heading already centers its candidate
scan on the bearing to latest_goal_xy (see that method's docstring)
whenever a goal is set, REGARDLESS of use_global_planner - so with the
global planner on, the carrot point handed to SmacPlannerHybrid now tracks
your clicked goal instead of just "wherever's open ahead", and
SmacPlannerHybrid plans a real, kinematically-valid path to it. Pass
use_global_planner:=false to switch to the newer straight-segment
technique (_split_into_clear_segments) instead, same as goal_navigate.
launch.py.

Speed/acceleration/turning-radius are NOT set here - they come entirely
from teb_controller.yaml (max_vel_x, max_vel_x_backwards, max_vel_theta,
acc_lim_x, min_turning_radius - see that file's own comments), the same
config file real_navigate.launch.py/mppi_navigate.launch.py/
goal_navigate.launch.py all share. Change the vehicle's actual driving
speed there, not in this launch file.

RViz reuses local_nav/rviz/bag_navigate.rviz as-is (same choice as
goal_navigate.launch.py): it already has Local Costmap/Global Costmap
displays and a "2D Goal Pose" tool wired to /goal_pose - exactly
carrot_path_publisher's goal topic, no new rviz config needed. Its
Steering/Steering Actual vs Target marker displays are real-hardware-
specific and should actually show data here (unlike in the sim launch).

self_hit_filter is included by default even though it was built for a
Gazebo-only artifact (the sim LiDAR mount sitting closer to the chassis
than the real one) - it's a narrow box + 1m range filter, cheap and inert
if the real point cloud has no self-hits, but this should be verified
against real data rather than assumed either way.

SAFETY: verify sender_port/feedback_port match the actual physical wiring
before running with loopback:=false - the two default to the same
placeholder because that's what uart_sender_node.py/uart_bridge.py's own
defaults already were, not because they're known to share a port.
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
    pkg_local_nav = get_package_share_directory('local_nav')
    pkg_velodyne_pointcloud = get_package_share_directory('velodyne_pointcloud')

    costmap_params = os.path.join(pkg_local_nav, 'config', 'local_costmap.yaml')
    teb_params = os.path.join(pkg_local_nav, 'config', 'teb_controller.yaml')
    planner_costmap_params = os.path.join(pkg_local_nav, 'config', 'planner_costmap.yaml')
    planner_server_params = os.path.join(pkg_local_nav, 'config', 'smac_planner_server.yaml')
    rviz_config = os.path.join(pkg_local_nav, 'rviz', 'bag_navigate.rviz')

    carrot_distance_arg = DeclareLaunchArgument('carrot_distance', default_value='13.0')
    scan_distance_arg = DeclareLaunchArgument('scan_distance', default_value='13.0')
    loopback_arg = DeclareLaunchArgument(
        'loopback', default_value='true',
        description='uart_sender_node loopback mode - true exercises the full code path '
                    'without writing to the real serial port. Set false only once /steering_angle, '
                    '/odom, and the costmap have all been watched and look correct.')
    sender_port_arg = DeclareLaunchArgument('sender_port', default_value='/dev/ttyUSB0')
    feedback_port_arg = DeclareLaunchArgument('feedback_port', default_value='/dev/ttyUSB0')
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')
    use_global_planner_arg = DeclareLaunchArgument(
        'use_global_planner', default_value='true',
        description='true (default): plan through planner_server/SmacPlannerHybrid to a '
                    'carrot point that tracks your RViz "2D Goal Pose" click. false: skip the '
                    'global planner, use _split_into_clear_segments to build the whole path '
                    'to the clicked goal directly instead.')
    centering_weight_arg = DeclareLaunchArgument('centering_weight', default_value='2.0')
    corridor_check_distance_arg = DeclareLaunchArgument('corridor_check_distance', default_value='3.0')
    waypoint_step_m_arg = DeclareLaunchArgument('waypoint_step_m', default_value='1.5')
    max_waypoint_steps_arg = DeclareLaunchArgument('max_waypoint_steps', default_value='50')
    goal_xy_tolerance_arg = DeclareLaunchArgument('goal_xy_tolerance', default_value='0.5')
    use_lane_following_arg = DeclareLaunchArgument(
        'use_lane_following', default_value='false',
        description='true: navigate along the medial-axis skeleton of the local costmap '
                    '(_build_lane_path) instead of either other technique - exactly centered, '
                    'no oscillation risk, independent of corridor width.')
    lane_simplify_epsilon_arg = DeclareLaunchArgument('lane_simplify_epsilon', default_value='0.3')

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

    # Not the stock velodyne-all-nodes-VLP16-launch.py: that hardcodes
    # calibration to the standard VLP16db.yaml with no way to override it
    # from outside. This vehicle's real device_ip/calibration/min_range
    # were confirmed against the ROS1 real-hardware launch (see
    # real_navigate.launch.py's own comment for the full trail):
    #   - calibration: VLP16_hires_db.yaml, not VLP16db.yaml - Puck Hi-Res
    #     unit (1.33deg vertical spacing), standard file has wrong
    #     per-laser vertical angle corrections for it.
    #   - device_ip: "" (accept from any source).
    #   - min_range: 0.4, not the stock default 0.9.
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

    wheel_odometry = Node(
        package='local_nav',
        executable='wheel_odometry',
        name='wheel_odometry',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    # use_sim_time: False here (and on every node in this launch file) is
    # not just a stylistic default - local_costmap.yaml/planner_costmap.yaml/
    # teb_controller.yaml all hardcode use_sim_time: true (written for
    # Gazebo, which publishes /clock). On real hardware nothing publishes
    # /clock at all, so a use_sim_time:true node's ROS clock never advances
    # past zero, and the costmap's observation-staleness check silently
    # rejects every point cloud forever - see real_navigate.launch.py's own
    # comment for the full history of this bug. The {'use_sim_time': False}
    # dict must come after teb_params in this list - later entries win.
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
        parameters=[planner_costmap_params, planner_server_params, {'use_sim_time': False}],
        condition=IfCondition(LaunchConfiguration('use_global_planner')),
    )

    # Split in two (rather than one lifecycle_manager with a launch-time-
    # conditional node_names list) because node_names needs a concrete
    # Python list at description-build time - LaunchConfiguration values
    # aren't resolved until launch time. Only one of these two actually
    # launches (IfCondition/UnlessCondition on the same argument).
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
        velodyne_driver_node,
        velodyne_transform_node,
        velodyne_static_tf,
        segment_ground,
        self_hit_filter,
        uart_bridge,
        wheel_odometry,
        controller_server,
        planner_server,
        lifecycle_manager,
        lifecycle_manager_no_planner,
        carrot_path_publisher,
        steering_uart_bridge,
        uart_sender,
        rviz2,
    ])
