"""Real-hardware counterpart of mppi_navigate.launch.py: same costmap ->
planner_server (NavFn) -> DWB stack, but driven by the real Velodyne and
real wheel feedback instead of Gazebo, and actuating over real UART
instead of ackermann_steering_controller.

Differences from the sim launch:
  - velodyne (real VLP16 driver, not Gazebo's simulated LiDAR plugin) +
    velodyne_static_tf (base_link -> velodyne, ground_segmentation's own
    node, matching what golfcart_bringup.launch.py uses).
  - wheel_odometry: neither the real vehicle nor its existing reactive
    pipeline has ever needed odom->base_link TF or a nav_msgs/Odometry -
    that pipeline works entirely in sensor-relative space. This stack's
    costmaps (global_frame: odom) and DWB (velocity feedback via its
    odom_topic param) both need one, so wheel_odometry integrates
    /wheel_uart (via uart_bridge, real wheel feedback: steering degrees +
    speed km/h, both already calibrated by the vehicle's MCU) into
    bicycle-model dead-reckoning odometry - see wheel_odometry.py.
  - controller_server's cmd_vel is remapped to a neutral /cmd_vel (not
    /ackermann_steering_controller/reference_unstamped, which only exists
    in the Gazebo sim) - steering_uart_bridge subscribes that same topic
    and publishes uart_sender_node's actual contract (/steering_angle,
    /final_red_detected, /final_yellow_detected).
  - uart_sender_node (real hardware, team's unmodified code) actually
    runs here, with loopback:=true by default - it exercises the exact
    same code path as talking to the real serial port, but skips the
    actual write, so the first run can be watched safely before ever
    setting loopback:=false.

self_hit_filter is included by default even though it was built for a
Gazebo-only artifact (the sim LiDAR mount sitting closer to the chassis
than the real one) - it's a narrow box + 1m range filter, cheap and
inert if the real point cloud has no self-hits, but this should be
verified against real data (e.g. replay the team's velodyne_uart_bag and
check /non_ground_points for a ring centered on the vehicle, the same
check that found the sim issue) rather than assumed either way.

SAFETY: verify sender_port/feedback_port match the actual physical
wiring before running with loopback:=false - the two default to the same
placeholder because that's what uart_sender_node.py/uart_bridge.py's own
defaults already were, not because they're known to share a port.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_local_nav = get_package_share_directory('local_nav')
    pkg_velodyne_pointcloud = get_package_share_directory('velodyne_pointcloud')

    costmap_params = os.path.join(pkg_local_nav, 'config', 'local_costmap.yaml')
    dwb_params = os.path.join(pkg_local_nav, 'config', 'dwb_controller.yaml')
    planner_costmap_params = os.path.join(pkg_local_nav, 'config', 'planner_costmap.yaml')
    planner_server_params = os.path.join(pkg_local_nav, 'config', 'planner_server.yaml')

    carrot_distance_arg = DeclareLaunchArgument('carrot_distance', default_value='9.0')
    scan_distance_arg = DeclareLaunchArgument('scan_distance', default_value='9.0')
    dwb_sim_time_arg = DeclareLaunchArgument('dwb_sim_time', default_value='2.5')
    loopback_arg = DeclareLaunchArgument(
        'loopback', default_value='true',
        description='uart_sender_node loopback mode - true exercises the full code path '
                    'without writing to the real serial port. Set false only once /steering_angle, '
                    '/odom, and the costmap have all been watched and look correct.')
    sender_port_arg = DeclareLaunchArgument('sender_port', default_value='/dev/ttyUSB0')
    feedback_port_arg = DeclareLaunchArgument('feedback_port', default_value='/dev/ttyUSB0')

    carrot_distance = ParameterValue(LaunchConfiguration('carrot_distance'), value_type=float)
    scan_distance = ParameterValue(LaunchConfiguration('scan_distance'), value_type=float)
    dwb_sim_time = ParameterValue(LaunchConfiguration('dwb_sim_time'), value_type=float)
    loopback = ParameterValue(LaunchConfiguration('loopback'), value_type=bool)

    # Not the stock velodyne-all-nodes-VLP16-launch.py: that hardcodes
    # calibration to the standard VLP16db.yaml with no way to override it
    # from outside. This vehicle's real device_ip/calibration/min_range
    # were confirmed against the ROS1 real-hardware launch
    # (catkin_ws/.../ground_segmentation/launch/lidar_gps_devices.launch ->
    # velodyne_pointcloud/launch/VLP16_points.launch's actual defaults):
    #   - calibration: VLP16_hires_db.yaml, not VLP16db.yaml - this is a
    #     Velodyne Puck Hi-Res unit (1.33deg vertical spacing), and the
    #     standard calibration file has the wrong per-laser vertical angle
    #     corrections for it, which would geometrically distort every
    #     point cloud.
    #   - device_ip: "" (accept from any source) rather than the stock
    #     ROS2 default's 192.168.1.201 (which only works if the sensor's
    #     IP was never changed from a specific assumption).
    #   - min_range: 0.4, not the stock default 0.9 (self_hit_filter/
    #     segment_ground already handle near-field returns; no reason to
    #     discard more of the near field than the real launch did).
    # laserscan_node is dropped entirely - nothing in this pipeline
    # consumes the 2D LaserScan it produces, only /velodyne_points.
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
            'rpm': 600.0,
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
    # dwb_controller.yaml all hardcode use_sim_time: true (written for
    # Gazebo, which publishes /clock). On real hardware nothing publishes
    # /clock at all, so a use_sim_time:true node's ROS clock never advances
    # past zero - every real, genuinely-fresh sensor timestamp then looks
    # wildly mismatched against that stuck clock, and the costmap's
    # observation-staleness check silently rejects every single point cloud,
    # forever - a real costmap that never accumulates a single lethal cell,
    # with no error logged. This is the same class of bug that caused the
    # identical symptom in bag_navigate.launch.py (there, the fix was the
    # opposite: use_sim_time true + --clock, to match the bag's own
    # recorded time instead of wall-clock). Confirmed via
    # /local_costmap/costmap coming back 100% free before this fix.
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[
            costmap_params, dwb_params,
            {'FollowPath.sim_time': dwb_sim_time, 'use_sim_time': False},
        ],
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

    return LaunchDescription([
        carrot_distance_arg,
        scan_distance_arg,
        dwb_sim_time_arg,
        loopback_arg,
        sender_port_arg,
        feedback_port_arg,
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
        carrot_path_publisher,
        steering_uart_bridge,
        uart_sender,
    ])
