"""Real-hardware counterpart of sensors_bringup.launch.py: real Velodyne
(velodyne_driver + velodyne_transform, real UDP packets over the network,
not a Gazebo plugin) + velodyne_static_tf (base_link->velodyne static TF -
real hardware has no URDF joint tree to supply this automatically, unlike
sim) + segment_ground (torch-based ground removal, TOGGLEABLE via
ground_filter, same trade-off as the sim launch) + self_hit_filter (always
on) + uart_bridge (real wheel feedback over serial) + wheel_odometry
(dead-reckoning /odom, TF publishing now OFF - see below) + chcnav's own
real_gps.launch.py (UDP GNSS/INS device -> /imu, /fix) + chcnav_static_tf
(base_link->imu_mount_link/gps_mount_link static TF - real hardware has no
robot_state_publisher, unlike sim) + ekf_node (robot_localization, fuses
wheel_odometry's /odom with /imu for yaw-drift correction, becomes the SOLE
odom->base_link broadcaster - wheel_odometry's own TF publishing is
disabled via publish_tf:False below to avoid two nodes fighting over the
same transform). GPS (/fix) is not fused yet - stage 1 is IMU only.

No radar here - the old workspace's real-hardware launches never used
radar_sim (that's Gazebo-only, a stand-in for a sensor with no native
Gazebo plugin) nor a real radar driver (smartmicro_ros2_radars/
umrr_ros2_driver exists in the old workspace's vendored source but was
never wired into any real launch there either - out of scope here too).

ground_filter:=true (default): segment_ground runs, self_hit_filter reads
its /non_ground_points output. ground_filter:=false: segment_ground is
skipped (the slowest node in this pipeline - torch RANSAC), self_hit_filter
reads raw /velodyne_points instead - confirmed in this same workspace's
sim testing that this trade-off can starve downstream planning of a clean
costmap; same caveat applies here.

SAFETY: verify sender_port/feedback_port match the actual physical wiring.
This launch only reads from the feedback port (uart_bridge, wheel
odometry) - it has no steering/actuation node at all, that's added
separately by whichever navigate-algo launch you bring up on top (real_
navigate_algo_theta_star.launch.py), same split as the sim workflow
(localize.launch.py has no controller_server/planner_server either).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_velodyne_pointcloud = get_package_share_directory('velodyne_pointcloud')
    pkg_chcnav = get_package_share_directory('chcnav')
    pkg_buggy_perception = get_package_share_directory('buggy_perception')

    ekf_params = os.path.join(pkg_buggy_perception, 'config', 'ekf.yaml')

    feedback_port_arg = DeclareLaunchArgument('feedback_port', default_value='/dev/ttyUSB0')
    ground_filter_arg = DeclareLaunchArgument(
        'ground_filter', default_value='true',
        description='false: skip segment_ground (the slow torch-based node) entirely - '
                    'self_hit_filter runs on raw /velodyne_points instead. See this '
                    "file's docstring for the trade-off.")
    udp_port_arg = DeclareLaunchArgument(
        'udp_port', default_value='7531',
        description="chcnav CGI device's configured UDP port - see chcnav/launch/real_gps.launch.py.")

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
        package='buggy_perception',
        executable='velodyne_static_tf',
        name='velodyne_static_tf',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    segment_ground = Node(
        package='buggy_perception',
        executable='segment_ground',
        name='segment_ground',
        output='screen',
        parameters=[{'use_sim_time': False}],
        condition=IfCondition(LaunchConfiguration('ground_filter')),
    )

    self_hit_filter = Node(
        package='buggy_perception',
        executable='self_hit_filter',
        name='self_hit_filter',
        output='screen',
        parameters=[{'use_sim_time': False}],
        remappings=[
            ('/non_ground_points', non_ground_points_source),
        ],
    )

    uart_bridge = Node(
        package='buggy_uart',
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
        package='buggy_perception',
        executable='wheel_odometry',
        name='wheel_odometry',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            # ekf_node below is now the sole odom->base_link broadcaster.
            'publish_tf': False,
        }],
    )

    chcnav_gps = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_chcnav, 'launch', 'real_gps.launch.py')
        ),
        launch_arguments={
            'udp_port': LaunchConfiguration('udp_port'),
        }.items(),
    )

    chcnav_static_tf = Node(
        package='buggy_perception',
        executable='chcnav_static_tf',
        name='chcnav_static_tf',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_params, {'use_sim_time': False}],
        remappings=[
            ('odom0', '/odom'),
        ],
    )

    return LaunchDescription([
        feedback_port_arg,
        ground_filter_arg,
        udp_port_arg,
        velodyne_driver_node,
        velodyne_transform_node,
        velodyne_static_tf,
        segment_ground,
        self_hit_filter,
        uart_bridge,
        wheel_odometry,
        chcnav_gps,
        chcnav_static_tf,
        ekf_node,
    ])
