"""Full sensor + perception bringup: Gazebo spawn (buggy_description) ->
Velodyne (URDF/Gazebo plugin, always on) -> segment_ground (torch-based
ground removal, TOGGLEABLE via ground_filter) -> self_hit_filter (always
on - excludes the vehicle's own chassis returns regardless of ground_filter)
-> radar_sim (UMRR-A4 stand-in, fed from self_hit_filter's output).

odom->base_link TF: previously ackermann_steering_controller's own odom TF,
relayed onto /tf by tf_odom_relay.py. Now an ekf_node (robot_localization,
buggy_perception/config/ekf.yaml) fuses that controller's odometry topic
with /imu (Gazebo's native IMU plugin on imu_mount_link, wired via
buggy_description/urdf/buggy.gps_imu.xacro - already present in the spawned
URDF, no extra node needed here for that part) for yaw-drift correction,
and is now the SOLE broadcaster of that transform - see
buggy_description/config/ackermann_controllers.yaml's enable_odom_tf:false.
tf_odom_relay.py is unused in this launch as of that change (real-hardware
equivalent: real_sensors_bringup.launch.py's wheel_odometry publish_tf:False
+ its own ekf_node).

ground_filter:=true (default): segment_ground runs, self_hit_filter reads
its /non_ground_points output - matches the old workspace's normal mode.
ground_filter:=false: segment_ground is skipped entirely (it's the
slowest node in this pipeline - torch RANSAC), self_hit_filter reads raw
/velodyne_points instead - faster, but ground plane returns show up as
spurious obstacles/radar targets. Same trade-off as the old workspace's
various ground_filter args (see buggy_mapping's real_closed_loop_navigate_
slam.launch.py docstring in the old workspace for the fullest writeup of
this trade-off).

radar_sim's output topic name (/non_ground_points_filtered) never
changes regardless of ground_filter - only self_hit_filter's INPUT
remaps between the two upstream sources.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_buggy_description = get_package_share_directory('buggy_description')
    pkg_buggy_perception = get_package_share_directory('buggy_perception')

    ekf_params = os.path.join(pkg_buggy_perception, 'config', 'ekf.yaml')

    world_file_arg = DeclareLaunchArgument(
        'world_file', default_value='rect_loop_track.world',
        description='World file under buggy_description/worlds/.')
    spawn_x_arg = DeclareLaunchArgument('spawn_x', default_value='5.0')
    spawn_y_arg = DeclareLaunchArgument('spawn_y', default_value='5.0')
    spawn_yaw_arg = DeclareLaunchArgument('spawn_yaw', default_value='0.0')
    gui_arg = DeclareLaunchArgument('gui', default_value='true')
    ground_filter_arg = DeclareLaunchArgument(
        'ground_filter', default_value='true',
        description='false: skip segment_ground (the slow torch-based node) entirely - '
                    'self_hit_filter runs on raw /velodyne_points instead. See this '
                    "file's docstring for the trade-off.")
    odom0_topic_arg = DeclareLaunchArgument(
        'odom0_topic', default_value='/ackermann_steering_controller/odometry',
        description="ekf_node's wheel-odometry input - steering_controllers_library's "
                    "own nav_msgs/Odometry topic (separate from its ~/tf_odometry, which "
                    "is now unused - see enable_odom_tf:false). Override if this doesn't "
                    "match what 'ros2 topic list' shows on your ros2_control version.")

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
            'gui': LaunchConfiguration('gui'),
        }.items(),
    )

    segment_ground = Node(
        package='buggy_perception',
        executable='segment_ground',
        name='segment_ground',
        output='screen',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('ground_filter')),
    )

    self_hit_filter = Node(
        package='buggy_perception',
        executable='self_hit_filter',
        name='self_hit_filter',
        output='screen',
        parameters=[{'use_sim_time': True}],
        remappings=[
            ('/non_ground_points', non_ground_points_source),
        ],
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_params, {'use_sim_time': True}],
        remappings=[
            ('odom0', LaunchConfiguration('odom0_topic')),
        ],
    )

    radar_sim = Node(
        package='buggy_sensors',
        executable='radar_sim',
        name='radar_sim',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        world_file_arg,
        spawn_x_arg,
        spawn_y_arg,
        spawn_yaw_arg,
        gui_arg,
        ground_filter_arg,
        odom0_topic_arg,
        gazebo,
        segment_ground,
        self_hit_filter,
        ekf_node,
        radar_sim,
    ])
