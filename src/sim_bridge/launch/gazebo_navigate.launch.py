"""Closed-loop navigation using the team's real perception/planning nodes
(ground_segmentation, uart's topic_relay_node) against Gazebo simulation
instead of real hardware/bag playback.

  buggy_description (Gazebo) -> /velodyne_points
    -> segment_ground -> obstacle_detector -> /zone_obstacle_status
                       -> front_view_occupancy_map -> /target_lateral_position
    -> steering_calculator_node -> /steering_angle
  /zone_obstacle_status -> topic_relay_node -> /final_red_detected, /final_yellow_detected
  /steering_angle + /final_red_detected + /final_yellow_detected
    -> sim_actuation_bridge -> Twist -> ackermann_steering_controller

velodyne_static_tf is NOT launched here: buggy_description's URDF already
provides a real base_link -> lidar_mount_link transform via
robot_state_publisher, and our simulated sensor publishes in the
lidar_mount_link frame (not "velodyne"), so their separate static
broadcaster would be redundant and name-mismatched.

Parameters below are copied verbatim from
ground_segmentation/launch/golfcart_bringup.launch.py (their real-hardware
tuning) - not re-tuned for the sim track. If something needs adjusting for
the sim environment specifically, that's a deliberate, separately-flagged
change, not a silent divergence.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_buggy_description = get_package_share_directory('buggy_description')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_buggy_description, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': os.path.join(
                pkg_buggy_description, 'worlds', 'closed_loop_track.world'),
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

    front_view_occupancy_map = Node(
        package='ground_segmentation',
        executable='front_view_occupancy_map',
        name='front_view_occupancy_map',
        output='screen',
        parameters=[{
            'roi_x_min': 0.5,
            'roi_x_max': 15.0,
            'roi_y_min': -3.0,
            'roi_y_max': 3.0,
            'roi_z_min': -0.5,
            'roi_z_max': 2.0,
            'num_bins': 60,
            'occupancy_threshold': 17,
            'update_rate': 10.0,
            'visualize_bins': True,
            'visualize_center': True,
            'bin_marker_height': -1.0,
        }],
    )

    obstacle_detector = Node(
        package='ground_segmentation',
        executable='obstacle_detector',
        name='obstacle_detector',
        output='screen',
        parameters=[{
            'min_distance_threshold': 0.12,
        }],
    )

    steering_calculator = Node(
        package='ground_segmentation',
        executable='steering_calculator_node',
        name='steering_calculator_node',
        output='screen',
        parameters=[{
            'wheelbase': 1.6,
            'max_steering_angle': 20.0,
            'steering_gain': 0.4,
            'lookahead_distance': 6.0,
            'enable_visualization': True,
            'wheel_visualization_size': 0.5,
            'steering_kp': 0.6,
            'steering_ki': 0.1,
            'steering_kd': 0.2,
            'small_error_threshold': 5.0,
            'fine_tuning_factor': 1.2,
            'max_speed': 4.0,
            'min_speed': 0.5,
            'speed_reduction_factor': 0.5,
            'calculation_rate': 10.0,
        }],
    )

    topic_relay = Node(
        package='uart',
        executable='topic_relay_node',
        name='topic_relay_node',
        output='screen',
    )

    sim_actuation_bridge = Node(
        package='sim_bridge',
        executable='sim_actuation_bridge',
        name='sim_actuation_bridge',
        output='screen',
    )

    return LaunchDescription([
        gazebo,
        segment_ground,
        front_view_occupancy_map,
        obstacle_detector,
        steering_calculator,
        topic_relay,
        sim_actuation_bridge,
    ])
