"""
Launch file to start all golf cart ground segmentation and UART nodes.

Nodes launched:
  1. velodyne_static_tf       – Static TF broadcaster (base_link → velodyne)
  2. segment_ground           – LiDAR ground segmentation (RANSAC)
  3. front_view_occupancy_map – Lateral occupancy histogram & free-space centre
  4. obstacle_detector        – DBSCAN clustering & zone obstacle status
  5. steering_calculator_node – Pure-pursuit steering + PID control
  6. uart_bridge              – UART serial receiver (wheel feedback → /wheel_uart)
  7. uart_sender_node         – UART serial sender (steering + zone → MCU)
  8. dms_system               – Driver monitoring system (camera + dms node)
  9. rviz2                    – Visualization (optional, controlled by launch arg)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Launch arguments ────────────────────────────────────────────
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz2 with the golf cart demo config'
    )

    pkg_share = get_package_share_directory('ground_segmentation')
    rviz_config = os.path.join(pkg_share, 'rviz', 'golfcart_demo2.rviz')
    dms_share = get_package_share_directory('dms_adas_mediapipe')
    dms_launch = os.path.join(dms_share, 'launch', 'dms_system.launch.py')

    # ── RViz2 ───────────────────────────────────────────────────────
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen',
    )

    # ── 1. Static TF: base_link → velodyne ──────────────────────────
    velodyne_static_tf = Node(
        package='ground_segmentation',
        executable='velodyne_static_tf',
        name='velodyne_static_tf',
        output='screen',
    )

    # ── 2. Ground segmentation ──────────────────────────────────────
    segment_ground = Node(
        package='ground_segmentation',
        executable='segment_ground',
        name='segment_ground',
        output='screen',
    )

    # ── 3. Front-view occupancy map ─────────────────────────────────
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

    # ── 4. Obstacle detector ────────────────────────────────────────
    obstacle_detector = Node(
        package='ground_segmentation',
        executable='obstacle_detector',
        name='obstacle_detector',
        output='screen',
        parameters=[{
            'min_distance_threshold': 0.12,
        }],
    )

    # ── 5. Steering calculator ──────────────────────────────────────
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

    # ── 6. UART bridge (wheel feedback receiver) ────────────────────
    # uart_bridge = Node(
    #     package='uart',
    #     executable='uart_bridge',
    #     name='uart_bridge',
    #     output='screen',
    #     parameters=[{
    #         'port': '/dev/ttyUSB0',
    #         'baud': 115200,
    #     }],
    # )

    # ── 8. DMS system (camera + driver monitoring) ──────────────────
    dms_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(dms_launch)
    )

    # # ── 7. UART sender (steering + zone commands) ──────────────────
    uart_sender = Node(
        package='uart',
        executable='uart_sender_node',
        name='uart_sender_node',
        output='screen',
        parameters=[{
            'port': '/dev/ttyUSB0',
            'baudrate': 115200,
            'loopback': False,
        }],
    )

    # ── 7b. Topic relay (bridge sensor signals to UART control topics) ──
    topic_relay = Node(
        package='uart',
        executable='topic_relay_node',
        name='topic_relay_node',
        output='screen',
    )

    return LaunchDescription([
        rviz_arg,
        velodyne_static_tf,
        segment_ground,
        front_view_occupancy_map,
        obstacle_detector,
        steering_calculator,
        dms_system,
        topic_relay,
        #uart_bridge,
        uart_sender,
        rviz2,
    ])
