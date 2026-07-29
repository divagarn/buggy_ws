from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('dms_adas_mediapipe')
    params_file = os.path.join(pkg_share, 'config', 'dms_params.yaml')

    return LaunchDescription([
        Node(
            package='dms_adas_mediapipe',
            executable='dms_node',
            name='dms_node',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='dms_adas_mediapipe',
            executable='gstreamer_camera_node',
            name='gstreamer_camera_node',
            output='screen',
            parameters=[params_file],
        ),
    ])
