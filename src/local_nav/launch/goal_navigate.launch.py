"""Demo launch for the goal-seeking straight-segment technique (see
carrot_path_publisher.py's "Goal-seeking" docstring section and
_split_into_clear_segments): Gazebo -> segment_ground -> local costmap ->
carrot_path_publisher builds a whole multi-waypoint path to a chosen
/goal_pose in one shot (no SmacPlannerHybrid/planner_server involved) ->
TEB -> ackermann_steering_controller.

Unlike mppi_navigate.launch.py (which defaults use_global_planner:=true,
planner_server/SmacPlannerHybrid computing the reference path), this
launch defaults use_global_planner:=false specifically to exercise the
new technique end to end: draw/see the live local costmap in RViz, click
"2D Goal Pose" to publish a goal, and watch carrot_path_publisher's
_split_into_clear_segments (centering-weighted, see that method's
docstring for why plain closest-to-goal scoring alone wasn't enough) plan
and drive there. use_global_planner can still be overridden true from the
command line to compare directly against the SmacPlannerHybrid path in
the same world.

RViz reuses local_nav/rviz/bag_navigate.rviz as-is: it already has Local
Costmap/Global Costmap displays (the obstacle map this technique reasons
over) and a "2D Goal Pose" tool wired to /goal_pose, which is exactly
carrot_path_publisher's goal topic - no new rviz config needed. Its
Steering/Steering Actual vs Target marker displays are for the real-
hardware/bag bridge and simply show nothing here; harmless.

Tunable from the command line, e.g.:
    ros2 launch local_nav goal_navigate.launch.py centering_weight:=3.0
  - centering_weight / corridor_check_distance: how strongly and how far
    the picked path is pulled away from walls toward the corridor middle.
  - waypoint_step_m / max_waypoint_steps: how finely-grained and how long
    a path _split_into_clear_segments is allowed to build per request_plan
    cycle.
  - goal_xy_tolerance: how close counts as "arrived" (stops replanning,
    holds until a new /goal_pose arrives).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_buggy_description = get_package_share_directory('buggy_description')
    pkg_local_nav = get_package_share_directory('local_nav')

    costmap_params = os.path.join(pkg_local_nav, 'config', 'local_costmap.yaml')
    teb_params = os.path.join(pkg_local_nav, 'config', 'teb_controller.yaml')
    planner_costmap_params = os.path.join(pkg_local_nav, 'config', 'planner_costmap.yaml')
    planner_server_params = os.path.join(pkg_local_nav, 'config', 'smac_planner_server.yaml')
    rviz_config = os.path.join(pkg_local_nav, 'rviz', 'bag_navigate.rviz')

    carrot_distance_arg = DeclareLaunchArgument('carrot_distance', default_value='13.0')
    scan_distance_arg = DeclareLaunchArgument('scan_distance', default_value='13.0')
    world_file_arg = DeclareLaunchArgument(
        'world_file', default_value='rect_loop_track.world',
        description='World file under buggy_description/worlds/, e.g. '
                    'rect_loop_track.world, varying_width_track.world, closed_loop_track2.world')
    spawn_x_arg = DeclareLaunchArgument('spawn_x', default_value='0.0')
    spawn_y_arg = DeclareLaunchArgument('spawn_y', default_value='-9.5')
    spawn_yaw_arg = DeclareLaunchArgument('spawn_yaw', default_value='0.0')
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')
    # true (unlike mppi_navigate.launch.py's default false): this launch
    # exists specifically to demo/exercise the goal-seeking straight-
    # segment technique instead of SmacPlannerHybrid. Override to true to
    # compare directly against the global-planner path in the same world.
    use_global_planner_arg = DeclareLaunchArgument(
        'use_global_planner', default_value='false',
        description='false (default here): carrot_path_publisher builds the whole path to '
                    '/goal_pose itself via _split_into_clear_segments, no planner_server '
                    'involved. true: fall back to planner_server/SmacPlannerHybrid instead.')
    centering_weight_arg = DeclareLaunchArgument('centering_weight', default_value='2.0')
    corridor_check_distance_arg = DeclareLaunchArgument('corridor_check_distance', default_value='3.0')
    waypoint_step_m_arg = DeclareLaunchArgument('waypoint_step_m', default_value='1.5')
    max_waypoint_steps_arg = DeclareLaunchArgument('max_waypoint_steps', default_value='50')
    goal_xy_tolerance_arg = DeclareLaunchArgument('goal_xy_tolerance', default_value='0.5')
    # Independent of use_global_planner - wins over both other techniques
    # whenever True and a goal is set (see carrot_path_publisher's module
    # docstring, "use_lane_following" paragraph).
    use_lane_following_arg = DeclareLaunchArgument(
        'use_lane_following', default_value='false',
        description='true: navigate along the medial-axis skeleton of the local costmap '
                    '(_build_lane_path) instead of either other technique - exactly centered, '
                    'no oscillation risk, independent of corridor width.')
    lane_simplify_epsilon_arg = DeclareLaunchArgument('lane_simplify_epsilon', default_value='0.3')

    # ParameterValue(..., value_type=...) coerces the launch argument's
    # string substitution to an actual typed parameter - without it,
    # carrot_path_publisher would get a string and reject it with a
    # "Wrong parameter type" error at configure time.
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

    # ground_segmentation is the team's unmodified real-hardware code; it
    # does no self-hit filtering because the real Velodyne mount doesn't
    # clip the chassis the way it does in Gazebo. Interposing this filter
    # downstream keeps their node untouched while stopping the self-hit
    # blob from reaching the costmap.
    self_hit_filter = Node(
        package='local_nav',
        executable='self_hit_filter',
        name='self_hit_filter',
        output='screen',
    )

    # ackermann_steering_controller publishes its odom transform on
    # /ackermann_steering_controller/tf_odometry (tf2_msgs/TFMessage)
    # instead of the standard /tf topic, so tf2_ros buffer clients
    # (including the costmap below) can't see the odom->base_link
    # transform without this relay.
    tf_odom_relay = Node(
        package='local_nav',
        executable='tf_odom_relay',
        name='tf_odom_relay',
        output='screen',
    )

    # UMRR-A4 Type 171 Automotive radar stand-in - Gazebo Classic has no
    # native mmWave radar plugin, so this derives a sparse, FOV-limited
    # target list from the same obstacle geometry the Velodyne already
    # sees and publishes it on the real driver's exact topic/message
    # contract. Fused into local_costmap as a second obstacle_layer
    # source (see local_costmap.yaml's radar_targets entry).
    radar_sim = Node(
        package='local_nav',
        executable='radar_sim',
        name='radar_sim',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # controller_server creates and owns its own "local_costmap" internally
    # by default - a separate nav2_costmap_2d node is redundant. Passing
    # both yaml files to controller_server directly is the correct wiring.
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

    # Only launched when use_global_planner:=true is passed on the command
    # line - not used by the goal-seeking technique this launch defaults
    # to, but kept available for a direct side-by-side comparison.
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[planner_costmap_params, planner_server_params],
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
        }],
    )

    # Real-hardware dry run: converts the exact same TEB Twist into the
    # real vehicle's actual contract. Harmless alongside Gazebo.
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
        carrot_distance_arg,
        scan_distance_arg,
        world_file_arg,
        spawn_x_arg,
        spawn_y_arg,
        spawn_yaw_arg,
        rviz_arg,
        use_global_planner_arg,
        centering_weight_arg,
        corridor_check_distance_arg,
        waypoint_step_m_arg,
        max_waypoint_steps_arg,
        goal_xy_tolerance_arg,
        use_lane_following_arg,
        lane_simplify_epsilon_arg,
        gazebo,
        segment_ground,
        self_hit_filter,
        tf_odom_relay,
        radar_sim,
        controller_server,
        planner_server,
        lifecycle_manager,
        lifecycle_manager_no_planner,
        carrot_path_publisher,
        steering_uart_bridge,
        rviz2,
    ])
