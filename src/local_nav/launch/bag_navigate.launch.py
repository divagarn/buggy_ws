"""Replay-based test of the local nav stack against real recorded sensor
data (bags/velodyne_uart_bag), instead of Gazebo or the physical vehicle.

Run the bag separately, in another terminal, so you control looping/rate/
pause independently of this launch. --clock is required (see note below):

    ros2 bag play /home/divagar/ros2_r_ws/buggy_demo_01/bags/velodyne_uart_bag --clock -l

Then this launch file brings up the rest: velodyne_static_tf (base_link ->
velodyne - no live driver needed, the bag already has /velodyne_points) ->
segment_ground -> self_hit_filter -> wheel_odometry (/wheel_uart -> /odom +
TF, same integrator as real_navigate.launch.py) -> costmaps -> planner_server
(SmacPlannerHybrid) -> TEB -> steering_uart_bridge (still publishes /steering_angle etc
so you can watch what WOULD be sent, even though nothing here actually
drives the vehicle - see below) + RViz.

This is an OPEN-LOOP test: the bag replays a fixed, already-recorded
sensor stream regardless of what our stack decides to do, so it validates
perception/planning quality (does segment_ground/self_hit_filter/costmap/
SmacPlannerHybrid/TEB behave sensibly against real data?) but NOT closed-loop control -
the vehicle's actual recorded motion in the bag won't change based on our
computed commands. There is no uart_sender_node or uart_bridge here: the
bag already supplies /wheel_uart directly (no live serial device to read
from), and there's no real vehicle to actuate.

use_sim_time is True throughout, matching the recorded data: every message
in the bag keeps its ORIGINAL recording-time header.stamp (Feb 2026), which
is many months behind today's wall-clock time. With use_sim_time False (an
earlier, wrong attempt at this launch file), every node's "now" was the
real wall clock, so every observation looked absurdly stale - the costmap
still published on its own timer (all-free, since nothing ever passed the
staleness check), but marking silently rejected 100% of point cloud
observations, every single cycle, with no error - /plan, /odom, and TF all
looked fine since they don't run that same staleness comparison, which is
what made this look like "lidar data received, just not reaching the
costmap" rather than a clock problem. `ros2 bag play --clock` publishes
/clock advancing at the bag's own original pace, and use_sim_time: true
here makes every node treat that (not the wall clock) as "now" - so a
bag-timestamped observation and "now" agree, and marking works.

metadata.yaml note: /wheel_uart's bag metadata originally listed
'camera_publisher_msgs/msg/SteerSpeed', a package that doesn't exist in
this workspace - ros2 bag play was silently dropping the topic entirely.
The .db3 file's own topics table said 'ground_segmentation_msgs/msg/
SteerSpeed' instead (this workspace's actual message package - almost
certainly the recorder was updated after a package rename without
regenerating metadata.yaml), and after correcting metadata.yaml to match,
replayed /wheel_uart decodes to physically sane values (steering within
+-20deg, smoothly varying; speed in a believable low km/h range) -
confirming the fix, not just papering over the type-not-found warning.
The original file is preserved as metadata.yaml.bak.
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

    costmap_params = os.path.join(pkg_local_nav, 'config', 'local_costmap.yaml')
    teb_params = os.path.join(pkg_local_nav, 'config', 'teb_controller.yaml')
    planner_costmap_params = os.path.join(pkg_local_nav, 'config', 'planner_costmap.yaml')
    planner_server_params = os.path.join(pkg_local_nav, 'config', 'smac_planner_server.yaml')
    rviz_config = os.path.join(pkg_local_nav, 'rviz', 'bag_navigate.rviz')

    carrot_distance_arg = DeclareLaunchArgument('carrot_distance', default_value='13.0')
    scan_distance_arg = DeclareLaunchArgument('scan_distance', default_value='13.0')
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')
    # false: skip planner_server (SmacPlannerHybrid) entirely and drive TEB
    # off carrot_path_publisher's own straight-line-to-picked-goal fallback
    # instead (see that node's use_global_planner param/docstring).
    use_global_planner_arg = DeclareLaunchArgument(
        'use_global_planner', default_value='true',
        description='true: plan through planner_server/SmacPlannerHybrid before handing '
                    'TEB a path (default). false: skip the global planner, TEB follows a '
                    'straight line to the same obstacle-aware goal instead.')
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
    use_global_planner = ParameterValue(LaunchConfiguration('use_global_planner'), value_type=bool)
    centering_weight = ParameterValue(LaunchConfiguration('centering_weight'), value_type=float)
    corridor_check_distance = ParameterValue(LaunchConfiguration('corridor_check_distance'), value_type=float)
    waypoint_step_m = ParameterValue(LaunchConfiguration('waypoint_step_m'), value_type=float)
    max_waypoint_steps = ParameterValue(LaunchConfiguration('max_waypoint_steps'), value_type=int)
    goal_xy_tolerance = ParameterValue(LaunchConfiguration('goal_xy_tolerance'), value_type=float)
    use_lane_following = ParameterValue(LaunchConfiguration('use_lane_following'), value_type=bool)
    lane_simplify_epsilon = ParameterValue(LaunchConfiguration('lane_simplify_epsilon'), value_type=float)

    velodyne_static_tf = Node(
        package='ground_segmentation',
        executable='velodyne_static_tf',
        name='velodyne_static_tf',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    segment_ground = Node(
        package='ground_segmentation',
        executable='segment_ground',
        name='segment_ground',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    self_hit_filter = Node(
        package='local_nav',
        executable='self_hit_filter',
        name='self_hit_filter',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    wheel_odometry = Node(
        package='local_nav',
        executable='wheel_odometry',
        name='wheel_odometry',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[costmap_params, teb_params, {'use_sim_time': True}],
        remappings=[
            ('cmd_vel', '/cmd_vel'),
        ],
    )

    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[planner_costmap_params, planner_server_params, {'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('use_global_planner')),
    )

    # Split in two (rather than one lifecycle_manager with a launch-time-
    # conditional node_names list) because node_names needs a concrete
    # Python list at description-build time - LaunchConfiguration values
    # aren't resolved until launch time, so there's no way to branch on
    # use_global_planner's actual value here. Only one of these two actually
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
            'odom_topic': '/odom',
            'use_global_planner': use_global_planner,
            'centering_weight': centering_weight,
            'corridor_check_distance': corridor_check_distance,
            'waypoint_step_m': waypoint_step_m,
            'max_waypoint_steps': max_waypoint_steps,
            'goal_xy_tolerance': goal_xy_tolerance,
            'use_lane_following': use_lane_following,
            'lane_simplify_epsilon': lane_simplify_epsilon,
            'use_sim_time': True,
        }],
    )

    steering_uart_bridge = Node(
        package='local_nav',
        executable='steering_uart_bridge',
        name='steering_uart_bridge',
        output='screen',
        parameters=[{
            'cmd_vel_topic': '/cmd_vel',
            'use_sim_time': True,
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
        rviz_arg,
        use_global_planner_arg,
        centering_weight_arg,
        corridor_check_distance_arg,
        waypoint_step_m_arg,
        max_waypoint_steps_arg,
        goal_xy_tolerance_arg,
        use_lane_following_arg,
        lane_simplify_epsilon_arg,
        velodyne_static_tf,
        segment_ground,
        self_hit_filter,
        wheel_odometry,
        controller_server,
        planner_server,
        lifecycle_manager,
        lifecycle_manager_no_planner,
        carrot_path_publisher,
        steering_uart_bridge,
        rviz2,
    ])
