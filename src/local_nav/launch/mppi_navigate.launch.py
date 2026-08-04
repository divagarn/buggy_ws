"""Map-free local navigation: Gazebo -> segment_ground -> local costmap
-> planner_server (SmacPlannerHybrid, real path planning ~5m ahead)
-> TEB (teb_local_planner::TebLocalPlannerROS) -> ackermann_steering_controller.

Global planner history: NavfnPlanner (smac_planner_server.yaml's predecessor,
planner_server.yaml, kept for reference/rollback) -> SmacPlannerHybrid.
NavfnPlanner's Dijkstra search has no notion of the vehicle's kinematics, so
its shortest path could bend tighter than this Ackermann buggy can actually
steer - drivability was left entirely to TEB to fix up downstream.
SmacPlannerHybrid (Hybrid-A*) searches motion primitives directly, so
minimum_turning_radius (4.4m, same value as TEB's) is a hard constraint on
the global plan itself.

Local controller history: MPPI -> DWB -> TEB.
- MPPI's CostCritic was a soft weighted penalty, and PathAlignCritic's
  pull toward the reference path (which could point straight through a
  wall) was winning that tug-of-war often enough to still hit walls even
  after the self-hit-filter fix. mppi_controller.yaml kept for reference.
- DWB's BaseObstacle critic fixed that (hard-rejects any candidate
  trajectory overlapping a lethal cell outright), but DWB has no native
  Ackermann motion model - it only caps angular velocity, which can still
  imply an unrealistic steering angle at low speed (curvature = wz/vx
  blows up as vx -> 0), the reason dwb_controller.yaml needed a vx_min
  epsilon workaround. dwb_controller.yaml kept for reference.
- TEB (teb_controller.yaml) has native car-like support:
  min_turning_radius/wheelbase are real constraints the trajectory
  optimizer enforces directly, not a per-sample ratio - no equivalent
  division-by-zero case, no epsilon workaround needed.

Naive straight-line reference path -> real planning: carrot_path_publisher
used to just extend a straight line ahead of the vehicle and hand it to
the controller as-is, trusting the controller to deviate around whatever
was in the way. That's not path *planning*, and a straight line into a
wall gave the controller a reference it fundamentally couldn't reconcile
with obstacle avoidance. planner_server (SmacPlannerHybrid) now computes
a real, obstacle-avoiding path through the rolling local costmap out to
~5m ahead, and that computed path is what gets handed to the local
controller - not a straight line.

Unlike navigate.launch.py / gazebo_navigate.launch.py, this path does NOT
use obstacle_detector, steering_calculator_node, or topic_relay_node - the
local planner + controller handle obstacle avoidance directly from the
costmap (built from /non_ground_points_filtered), so the reactive
lateral-histogram chain isn't in the loop at all here.

Two tunable distances, overridable from the command line
(`ros2 launch local_nav mppi_navigate.launch.py carrot_distance:=8.0 ...`):
  - carrot_distance: how far ahead the /plan goal (sent to SmacPlannerHybrid)
    is placed.
  - scan_distance: how far ahead carrot_path_publisher looks when scanning
    for the clearest turn direction - kept larger than carrot_distance so
    a corner is noticed with enough room left to actually turn into it.
TEB's own local horizon (unlike DWB's single dwb_sim_time number) is
governed by several params in teb_controller.yaml directly
(max_global_plan_lookahead_dist, dt_ref, max_samples) - no single
launch-time override for it here.
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
    # World file name only (not a full path) - looked up under
    # buggy_description/worlds/. Defaults match rect_loop_track.world's own
    # spawn point (bottom corridor centerline); override all four together
    # when switching to a world with a different layout, e.g.
    # varying_width_track.world's south leg uses the same spawn as
    # rect_loop_track.world (both are 7m there) so the defaults work for
    # either, but closed_loop_track2.world (Building Editor floor plan)
    # needs spawn_x:=5.0 spawn_y:=5.0.
    world_file_arg = DeclareLaunchArgument(
        'world_file', default_value='rect_loop_track.world',
        description='World file under buggy_description/worlds/, e.g. '
                    'rect_loop_track.world, varying_width_track.world, closed_loop_track2.world')
    spawn_x_arg = DeclareLaunchArgument('spawn_x', default_value='0.0')
    spawn_y_arg = DeclareLaunchArgument('spawn_y', default_value='-9.5')
    spawn_yaw_arg = DeclareLaunchArgument('spawn_yaw', default_value='0.0')
    # false: skip planner_server (SmacPlannerHybrid) entirely and drive TEB
    # off carrot_path_publisher's own straight-line-to-picked-goal fallback
    # instead (see that node's use_global_planner param/docstring).
    use_global_planner_arg = DeclareLaunchArgument(
        'use_global_planner', default_value='true',
        description='true: plan through planner_server/SmacPlannerHybrid before handing '
                    'TEB a path (default). false: skip the global planner, TEB follows a '
                    'straight line to the same obstacle-aware goal instead.')
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')
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
    # (including the costmap above) can't see the odom->base_link
    # transform without this relay.
    tf_odom_relay = Node(
        package='local_nav',
        executable='tf_odom_relay',
        name='tf_odom_relay',
        output='screen',
    )

    # UMRR-A4 Type 171 Automotive radar (smartmicro_ros2_radars) stand-in -
    # Gazebo Classic has no native mmWave radar plugin, so this derives a
    # sparse, FOV-limited target list from the same obstacle geometry the
    # Velodyne already sees and publishes it on the real driver's exact
    # topic/message contract. Fused into local_costmap/planner_costmap as
    # a second obstacle_layer source (see those yamls' radar_targets
    # entry) - sim-only, real_navigate.launch.py uses the actual
    # umrr_ros2_driver node instead once that's wired in.
    radar_sim = Node(
        package='local_nav',
        executable='radar_sim',
        name='radar_sim',
        output='screen',
        # Unlike self_hit_filter (which just passes through
        # ground_segmentation's already-sim-timed header unchanged), this
        # node stamps brand-new messages via self.get_clock().now() each
        # cycle - without use_sim_time, that's wall-clock time while the
        # rest of the stack (TF, costmap) runs on Gazebo's sim clock,
        # which silently drops every observation as "earlier than all the
        # data in the transform cache" (same use_sim_time bug class fixed
        # in bag_navigate.launch.py/real_navigate.launch.py earlier).
        parameters=[{'use_sim_time': True}],
    )

    # controller_server creates and owns its own "local_costmap" internally
    # by default (confirmed via its startup log: "[local_costmap.local_costmap]:
    # Creating Costmap" printed from within the controller_server process) -
    # a separate nav2_costmap_2d node is redundant, and that standalone
    # executable hardcodes its own internal node name ("costmap") regardless
    # of launch-time remapping, which doesn't match this config's
    # "local_costmap" naming at all. Passing both yaml files to
    # controller_server directly is the correct, simpler wiring.
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

    # planner_server creates and owns its own "global_costmap" internally,
    # same as controller_server does for "local_costmap" - a separate
    # rolling window fed by the same /non_ground_points_filtered, used only
    # to plan the ~5m-ahead reference path (see planner_costmap.yaml).
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

    # Real-hardware dry run: converts the exact same DWB Twist into the
    # real vehicle's actual contract (/steering_angle degrees,
    # /final_red_detected, /final_yellow_detected - what uart_sender_node.py
    # subscribes to). Harmless alongside Gazebo - it's a separate topic
    # pair that doesn't affect ackermann_steering_controller - and lets us
    # watch/validate what would actually be sent to the real cart before
    # ever running this on it.
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
        use_global_planner_arg,
        rviz_arg,
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
