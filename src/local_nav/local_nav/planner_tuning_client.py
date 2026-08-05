"""Standalone SmacPlannerHybrid tuning tool - no Gazebo, no robot, no TF
chain, just map_server + planner_server + this client (see
tune_planner_standalone.launch.py). Lets you iterate on
smac_planner_server.yaml (tolerance, minimum_turning_radius, penalties,
motion_model_for_search, ...) by editing the yaml, restarting the launch,
and clicking two points in RViz - none of Gazebo's per-test startup time
or this project's observed CPU/real-time-factor overhead applies here,
since nothing is actually simulating physics or sensors.

Use RViz's "2D Pose Estimate" to set a START point, then "2D Goal Pose" to
set a GOAL - each goal click immediately calls planner_server's
compute_path_to_pose action with use_start=True (both poses given
explicitly, exactly as clicked), so no fake TF/robot pose is needed at
all. Logs success/failure, planning_time, and path length to the
console, and republishes the resulting path on plan_topic (default
/tuning_plan) for RViz's Path display. Re-click "2D Goal Pose" as many
times as you want from the same start; re-click "2D Pose Estimate" to
try a different start.

(The global costmap will log periodic "Timed out waiting for transform
from base_link to odom" warnings the whole time this runs - harmless
here, it's costmap_2d's own robot-pose-for-obstacle-clearing lookup,
which has nothing to do with the explicit start/goal poses this tool
actually plans between.)
"""

import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.node import Node


class PlannerTuningClient(Node):

    def __init__(self):
        super().__init__('planner_tuning_client')

        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('initialpose_topic', '/initialpose')
        self.declare_parameter('plan_topic', '/tuning_plan')
        self.declare_parameter('planner_id', 'GridBased')

        self.planner_id = self.get_parameter('planner_id').value
        self.start_pose = None

        self.plan_pub = self.create_publisher(Path, self.get_parameter('plan_topic').value, 1)
        self.create_subscription(
            PoseWithCovarianceStamped, self.get_parameter('initialpose_topic').value,
            self.initialpose_callback, 10)
        self.create_subscription(
            PoseStamped, self.get_parameter('goal_topic').value, self.goal_callback, 10)

        self.action_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')

        self.get_logger().info(
            'Planner tuning client ready - set a start with "2D Pose Estimate", then a goal '
            'with "2D Goal Pose". Edit smac_planner_server.yaml and restart the launch to '
            'try different parameters.')

    def initialpose_callback(self, msg):
        self.start_pose = PoseStamped(header=msg.header, pose=msg.pose.pose)
        self.get_logger().info(
            f'Start set: ({self.start_pose.pose.position.x:.2f}, '
            f'{self.start_pose.pose.position.y:.2f})')

    def goal_callback(self, msg):
        if self.start_pose is None:
            self.get_logger().warn('No start pose set yet - use "2D Pose Estimate" first.')
            return
        if not self.action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('compute_path_to_pose action server not available.')
            return

        goal = ComputePathToPose.Goal()
        goal.goal = msg
        goal.start = self.start_pose
        goal.use_start = True
        goal.planner_id = self.planner_id

        self._request_time = time.time()
        future = self.action_client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by planner_server.')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future):
        wall_elapsed = time.time() - self._request_time
        result = future.result()
        status = result.status
        path = result.result.path
        planning_time = (
            result.result.planning_time.sec + result.result.planning_time.nanosec / 1e9)

        if status != GoalStatus.STATUS_SUCCEEDED or not path.poses:
            self.get_logger().warn(
                f'Planning FAILED (status={status}, wall time {wall_elapsed:.2f}s) - '
                'no path found between the given start/goal with current '
                'smac_planner_server.yaml settings.')
            return

        length = 0.0
        for a, b in zip(path.poses, path.poses[1:]):
            length += math.hypot(
                b.pose.position.x - a.pose.position.x, b.pose.position.y - a.pose.position.y)

        self.get_logger().info(
            f'Planning SUCCEEDED: {len(path.poses)} poses, {length:.2f}m path length, '
            f'planner-reported planning_time={planning_time:.3f}s, '
            f'wall time {wall_elapsed:.2f}s.')
        self.plan_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = PlannerTuningClient()
    rclpy.spin(node)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
