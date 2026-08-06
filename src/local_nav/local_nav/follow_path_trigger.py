"""Sends whatever path planner_tuning_client publishes on plan_topic
straight to controller_server's FollowPath action - the missing link that
makes tune_planner_standalone.launch.py exercise TEB (teb_controller.yaml)
against a global plan, not just SmacPlannerHybrid in isolation. See that
launch file's docstring for the full combined global+local tuning
workflow: click a start/goal in RViz, planner_tuning_client computes the
global plan, THIS node immediately hands it to controller_server, which
drives kinematic_sim's fake robot along it - watch it move in RViz, tune
teb_controller.yaml, restart, repeat.

Logs FollowPath's feedback (distance_to_goal, speed) periodically and the
final result, same spirit as planner_tuning_client's success/failure/
timing logging for the global planner.
"""

import time

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.node import Node


class FollowPathTrigger(Node):

    def __init__(self):
        super().__init__('follow_path_trigger')

        self.declare_parameter('plan_topic', '/tuning_plan')
        self.declare_parameter('controller_id', 'FollowPath')
        self.declare_parameter('feedback_log_period_sec', 1.0)

        self.controller_id = self.get_parameter('controller_id').value
        self.feedback_log_period = self.get_parameter('feedback_log_period_sec').value
        self._last_feedback_log = 0.0
        self._goal_handle = None

        self.create_subscription(
            Path, self.get_parameter('plan_topic').value, self.plan_callback, 1)
        self.action_client = ActionClient(self, FollowPath, 'follow_path')

        self.get_logger().info(
            f'Follow-path trigger ready - will send every path received on '
            f'{self.get_parameter("plan_topic").value} to controller_server.')

    def plan_callback(self, msg):
        if not msg.poses:
            return
        if not self.action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('follow_path action server not available.')
            return

        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()

        goal = FollowPath.Goal()
        goal.path = msg
        goal.controller_id = self.controller_id

        self._start_time = time.time()
        send_future = self.action_client.send_goal_async(goal, feedback_callback=self._on_feedback)
        send_future.add_done_callback(self._on_goal_response)
        self.get_logger().info(f'Sending {len(msg.poses)}-pose path to controller_server.')

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('FollowPath goal rejected by controller_server.')
            return
        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_feedback(self, feedback_msg):
        now = time.time()
        if now - self._last_feedback_log < self.feedback_log_period:
            return
        self._last_feedback_log = now
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'FollowPath progress: distance_to_goal={fb.distance_to_goal:.2f}m, '
            f'speed={fb.speed:.2f}m/s.')

    def _on_result(self, future):
        elapsed = time.time() - self._start_time
        result = future.result()
        status = result.status
        self._goal_handle = None
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'FollowPath SUCCEEDED in {elapsed:.1f}s.')
        else:
            self.get_logger().warn(f'FollowPath ended with status={status} after {elapsed:.1f}s.')


def main(args=None):
    rclpy.init(args=args)
    node = FollowPathTrigger()
    rclpy.spin(node)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
