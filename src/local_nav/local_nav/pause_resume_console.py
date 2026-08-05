"""Interactive terminal control for the closed-loop map navigation launches
(closed_loop_navigate_slam.launch.py / real_closed_loop_navigate_slam.launch.py):
type ":pause_n_l" + Enter in the SAME terminal the launch is running in to
halt navigation - planner_server/controller_server are deactivated via
nav2_lifecycle_manager's own manage_nodes service (PAUSE), and this node
also publishes one immediate zero Twist as a belt-and-suspenders stop.
While paused, use RViz's "2D Pose Estimate" tool to give slam_toolbox the
vehicle's actual pose (slam_toolbox's localization mode listens on
/initialpose the same way AMCL does) - then type ":resume" + Enter to
reactivate planner_server/controller_server and carry on.

This deliberately does NOT touch slam_toolbox itself - it keeps running
and scan-matching throughout. Only the planner_server/controller_server
managed by lifecycle_manager_local_nav get paused/resumed; slam_toolbox is
a plain (non-lifecycle) node in this launch, and setting a correct
/initialpose while it's live is the normal, supported slam_toolbox
workflow - no separate pause step needed for it.

Reads stdin directly (a background thread, so it doesn't block rclpy's
executor) rather than a ROS topic/service, since this is meant to be typed
straight into the launch's own terminal, matching how the operator is
already watching that terminal's log output.
"""

import sys
import threading

import rclpy
from geometry_msgs.msg import Twist
from nav2_msgs.srv import ManageLifecycleNodes
from rclpy.node import Node


class PauseResumeConsole(Node):

    def __init__(self):
        super().__init__('pause_resume_console')

        self.declare_parameter('lifecycle_manager_name', 'lifecycle_manager_local_nav')
        self.declare_parameter('cmd_vel_topic', '/ackermann_steering_controller/reference_unstamped')

        manager_name = self.get_parameter('lifecycle_manager_name').value
        self.manage_nodes_client = self.create_client(
            ManageLifecycleNodes, f'/{manager_name}/manage_nodes')
        self.cmd_pub = self.create_publisher(
            Twist, self.get_parameter('cmd_vel_topic').value, 1)

        self.get_logger().info(
            'Pause/resume console ready - type ":pause_n_l" + Enter in this terminal to '
            'stop navigation and set a correct pose via RViz\'s "2D Pose Estimate", then '
            '":resume" + Enter to continue.')

        self._stdin_thread = threading.Thread(target=self._read_stdin, daemon=True)
        self._stdin_thread.start()

    def _read_stdin(self):
        for line in sys.stdin:
            command = line.strip().lower()
            if command == ':pause_n_l':
                self._pause()
            elif command == ':resume':
                self._resume()
            elif command:
                self.get_logger().warn(
                    f'Unrecognized command "{command}" - use ":pause_n_l" or ":resume".')

    def _call_manage_nodes(self, command_value, label):
        if not self.manage_nodes_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(
                f'{label} FAILED: lifecycle manager service not available.')
            return
        request = ManageLifecycleNodes.Request()
        request.command = command_value
        future = self.manage_nodes_client.call_async(request)
        future.add_done_callback(lambda f, label=label: self._on_manage_nodes_done(f, label))

    def _on_manage_nodes_done(self, future, label):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f'{label} succeeded.')
            else:
                self.get_logger().error(f'{label} FAILED (service returned success=false).')
        except Exception as exc:
            self.get_logger().error(f'{label} FAILED: {exc}')

    def _pause(self):
        self.get_logger().warn(
            'PAUSING navigation - deactivating planner_server/controller_server. '
            'Set the vehicle\'s real pose with RViz\'s "2D Pose Estimate" tool now, '
            'then type ":resume".')
        self.cmd_pub.publish(Twist())
        self._call_manage_nodes(ManageLifecycleNodes.Request.PAUSE, 'Pause')

    def _resume(self):
        self.get_logger().warn('RESUMING navigation - reactivating planner_server/controller_server.')
        self._call_manage_nodes(ManageLifecycleNodes.Request.RESUME, 'Resume')


def main(args=None):
    rclpy.init(args=args)
    node = PauseResumeConsole()
    rclpy.spin(node)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
