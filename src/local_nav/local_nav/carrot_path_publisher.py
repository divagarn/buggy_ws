#!/usr/bin/env python3
"""Local path planner front-end.

Earlier version of this node just synthesized a naive straight line ahead
of the vehicle and handed it to the local controller (MPPI/DWB) as the
FollowPath reference, trusting the controller to deviate from it around
obstacles. In practice the controller's path-following critics/costs could
still pull it into a wall that the naive line ran straight through - there
was no actual path *planning* happening, just reactive trajectory scoring.

This version asks planner_server (nav2_navfn_planner, over the rolling
local costmap built from /non_ground_points_filtered) to compute a real,
obstacle-avoiding path from the vehicle's current pose out to a goal point
~carrot_distance meters ahead, then hands that computed path to the local
controller instead. Re-requested on a timer so the goal continuously
tracks ahead of the vehicle's latest pose.

The goal direction is NOT simply "current heading" - that degenerates
exactly at real corners, where the corridor bends away from wherever the
vehicle currently happens to be pointed, and NavFn can only ever try to
reach a point straight ahead: it can bend the path a little to dodge the
nearest wall, but there's nothing pulling it to actually commit to the
turn, so the vehicle just nudges along the wall instead of turning.
_pick_goal_heading scans a spread of candidate headings against
/global_costmap/costmap (the same costmap NavfnPlanner itself plans over)
and measures how far each stays clear of lethal/inflated/unknown cells,
then targets whichever direction is genuinely open - biased toward
current heading so it doesn't zigzag when several directions are equally
clear.

The scan distance (scan_distance) is deliberately larger than the actual
placed goal distance (carrot_distance): with a car-like Ackermann vehicle
that has a real minimum turning radius (4.4m here) and can't rotate in
place or reverse out of a mistake, waiting until the wall is only
carrot_distance away to notice the corner leaves no room left to complete
the arc - DWB ends up maxed out on steering with almost no forward speed,
unable to finish the turn, and just freezes. Scanning further out lets it
notice the corner and start bending toward it while there's still enough
room to sweep a physically achievable turn.
"""

import math

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, FollowPath

LETHAL_THRESHOLD = 90  # nav2_costmap_2d: 253 lethal / 254 inscribed / 0-252 inflation decay


class CarrotPathPublisher(Node):

    def __init__(self):
        super().__init__('carrot_path_publisher')

        self.carrot_distance = self.declare_parameter('carrot_distance', 5.0).value
        self.scan_distance = self.declare_parameter('scan_distance', 9.0).value
        self.resend_period_sec = self.declare_parameter('resend_period_sec', 1.0).value
        self.controller_id = self.declare_parameter('controller_id', 'FollowPath').value
        self.planner_id = self.declare_parameter('planner_id', 'GridBased').value
        odom_topic = self.declare_parameter(
            'odom_topic', '/ackermann_steering_controller/odometry').value

        self.latest_odom = None
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)

        self.latest_costmap = None
        costmap_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(
            OccupancyGrid, '/global_costmap/costmap', self.costmap_callback, costmap_qos)

        self._planner_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')
        self._controller_client = ActionClient(self, FollowPath, 'follow_path')

        # Same computed Path also published to a plain topic for RViz -
        # the action goals themselves aren't visible there (Fixed Frame:
        # odom to see it).
        self.plan_pub = self.create_publisher(Path, '/plan', 1)

        self._planning_in_progress = False

        self.timer = self.create_timer(self.resend_period_sec, self.request_plan)

        self.get_logger().info(
            f'Local path planner initialized (goal {self.carrot_distance}m ahead, '
            f'scanning {self.scan_distance}m ahead for turns, replans every '
            f'{self.resend_period_sec}s via planner_server/NavfnPlanner, '
            f'odom from {odom_topic})'
        )

    def odom_callback(self, msg):
        self.latest_odom = msg

    def costmap_callback(self, msg):
        self.latest_costmap = msg

    def _yaw_from_quaternion(self, q):
        # Z-axis yaw only, matches quaternion_from_yaw used elsewhere in this project
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _yaw_to_quaternion(self, yaw):
        return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

    def _clear_distance(self, x0, y0, heading, max_dist, step=0.2):
        """March from (x0, y0) along heading in the costmap frame, return
        distance to the first lethal/inflated/unknown cell, capped at
        max_dist. Costmap frame matches x0/y0's frame (odom)."""
        costmap = self.latest_costmap
        info = costmap.info
        w, h = info.width, info.height
        res = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y
        data = costmap.data
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        d = 0.0
        while d < max_dist:
            d += step
            gx = int((x0 + d * cos_h - ox) / res)
            gy = int((y0 + d * sin_h - oy) / res)
            if gx < 0 or gx >= w or gy < 0 or gy >= h:
                return d  # ran off the known window - treat edge as the limit
            val = data[gy * w + gx]
            if val < 0 or val >= LETHAL_THRESHOLD:
                return d
        return max_dist

    def _pick_goal_heading(self, x0, y0, current_yaw):
        """Return (heading, goal_distance) - scans candidate headings
        around current_yaw out to scan_distance (far enough to notice a
        corner before running out of room to turn into it) and picks
        whichever stays clear the longest, biased toward current_yaw so it
        doesn't zigzag between two similarly-open directions. The returned
        goal_distance is capped at carrot_distance - scan_distance is only
        used to pick the direction, not to place a far-off goal NavFn/DWB
        would have to commit to immediately."""
        if self.latest_costmap is None:
            return current_yaw, self.carrot_distance

        best_heading = current_yaw
        best_dist = self._clear_distance(x0, y0, current_yaw, self.scan_distance)

        # Widening spread: nearer-to-current-heading options are checked
        # first and only displaced by a clearly better (not just marginally
        # different) alternative, so it doesn't turn every cycle chasing
        # noise between near-equal directions.
        for offset_deg in (15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90):
            heading = current_yaw + math.radians(offset_deg)
            dist = self._clear_distance(x0, y0, heading, self.scan_distance)
            if dist > best_dist + 0.3:
                best_dist = dist
                best_heading = heading

        goal_dist = max(min(best_dist, self.carrot_distance), 1.0)
        return best_heading, goal_dist

    def request_plan(self):
        if self._planning_in_progress:
            return
        if self.latest_odom is None:
            self.get_logger().warn('No odometry received yet, skipping plan request', throttle_duration_sec=5.0)
            return
        if not self._planner_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warn('compute_path_to_pose action server not available yet', throttle_duration_sec=5.0)
            return

        pose = self.latest_odom.pose.pose
        yaw = self._yaw_from_quaternion(pose.orientation)
        header = self.latest_odom.header  # frame_id "odom", matches costmaps' global_frame

        goal_yaw, goal_dist = self._pick_goal_heading(pose.position.x, pose.position.y, yaw)
        qx, qy, qz, qw = self._yaw_to_quaternion(goal_yaw)

        start = PoseStamped()
        start.header = header
        start.pose = pose

        goal_pose = PoseStamped()
        goal_pose.header = header
        goal_pose.pose.position.x = pose.position.x + goal_dist * math.cos(goal_yaw)
        goal_pose.pose.position.y = pose.position.y + goal_dist * math.sin(goal_yaw)
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.x = qx
        goal_pose.pose.orientation.y = qy
        goal_pose.pose.orientation.z = qz
        goal_pose.pose.orientation.w = qw

        goal = ComputePathToPose.Goal()
        goal.goal = goal_pose
        goal.start = start
        goal.use_start = True
        goal.planner_id = self.planner_id

        self._planning_in_progress = True
        send_future = self._planner_client.send_goal_async(goal)
        send_future.add_done_callback(self._plan_goal_response)

    def _plan_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('compute_path_to_pose goal rejected', throttle_duration_sec=5.0)
            self._planning_in_progress = False
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._plan_result)

    def _plan_result(self, future):
        self._planning_in_progress = False
        try:
            path = future.result().result.path
        except Exception as exc:
            self.get_logger().warn(f'compute_path_to_pose failed: {exc}', throttle_duration_sec=5.0)
            return

        if not path.poses:
            self.get_logger().warn('Planner returned an empty path', throttle_duration_sec=5.0)
            return

        self.plan_pub.publish(path)

        if not self._controller_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warn('follow_path action server not available yet', throttle_duration_sec=5.0)
            return

        follow_goal = FollowPath.Goal()
        follow_goal.path = path
        follow_goal.controller_id = self.controller_id
        self._controller_client.send_goal_async(follow_goal)


def main(args=None):
    rclpy.init(args=args)
    node = CarrotPathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
