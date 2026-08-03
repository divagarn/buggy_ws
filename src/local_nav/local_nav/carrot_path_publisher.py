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

Goal heading smoothing (added when the global planner became
SmacPlannerHybrid): NavfnPlanner is a Dijkstra search over position only -
it completely ignores the goal PoseStamped's orientation, so a picked
heading that flipped a few degrees cycle to cycle (raycast noise near
_pick_goal_heading's switching margin, not just real corners) had no
effect on the planned path shape. SmacPlannerHybrid is a real kinematic
planner where the goal orientation is a hard endpoint constraint the curve
must terminate facing - the same per-cycle heading noise that was
invisible before now reshapes the entire planned curve every ~1s
(measured live: end-of-path heading oscillating by 5-15deg practically
every cycle, even mid-corridor with nothing nearby).

_smoothed_yaw damps that small, noise-scale wobble but deliberately does
NOT lag on a large, genuine heading change (see GOAL_HEADING_SNAP_DEG): a
first pass here used a plain low-pass filter on every change, which
technically fixed the noise but then took 5-8 cycles (5-8s) to converge on
a real ~90deg corner - during which the vehicle kept driving forward at
up to 2.2m/s, eating right into the extra reaction room scan_distance was
supposed to buy it. Confirmed live: the vehicle drove itself into an
inside corner with no >=4.4m-radius Dubin path left to reach a large,
obviously-open area it should have caught much earlier, and
SmacPlannerHybrid returned "no valid path found" for minutes straight
(not a bug - genuinely no forward-only path existed from that position).
Large, clearly-real corrections now snap through in one cycle; only
sub-threshold wobble gets damped.

Centerline tracking (straight sections only): the costmap-inflation-
gradient approach to centering (local_costmap.yaml/planner_costmap.yaml's
inflation_radius tuned to roughly half the corridor width, see those
files' comments) is an indirect approximation - it depends on a specific
corridor width and breaks if width varies along the route. _centerline_lateral_bias()
instead directly clusters /non_ground_points_filtered into a left-wall and
a right-wall per forward-distance slice and biases the goal toward their
literal midpoint - a more precise center for confirmed straight corridor
segments than an inflation-gradient guess. Deliberately NOT used at
corners/openings/T-junctions, where one side's wall may be missing,
farther away, or absent entirely: the confidence check below (enough
consecutive both-sides bins, roughly consistent width) is what SmacPlannerHybrid's
turning-radius search and carrot_path_publisher's gap-scan already handle
correctly (see this whole module's history above) - this feature only
ever nudges the goal sideways when it's confident it's looking at a plain
two-wall corridor, and falls back to zero bias (i.e. today's behavior)
the instant that confidence check fails, rather than trying to be clever
about partial/ambiguous wall data.
"""

import math

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from nav2_msgs.action import ComputePathToPose, FollowPath

LETHAL_THRESHOLD = 90  # nav2_costmap_2d: 253 lethal / 254 inscribed / 0-252 inflation decay

# base_link -> velodyne static mount offset (velodyne_static_tf.py), no
# rotation currently (roll=pitch=yaw=0), so sensor-frame -> base_link-frame
# is just this translation. Same values as self_hit_filter.py, duplicated
# rather than imported since that file is a standalone node script, not a
# shared library, and it's only two numbers.
SENSOR_OFFSET_X = 1.73
SENSOR_OFFSET_Y = 0.0


class CarrotPathPublisher(Node):

    def __init__(self):
        super().__init__('carrot_path_publisher')

        self.carrot_distance = self.declare_parameter('carrot_distance', 5.0).value
        self.scan_distance = self.declare_parameter('scan_distance', 9.0).value
        self.resend_period_sec = self.declare_parameter('resend_period_sec', 1.0).value
        self.controller_id = self.declare_parameter('controller_id', 'FollowPath').value
        self.planner_id = self.declare_parameter('planner_id', 'GridBased').value
        # Low-pass filter weight applied to the picked goal heading each
        # cycle (see module docstring's "Goal heading smoothing" note) -
        # 1.0 disables smoothing entirely (old NavfnPlanner-era behavior).
        self.goal_heading_alpha = self.declare_parameter('goal_heading_alpha', 0.4).value
        # Above this, a heading change is treated as a real corner (snap
        # immediately) rather than frame-to-frame raycast noise (damp via
        # goal_heading_alpha) - comfortably above the ~5-15deg noise band
        # measured live mid-corridor, comfortably below a genuine turn.
        self.goal_heading_snap_deg = self.declare_parameter('goal_heading_snap_deg', 25.0).value
        # How much farther a candidate heading must stay clear than the
        # current best before it's allowed to win (see _pick_goal_heading).
        # 0.3m was tuned back when NavfnPlanner ignored goal orientation
        # entirely, so a spurious switch only nudged the *reference* path a
        # little and MPPI/DWB/TEB just tracked it - harmless. It's too
        # small now: normal raycast/costmap noise on a wide-open straight
        # (e.g. the 7m corridor in rect_loop_track.world) still exceeds
        # 0.3m often enough to swap between +-15deg candidates almost every
        # cycle, and since the goal orientation is now a hard endpoint
        # constraint for SmacPlannerHybrid, that swap visibly S-curves the
        # vehicle down what should be a straight line. Real corners show a
        # multi-meter clearance gap between continuing straight and turning
        # (measured live: often 5m+), so raising this doesn't blunt corner
        # detection at all.
        self.heading_switch_margin_m = self.declare_parameter('heading_switch_margin_m', 1.5).value
        self._smoothed_yaw = None
        odom_topic = self.declare_parameter(
            'odom_topic', '/ackermann_steering_controller/odometry').value

        # Centerline tracking params - see module docstring's "Centerline
        # tracking" note. Distances in base_link-frame (forward=x, left=y).
        self.centerline_enabled = self.declare_parameter('centerline_enabled', True).value
        self.centerline_bin_size = self.declare_parameter('centerline_bin_size', 1.0).value
        # Below this many both-sides bins, there isn't enough confirmed
        # corridor ahead to trust a centerline - falls back to zero bias.
        self.centerline_min_bins = self.declare_parameter('centerline_min_bins', 4).value
        # A "wall" candidate must be within this lateral band to count -
        # too close (< side_min) risks catching the vehicle's own
        # near-field noise self_hit_filter didn't fully remove; too far
        # (> side_max) risks pairing with something that isn't actually
        # this corridor's wall (e.g. a distant wall past an opening).
        self.centerline_side_min = self.declare_parameter('centerline_side_min', 0.5).value
        self.centerline_side_max = self.declare_parameter('centerline_side_max', 5.0).value
        # Corridor width (left_wall - right_wall) must stay within this
        # much of itself across all valid bins - a real straight corridor
        # has near-constant width; a corner or opening does not, and this
        # is what actually keeps the feature out of corners (not a
        # separate "is this a corner" flag).
        self.centerline_width_tolerance = self.declare_parameter('centerline_width_tolerance', 1.0).value
        # Sanity cap on how far the goal can be nudged sideways in one
        # cycle, independent of the confidence check above.
        self.centerline_max_bias = self.declare_parameter('centerline_max_bias', 2.0).value
        self.latest_cloud_xy = None
        self.create_subscription(
            PointCloud2, '/non_ground_points_filtered', self.cloud_callback, 2)

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
        # Visualizes what _centerline_lateral_bias() actually detected each
        # cycle (base_link frame) - green when confident enough to bias the
        # goal, red when it bailed out (not enough both-sides bins, or
        # width too inconsistent - see that method's docstring). Lets you
        # watch the real detected wall midpoints against the track's own
        # lane stripe/geometry instead of just trusting the math.
        self.centerline_marker_pub = self.create_publisher(Marker, '/centerline_marker', 1)

        self._planning_in_progress = False

        self.timer = self.create_timer(self.resend_period_sec, self.request_plan)

        self.get_logger().info(
            f'Local path planner initialized (goal {self.carrot_distance}m ahead, '
            f'scanning {self.scan_distance}m ahead for turns, replans every '
            f'{self.resend_period_sec}s via planner_server/SmacPlannerHybrid, '
            f'goal_heading_alpha={self.goal_heading_alpha}, '
            f'centerline_enabled={self.centerline_enabled}, odom from {odom_topic})'
        )

    def odom_callback(self, msg):
        self.latest_odom = msg

    def costmap_callback(self, msg):
        self.latest_costmap = msg

    def cloud_callback(self, msg):
        if msg.width == 0:
            self.latest_cloud_xy = None
            return
        pts = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
        # Sensor frame -> base_link frame (forward=x, left=y), same
        # translation-only conversion as self_hit_filter.py.
        xy = pts[:, :2].copy()
        xy[:, 0] += SENSOR_OFFSET_X
        xy[:, 1] += SENSOR_OFFSET_Y
        self.latest_cloud_xy = xy

    def _centerline_lateral_bias(self, max_forward):
        """Return a lateral offset (base_link-frame, +y=left) to nudge the
        goal toward the literal midpoint between confirmed left/right
        corridor walls, or None if not confident this is a plain straight
        corridor (see module docstring's "Centerline tracking" note)."""
        if not self.centerline_enabled or self.latest_cloud_xy is None:
            return None

        xy = self.latest_cloud_xy
        bin_edges = np.arange(1.0, max_forward, self.centerline_bin_size)
        bin_xs = []
        centers = []
        widths = []
        for x_lo in bin_edges:
            x_hi = x_lo + self.centerline_bin_size
            in_bin = xy[(xy[:, 0] >= x_lo) & (xy[:, 0] < x_hi)]
            if in_bin.shape[0] == 0:
                continue
            y = in_bin[:, 1]
            left = y[(y > self.centerline_side_min) & (y < self.centerline_side_max)]
            right = y[(y < -self.centerline_side_min) & (y > -self.centerline_side_max)]
            if left.size == 0 or right.size == 0:
                continue
            left_wall = left.min()   # nearest point on the left
            right_wall = right.max()  # nearest point on the right (least negative)
            bin_xs.append(x_lo + self.centerline_bin_size / 2.0)
            centers.append((left_wall + right_wall) / 2.0)
            widths.append(left_wall - right_wall)

        confident = (
            len(centers) >= self.centerline_min_bins
            and max(widths) - min(widths) <= self.centerline_width_tolerance
        ) if centers else False

        bias = None
        if confident:
            bias = float(np.mean(centers))
            bias = max(-self.centerline_max_bias, min(self.centerline_max_bias, bias))

        self._publish_centerline_marker(bin_xs, centers, confident)
        return bias

    def _publish_centerline_marker(self, bin_xs, centers, confident):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "centerline_detected"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.15
        if confident:
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 0.0, 1.0
        else:
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.0, 0.0, 0.6
        for x, y in zip(bin_xs, centers):
            p = Point()
            # float(): numpy.float32/float64 (from np.arange/array.min/max
            # above) fail geometry_msgs' field type assertion, which isn't
            # caught anywhere upstream - it just kills the node outright.
            p.x, p.y, p.z = float(x), float(y), 0.1
            marker.points.append(p)
        self.centerline_marker_pub.publish(marker)

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
            if dist > best_dist + self.heading_switch_margin_m:
                best_dist = dist
                best_heading = heading

        goal_dist = max(min(best_dist, self.carrot_distance), 1.0)

        # Low-pass filter on the heading itself (angle-wrap-safe): converge
        # toward best_heading over a few cycles rather than snapping to it,
        # so a genuinely better opening still wins out but frame-to-frame
        # raycast noise doesn't reshape SmacPlannerHybrid's entire curve
        # every cycle. See module docstring.
        if self._smoothed_yaw is None:
            self._smoothed_yaw = best_heading
        else:
            diff = math.atan2(
                math.sin(best_heading - self._smoothed_yaw),
                math.cos(best_heading - self._smoothed_yaw))
            if abs(math.degrees(diff)) >= self.goal_heading_snap_deg:
                self._smoothed_yaw = best_heading  # real corner - commit now, don't lag
            else:
                self._smoothed_yaw += self.goal_heading_alpha * diff

        return self._smoothed_yaw, goal_dist

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

        goal_x = pose.position.x + goal_dist * math.cos(goal_yaw)
        goal_y = pose.position.y + goal_dist * math.sin(goal_yaw)

        # Straight-corridor centerline nudge (see module docstring) - bias
        # is in base_link-frame (+y=left of the vehicle's current actual
        # heading, not goal_yaw), so it's rotated by the vehicle's current
        # yaw, not goal_yaw, before being added to the goal position.
        lateral_bias = self._centerline_lateral_bias(goal_dist)
        if lateral_bias is not None:
            goal_x += -math.sin(yaw) * lateral_bias
            goal_y += math.cos(yaw) * lateral_bias

        goal_pose = PoseStamped()
        goal_pose.header = header
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
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
