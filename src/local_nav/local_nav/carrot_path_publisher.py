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

Goal-seeking (/goal_pose, e.g. RViz's "2D Goal Pose"): everything above is
purely reactive - it has no notion of a destination, just "stay clear of
obstacles, keep going". Confirmed live (tools/carrot_heading_tuner.py, an
L/T-shaped test corridor) that this made every carrot goal converge on the
same left-biased route regardless of where the actual destination was, and
that it would happily drive straight through/past a goal that didn't
happen to also be wherever looked clearest. With a goal set,
_pick_goal_heading centers its candidate scan on the bearing to that goal
instead of current heading (widened to +-165deg, since reaching a goal
that's behind a corridor bend can require a heading nowhere near the
direct bearing until the corridor opens up) and caps goal_distance at the
remaining distance to the goal so it stops there rather than overshooting.
request_plan itself stops replanning entirely once within goal_xy_tolerance
of the goal, holding until a new /goal_pose arrives.

use_global_planner=False with a goal set goes one step further:
_split_into_clear_segments builds a whole multi-waypoint polyline to the
goal in a single request_plan() cycle (by calling _pick_goal_heading
repeatedly, waypoint_step_m at a time, internally - not one carrot per
real replan tick), so TEB gets a path that already bends around a corner
instead of a single short segment it'd otherwise only see one piece of
per cycle.

use_lane_following=True (independent of use_global_planner - checked
first in request_plan, wins over both other modes whenever a goal is
set) is a third, structurally different technique: _build_lane_path
computes the medial axis (skeleton) of the local costmap's free space -
the set of points equidistant from the nearest walls on every side, i.e.
the true geometric centerline of every corridor/junction/turn in the
map - and searches along it from the vehicle's position to the goal,
instead of picking headings step by step. Confirmed live
(tools/carrot_heading_tuner.py) that _split_into_clear_segments/
_pick_goal_heading's centering_weight has a real ceiling (oscillates
above ~2.5 on a test corridor) and still only weakly centers in wide-open
areas; the skeleton is exactly centered everywhere, independent of
corridor width, with no oscillation risk since it's a real graph search,
not a greedy per-step heuristic. The raw skeleton path is pixel-jagged
(a discretized 1-cell-wide ridge), so it's simplified via Douglas-Peucker
(bounded perpendicular deviation from the true skeleton shape, NOT "cut
straight to whichever point is still technically clear" - that greedy
alternative was tried first and throws away the whole centering benefit,
producing the exact same corner-hugging diagonal _split_into_clear_
segments already had a problem with, confirmed by rendering both side by
side) before being handed to _build_straight_path the same as the other
two modes' waypoint lists.

auto_return_to_start (buggy_mapping's closed-loop navigation): drive the
whole mapped loop once and stop back where you started, from a single
"2D Goal Pose" click, rather than needing a second manual goal for the
return leg or looping forever. start_pose_xy is recorded once, from the
node's very first odometry message. When the first goal is reached,
request_plan auto-chains exactly one more goal - start_pose_xy - instead
of holding; reaching THAT one (detected by the just-reached goal already
being start_pose_xy) holds for real. Works with all three techniques
above, since it only ever changes what latest_goal_xy is, the same input
each of them already reads.
"""

import math

import networkx as nx
import numpy as np
import rclpy
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point, PoseArray, Pose
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

# Waypoint spacing for _build_straight_path (use_global_planner=False mode).
STRAIGHT_PATH_STEP_M = 0.5


class CarrotPathPublisher(Node):

    def __init__(self):
        super().__init__('carrot_path_publisher')

        self.carrot_distance = self.declare_parameter('carrot_distance', 5.0).value
        self.scan_distance = self.declare_parameter('scan_distance', 9.0).value
        self.resend_period_sec = self.declare_parameter('resend_period_sec', 1.0).value
        self.controller_id = self.declare_parameter('controller_id', 'FollowPath').value
        self.planner_id = self.declare_parameter('planner_id', 'GridBased').value
        # False skips planner_server (SmacPlannerHybrid) entirely: instead of
        # ComputePathToPose, a plain interpolated straight line from the
        # current pose to the same obstacle-aware goal _pick_goal_heading
        # already picks (see that method - it's still costmap-driven, not a
        # blind forward line) is handed straight to TEB via FollowPath. TEB's
        # own local-costmap obstacle avoidance is what keeps it off walls in
        # this mode, with no kinematic global search upstream of it. See
        # _build_straight_path.
        self.use_global_planner = self.declare_parameter('use_global_planner', True).value
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
        # Goal-seeking mode only (see _pick_goal_heading docstring):
        # "closest to the goal" alone has no preference for staying away
        # from walls, so around a bend it'll hug the inside edge as tight
        # as clear_distance allows - confirmed live with
        # tools/carrot_heading_tuner.py by drawing an actual curved
        # corridor and watching the picked path graze the inner wall the
        # whole way around instead of favoring the middle. 0 disables
        # centering entirely. 2.0 (rather than an equal-weight 1.0) is
        # deliberate: measured on a wide-then-narrow test corridor in that
        # same tool, 2.0 roughly halved the path's peak offset from the
        # true centerline (4.36m -> 2.06m) with no regression on a tight
        # 90deg corridor test. Don't casually raise this further - above
        # ~2.5 in the same test the vehicle got torn between "avoid the
        # left wall" and "avoid the right wall" candidates scoring nearly
        # equally and oscillated in place rather than committing to
        # either; this is a real ceiling on the greedy single-step scoring
        # here, not just an untried "more is better" knob. corridor_check_
        # distance caps how far sideways is worth checking - reward stops
        # accruing past this much clearance on the tighter side, so a
        # candidate in wide-open space doesn't keep scoring better forever
        # just for being in the middle of nowhere.
        self.centering_weight = self.declare_parameter('centering_weight', 2.0).value
        self.corridor_check_distance = self.declare_parameter('corridor_check_distance', 3.0).value
        # use_global_planner=False + a real /goal_pose: how far each of
        # _split_into_clear_segments' internal steps advances (independent
        # of carrot_distance - this is building a whole path in one
        # request_plan() cycle, not one per-cycle carrot), and how many
        # such steps it's allowed to take before giving up on a still-
        # distant goal. 1.5m keeps individual steps small enough to
        # reliably corner (see that method's docstring for why large,
        # greedy steps fail); 50 steps at 1.5m covers 75m of travel, far
        # more than one scan_distance/carrot_distance-scale corridor
        # segment would ever need - cheap to compute (all raycasts stay
        # within latest_costmap's own rolling window), so a generous
        # budget costs nothing but a few extra iterations once actually
        # near the goal.
        self.waypoint_step_m = self.declare_parameter('waypoint_step_m', 1.5).value
        self.max_waypoint_steps = self.declare_parameter('max_waypoint_steps', 50).value
        # Independent of use_global_planner - see module docstring's
        # "use_lane_following" paragraph for why this is a structurally
        # different technique (medial-axis skeleton search) rather than
        # another tuning of the greedy step-by-step approach. Wins over
        # both other modes in request_plan whenever True and a goal is set.
        self.use_lane_following = self.declare_parameter('use_lane_following', False).value
        # Douglas-Peucker tolerance (max perpendicular deviation, meters)
        # for simplifying the raw pixel-jagged skeleton path down to a few
        # waypoints - see _build_lane_path's docstring for why this must
        # be deviation-bounded simplification, not greedy "shortcut to
        # whichever point is still clear".
        self.lane_simplify_epsilon = self.declare_parameter('lane_simplify_epsilon', 0.3).value
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
        # Recorded once, from the vehicle's very first odometry message -
        # see odom_callback and auto_return_to_start below.
        self.start_pose_xy = None
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)

        # Optional fixed destination (RViz "2D Goal Pose" publishes here by
        # convention). Without one, _pick_goal_heading falls back to its
        # original purely-reactive behavior (scan around current heading,
        # go wherever's clearest, no notion of "arriving" anywhere) - see
        # that method's docstring for why a goal changes the scan entirely.
        self.latest_goal_xy = None
        self.goal_xy_tolerance = self.declare_parameter('goal_xy_tolerance', 0.5).value
        # For closed-loop map-based navigation (buggy_mapping): drive one
        # full lap and stop back where you started, rather than needing a
        # second manual goal click for the return leg or looping forever.
        # When the FIRST /goal_pose goal is reached, automatically chains
        # exactly one more goal - start_pose_xy - instead of holding; once
        # THAT one is reached (i.e. we're back at the recorded start), it
        # holds for real. See request_plan's "goal reached" branch.
        self.auto_return_to_start = self.declare_parameter('auto_return_to_start', False).value
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_callback, 1)

        # /global_costmap/costmap is planner_server's own internal costmap -
        # when use_global_planner is False, planner_server isn't launched at
        # all (see mppi/real/bag_navigate.launch.py's IfCondition on
        # planner_server), so that topic never publishes and latest_costmap
        # stays None forever. _pick_goal_heading's very first check
        # (latest_costmap is None -> return current_yaw unchanged) then
        # fires every single cycle - not "scanning and finding nothing
        # open", just never scanning at all, which is why the straight-line
        # fallback previously drove dead ahead through obstacles instead of
        # turning. controller_server's own local_costmap runs unconditionally
        # regardless of use_global_planner, so that's the correct source to
        # fall back to here.
        default_costmap_topic = (
            '/global_costmap/costmap' if self.use_global_planner else '/local_costmap/costmap')
        costmap_topic = self.declare_parameter('costmap_topic', default_costmap_topic).value
        self.latest_costmap = None
        costmap_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(
            OccupancyGrid, costmap_topic, self.costmap_callback, costmap_qos)

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
        # The chosen /goal_pose itself (a plain sphere) - republished every
        # request_plan cycle (not just once in goal_pose_callback) because
        # this publisher, like plan_pub/centerline_marker_pub above, has no
        # transient-local durability: a late-joining RViz would otherwise
        # never see a goal that was set before it connected.
        self.goal_marker_pub = self.create_publisher(Marker, '/goal_marker', 1)
        # use_lane_following's specific path (see _build_lane_path) - drawn
        # separately from the generic /plan Path (which every mode
        # publishes identically) so you can actually tell, in RViz, that a
        # given route came from the skeleton search and not the greedy
        # step-by-step techniques.
        self.lane_marker_pub = self.create_publisher(Marker, '/lane_marker', 1)
        # The actual subgoal waypoints the straight-segment/lane-following
        # techniques split the main /goal_pose into, as plain data (not a
        # Marker) - "ros2 topic echo /sub_goals" or an RViz PoseArray
        # display shows exactly which intermediate points the controller is
        # being aimed at, one at a time, instead of just the dense
        # interpolated /plan. Not published for SmacPlannerHybrid: that
        # technique hands back one dense path from the action server, it
        # doesn't produce independently-planned subgoals the way the other
        # two do.
        self.sub_goals_pub = self.create_publisher(PoseArray, '/sub_goals', 1)

        self._planning_in_progress = False

        self.timer = self.create_timer(self.resend_period_sec, self.request_plan)

        planner_desc = (
            'via planner_server/SmacPlannerHybrid' if self.use_global_planner
            else 'via a straight line to the picked goal (global planner disabled)'
        )
        if self.use_lane_following:
            planner_desc += ' (use_lane_following: wins over this whenever a goal is set)'
        self.get_logger().info(
            f'Local path planner initialized (goal {self.carrot_distance}m ahead, '
            f'scanning {self.scan_distance}m ahead for turns, replans every '
            f'{self.resend_period_sec}s {planner_desc}, '
            f'goal_heading_alpha={self.goal_heading_alpha}, '
            f'centerline_enabled={self.centerline_enabled}, odom from {odom_topic})'
        )

    def odom_callback(self, msg):
        if self.start_pose_xy is None:
            # Recorded once, from the very first odometry message - "where
            # it was start" for auto_return_to_start, not wherever the
            # vehicle happens to be when the first /goal_pose arrives.
            self.start_pose_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
            self.get_logger().info(
                f'Start pose recorded: ({self.start_pose_xy[0]:.2f}, {self.start_pose_xy[1]:.2f})')
        self.latest_odom = msg

    def goal_pose_callback(self, msg):
        self.latest_goal_xy = (msg.pose.position.x, msg.pose.position.y)
        self._smoothed_yaw = None  # don't drag the old (possibly reactive) heading into a fresh goal
        self.get_logger().info(
            f'New goal received: ({self.latest_goal_xy[0]:.2f}, {self.latest_goal_xy[1]:.2f})')

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

    def _publish_goal_marker(self, header):
        marker = Marker()
        marker.header = header
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.latest_goal_xy[0]
        marker.pose.position.y = self.latest_goal_xy[1]
        marker.pose.position.z = 0.3
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.5
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.0, 1.0, 0.9
        self.goal_marker_pub.publish(marker)

    def _publish_lane_marker(self, header, waypoints_xy):
        marker = Marker()
        marker.header = header
        marker.ns = "lane"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.12
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 1.0, 1.0
        for x, y in waypoints_xy:
            p = Point()
            p.x, p.y, p.z = float(x), float(y), 0.05
            marker.points.append(p)
        self.lane_marker_pub.publish(marker)

    def _publish_sub_goals(self, header, waypoints_xy):
        """waypoints_xy: the subgoal list a technique split the main goal
        into (_split_into_clear_segments or _build_lane_path's simplified
        output) - each pose's orientation points at the next waypoint (last
        one repeats the previous segment's heading, or stays identity if
        there's only one point)."""
        array = PoseArray()
        array.header = header
        n = len(waypoints_xy)
        for i, (x, y) in enumerate(waypoints_xy):
            pose = Pose()
            pose.position.x, pose.position.y = float(x), float(y)
            if i + 1 < n:
                nx, ny = waypoints_xy[i + 1]
                heading = math.atan2(ny - y, nx - x)
            elif n > 1:
                px, py = waypoints_xy[i - 1]
                heading = math.atan2(y - py, x - px)
            else:
                heading = 0.0
            qx, qy, qz, qw = self._yaw_to_quaternion(heading)
            pose.orientation.x, pose.orientation.y = qx, qy
            pose.orientation.z, pose.orientation.w = qz, qw
            array.poses.append(pose)
        self.sub_goals_pub.publish(array)

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

    def _pick_goal_heading(self, x0, y0, current_yaw, smoothed_yaw):
        """Return (new_smoothed_yaw, goal_distance).

        With no latest_goal_xy set, this is the original purely-reactive
        picker: scans candidate headings around current_yaw out to
        scan_distance and picks whichever stays clear the longest, biased
        toward current_yaw so it doesn't zigzag between two similarly-open
        directions. That version has NO notion of a destination - it just
        keeps going wherever's open, turning only when the costmap forces
        it to, and ties between equally-open directions are broken by
        whichever offset happens to be checked first (the fixed
        (15,-15,30,-30,...) order below), which in practice means it
        favors turning left at a symmetric junction for no reason tied to
        where you actually want to go. Fine for "just don't hit anything",
        useless for "go to this specific point" - confirmed with
        tools/carrot_heading_tuner.py on an L/T-shaped corridor: all three
        of a north goal, a south goal, and a "stay in this corridor" goal
        produced the *same* left-biased path, because nothing here ever
        looked at where any of those goals actually were.

        With latest_goal_xy set, the scan is centered on the bearing to
        that goal instead of current_yaw, widened to +-165deg (from +-90):
        reaching a goal that's, say, due north from deep inside an
        east-west corridor legitimately requires committing to a heading
        nowhere near the direct bearing until the corridor opens up, and
        the narrower spread used for the no-goal case couldn't find that
        far a candidate at all.

        Crucially, goal-seeking mode canNOT just reuse the no-goal case's
        "pick whichever candidate has the longest raw clear_distance,
        re-centered on the goal bearing instead of current heading" - raw
        clearance has no idea a candidate is actually useful, only that
        it's open, and will happily prefer heading almost straight back
        down the empty corridor already driven through over continuing
        toward the goal, since "already-driven-through empty corridor"
        reads as more clear than "corridor ahead, not yet explored,
        curving toward the goal". Confirmed live with
        tools/carrot_heading_tuner.py: a raw-clearance version of this,
        merely re-centered on the goal bearing, made the simulated vehicle
        oscillate in place, repeatedly "deciding" a ~180deg reversal was
        the best option, and it never reached an L-shaped corridor's goal
        at all.

        So goal-seeking mode scores each candidate by the resulting
        distance to the goal after advancing along it (by however far
        _clear_distance says is safe, capped at carrot_distance/remaining
        distance to goal) - not by raw clearance. A direct, unobstructed
        shot at the goal scores 0 (lands exactly on it) and always wins.
        When blocked, whichever candidate's reachable endpoint ends up
        closest to the goal wins - reversing away from the goal always
        scores worse than continuing forward, no matter how much open
        corridor is behind the vehicle, which is exactly the missing
        constraint above. heading_switch_margin_m is reused as-is (still
        meters, still "how much better before switching"), just measured
        against this distance-to-goal score instead of raw clearance.

        Distance-to-goal alone still isn't enough for a good-looking path,
        just a legal one: it has no preference for staying away from
        walls, so around a bend it'll happily hug the inside edge as tight
        as _clear_distance allows, since that's the shortest technically-
        clear route - confirmed live by drawing an actual curved corridor
        in tools/carrot_heading_tuner.py and watching the picked path
        graze the inner wall the whole way around instead of favoring the
        middle. candidate_centering adds a second term: two perpendicular
        raycasts at each candidate's own prospective ENDPOINT (not
        (x0,y0) - checked there first, it had zero effect on the winner,
        since that's the same fixed point for every candidate and mostly
        just measures "how open is it here", not "does this specific
        candidate end up near a wall"), capped at corridor_check_distance,
        scored by whichever side is tighter (min, not average - wide open
        on one side shouldn't hide grazing a wall on the other). Weighted
        by centering_weight and added directly to the goal-distance score
        (both already in meters).

        smoothed_yaw is threaded through explicitly (returned alongside
        goal_distance) rather than read/written on self._smoothed_yaw
        directly, so _split_into_clear_segments can run its own,
        independent smoothing thread when it calls this repeatedly to
        build a whole multi-step path in one request_plan() cycle, without
        clobbering the live per-cycle smoothing state this method's other
        caller (request_plan's reactive/no-destination path) relies on."""
        has_goal = self.latest_goal_xy is not None
        if has_goal:
            gx, gy = self.latest_goal_xy
            dist_to_goal = math.hypot(gx - x0, gy - y0)
            base_heading = math.atan2(gy - y0, gx - x0)
        else:
            dist_to_goal = None
            base_heading = current_yaw

        if self.latest_costmap is None:
            return base_heading, (min(self.carrot_distance, dist_to_goal) if has_goal else self.carrot_distance)

        def candidate_advance(clear):
            if has_goal:
                return max(min(clear, self.carrot_distance, dist_to_goal), 0.0)
            return max(min(clear, self.carrot_distance), 1.0)

        def candidate_centering(heading, ex, ey):
            # Checked at the candidate's prospective ENDPOINT (ex,ey), not
            # (x0,y0) - (x0,y0) is the same fixed point for every
            # candidate, so checking there mostly measures "how open is it
            # here" rather than "does THIS candidate's own endpoint sit
            # near a wall", which barely distinguishes between candidates
            # at all (confirmed live: had zero effect on the winning
            # heading until moved to the endpoint).
            if self.centering_weight == 0.0:
                return 0.0
            left = self._clear_distance(ex, ey, heading + math.pi / 2, self.corridor_check_distance)
            right = self._clear_distance(ex, ey, heading - math.pi / 2, self.corridor_check_distance)
            return min(left, right)

        def candidate_score(heading, clear):
            if not has_goal:
                return clear  # original behavior: bigger clearance wins outright
            advance = candidate_advance(clear)
            ex = x0 + advance * math.cos(heading)
            ey = y0 + advance * math.sin(heading)
            goal_term = -math.hypot(gx - ex, gy - ey)  # bigger (closer to 0) is better, like clear
            return goal_term + self.centering_weight * candidate_centering(heading, ex, ey)

        best_heading = base_heading
        best_clear = self._clear_distance(x0, y0, base_heading, self.scan_distance)
        best_score = candidate_score(base_heading, best_clear)

        # Widening spread: nearer-to-preferred options are checked first
        # and only displaced by a clearly better (not just marginally
        # different) alternative, so it doesn't turn every cycle chasing
        # noise between near-equal directions. Goal-seeking mode needs the
        # wider +-165deg spread (see docstring above); the no-goal
        # reactive mode keeps the original +-90deg - deliberately can't
        # see "behind" it, since without a destination there's nothing to
        # justify considering a near-reverse turn.
        offsets_deg = (
            (15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90, 105, -105, 120, -120, 135, -135, 150, -150, 165, -165)
            if has_goal else
            (15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90))
        for offset_deg in offsets_deg:
            heading = base_heading + math.radians(offset_deg)
            clear = self._clear_distance(x0, y0, heading, self.scan_distance)
            score = candidate_score(heading, clear)
            if score > best_score + self.heading_switch_margin_m:
                best_score = score
                best_clear = clear
                best_heading = heading

        goal_dist = candidate_advance(best_clear)

        # Low-pass filter on the heading itself (angle-wrap-safe): converge
        # toward best_heading over a few cycles rather than snapping to it,
        # so a genuinely better opening still wins out but frame-to-frame
        # raycast noise doesn't reshape SmacPlannerHybrid's entire curve
        # every cycle. See module docstring.
        if smoothed_yaw is None:
            smoothed_yaw = best_heading
        else:
            diff = math.atan2(
                math.sin(best_heading - smoothed_yaw),
                math.cos(best_heading - smoothed_yaw))
            if abs(math.degrees(diff)) >= self.goal_heading_snap_deg:
                smoothed_yaw = best_heading  # real corner - commit now, don't lag
            else:
                smoothed_yaw += self.goal_heading_alpha * diff

        return smoothed_yaw, goal_dist

    def _split_into_clear_segments(self, x0, y0, gx, gy, max_steps):
        """use_global_planner=False + a real /goal_pose set: instead of one
        straight segment capped at whatever's reachable this single cycle
        (the old behavior, still used below when there's no fixed goal to
        aim a multi-segment path at), build a whole multi-waypoint polyline
        from (x0,y0) to (gx,gy) in one shot by repeatedly calling
        _pick_goal_heading and advancing waypoint_step_m at a time - i.e.
        this IS request_plan's own per-cycle logic, just run several times
        back-to-back within a single cycle instead of once per real
        resend_period_sec tick, so TEB gets a longer, already-bent
        reference path instead of one short segment it has to wait several
        real cycles to see the rest of.

        Two earlier versions of this tried to be a proper geometric
        "insert one detour waypoint, recurse" splitter with their own
        scoring loop, and both were wrong in ways only found by actually
        running them (tools/carrot_heading_tuner.py) on an L/T-junction,
        not by inspection:
          1. Scored each candidate by the goal-distance reached after
             advancing the candidate's own FULL clear_distance. A diagonal
             heading with only ~6m of clearance can score better (ends up
             numerically closer to the goal) than continuing another ~12m
             down the actual corridor at ~0deg, because the diagonal
             points more directly at the goal - so one greedy jump locks
             onto the diagonal, quickly runs out of room hugging the
             inside wall, and never finds its way through.
          2. Capped each step's advance at a small fixed distance to force
             many small steps (the right instinct), but then floored that
             capped advance at a minimum step so it was never allowed to
             stay still - which, once genuinely cornered (clearance near
             zero), forced it forward anyway, punching straight through
             the wall in small increments, and every step after that
             started from an already-inside-a-wall position, so its own
             clear_distance readings were meaningless from then on.

        _pick_goal_heading doesn't have either problem: its own
        goal_distance is already bounded by real measured clearance (never
        forced past what's actually open), and calling it repeatedly with
        a small waypoint_step_m re-evaluates from a fresh position often
        enough that a momentarily-attractive diagonal never gets to commit
        the vehicle too far into a corner before the next call corrects
        course. Confirmed live: this is the version that actually reaches
        the goal cleanly on the L/T-junction test.

        Returns a list of (x, y) waypoints starting with (x0,y0) - ending
        with (gx,gy) if it converges within max_steps, or wherever it got
        to otherwise (TEB's own local-costmap obstacle avoidance is the
        final backstop here, same as it always was for the single-segment
        case, so a shorter-than-hoped polyline is still strictly better
        than none)."""
        x, y = x0, y0
        yaw = math.atan2(gy - y0, gx - x0)
        smoothed_yaw = None  # independent of self._smoothed_yaw - see _pick_goal_heading's docstring
        waypoints = [(x, y)]
        for _ in range(max_steps):
            dist = math.hypot(gx - x, gy - y)
            if dist < STRAIGHT_PATH_STEP_M:
                break
            smoothed_yaw, goal_dist = self._pick_goal_heading(x, y, yaw, smoothed_yaw)
            advance = min(self.waypoint_step_m, goal_dist, dist)
            x = x + advance * math.cos(smoothed_yaw)
            y = y + advance * math.sin(smoothed_yaw)
            yaw = smoothed_yaw
            waypoints.append((x, y))
            if math.hypot(gx - x, gy - y) < STRAIGHT_PATH_STEP_M:
                break
        return waypoints

    def _build_lane_path(self, x0, y0, gx, gy):
        """use_lane_following=True + a real /goal_pose set: navigate along
        the medial axis (skeleton) of latest_costmap's free space instead
        of picking headings step by step - see module docstring's
        "use_lane_following" paragraph for the full rationale. Returns a
        list of (x, y) waypoints from (x0,y0) to (gx,gy), or None if no
        skeleton path connects them (logged, caller falls through to
        whatever the other modes would have done - see request_plan)."""
        info = self.latest_costmap.info
        w, h = info.width, info.height
        res = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y
        grid = np.array(self.latest_costmap.data, dtype=np.int16).reshape(h, w)

        free = (grid >= 0) & (grid < LETHAL_THRESHOLD)
        skeleton = skeletonize(free)
        ys, xs = np.nonzero(skeleton)
        if len(ys) == 0:
            return None

        pts = np.stack([ys, xs], axis=1)
        tree = cKDTree(pts)

        graph = nx.Graph()
        idx_of = {(int(y), int(x)): True for y, x in pts}
        for y, x in pts:
            y, x = int(y), int(x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    if (y + dy, x + dx) in idx_of:
                        graph.add_edge((y, x), (y + dy, x + dx), weight=math.hypot(dy, dx) * res)

        def nearest_skeleton_cell(wx, wy):
            cell_y = (wy - oy) / res
            cell_x = (wx - ox) / res
            _, i = tree.query([cell_y, cell_x])
            return (int(pts[i][0]), int(pts[i][1]))

        start_cell = nearest_skeleton_cell(x0, y0)
        goal_cell = nearest_skeleton_cell(gx, gy)

        try:
            path_cells = nx.shortest_path(graph, start_cell, goal_cell, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        path_xy = [(ox + (cx + 0.5) * res, oy + (cy + 0.5) * res) for cy, cx in path_cells]
        return self._simplify_path_dp(path_xy, self.lane_simplify_epsilon)

    def _simplify_path_dp(self, points, epsilon):
        """Ramer-Douglas-Peucker simplification: keeps the path within
        epsilon meters (perpendicular distance) of the original polyline's
        shape at every point, discarding only points that don't bend the
        path more than that - NOT the same as "shortcut to the farthest
        point that's still technically collision-clear" (a materially
        worse alternative that was tried first for _build_lane_path - it
        throws away the skeleton's centering entirely, since a wall-
        hugging diagonal shortcut can easily still be "technically clear".
        Deviation-bounded simplification is what actually preserves the
        skeleton's true centered shape while still cutting hundreds of
        raw pixel-jagged skeleton cells down to a handful of waypoints)."""
        if len(points) < 3:
            return points

        def perpendicular_distance(p, a, b):
            (px, py), (ax, ay), (bx, by) = p, a, b
            dx, dy = bx - ax, by - ay
            if dx == 0.0 and dy == 0.0:
                return math.hypot(px - ax, py - ay)
            t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
            proj_x, proj_y = ax + t * dx, ay + t * dy
            return math.hypot(px - proj_x, py - proj_y)

        dmax = 0.0
        index = 0
        for i in range(1, len(points) - 1):
            d = perpendicular_distance(points[i], points[0], points[-1])
            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            left = self._simplify_path_dp(points[:index + 1], epsilon)
            right = self._simplify_path_dp(points[index:], epsilon)
            return left[:-1] + right
        return [points[0], points[-1]]

    def _build_straight_path(self, header, waypoints_xy):
        """Interpolates a nav_msgs/Path at STRAIGHT_PATH_STEP_M spacing
        through a polyline of (x, y) waypoints (either the single
        [start, goal] pair from the old no-destination fallback, or
        _split_into_clear_segments' multi-segment result). Orientation at
        each point faces that segment's own direction, not one fixed
        heading for the whole path, so the path itself actually bends at
        each detour waypoint - TEB tracks orientation as well as position,
        so a single fixed heading across a bent polyline would fight the
        turn instead of describing it."""
        path = Path()
        path.header = header
        for seg_idx in range(len(waypoints_xy) - 1):
            (sx, sy), (ex, ey) = waypoints_xy[seg_idx], waypoints_xy[seg_idx + 1]
            seg_dist = math.hypot(ex - sx, ey - sy)
            seg_yaw = math.atan2(ey - sy, ex - sx)
            qx, qy, qz, qw = self._yaw_to_quaternion(seg_yaw)
            steps = max(1, int(seg_dist / STRAIGHT_PATH_STEP_M))
            start_i = 0 if seg_idx == 0 else 1  # skip the point shared with the previous segment
            for i in range(start_i, steps + 1):
                t = i / steps
                pose_stamped = PoseStamped()
                pose_stamped.header = header
                pose_stamped.pose.position.x = sx + t * (ex - sx)
                pose_stamped.pose.position.y = sy + t * (ey - sy)
                pose_stamped.pose.position.z = 0.0
                pose_stamped.pose.orientation.x = qx
                pose_stamped.pose.orientation.y = qy
                pose_stamped.pose.orientation.z = qz
                pose_stamped.pose.orientation.w = qw
                path.poses.append(pose_stamped)
        return path

    def _dispatch_follow_path(self, path):
        self.plan_pub.publish(path)

        if not self._controller_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warn('follow_path action server not available yet', throttle_duration_sec=5.0)
            return

        follow_goal = FollowPath.Goal()
        follow_goal.path = path
        follow_goal.controller_id = self.controller_id
        self._controller_client.send_goal_async(follow_goal)

    def request_plan(self):
        if self.latest_odom is None:
            self.get_logger().warn('No odometry received yet, skipping plan request', throttle_duration_sec=5.0)
            return

        if self.latest_goal_xy is not None:
            # Republished every cycle (not just once in goal_pose_callback)
            # since this publisher has no transient-local durability - a
            # late-joining RViz would otherwise never see a goal that was
            # set before it connected. Kept alive through the "holding"
            # early-return below too, so the marker doesn't disappear once
            # arrived.
            self._publish_goal_marker(self.latest_odom.header)
            pose_now = self.latest_odom.pose.pose
            remaining = math.hypot(
                self.latest_goal_xy[0] - pose_now.position.x,
                self.latest_goal_xy[1] - pose_now.position.y)
            if remaining <= self.goal_xy_tolerance:
                # auto_return_to_start (closed-loop map navigation): the
                # FIRST goal reached auto-chains one more goal - the
                # recorded start_pose_xy - instead of holding, so one
                # manual "2D Goal Pose" click drives the whole lap and
                # back, not just the outbound leg. Detected by whether the
                # goal just reached IS start_pose_xy already (the return
                # leg) or not (the outbound leg, or any leg before it) -
                # only ever chains once per goal, so this can't loop
                # forever even if start_pose_xy and a later manually-set
                # goal happen to coincide.
                if self.auto_return_to_start and self.start_pose_xy is not None:
                    dist_to_start = math.hypot(
                        self.start_pose_xy[0] - self.latest_goal_xy[0],
                        self.start_pose_xy[1] - self.latest_goal_xy[1])
                    if dist_to_start > self.goal_xy_tolerance:
                        self.latest_goal_xy = self.start_pose_xy
                        self._smoothed_yaw = None
                        self.get_logger().info(
                            'Leg complete - returning to recorded start '
                            f'({self.start_pose_xy[0]:.2f}, {self.start_pose_xy[1]:.2f}).')
                        return
                    self.get_logger().info(
                        'Closed loop complete - back at start, holding.', throttle_duration_sec=5.0)
                    return
                # Deliberately don't clear latest_goal_xy here: doing so would
                # make _pick_goal_heading fall back to its reactive, no-
                # destination mode next cycle and drive off again. Holding
                # means "keep re-detecting arrived and doing nothing" until a
                # genuinely new /goal_pose overwrites latest_goal_xy.
                self.get_logger().info(
                    'Goal reached (within %.2fm) - holding, no further replans until a '
                    'new /goal_pose arrives.' % self.goal_xy_tolerance, throttle_duration_sec=5.0)
                return

        # use_lane_following wins over both other modes whenever a goal is
        # set (see module docstring) - falls through to whatever
        # use_global_planner would otherwise do if no costmap is available
        # yet or no skeleton path connects start to goal (e.g. genuinely
        # disconnected free-space regions in the current rolling window).
        if self.use_lane_following and self.latest_goal_xy is not None:
            pose = self.latest_odom.pose.pose
            header = self.latest_odom.header
            gx, gy = self.latest_goal_xy
            if self.latest_costmap is not None:
                waypoints_xy = self._build_lane_path(pose.position.x, pose.position.y, gx, gy)
                if waypoints_xy is not None:
                    self._publish_lane_marker(header, waypoints_xy)
                    self._publish_sub_goals(header, waypoints_xy)
                    path = self._build_straight_path(header, waypoints_xy)
                    self._dispatch_follow_path(path)
                    return
                self.get_logger().warn(
                    'use_lane_following: no skeleton path found connecting current pose to '
                    'goal through this costmap window - falling back to the straight-segment '
                    'technique this cycle.', throttle_duration_sec=5.0)

        # use_global_planner=False with a real destination set: build the
        # whole straight-segment polyline to it in one shot (see
        # _split_into_clear_segments) rather than the single
        # carrot-distance-capped segment used below - that single-segment
        # path is deliberately reachable-by-construction every cycle (its
        # endpoint is chosen specifically because the line to it is clear),
        # so applying the splitter to IT would never find anything to
        # split; it only does something useful aimed at the actual,
        # possibly-much-farther-than-carrot_distance goal.
        if not self.use_global_planner and self.latest_goal_xy is not None:
            pose = self.latest_odom.pose.pose
            header = self.latest_odom.header
            gx, gy = self.latest_goal_xy
            if self.latest_costmap is None:
                waypoints_xy = [(pose.position.x, pose.position.y), (gx, gy)]
            else:
                waypoints_xy = self._split_into_clear_segments(
                    pose.position.x, pose.position.y, gx, gy, self.max_waypoint_steps)
            self._publish_sub_goals(header, waypoints_xy)
            path = self._build_straight_path(header, waypoints_xy)
            self._dispatch_follow_path(path)
            return

        if self.use_global_planner:
            if self._planning_in_progress:
                return
            if not self._planner_client.wait_for_server(timeout_sec=0.2):
                self.get_logger().warn('compute_path_to_pose action server not available yet', throttle_duration_sec=5.0)
                return

        pose = self.latest_odom.pose.pose
        yaw = self._yaw_from_quaternion(pose.orientation)
        header = self.latest_odom.header  # frame_id "odom", matches costmaps' global_frame

        self._smoothed_yaw, goal_dist = self._pick_goal_heading(
            pose.position.x, pose.position.y, yaw, self._smoothed_yaw)
        goal_yaw = self._smoothed_yaw
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

        if not self.use_global_planner:
            # latest_goal_xy is None here (that case returns earlier, via
            # _split_into_clear_segments) - no real destination to split
            # toward, just the single reactive carrot segment.
            waypoints_xy = [
                (start.pose.position.x, start.pose.position.y),
                (goal_pose.pose.position.x, goal_pose.pose.position.y)]
            path = self._build_straight_path(header, waypoints_xy)
            self._dispatch_follow_path(path)
            return

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

        self._dispatch_follow_path(path)


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
