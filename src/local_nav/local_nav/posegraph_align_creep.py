"""Automatic drive-to-align for the slam_toolbox pause/resume workflow
(closed_loop_navigate_slam.launch.py / real_closed_loop_navigate_slam.launch.py):
replaces manual teleop-forward-a-bit (see pause_resume_console.py's
docstring for why SOME motion is required at all - slam_toolbox's
minimum_travel_distance/minimum_travel_heading gates mean it won't
re-process a scan against a freshly-set /initialpose until the vehicle
has actually moved) by driving the vehicle along the ACTUAL route recorded
in the saved pose graph, instead of just blindly forward.

Why the pose graph specifically, not just "drive forward": a straight
line might immediately point the vehicle off the mapped track (into a
wall, or into unmapped space) depending on which way it happens to be
facing after your "2D Pose Estimate" click - the pose graph's vertex
sequence IS the exact path that was actually driven while mapping, so
following it guarantees the vehicle stays on real, previously-scanned
ground the whole time, giving slam_toolbox's scan matcher the same kind
of genuine, distinct viewpoints it saw originally.

How waypoints are obtained: slam_toolbox continuously publishes its own
pose graph on /slam_toolbox/graph_visualization (a MarkerArray - one
SPHERE marker per graph vertex, id = vertex id = original driving order)
regardless of pause/resume state. This node takes a fresh snapshot of
those the moment /initialpose is set, finds the vertex closest to the
newly-set pose, and drives forward through the following vertices in
increasing id order (i.e. continuing in the SAME direction the map was
originally driven from wherever you just placed the vehicle).

Steering is real pure pursuit (curvature = 2*sin(alpha)/L, steering_angle
= atan(wheelbase*curvature)), not a naive proportional-heading-error
Twist - this vehicle's steering is mechanically limited to +/-max_steer_rad
(buggy.urdf.xacro's max_steer_rad, 20 degrees by default - wheelbase 1.6m
gives a ~4.4m minimum turning radius), and a naive controller can easily
ask for a turn tighter than that, which just saturates against the limit
and drives roughly straight regardless of what the route actually needs -
exactly the "not aligning" failure this replaced. Picks a lookahead point
at approximately lookahead_distance ahead along the remaining route
(walking forward through waypoints, not just the immediate next one),
same as any standard Ackermann path-follower, so the requested curvature
stays sane relative to actual waypoint spacing instead of chasing every
small zigzag between consecutive (only ~0.3m apart) recorded vertices.
Same live-/scan obstacle check as localization_creep.py.

This does NOT call :resume itself and does NOT touch slam_toolbox or
planner_server/controller_server's lifecycle state - unlike AMCL's
localization_creep, the operator stays in the loop here: once this node
stops (waypoints exhausted, max_creep_time_sec safety cap, or an
obstacle blocks it for too long), it logs a message and waits for you to
check RViz and type ":resume" yourself if it looks right, matching how
pause_resume_console.py already works.
"""

import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray


class PosegraphAlignCreep(Node):

    def __init__(self):
        super().__init__('posegraph_align_creep')

        self.declare_parameter('cmd_vel_topic', '/ackermann_steering_controller/reference_unstamped')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('graph_topic', '/slam_toolbox/graph_visualization')
        self.declare_parameter('initialpose_topic', '/initialpose')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('creep_speed', 0.6)
        self.declare_parameter('min_forward_clearance', 3.0)
        self.declare_parameter('forward_cone_deg', 30.0)
        self.declare_parameter('waypoint_reached_radius', 1.0)
        self.declare_parameter('num_waypoints_to_traverse', 15)
        self.declare_parameter('lookahead_distance', 1.5)
        self.declare_parameter('wheelbase', 1.6)
        self.declare_parameter('max_steer_rad', 0.349066)
        self.declare_parameter('max_creep_time_sec', 30.0)
        self.declare_parameter('control_rate_hz', 10.0)

        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.creep_speed = self.get_parameter('creep_speed').value
        self.min_forward_clearance = self.get_parameter('min_forward_clearance').value
        self.forward_cone_rad = math.radians(self.get_parameter('forward_cone_deg').value)
        self.waypoint_reached_radius = self.get_parameter('waypoint_reached_radius').value
        self.num_waypoints_to_traverse = self.get_parameter('num_waypoints_to_traverse').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.max_steer_rad = self.get_parameter('max_steer_rad').value
        self.max_creep_time_sec = self.get_parameter('max_creep_time_sec').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 1)
        self.latest_graph_vertices = []  # sorted [(id, x, y), ...]
        self.create_subscription(
            MarkerArray, self.get_parameter('graph_topic').value, self.graph_callback, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, self.get_parameter('initialpose_topic').value,
            self.initialpose_callback, 10)
        self.min_forward_range = None
        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value, self.scan_callback,
            qos_profile_sensor_data)

        self.active = False
        self.route = []
        self.route_index = 0
        self.start_time = None

        rate = self.get_parameter('control_rate_hz').value
        self.create_timer(1.0 / rate, self.control_tick)

        self.get_logger().info(
            'Posegraph align creep ready - waiting for "2D Pose Estimate" in RViz to start '
            'driving along the recorded route.')

    def graph_callback(self, msg):
        vertices = [
            (m.id, m.pose.position.x, m.pose.position.y)
            for m in msg.markers
            if m.type == 2 and m.action == 0  # SPHERE, ADD - graph vertex nodes
        ]
        if vertices:
            self.latest_graph_vertices = sorted(vertices, key=lambda v: v[0])

    def scan_callback(self, msg):
        if not msg.ranges:
            return
        min_range = None
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):
                continue
            angle = msg.angle_min + i * msg.angle_increment
            if abs(angle) <= self.forward_cone_rad:
                if min_range is None or r < min_range:
                    min_range = r
        self.min_forward_range = min_range

    def initialpose_callback(self, msg):
        if not self.latest_graph_vertices:
            self.get_logger().warn(
                'No pose graph vertices seen yet on graph_topic - cannot plan an align route. '
                'Is slam_toolbox running and has it published at least once?')
            return

        px, py = msg.pose.pose.position.x, msg.pose.pose.position.y
        nearest_idx = min(
            range(len(self.latest_graph_vertices)),
            key=lambda i: math.hypot(
                self.latest_graph_vertices[i][1] - px, self.latest_graph_vertices[i][2] - py))

        # Wrap around rather than slice - this is a CLOSED loop track, so
        # the vertex nearest the click point can just as easily be near
        # the END of the recorded sequence (a full lap later, back near
        # the start) as the beginning. A plain slice left almost no route
        # to drive when that happened (observed: nearest vertex landed at
        # index 492 of 495, leaving exactly 1 waypoint). Continuing past
        # the last vertex by wrapping to vertex 0 is the geometrically
        # correct "keep going forward" on a loop.
        n = len(self.latest_graph_vertices)
        route_len = min(self.num_waypoints_to_traverse + 1, n)
        self.route = [
            self.latest_graph_vertices[(nearest_idx + i) % n] for i in range(route_len)]
        self.route_index = 1 if len(self.route) > 1 else 0  # skip the vertex we're already at
        self.active = True
        self.start_time = self.get_clock().now()
        self.get_logger().info(
            f'New pose set - driving through {len(self.route) - self.route_index} route '
            f'waypoints starting from graph vertex {self.route[0][0]}.')

    def _current_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except Exception:
            return None
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return tf.transform.translation.x, tf.transform.translation.y, yaw

    def control_tick(self):
        if not self.active:
            return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed >= self.max_creep_time_sec:
            self._stop('TIMED OUT')
            return

        pose = self._current_pose()
        if pose is None:
            self.cmd_pub.publish(Twist())
            return
        x, y, yaw = pose

        # Advance past any waypoints already behind/at the vehicle before
        # picking a lookahead target - keeps the route progressing even if
        # several waypoints fall within waypoint_reached_radius at once.
        while (self.route_index < len(self.route)
               and math.hypot(self.route[self.route_index][1] - x,
                               self.route[self.route_index][2] - y) <= self.waypoint_reached_radius):
            self.route_index += 1

        if self.route_index >= len(self.route):
            self._stop('finished the planned route')
            return

        # Lookahead target: walk forward from route_index for the first
        # waypoint at or beyond lookahead_distance, or the last remaining
        # one if the whole rest of the route is closer than that.
        target = self.route[-1]
        for i in range(self.route_index, len(self.route)):
            wx, wy = self.route[i][1], self.route[i][2]
            if math.hypot(wx - x, wy - y) >= self.lookahead_distance:
                target = self.route[i]
                break

        dx, dy = target[1] - x, target[2] - y
        lookahead_actual = math.hypot(dx, dy)
        alpha = math.atan2(dy, dx) - yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        # Pure pursuit: curvature = 2*sin(alpha)/L, steering = atan(wheelbase*curvature),
        # then converted back to the angular.z this Twist-based controller expects,
        # via the same bicycle relationship - so the requested steering angle never
        # exceeds max_steer_rad regardless of how sharp alpha is.
        curvature = 2.0 * math.sin(alpha) / max(lookahead_actual, 0.1)
        steering_angle = math.atan(self.wheelbase * curvature)
        steering_angle = max(-self.max_steer_rad, min(self.max_steer_rad, steering_angle))

        blocked = (
            self.min_forward_range is not None
            and self.min_forward_range < self.min_forward_clearance
        )
        cmd = Twist()
        if not blocked:
            cmd.linear.x = self.creep_speed
            cmd.angular.z = self.creep_speed * math.tan(steering_angle) / self.wheelbase
        self.cmd_pub.publish(cmd)

    def _stop(self, reason):
        self.cmd_pub.publish(Twist())
        self.active = False
        self.get_logger().info(
            f'Posegraph align creep stopped ({reason}) after covering '
            f'{self.route_index}/{len(self.route)} waypoints. Check RViz - if the live scan '
            'lines up with the map, type ":resume". If not, set "2D Pose Estimate" again to '
            'retry.')


def main(args=None):
    rclpy.init(args=args)
    node = PosegraphAlignCreep()
    rclpy.spin(node)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
