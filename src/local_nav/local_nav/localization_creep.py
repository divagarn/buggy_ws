"""Drives the vehicle slowly forward, obstacle-checked against live /scan,
right after AMCL's global localization is triggered - see
closed_loop_navigate.launch.py's docstring for the full problem this
solves: on a geometrically repetitive track (e.g. a near-regular hexagonal
loop), a single stationary scan can look nearly identical from several
different spots on the loop, so AMCL can converge CONFIDENTLY to the wrong
one. Real, sustained motion across genuinely different viewpoints is what
breaks that ambiguity - candidate poses that don't actually lie on the
vehicle's true path fall out of the particle filter as it drives, while a
stationary vehicle gives AMCL nothing to disambiguate with.

This does not do any real path planning - planner_server/controller_server
aren't up yet at this point (see localization_settle_time). It just drives
straight ahead at a fixed slow speed, backing off to a stop whenever
anything gets closer than min_forward_clearance in front of it, until
EITHER it has covered min_creep_distance AND AMCL's positional covariance
has dropped under convergence_covariance_threshold (a real, confirmed
convergence), or max_creep_time_sec elapses regardless (a safety cap - if
localization still hasn't resolved by then, something is wrong, and
planner_server comes up per its own timer with whatever confidence AMCL
currently has rather than blocking forever).

KNOWN LIMITATION: straight-line-only creep can't out-maneuver a wall
directly ahead of the spawn point, and won't disambiguate a loop whose
symmetric segments are longer than min_creep_distance - it's a best-effort
LiDAR-only mitigation, not a substitute for an absolute position fix (e.g.
GPS/INS) which would resolve this unambiguously regardless of track shape.
"""

import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class LocalizationCreep(Node):

    def __init__(self):
        super().__init__('localization_creep')

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('amcl_pose_topic', '/amcl_pose')
        self.declare_parameter('creep_speed', 0.4)
        self.declare_parameter('min_forward_clearance', 3.0)
        self.declare_parameter('forward_cone_deg', 30.0)
        self.declare_parameter('min_creep_distance', 6.0)
        self.declare_parameter('convergence_covariance_threshold', 1.0)
        self.declare_parameter('max_creep_time_sec', 20.0)
        self.declare_parameter('control_rate_hz', 10.0)

        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.creep_speed = self.get_parameter('creep_speed').value
        self.min_forward_clearance = self.get_parameter('min_forward_clearance').value
        self.forward_cone_rad = math.radians(self.get_parameter('forward_cone_deg').value)
        self.min_creep_distance = self.get_parameter('min_creep_distance').value
        self.covariance_threshold = self.get_parameter('convergence_covariance_threshold').value
        self.max_creep_time_sec = self.get_parameter('max_creep_time_sec').value

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 1)
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)
        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value, self.scan_callback,
            qos_profile_sensor_data)
        self.create_subscription(
            PoseWithCovarianceStamped, self.get_parameter('amcl_pose_topic').value,
            self.amcl_pose_callback, 10)

        self.last_odom_stamp = None
        self.distance_traveled = 0.0
        self.min_forward_range = None
        self.latest_covariance_xx = float('inf')
        self.latest_covariance_yy = float('inf')
        self.start_time = self.get_clock().now()
        self.done = False

        rate = self.get_parameter('control_rate_hz').value
        self.create_timer(1.0 / rate, self.control_tick)

        self.get_logger().info(
            f'Localization creep starting: speed={self.creep_speed}m/s, '
            f'min_forward_clearance={self.min_forward_clearance}m, '
            f'min_creep_distance={self.min_creep_distance}m, '
            f'convergence_covariance_threshold={self.covariance_threshold}, '
            f'max_creep_time_sec={self.max_creep_time_sec}s')

    def odom_callback(self, msg):
        now = self.get_clock().now()
        if self.last_odom_stamp is not None:
            dt = (now - self.last_odom_stamp).nanoseconds / 1e9
            if dt > 0:
                self.distance_traveled += abs(msg.twist.twist.linear.x) * dt
        self.last_odom_stamp = now

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

    def amcl_pose_callback(self, msg):
        cov = msg.pose.covariance
        self.latest_covariance_xx = cov[0]
        self.latest_covariance_yy = cov[7]

    def control_tick(self):
        if self.done:
            return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        converged = (
            self.distance_traveled >= self.min_creep_distance
            and max(self.latest_covariance_xx, self.latest_covariance_yy) <= self.covariance_threshold
        )
        timed_out = elapsed >= self.max_creep_time_sec

        if converged or timed_out:
            self.cmd_pub.publish(Twist())
            self.done = True
            if converged:
                self.get_logger().info(
                    f'Localization creep converged after {self.distance_traveled:.2f}m, '
                    f'{elapsed:.1f}s (covariance xx={self.latest_covariance_xx:.3f} '
                    f'yy={self.latest_covariance_yy:.3f}).')
            else:
                self.get_logger().warn(
                    f'Localization creep TIMED OUT after {elapsed:.1f}s having traveled '
                    f'{self.distance_traveled:.2f}m (covariance xx={self.latest_covariance_xx:.3f} '
                    f'yy={self.latest_covariance_yy:.3f}) - localization may still be wrong. '
                    'Check the Particle Cloud in RViz before trusting navigation.')
            # Exit the process (rather than idling) so a launch-side
            # OnProcessExit handler can bring up planner_server/
            # controller_server exactly when creep genuinely finishes,
            # instead of guessing a wall-clock duration to wait - this
            # node's own timing is on the ROS/sim clock (use_sim_time),
            # which runs at whatever rate Gazebo's real-time factor
            # allows, not wall-clock time.
            rclpy.shutdown()
            return

        blocked = (
            self.min_forward_range is not None
            and self.min_forward_range < self.min_forward_clearance
        )
        cmd = Twist()
        if not blocked:
            cmd.linear.x = self.creep_speed
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationCreep()
    rclpy.spin(node)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
