"""Lightweight fake-robot kinematic integrator for standalone local
(TEB/controller_server) tuning - see tune_planner_standalone.launch.py's
docstring for the full workflow. This is NOT a physics simulator (no
collisions, no wheel slip, no actuator limits beyond what upstream nodes
already impose) - it exists purely so controller_server's FollowPath
action has a moving robot to control and a live map->odom->base_link TF
chain to compute against, without needing Gazebo (whose per-test startup
time and CPU/real-time-factor overhead this whole tuning path was built
to avoid - see tune_planner_standalone.launch.py's own docstring).

Subscribes to cmd_vel_topic (whatever controller_server's FollowPath
computes) and integrates a plain unicycle model (x/y/yaw from
linear.x/angular.z) at control_rate_hz - deliberately not the vehicle's
Ackermann/bicycle model, since TEB's own internal model already
constrains what velocity commands it produces (min_turning_radius etc. in
teb_controller.yaml), so re-imposing that constraint here would only
mask a bad TEB output as a bad kinematic response and confuse tuning.

Publishes nav_msgs/Odometry on odom_topic (matching
teb_controller.yaml's odom_topic: /odom), broadcasts odom_frame-
>base_frame TF dynamically, and publishes a visualization_msgs/Marker
(marker_topic) so RViz shows something moving - there is no robot model
loaded in this standalone tool.

Also subscribes to initialpose_topic (RViz "2D Pose Estimate") to reset
its internal pose - the same click that sets planner_tuning_client's
planning start also repositions the fake robot to match, so the global
plan's start and the local controller's actual position stay consistent.
"""

import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, Twist, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker


class KinematicSim(Node):

    def __init__(self):
        super().__init__('kinematic_sim')

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('initialpose_topic', '/initialpose')
        self.declare_parameter('marker_topic', '/fake_robot_marker')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('control_rate_hz', 30.0)

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.w = 0.0

        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, self.get_parameter('odom_topic').value, 10)
        self.marker_pub = self.create_publisher(
            Marker, self.get_parameter('marker_topic').value, 1)

        self.create_subscription(
            Twist, self.get_parameter('cmd_vel_topic').value, self.cmd_vel_callback, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, self.get_parameter('initialpose_topic').value,
            self.initialpose_callback, 10)

        self.last_tick = self.get_clock().now()
        rate = self.get_parameter('control_rate_hz').value
        self.create_timer(1.0 / rate, self.tick)

        self.get_logger().info(
            f'Kinematic sim ready - fake robot starts at (0, 0, 0) in {self.odom_frame}, '
            f'reset via "2D Pose Estimate", driven by {self.get_parameter("cmd_vel_topic").value}.')

    def cmd_vel_callback(self, msg):
        self.v = msg.linear.x
        self.w = msg.angular.z

    def initialpose_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.get_logger().info(f'Fake robot reset to ({self.x:.2f}, {self.y:.2f}).')

    def tick(self):
        now = self.get_clock().now()
        dt = (now - self.last_tick).nanoseconds / 1e9
        self.last_tick = now
        if dt <= 0.0:
            return

        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.w * dt

        qz = math.sin(self.yaw / 2.0)
        qw = math.cos(self.yaw / 2.0)
        orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = orientation
        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = now.to_msg()
        tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.rotation = orientation
        self.tf_broadcaster.sendTransform(tf)

        marker = Marker()
        marker.header.stamp = now.to_msg()
        marker.header.frame_id = self.odom_frame
        marker.ns = 'fake_robot'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = self.x
        marker.pose.position.y = self.y
        marker.pose.orientation = orientation
        marker.scale.x = 1.0
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.r = 0.1
        marker.color.g = 0.9
        marker.color.b = 0.2
        marker.color.a = 1.0
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = KinematicSim()
    rclpy.spin(node)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
