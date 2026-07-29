#!/usr/bin/env python3
"""Local nav stack's real-hardware actuation bridge.

The local nav pipeline's controller (DWB) publishes a geometry_msgs/Twist
- in Gazebo, that goes straight to ackermann_steering_controller and
actually moves the simulated vehicle. On the real golf cart there is no
ackermann_steering_controller: the physical steering/motor hardware is
driven by uart_sender_node.py (unmodified team code), which subscribes to
/steering_angle (std_msgs/Float32, degrees) and the /final_red_detected /
/final_yellow_detected (std_msgs/Bool) stop/caution flags and writes them
out over serial. Without this bridge, switching this pipeline from Gazebo
to the real vehicle has no path to actually move it - DWB's Twist would
have nowhere real to go.

Converts the same (linear.x, angular.z) Twist DWB publishes into that
contract - the exact inverse of sim_bridge/sim_actuation_bridge.py's
Twist synthesis, so the sign convention it already validates (SteerSpeed
positive degrees = right vs ROS Twist positive angular.z = left) is
preserved rather than re-derived and possibly getting the sign backwards.

Hard-clamps to the vehicle's actual physical steering limit (+-20deg,
buggy.urdf.xacro's max_steer_rad = 0.349066 = 20deg). DWB only caps
angular.z (wz), not the resulting steering angle directly, so at low
forward speed a modest wz implies an enormous angle
(atan(wheelbase * wz / speed) blows up as speed -> 0) - confirmed ~40deg
commands during simulation testing at low speed, and it would go higher
still (measured up to ~86deg at DWB's min_vel_x with wz near its cap).
Gazebo's URDF joint limits silently clamp this in simulation, but
uart_sender_node.py (real hardware) has no clamping at all and would
forward whatever it's given straight to the actuator - this clamp is
what stands in for that missing hardware-side safety limit.

Also replicates the old reactive pipeline's steering markers
(ground_segmentation/steering_calculator_node.py), same geometry/colors/
text, so RViz looks the same across the old and new stacks:
  - /steering_visualization: steering wheel + front wheels + command text
    + a direction arrow, all oriented by our computed steering angle.
  - /steering_actual_vs_target: blue arrow = our commanded ("target")
    angle, red arrow = real wheel feedback ("actual") from /wheel_uart,
    with a text readout of the difference. Only published when recent
    /wheel_uart feedback exists (same have_recent_feedback gate the old
    code used) - Gazebo sim has no such feedback loop, so this stays
    silent there; real-hardware and bag-replay modes both have it.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool
from visualization_msgs.msg import Marker, MarkerArray
from ground_segmentation_msgs.msg import SteerSpeed


class SteeringUartBridge(Node):

    def __init__(self):
        super().__init__('steering_uart_bridge')

        self.wheelbase = self.declare_parameter('wheelbase', 1.6).value
        self.min_forward_speed = self.declare_parameter('min_forward_speed', 0.05).value
        self.max_steering_deg = self.declare_parameter('max_steering_deg', 20.0).value
        self.enable_visualization = self.declare_parameter('enable_visualization', True).value
        self.wheel_visualization_size = self.declare_parameter('wheel_visualization_size', 0.5).value
        cmd_vel_topic = self.declare_parameter(
            'cmd_vel_topic', '/ackermann_steering_controller/reference_unstamped').value

        self.steering_pub = self.create_publisher(Float32, '/steering_angle', 1)
        self.red_pub = self.create_publisher(Bool, '/final_red_detected', 1)
        self.yellow_pub = self.create_publisher(Bool, '/final_yellow_detected', 1)
        self.visualization_pub = self.create_publisher(MarkerArray, '/steering_visualization', 1)
        self.actual_vs_target_pub = self.create_publisher(MarkerArray, '/steering_actual_vs_target', 1)

        # Real wheel feedback (steering deg + speed km/h) for the actual-vs-
        # target comparison - only present in real-hardware/bag-replay
        # modes, never published in Gazebo sim.
        self.current_steering = 0.0
        self.current_speed = 0.0
        self.last_feedback_time = self.get_clock().now()

        self.create_subscription(Twist, cmd_vel_topic, self.cmd_vel_callback, 1)
        self.create_subscription(SteerSpeed, '/wheel_uart', self.wheel_feedback_callback, 1)

        self.get_logger().info(
            f'Steering/UART bridge initialized (wheelbase={self.wheelbase}m, '
            f'listening on {cmd_vel_topic})'
        )

    def wheel_feedback_callback(self, msg):
        self.current_steering = msg.steering
        self.current_speed = msg.speed
        self.last_feedback_time = self.get_clock().now()

    def _secs_since(self, stamp):
        return (self.get_clock().now() - stamp).nanoseconds / 1e9

    def _quaternion_from_euler(self, roll, pitch, yaw):
        cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
        cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )

    def cmd_vel_callback(self, msg):
        speed_ms = msg.linear.x
        angular_z = msg.angular.z

        if abs(speed_ms) < 1e-3:
            steering_deg = 0.0
        else:
            # Inverse of sim_actuation_bridge.py's
            # angular_z = -tan(steering_rad) * speed_ms / wheelbase
            steering_rad = math.atan2(-angular_z * self.wheelbase, speed_ms)
            steering_deg = math.degrees(steering_rad)

        steering_deg = max(-self.max_steering_deg, min(self.max_steering_deg, steering_deg))
        self.steering_pub.publish(Float32(data=steering_deg))

        # red_detected=True is uart_sender_node's STOP state. DWB commanding
        # forward speed below min_forward_speed means it found no safe
        # trajectory (frozen, or goal reached) - treat that as stop rather
        # than let the real vehicle creep indefinitely at a near-zero crawl.
        stopped = speed_ms < self.min_forward_speed
        self.red_pub.publish(Bool(data=stopped))
        self.yellow_pub.publish(Bool(data=False))

        if self.enable_visualization:
            self.visualize_steering(steering_deg, speed_ms * 3.6)

    def visualize_steering(self, steering_angle, speed=0.0):
        """Steering wheel + front wheels + command text + direction arrow,
        matching steering_calculator_node.py's visualize_steering exactly."""
        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.header.frame_id = "base_link"
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.ns = "steering"
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        self.visualization_pub.publish(marker_array)

        marker_array = MarkerArray()

        wheel = Marker()
        wheel.header.frame_id = "base_link"
        wheel.header.stamp = self.get_clock().now().to_msg()
        wheel.ns = "steering"
        wheel.id = 0
        wheel.type = Marker.CYLINDER
        wheel.action = Marker.ADD
        wheel.pose.position.x = 0.5
        wheel.pose.position.y = -0.4
        wheel.pose.position.z = 1.0
        q = self._quaternion_from_euler(0, np.radians(90), np.radians(steering_angle))
        wheel.pose.orientation.x, wheel.pose.orientation.y = q[0], q[1]
        wheel.pose.orientation.z, wheel.pose.orientation.w = q[2], q[3]
        wheel.scale.x = self.wheel_visualization_size
        wheel.scale.y = self.wheel_visualization_size
        wheel.scale.z = 0.1
        wheel.color.r, wheel.color.g, wheel.color.b, wheel.color.a = 0.2, 0.2, 0.2, 0.8
        marker_array.markers.append(wheel)

        left_wheel = Marker()
        left_wheel.header.frame_id = "base_link"
        left_wheel.header.stamp = self.get_clock().now().to_msg()
        left_wheel.ns = "steering"
        left_wheel.id = 1
        left_wheel.type = Marker.CYLINDER
        left_wheel.action = Marker.ADD
        left_wheel.pose.position.x = self.wheelbase
        left_wheel.pose.position.y = 0.8
        left_wheel.pose.position.z = 0.3
        q = self._quaternion_from_euler(0, np.radians(90), np.radians(steering_angle))
        left_wheel.pose.orientation.x, left_wheel.pose.orientation.y = q[0], q[1]
        left_wheel.pose.orientation.z, left_wheel.pose.orientation.w = q[2], q[3]
        left_wheel.scale.x = 0.6
        left_wheel.scale.y = 0.6
        left_wheel.scale.z = 0.2
        left_wheel.color.r, left_wheel.color.g = 0.1, 0.1
        left_wheel.color.b, left_wheel.color.a = 0.1, 0.8
        marker_array.markers.append(left_wheel)

        right_wheel = Marker()
        right_wheel.header.frame_id = "base_link"
        right_wheel.header.stamp = self.get_clock().now().to_msg()
        right_wheel.ns = "steering"
        right_wheel.id = 2
        right_wheel.type = Marker.CYLINDER
        right_wheel.action = Marker.ADD
        right_wheel.pose.position.x = self.wheelbase
        right_wheel.pose.position.y = -0.8
        right_wheel.pose.position.z = 0.3
        right_wheel.pose.orientation = left_wheel.pose.orientation
        right_wheel.scale = left_wheel.scale
        right_wheel.color = left_wheel.color
        marker_array.markers.append(right_wheel)

        text = Marker()
        text.header.frame_id = "base_link"
        text.header.stamp = self.get_clock().now().to_msg()
        text.ns = "steering"
        text.id = 3
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = 1.0
        text.pose.position.y = 0.0
        text.pose.position.z = 1.5
        have_recent_feedback = self._secs_since(self.last_feedback_time) <= 1.0
        if have_recent_feedback:
            feedback_text = f" | Feedback: {self.current_steering:.1f}°, {self.current_speed:.1f} km/h"
        else:
            feedback_text = " | No feedback"
        text.text = f"Command: {steering_angle:.1f}° | Speed: {speed:.1f} km/h{feedback_text}"
        text.scale.z = 0.3
        text.color.r, text.color.g, text.color.b, text.color.a = 1.0, 1.0, 1.0, 1.0
        marker_array.markers.append(text)

        arrow = Marker()
        arrow.header.frame_id = "base_link"
        arrow.header.stamp = self.get_clock().now().to_msg()
        arrow.ns = "steering"
        arrow.id = 4
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose.position.x = self.wheelbase + 0.5
        arrow.pose.position.y = 0.0
        arrow.pose.position.z = 0.5
        q = self._quaternion_from_euler(0, 0, np.radians(steering_angle))
        arrow.pose.orientation.x, arrow.pose.orientation.y = q[0], q[1]
        arrow.pose.orientation.z, arrow.pose.orientation.w = q[2], q[3]
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 1.5, 0.3, 0.2
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 0.0, 0.8, 0.2, 0.8
        marker_array.markers.append(arrow)

        self.visualization_pub.publish(marker_array)

        if have_recent_feedback:
            self.visualize_steering_comparison(steering_angle, self.current_steering)

    def visualize_steering_comparison(self, target_steering, actual_steering):
        """Blue target vs red actual steering arrows + diff text, matching
        steering_calculator_node.py's visualize_steering_comparison exactly."""
        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.header.frame_id = "base_link"
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.ns = "steering_comparison"
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        self.actual_vs_target_pub.publish(marker_array)

        marker_array = MarkerArray()

        target_arrow = Marker()
        target_arrow.header.frame_id = "base_link"
        target_arrow.header.stamp = self.get_clock().now().to_msg()
        target_arrow.ns = "steering_comparison"
        target_arrow.id = 0
        target_arrow.type = Marker.ARROW
        target_arrow.action = Marker.ADD
        target_arrow.pose.position.x = 0.0
        target_arrow.pose.position.y = 0.0
        target_arrow.pose.position.z = 1.8
        q = self._quaternion_from_euler(0, np.radians(-90), np.radians(target_steering))
        target_arrow.pose.orientation.x, target_arrow.pose.orientation.y = q[0], q[1]
        target_arrow.pose.orientation.z, target_arrow.pose.orientation.w = q[2], q[3]
        target_arrow.scale.x, target_arrow.scale.y, target_arrow.scale.z = 1.0, 0.1, 0.1
        target_arrow.color.r, target_arrow.color.g = 0.0, 0.0
        target_arrow.color.b, target_arrow.color.a = 1.0, 0.8
        marker_array.markers.append(target_arrow)

        actual_arrow = Marker()
        actual_arrow.header.frame_id = "base_link"
        actual_arrow.header.stamp = self.get_clock().now().to_msg()
        actual_arrow.ns = "steering_comparison"
        actual_arrow.id = 1
        actual_arrow.type = Marker.ARROW
        actual_arrow.action = Marker.ADD
        actual_arrow.pose.position.x = 0.0
        actual_arrow.pose.position.y = 0.0
        actual_arrow.pose.position.z = 1.7
        q = self._quaternion_from_euler(0, np.radians(-90), np.radians(actual_steering))
        actual_arrow.pose.orientation.x, actual_arrow.pose.orientation.y = q[0], q[1]
        actual_arrow.pose.orientation.z, actual_arrow.pose.orientation.w = q[2], q[3]
        actual_arrow.scale = target_arrow.scale
        actual_arrow.color.r, actual_arrow.color.g = 1.0, 0.0
        actual_arrow.color.b, actual_arrow.color.a = 0.0, 0.8
        marker_array.markers.append(actual_arrow)

        diff_text = Marker()
        diff_text.header.frame_id = "base_link"
        diff_text.header.stamp = self.get_clock().now().to_msg()
        diff_text.ns = "steering_comparison"
        diff_text.id = 2
        diff_text.type = Marker.TEXT_VIEW_FACING
        diff_text.action = Marker.ADD
        diff_text.pose.position.x = 0.0
        diff_text.pose.position.y = 0.0
        diff_text.pose.position.z = 2.0
        steering_diff = actual_steering - target_steering
        diff_text.text = f"Target:{target_steering:.1f}°|Actual:{actual_steering:.1f}°|Diff: {steering_diff:.1f}°"
        diff_text.scale.z = 0.2
        diff_text.color.r, diff_text.color.g, diff_text.color.b, diff_text.color.a = 1.0, 1.0, 1.0, 1.0
        marker_array.markers.append(diff_text)

        self.actual_vs_target_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = SteeringUartBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
