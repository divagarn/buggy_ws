#!/usr/bin/env python3
"""Simulation stand-in for uart/uart_sender_node.py.

Subscribes to the exact same topics the real UART sender does -
/steering_angle directly (not /final_steering_angle; that's what
uart_sender_node.py actually reads, even though topic_relay_node also
relays steering onto /final_steering_angle), /final_red_detected,
/final_yellow_detected - and instead of writing them out over a serial
port, converts them into the geometry_msgs/Twist that Gazebo's
ackermann_steering_controller expects.

Deliberately matches uart_sender_node.py's current behavior of NOT
forcing a stop on stale data (that logic is commented out there too) -
keep this in sync if that changes.
"""

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist


class SimActuationBridge(Node):

    def __init__(self):
        super().__init__('sim_actuation_bridge')

        self.wheelbase = self.declare_parameter('wheelbase', 1.6).value
        self.normal_speed_kmh = self.declare_parameter('normal_speed_kmh', 8.0).value
        self.yellow_speed_kmh = self.declare_parameter('yellow_speed_kmh', 4.0).value

        self.yellow_detected = False
        self.red_detected = True
        self.degree = 0.0

        self.create_subscription(Float32, '/steering_angle', self.steering_callback, 1)
        self.create_subscription(Bool, '/final_yellow_detected', self.yellow_callback, 1)
        self.create_subscription(Bool, '/final_red_detected', self.red_callback, 1)

        self.cmd_pub = self.create_publisher(
            Twist, '/ackermann_steering_controller/reference_unstamped', 1)

        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz, matches uart_sender_node.py

        self.get_logger().info(f'Sim actuation bridge initialized (wheelbase={self.wheelbase}m)')

    def steering_callback(self, msg):
        self.degree = msg.data

    def yellow_callback(self, msg):
        self.yellow_detected = msg.data

    def red_callback(self, msg):
        self.red_detected = msg.data

    def timer_callback(self):
        if self.red_detected:
            speed_kmh = 0.0
        elif self.yellow_detected:
            speed_kmh = self.yellow_speed_kmh
        else:
            speed_kmh = self.normal_speed_kmh

        speed_ms = speed_kmh / 3.6
        steering_rad = math.radians(self.degree)

        # SteerSpeed convention: positive steering = right. ROS Twist
        # convention: positive angular.z = left. Flip sign converting
        # between the two (same as steering_calculator_node's own flip).
        angular_z = -math.tan(steering_rad) * speed_ms / self.wheelbase if speed_ms != 0 else 0.0

        twist = Twist()
        twist.linear.x = speed_ms
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = SimActuationBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
