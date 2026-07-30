#!/usr/bin/env python3
"""Manual keyboard teleop for the raw steering/UART path - bypasses the
whole nav stack (no segment_ground, no DWB, no planner) and publishes
directly to /steering_angle, /final_red_detected, /final_yellow_detected,
the exact same topics uart_sender_node.py subscribes to. Use this to
verify the buggy's steering servo actually responds to commands before
trusting the full autonomous stack - isolates "is the UART/hardware path
working at all" from "is the navigation logic correct".

Only useful with uart_sender_node actually running (real_navigate.launch.py,
or uart_sender_node on its own) and loopback:=false - otherwise nothing
physically happens, same as the rest of the stack.

Controls:
  a / LEFT   - steer left  (more negative)
  d / RIGHT  - steer right (more positive)
  s          - center steering (0 deg)
  g          - GO   (final_red_detected = False, allows motion)
  r          - STOP (final_red_detected = True, default/safe state)
  y          - toggle final_yellow_detected (caution flag)
  q / CTRL-C - quit (resets to STOP + 0 deg before exiting)

Steering is clamped to +-20deg, the vehicle's real physical limit
(buggy.urdf.xacro max_steer_rad = 0.349066 = 20deg) - same clamp
steering_uart_bridge.py enforces for the autonomous stack.
"""

import sys
import termios
import tty
import select

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool

STEERING_STEP = 2.0
MAX_STEERING_DEG = 20.0


def get_key(settings, timeout=0.1):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class SteeringTeleop(Node):
    def __init__(self):
        super().__init__('steering_teleop')
        self.steering_pub = self.create_publisher(Float32, '/steering_angle', 1)
        self.red_pub = self.create_publisher(Bool, '/final_red_detected', 1)
        self.yellow_pub = self.create_publisher(Bool, '/final_yellow_detected', 1)

        self.steering_deg = 0.0
        self.stopped = True
        self.yellow = False

        # uart_sender_node sends at 10Hz regardless of when it last heard
        # from us - republish continuously so our commanded state is
        # always what's fresh, not a single one-shot message.
        self.create_timer(0.1, self.publish_state)

    def publish_state(self):
        self.steering_pub.publish(Float32(data=self.steering_deg))
        self.red_pub.publish(Bool(data=self.stopped))
        self.yellow_pub.publish(Bool(data=self.yellow))

    def status_line(self):
        state = 'STOPPED' if self.stopped else 'GO'
        return f'steering={self.steering_deg:+.1f}deg  state={state}  yellow={self.yellow}'


def main():
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init()
    node = SteeringTeleop()

    print(__doc__)
    print(node.status_line())

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            key = get_key(settings, timeout=0.1)

            if key == 'a' or key == '\x1b[D':
                node.steering_deg = max(-MAX_STEERING_DEG, node.steering_deg - STEERING_STEP)
            elif key == 'd' or key == '\x1b[C':
                node.steering_deg = min(MAX_STEERING_DEG, node.steering_deg + STEERING_STEP)
            elif key == 's':
                node.steering_deg = 0.0
            elif key == 'g':
                node.stopped = False
            elif key == 'r':
                node.stopped = True
            elif key == 'y':
                node.yellow = not node.yellow
            elif key == 'q' or key == '\x03':
                break
            elif key == '':
                continue
            else:
                continue

            print(node.status_line())

    finally:
        # Always leave the vehicle in a safe, centered, stopped state on exit.
        node.steering_deg = 0.0
        node.stopped = True
        node.publish_state()
        print('\nExiting - steering centered, STOP sent.')
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
