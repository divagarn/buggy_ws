#!/usr/bin/env python3
"""Same as steering_teleop_standalone.py, but sends the ORIGINAL ROS1
6-byte packet format instead of the current ROS2 team code's 7-byte
format - no critical_flag byte, [START, yellow, red, angle_hi, angle_lo,
END]. Copied directly from catkin_ws/src/ros_uart_sender/src/
uart_interface.py's create_message(), not re-derived.

Use this if the 7-byte ROS2 format (steering_teleop_standalone.py) isn't
making the buggy respond - it tests whether the MCU firmware is still
expecting the older ROS1 format, where the extra critical_flag byte in
the newer format would shift the END byte position by one, breaking the
MCU's packet parsing.

Requires: pip install --user pyserial (already installed)

IMPORTANT: run standalone, NOT while uart_sender_node.py is also running
- both would fight over the same serial port.

Controls: same as steering_teleop_standalone.py
  a / LEFT   - steer left  (more negative)
  d / RIGHT  - steer right (more positive)
  s          - center steering (0 deg)
  g          - GO   (red_detected = False, allows motion)
  r          - STOP (red_detected = True, default/safe state)
  y          - toggle yellow_detected (caution flag)
  q / CTRL-C - quit (resets to STOP + 0 deg before exiting)
"""

import sys
import time
import termios
import tty
import select

import serial

# --- exact copy of catkin_ws/src/ros_uart_sender/src/uart_interface.py's
# protocol constants (ROS1) ---
PROTOCOL_START = 40  # '(' ASCII
PROTOCOL_END = 41    # ')' ASCII
OFFSET = -30
RESOLUTION = 0.01
DEGREE_RANGE = (-20, 20)

PORT = '/dev/ttyUSB0'
BAUDRATE = 115200
SEND_RATE_HZ = 10.0
STEERING_STEP = 2.0


def build_packet(yellow_detected, red_detected, degree):
    """Exact copy of ROS1's UartInterface.create_message() - 6 bytes,
    no critical_flag byte."""
    degree = max(min(degree, DEGREE_RANGE[1]), DEGREE_RANGE[0])
    raw_value = int((degree - OFFSET) / RESOLUTION)
    raw_value = max(min(raw_value, 32767), -32768)
    raw_value_bytes = raw_value.to_bytes(2, byteorder='big', signed=True)

    message = bytearray()
    message.append(PROTOCOL_START)
    message.append(1 if yellow_detected else 0)
    message.append(1 if red_detected else 0)
    message.extend(raw_value_bytes)
    message.append(PROTOCOL_END)
    return message


def get_key(settings, timeout=0.1):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def status_line(steering_deg, stopped, yellow):
    state = 'STOPPED' if stopped else 'GO'
    return f'steering={steering_deg:+.1f}deg  state={state}  yellow={yellow}'


def main():
    settings = termios.tcgetattr(sys.stdin)

    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    except serial.SerialException as e:
        print(f"Failed to open {PORT}: {e}")
        sys.exit(1)

    print(f"Connected to {PORT} @ {BAUDRATE} baud (ROS1 6-byte packet format)")
    print(__doc__)

    steering_deg = 0.0
    stopped = True
    yellow = False
    print(status_line(steering_deg, stopped, yellow))

    last_send = 0.0
    send_period = 1.0 / SEND_RATE_HZ

    try:
        while True:
            key = get_key(settings, timeout=0.05)

            if key == 'a' or key == '\x1b[D':
                steering_deg = max(-20.0, steering_deg - STEERING_STEP)
                print(status_line(steering_deg, stopped, yellow))
            elif key == 'd' or key == '\x1b[C':
                steering_deg = min(20.0, steering_deg + STEERING_STEP)
                print(status_line(steering_deg, stopped, yellow))
            elif key == 's':
                steering_deg = 0.0
                print(status_line(steering_deg, stopped, yellow))
            elif key == 'g':
                stopped = False
                print(status_line(steering_deg, stopped, yellow))
            elif key == 'r':
                stopped = True
                print(status_line(steering_deg, stopped, yellow))
            elif key == 'y':
                yellow = not yellow
                print(status_line(steering_deg, stopped, yellow))
            elif key == 'q' or key == '\x03':
                break

            now = time.time()
            if now - last_send >= send_period:
                packet = build_packet(yellow, stopped, steering_deg)
                ser.write(packet)
                last_send = now

    finally:
        steering_deg = 0.0
        stopped = True
        packet = build_packet(False, stopped, steering_deg)
        ser.write(packet)
        print('\nExiting - steering centered, STOP sent.')
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        ser.close()


if __name__ == '__main__':
    main()
