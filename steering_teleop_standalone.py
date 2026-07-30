#!/usr/bin/env python3
"""Standalone keyboard steering test - NO ROS2, NO rclpy, just pyserial
writing straight to the port. Isolates "is the raw UART/hardware path
working" from "is ROS2/topics/nodes routing correctly" - the lowest-level
test possible short of an oscilloscope.

Packet format and constants are copied directly from the team's
uart/uart/uart_interface.py (the code uart_sender_node.py actually uses),
not re-derived - byte-for-byte identical to what the real ROS2 stack
sends: [START, yellow, red, angle_hi, angle_lo, critical_flag, END].

Requires: pip install --user pyserial   (already installed on this
machine from the earlier uart_bridge/uart_sender_node fix)

IMPORTANT: run this standalone, NOT while uart_sender_node.py is also
running - both would fight over the same serial port.

Controls:
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

# --- exact copy of uart/uart/uart_interface.py's protocol constants ---
PROTOCOL_START = 40  # '(' ASCII
PROTOCOL_END = 41    # ')' ASCII
OFFSET = -30
RESOLUTION = 0.01
DEGREE_RANGE = (-20, 20)

PORT = '/dev/ttyUSB0'
BAUDRATE = 115200
SEND_RATE_HZ = 10.0  # matches uart_sender_node.py's own send timer
STEERING_STEP = 2.0


def build_packet(yellow_detected, red_detected, degree, critical_flag=False):
    """Exact copy of UartInterface.create_message()."""
    degree = max(min(degree, DEGREE_RANGE[1]), DEGREE_RANGE[0])
    raw_value = int((degree - OFFSET) / RESOLUTION)
    raw_value = max(min(raw_value, 32767), -32768)
    raw_value_bytes = raw_value.to_bytes(2, byteorder='big', signed=True)

    message = bytearray()
    message.append(PROTOCOL_START)
    message.append(1 if yellow_detected else 0)
    message.append(1 if red_detected else 0)
    message.extend(raw_value_bytes)
    message.append(1 if critical_flag else 0)
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

    print(f"Connected to {PORT} @ {BAUDRATE} baud")
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
