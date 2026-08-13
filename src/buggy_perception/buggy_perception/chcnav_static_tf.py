#!/usr/bin/env python3
"""Real-hardware counterpart of the gps_mount_link/imu_mount_link joints
that buggy_description's URDF already defines and Gazebo sim gets for free
via robot_state_publisher. Real launches have no robot_state_publisher
(this project's real-hardware convention is small per-sensor static-TF
nodes instead - see velodyne_static_tf.py), so without this there is no
base_link->imu_mount_link/gps_mount_link TF at all on real hardware, and
robot_localization's imu0 input would silently be dropped (it resolves
sensor_msgs/Imu.header.frame_id via TF to base_link_frame).

Offsets copied from the resolved values in buggy_description/urdf/
buggy.urdf.xacro's gps_mount_joint/imu_mount_joint (xacro-expanded:
gps 0 0 0.62, imu 0 0 0 - imu_mount_link sits at base_link's own origin,
gps_mount_link chassis_box_height above it). Keep in sync if the xacro
changes.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

GPS_MOUNT_Z = 0.62
IMU_MOUNT_Z = 0.0


class ChcnavStaticTF(Node):
    def __init__(self):
        super().__init__('chcnav_static_tf')

        broadcaster = StaticTransformBroadcaster(self)
        now = self.get_clock().now().to_msg()

        gps_tf = TransformStamped()
        gps_tf.header.stamp = now
        gps_tf.header.frame_id = 'base_link'
        gps_tf.child_frame_id = 'gps_mount_link'
        gps_tf.transform.translation.z = GPS_MOUNT_Z
        gps_tf.transform.rotation.w = 1.0

        imu_tf = TransformStamped()
        imu_tf.header.stamp = now
        imu_tf.header.frame_id = 'base_link'
        imu_tf.child_frame_id = 'imu_mount_link'
        imu_tf.transform.translation.z = IMU_MOUNT_Z
        imu_tf.transform.rotation.w = 1.0

        broadcaster.sendTransform([gps_tf, imu_tf])

        self.get_logger().info(
            f'Broadcasted static transforms base_link->gps_mount_link (z={GPS_MOUNT_Z}m) '
            f'and base_link->imu_mount_link (z={IMU_MOUNT_Z}m)'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ChcnavStaticTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
