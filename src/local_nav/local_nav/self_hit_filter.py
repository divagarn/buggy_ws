#!/usr/bin/env python3
"""Drop vehicle self-hit points from ground_segmentation's /non_ground_points,
before they reach the local costmap.

Simulation-only concern: ground_segmentation is the team's real-hardware
code and does no self-filtering (the real Velodyne mount doesn't clip the
chassis the way it does in Gazebo, where the sensor sits only ~15cm above
the roof). Without this, the LiDAR's self-hit returns on the chassis/hood
get classified as non-ground obstacles, forming a lethal costmap blob that
overlaps the robot's own footprint - every rollout starts already "in
collision" and MPPI's optimizer fails on every cycle regardless of motion
model. Box/range values are the same ones already validated against this
same buggy_description URDF in the original (pre-team-merge) workspace's
segment_ground.py self-footprint filter.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

SELF_FILTER_MIN_RANGE = 1.0
SELF_FILTER_X_MIN = -2.39
SELF_FILTER_X_MAX = 0.53
SELF_FILTER_Y_MIN = -0.72
SELF_FILTER_Y_MAX = 0.72
SELF_FILTER_Z_MIN = -1.05
SELF_FILTER_Z_MAX = -0.10


class SelfHitFilter(Node):
    def __init__(self):
        super().__init__('self_hit_filter')
        self.pub = self.create_publisher(PointCloud2, '/non_ground_points_filtered', 2)
        self.create_subscription(
            PointCloud2, '/non_ground_points', self.callback, 2)

    def callback(self, msg):
        if msg.width == 0:
            return
        pts = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
        xyz = pts[:, :3]
        too_close = np.linalg.norm(xyz, axis=1) < SELF_FILTER_MIN_RANGE
        inside_box = (
            (xyz[:, 0] > SELF_FILTER_X_MIN) & (xyz[:, 0] < SELF_FILTER_X_MAX) &
            (xyz[:, 1] > SELF_FILTER_Y_MIN) & (xyz[:, 1] < SELF_FILTER_Y_MAX) &
            (xyz[:, 2] > SELF_FILTER_Z_MIN) & (xyz[:, 2] < SELF_FILTER_Z_MAX)
        )
        kept = pts[~(too_close | inside_box)]

        out = PointCloud2()
        out.header = msg.header
        out.height = 1
        out.width = len(kept)
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = msg.point_step
        out.row_step = msg.point_step * len(kept)
        out.data = kept.tobytes()
        out.is_dense = True
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SelfHitFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
