"""STANDALONE TEST: identifies the center path in the LOCAL costmap's free
space, ahead of the robot, and publishes it as a marker - step 1 toward
feeding TEB a genuinely-centered reference path instead of
SmacPlannerHybrid's shortest-path output (see conversation this was built
from: TEB "collapsing" sometimes is suspected to be a bad-initial-guess
problem from a global path that isn't centered). Deliberately a separate
node from carrot_path_publisher/track_centerline_marker for now, to
verify the local-costmap version of this technique in isolation before
wiring it into actual navigation.

Different from the two other centerline mechanisms already in this
project:
  - carrot_path_publisher's /centerline_marker: a lateral BIAS (a single
    steering nudge), not a path.
  - track_centerline_marker's /track_centerline_marker: the WHOLE loop,
    computed once from the static map - global, not reactive to what's
    actually clear right now.
  - THIS node: skeletonizes the LIVE local costmap (same technique as
    both of the above - skimage skeletonize + free-space mask), then
    keeps only the piece of that skeleton in a forward cone ahead of the
    robot's current heading, ordered near-to-far. Reactive (recomputed
    on a timer against the live rolling costmap) and local (bounded by
    forward_distance/forward_half_angle_deg), unlike track_centerline_marker.

Run standalone alongside any of the existing closed-loop launches (needs
/local_costmap/costmap - published once controller_server is active - and
a odom_frame->base_frame TF, both already present):
    ros2 run local_nav local_centerline_marker
"""

import math

import networkx as nx
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from skimage.morphology import skeletonize
from visualization_msgs.msg import Marker

LETHAL_THRESHOLD = 90  # nav2_costmap_2d: 253 lethal / 254 inscribed / 0-252 inflation decay - same as carrot_path_publisher.py


class LocalCenterlineMarker(Node):

    def __init__(self):
        super().__init__('local_centerline_marker')

        self.declare_parameter('costmap_topic', '/local_costmap/costmap')
        self.declare_parameter('marker_topic', '/local_centerline_marker')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('forward_distance', 8.0)
        self.declare_parameter('forward_half_angle_deg', 60.0)
        self.declare_parameter('update_rate_hz', 2.0)

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.forward_distance = self.get_parameter('forward_distance').value
        self.forward_half_angle = math.radians(
            self.get_parameter('forward_half_angle_deg').value)

        self.latest_costmap = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.marker_pub = self.create_publisher(
            Marker, self.get_parameter('marker_topic').value, 1)
        self.create_subscription(
            OccupancyGrid, self.get_parameter('costmap_topic').value, self.costmap_callback, 1)

        rate = self.get_parameter('update_rate_hz').value
        self.create_timer(1.0 / rate, self.tick)

        self.get_logger().info(
            'Local centerline marker (TEST) ready - watching '
            f'{self.get_parameter("costmap_topic").value}, forward_distance='
            f'{self.forward_distance}m, forward_half_angle='
            f'{self.get_parameter("forward_half_angle_deg").value}deg.')

    def costmap_callback(self, msg):
        self.latest_costmap = msg

    def _robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame, rclpy.time.Time())
        except Exception:
            return None
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return tf.transform.translation.x, tf.transform.translation.y, yaw

    def tick(self):
        if self.latest_costmap is None:
            return
        pose = self._robot_pose()
        if pose is None:
            self.get_logger().warn(
                f'No {self.odom_frame}->{self.base_frame} transform yet.',
                throttle_duration_sec=5.0)
            return
        rx, ry, ryaw = pose

        msg = self.latest_costmap
        info = msg.info
        w, h = info.width, info.height
        res = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y
        grid = np.array(msg.data, dtype=np.int16).reshape(h, w)

        free = (grid >= 0) & (grid < LETHAL_THRESHOLD)
        skeleton = skeletonize(free)
        ys, xs = np.nonzero(skeleton)
        if len(ys) == 0:
            return

        # Skeleton pixels -> world (x, y), then keep only the ones in a
        # forward cone ahead of the robot's current heading - "forward" is
        # measured relative to the robot, not the costmap's own axes, since
        # the costmap doesn't rotate with the vehicle (see this file's
        # docstring / local_costmap.yaml's own comment on that).
        world_x = ox + (xs + 0.5) * res
        world_y = oy + (ys + 0.5) * res
        dx = world_x - rx
        dy = world_y - ry
        forward = dx * math.cos(ryaw) + dy * math.sin(ryaw)
        lateral = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
        distance = np.hypot(dx, dy)
        angle = np.abs(np.arctan2(lateral, forward))

        keep = (
            (forward > 0.0)
            & (distance <= self.forward_distance)
            & (angle <= self.forward_half_angle)
        )
        if not np.any(keep):
            return

        # Order near-to-far by forward distance - a reasonable path
        # ordering for a roughly-one-dimensional corridor ahead, without
        # needing a full graph-shortest-path search for this quick test.
        order = np.argsort(forward[keep])
        path_x = world_x[keep][order]
        path_y = world_y[keep][order]

        marker = Marker()
        marker.header.frame_id = self.odom_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'local_centerline_test'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.15
        marker.color.r = 0.0
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        marker.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in zip(path_x, path_y)]
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = LocalCenterlineMarker()
    rclpy.spin(node)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
