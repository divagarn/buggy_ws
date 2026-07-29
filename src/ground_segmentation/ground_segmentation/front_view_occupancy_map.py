#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from sensor_msgs_py import point_cloud2 as pc2

# NumPy compatibility for older transforms3d used by tf_transformations.
if not hasattr(np, "float"):
    np.float = float

from tf_transformations import quaternion_from_euler

class FrontViewOccupancyMap(Node):
    def __init__(self):
        super().__init__('front_view_occupancy_map')
        
        # Declare and get parameters
        self.declare_parameter('roi_x_min', 0.0)
        self.declare_parameter('roi_x_max', 15.0)
        self.declare_parameter('roi_y_min', -3.0)
        self.declare_parameter('roi_y_max', 3.0)
        self.declare_parameter('roi_z_min', -0.5)
        self.declare_parameter('roi_z_max', 2.0)
        self.declare_parameter('num_bins', 60)
        self.declare_parameter('occupancy_threshold', 17)
        self.declare_parameter('update_rate', 10.0)
        self.declare_parameter('visualize_bins', True)
        self.declare_parameter('visualize_center', True)
        self.declare_parameter('bin_marker_height', -1.0)

        self.roi_x_min = self.get_parameter('roi_x_min').value
        self.roi_x_max = self.get_parameter('roi_x_max').value
        self.roi_y_min = self.get_parameter('roi_y_min').value
        self.roi_y_max = self.get_parameter('roi_y_max').value
        self.roi_z_min = self.get_parameter('roi_z_min').value
        self.roi_z_max = self.get_parameter('roi_z_max').value
        self.num_bins = self.get_parameter('num_bins').value
        self.occupancy_threshold = self.get_parameter('occupancy_threshold').value
        self.update_rate = self.get_parameter('update_rate').value
        self.visualize_bins = self.get_parameter('visualize_bins').value
        self.visualize_center = self.get_parameter('visualize_center').value
        self.bin_marker_height = self.get_parameter('bin_marker_height').value
        
        # Create bins for the histogram
        self.bins = np.linspace(self.roi_y_min, self.roi_y_max, num=self.num_bins+1)
        self.bin_width = (self.roi_y_max - self.roi_y_min) / self.num_bins
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        
        # Subscribers
        self.subscription = self.create_subscription(
            PointCloud2, '/non_ground_points', self.non_ground_callback, 1)
        
        # Publishers
        self.occupancy_marker_pub = self.create_publisher(MarkerArray, '/occupancy_map_markers', 1)
        self.center_line_pub = self.create_publisher(MarkerArray, '/free_space_center_marker', 1)
        self.target_y_pub = self.create_publisher(Float32, '/target_lateral_position', 1)
        
        # Timer for fixed-rate processing
        self.timer = self.create_timer(1.0 / self.update_rate, self.process_and_publish)
        
        # Store latest point cloud
        self.latest_cloud = None
        self.last_update_time = self.get_clock().now()
        
        self.get_logger().info("Front View Occupancy Map node initialized")
    
    def non_ground_callback(self, msg):
        """Store the latest non-ground point cloud for processing"""
        self.latest_cloud = msg
        self.last_update_time = self.get_clock().now()
    
    def process_and_publish(self):
        """Process point cloud data and publish visualization markers at a fixed rate"""
        if self.latest_cloud is None:
            self.get_logger().warn("No non-ground point cloud data received yet", throttle_duration_sec=5.0)
            return
        
        # Check if the data is too old (more than 0.5 seconds)
        time_diff = (self.get_clock().now() - self.last_update_time).nanoseconds / 1e9
        if time_diff > 0.5:
            self.get_logger().warn("Non-ground point cloud data is too old", throttle_duration_sec=5.0)
            return
            
        try:
            # Convert the point cloud to a numpy array
            pc_data = np.array(list(pc2.read_points(self.latest_cloud, field_names=("x", "y", "z"), skip_nans=True)))
            if pc_data.ndim == 0 or len(pc_data) == 0:
                self.get_logger().info("Empty point cloud", throttle_duration_sec=3.0)
                return
            # Convert structured array (named fields) to regular Nx3 float32 array
            points = np.column_stack([pc_data['x'], pc_data['y'], pc_data['z']]).astype(np.float32)
            
            # Filter points within ROI
            mask = (points[:, 0] > self.roi_x_min) & \
                  (points[:, 0] < self.roi_x_max) & \
                  (points[:, 1] > self.roi_y_min) & \
                  (points[:, 1] < self.roi_y_max) & \
                  (points[:, 2] > self.roi_z_min) & \
                  (points[:, 2] < self.roi_z_max)
            roi_points = points[mask]
            
            if len(roi_points) == 0:
                self.get_logger().info("No points in ROI", throttle_duration_sec=3.0)
                # Publish default center position (0.0) when no obstacles detected
                float_msg = Float32()
                float_msg.data = 0.0
                self.target_y_pub.publish(float_msg)
                self.visualize_free_center(self.latest_cloud.header, 0.0)
                self.visualize_occupancy_bins(self.latest_cloud.header, np.zeros(self.num_bins))
                return
                
            # Extract lateral positions (Y values)
            lateral_points = roi_points[:, 1]
            
            # Create histogram
            counts, _ = np.histogram(lateral_points, bins=self.bins)
            
            # Find center of free space
            free_bins = np.where(counts < self.occupancy_threshold)[0]
            
            if len(free_bins) == 0:
                self.get_logger().warn("No free space found in front view", throttle_duration_sec=3.0)
                target_y = 0.0  # Default to center when no free space
            else:
                # Find the widest continuous free space
                free_bin_groups = self.find_continuous_groups(free_bins)
                if len(free_bin_groups) == 0:
                    target_y = 0.0
                else:
                    # Choose the widest group
                    widest_group = max(free_bin_groups, key=len)
                    # Choose the middle of the widest free area
                    target_y = np.mean(self.bin_centers[widest_group])
                    
                    # Try to find the best path considering obstacle distribution
                    # Find locations of obstacles (high counts)
                    obstacle_bins = np.where(counts >= self.occupancy_threshold)[0]
                    if len(obstacle_bins) > 0:
                        # Calculate center of obstacles
                        obstacle_center = np.mean(self.bin_centers[obstacle_bins])
                        
                        # If target is on same side as obstacle center, consider moving away slightly
                        if (target_y > 0 and obstacle_center > 0) or (target_y < 0 and obstacle_center < 0):
                            # Move target away from obstacles by adjusting up to 20%
                            adjustment = 0.2 * abs(target_y)
                            if target_y > 0:  # If target is positive (left), move more left
                                target_y += adjustment
                            else:  # If target is negative (right), move more right
                                target_y -= adjustment
                                
                            self.get_logger().info(f"Adjusted target away from obstacles: {target_y:.2f}m")
                
            # Limit target_y to within valid range
            target_y = max(min(target_y, self.roi_y_max - 0.5), self.roi_y_min + 0.5)
            
            # Publish target lateral position
            float_msg = Float32()
            float_msg.data = target_y
            self.target_y_pub.publish(float_msg)
            
            # Visualize the results
            if self.visualize_center:
                self.visualize_free_center(self.latest_cloud.header, target_y)
            
            if self.visualize_bins:
                self.visualize_occupancy_bins(self.latest_cloud.header, counts)
                
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {e}")

    def find_continuous_groups(self, indices):
        """Find continuous groups of bin indices"""
        if len(indices) == 0:
            return []
            
        # Sort indices first
        indices = sorted(indices)
        groups = []
        current_group = [indices[0]]
        
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                # Continuous with previous index
                current_group.append(indices[i])
            else:
                # Start a new group
                groups.append(current_group)
                current_group = [indices[i]]
                
        # Add the last group
        groups.append(current_group)
        return groups
    
    def visualize_occupancy_bins(self, header, counts):
        """Visualize the occupancy histogram as 3D bins"""
        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header = header
        clear_marker.ns = "occupancy_bins"
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        self.occupancy_marker_pub.publish(marker_array)
        
        # Create new markers
        marker_array = MarkerArray()
        
        # Normalize counts for better visualization
        max_count = max(counts) if max(counts) > 0 else 1
        normalized_counts = counts / max_count
        
        # Create a marker for each bin
        for i in range(len(counts)):
            marker = Marker()
            marker.header = header
            marker.ns = "occupancy_bins"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Position the marker at the center of the bin
            # Set x to be mid-point of ROI
            marker.pose.position.x = (self.roi_x_min + self.roi_x_max) / 2
            marker.pose.position.y = self.bin_centers[i]
            marker.pose.position.z = self.bin_marker_height / 2  # Center in Z
            marker.pose.orientation.w = 1.0
            
            # Set scale based on bin width and count
            marker.scale.x = self.roi_x_max - self.roi_x_min  # Full length of ROI
            marker.scale.y = self.bin_width * 0.9  # Slightly smaller than bin width to create gaps
            marker.scale.z = self.bin_marker_height  # Fixed height
            
            # Set color based on occupancy (red for occupied, green for free)
            if counts[i] < self.occupancy_threshold:
                # Free space - green
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.3
            else:
                # Occupied space - red, with intensity based on count
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.2 + 0.6 * normalized_counts[i]  # More opaque for higher counts
            
            marker_array.markers.append(marker)
        
        self.occupancy_marker_pub.publish(marker_array)

    def visualize_free_center(self, header, target_y):
        """Visualize the center of free space as a curved path"""
        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header = header
        clear_marker.ns = "free_center"
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        self.center_line_pub.publish(marker_array)
        
        # Create new markers
        marker_array = MarkerArray()
        
        # Path line marker (curved)
        path = Marker()
        path.header = header
        path.ns = "free_center"
        path.id = 0
        path.type = Marker.LINE_STRIP
        path.action = Marker.ADD
        path.pose.orientation.w = 1.0
        path.scale.x = 0.15  # Line width
        
        # Blue path line
        path.color.r = 0.0
        path.color.g = 0.2
        path.color.b = 1.0
        path.color.a = 0.8
        
        # Create a curved path that starts at vehicle and ends at target
        # The curve gradually transitions from current position to target position
        num_points = 20  # Number of points to create a smooth curve
        points = []
        
        # Calculate curve parameters - how much to curve the path
        # The curve should be more pronounced if target_y is further from center
        curve_intensity = abs(target_y) * 0.3  # Scale curve based on lateral distance
        
        # Direction of curve - positive if target is to the right, negative if to the left
        curve_direction = -1 if target_y > 0 else 1
        
        for i in range(num_points):
            # Parametric progress along path (0.0 to 1.0)
            t = i / (num_points - 1)
            
            # X coordinate increases linearly from 0 to max distance
            x = t * self.roi_x_max
            
            # Y coordinate follows a curve that:
            # 1. Starts at 0 (vehicle position)
            # 2. Gradually transitions to target_y
            # 3. Uses a smooth transition function
            
            # Smooth transition function (S-curve)
            smoothstep = t * t * (3 - 2 * t)
            
            # For curves, add a sine wave that peaks in the middle and goes to 0 at ends
            # The sine wave is stronger when the target is further from center
            sine_factor = np.sin(t * np.pi) * curve_intensity * curve_direction
            
            # Mix direct path with curved path based on distance
            if abs(target_y) > 0.5:  # Only add significant curves for larger offsets
                y = smoothstep * target_y + sine_factor * (1 - smoothstep)
            else:
                # For small offsets, use more direct path
                y = smoothstep * target_y
            
            points.append(Point(x=float(x), y=float(y), z=0.0))
        
        path.points = points
        marker_array.markers.append(path)
        
        # Calculate initial direction vector for the path (for arrow orientation)
        # Use the direction between the first two points of the curve
        if len(path.points) >= 2:
            init_x = path.points[1].x - path.points[0].x
            init_y = path.points[1].y - path.points[0].y
            direction_angle = np.arctan2(init_y, init_x)
        else:
            direction_angle = np.arctan2(target_y, 2.0)
        
        # Arrow marker showing initial direction of travel
        arrow = Marker()
        arrow.header = header
        arrow.ns = "free_center"
        arrow.id = 1
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        
        # Position at start of path
        arrow.pose.position.x = 1.0  # Place 1m ahead of vehicle
        arrow.pose.position.y = path.points[0].y if len(path.points) > 0 else 0.0
        arrow.pose.position.z = 0.5
        
        # Set orientation based on initial path direction
        q = quaternion_from_euler(0, 0, direction_angle)
        arrow.pose.orientation.x = q[0]
        arrow.pose.orientation.y = q[1]
        arrow.pose.orientation.z = q[2]
        arrow.pose.orientation.w = q[3]
        
        arrow.scale.x = 2.0  # Arrow length
        arrow.scale.y = 0.3  # Arrow width
        arrow.scale.z = 0.3  # Arrow height
        
        # Blue arrow
        arrow.color.r = 0.0
        arrow.color.g = 0.4
        arrow.color.b = 1.0
        arrow.color.a = 0.8
        
        marker_array.markers.append(arrow)
        
        # Find middle point of curve (for text placement)
        mid_point_idx = len(path.points) // 2
        mid_point_x = path.points[mid_point_idx].x if path.points else 3.0
        mid_point_y = path.points[mid_point_idx].y if path.points else target_y
        
        # Text label with the target y-position
        text = Marker()
        text.header = header
        text.ns = "free_center"
        text.id = 2
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        
        text.pose.position.x = mid_point_x  # Position at mid-point of curve
        text.pose.position.y = mid_point_y
        text.pose.position.z = 1.0
        text.pose.orientation.w = 1.0
        
        text.text = f"Target: {target_y:.2f}m"
        text.scale.z = 0.6  # Text size
        
        # White text
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        
        marker_array.markers.append(text)
        
        # Add a sphere at the end of the path for better visibility
        end_marker = Marker()
        end_marker.header = header
        end_marker.ns = "free_center"
        end_marker.id = 3
        end_marker.type = Marker.SPHERE
        end_marker.action = Marker.ADD
        
        # Position at end of path
        if len(path.points) > 0:
            end_marker.pose.position.x = path.points[-1].x
            end_marker.pose.position.y = path.points[-1].y
        else:
            end_marker.pose.position.x = self.roi_x_max
            end_marker.pose.position.y = target_y
        end_marker.pose.position.z = 0.5
        end_marker.pose.orientation.w = 1.0
        
        end_marker.scale.x = 0.5  # Sphere diameter
        end_marker.scale.y = 0.5
        end_marker.scale.z = 0.5
        
        # Cyan sphere
        end_marker.color.r = 0.0
        end_marker.color.g = 0.8
        end_marker.color.b = 1.0
        end_marker.color.a = 0.8
        
        marker_array.markers.append(end_marker)
        
        self.center_line_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = FrontViewOccupancyMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
