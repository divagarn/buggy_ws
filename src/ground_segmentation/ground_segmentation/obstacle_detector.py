#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point
from sensor_msgs_py import point_cloud2 as pc2
from sklearn.cluster import DBSCAN
from ground_segmentation_msgs.msg import ZoneObstacleStatus  # Import custom message


def _rgba(r, g, b, a):
    """Helper to create ColorRGBA with keyword args (ROS 2 doesn't accept positional)."""
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def _point(x, y, z):
    """Helper to create Point with keyword args (ROS 2 doesn't accept positional)."""
    return Point(x=float(x), y=float(y), z=float(z))


class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        
        # Vehicle dimensions (in meters)
        self.vehicle_width = (53 + 15) * 0.0254  # inches to meters (with buffer)
        self.vehicle_length = (111 + 15) * 0.0254  # inches to meters (with buffer)
        self.vehicle_height = 1.8  # estimated height in meters
        
        # Road parameters (in meters)
        self.road_width = 13 * 0.3048 + 7 * 0.0254  # 13ft 7in to meters
        
        # Detection ranges (in meters)
        self.min_detection_dist = (11 * 0.3048) + (10 * 0.0254)  # 11ft 10in
        self.max_detection_dist = 40.5 * 0.3048  # 40.5ft
        
        # Safety zones (in meters)
        self.danger_zone = 3.5  # Red zone - immediate danger
        self.warning_zone = 6.5  # Yellow zone - prepare to stop
        self.safe_zone = 10.0  # Green zone - proceed with caution
        
        # Filtering parameters
        self.declare_parameter('min_distance_threshold', 0.12)
        self.min_distance_threshold = self.get_parameter('min_distance_threshold').value
        self.get_logger().info(f"Minimum distance threshold set to {self.min_distance_threshold}m - objects closer will be filtered as noise")
        
        # Path parameters
        self.path_width = self.vehicle_width * 1.4  # Path slightly wider than vehicle
        
        # Visualization settings
        self.markers_ns = "obstacle_detector"
        # List of all marker namespaces for proper cleanup
        self.all_marker_namespaces = [
            self.markers_ns,               # Main object markers
            self.markers_ns + "_text",     # Text labels
            self.markers_ns + "_text_bg",  # Text backgrounds/glow effect
            self.markers_ns + "_outline",  # Wireframe outlines
        ]
        
        # ROS connections
        self.obstacle_pub = self.create_publisher(PointCloud2, '/detected_obstacles', 1)
        self.markers_pub = self.create_publisher(MarkerArray, '/obstacle_markers', 1)
        self.vehicle_marker_pub = self.create_publisher(MarkerArray, '/vehicle_markers', 1)
        self.path_marker_pub = self.create_publisher(MarkerArray, '/path_markers', 1)
        self.zone_status_pub = self.create_publisher(ZoneObstacleStatus, '/zone_obstacle_status', 1)
        
        # Flag to track whether we need to clear obstacles
        self.last_obstacle_count = 0
        self.has_published_obstacles = False

        # Cached vehicle marker array (built once, re-published with updated stamp)
        self._vehicle_markers_cache = None

        # Timer for continuous vehicle visualization (1 Hz)
        self.vehicle_viz_timer = self.create_timer(1.0, self.publish_vehicle_timer_callback)
        
        # PointCloud2 field layout constants for zero-copy binary
        self._xyz_fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        
        # Subscribers
        self.subscription = self.create_subscription(
            PointCloud2, '/non_ground_points', self.non_ground_callback, 1)
    
    # ── Fast binary PointCloud2 helpers ──────────────────────────────
    @staticmethod
    def _fast_read_xyz(msg):
        """Decode PointCloud2 to Nx3 float32 via direct buffer read — no per-point loop."""
        n = msg.width * msg.height
        if n == 0:
            return None
        step = msg.point_step
        data = bytes(msg.data)  # ensure writable copy
        # Find x/y/z byte offsets from fields
        off = {}
        for f in msg.fields:
            if f.name in ('x', 'y', 'z'):
                off[f.name] = f.offset
        if len(off) < 3:
            return None
        # Build a structured dtype that picks x/y/z out of each point_step block
        dt = np.dtype({'names': ['x', 'y', 'z'],
                       'formats': ['<f4', '<f4', '<f4'],
                       'offsets': [off['x'], off['y'], off['z']],
                       'itemsize': step})
        raw = np.frombuffer(data, dtype=dt, count=n)
        # Use float64 so numpy scalars are accepted by ROS 2 msg setters
        pts = np.column_stack([raw['x'], raw['y'], raw['z']]).astype(np.float64)
        # Remove NaN rows
        valid = np.isfinite(pts).all(axis=1)
        return pts[valid] if not valid.all() else pts

    @staticmethod
    def _fast_pub_xyz(header, pts, publisher):
        """Publish xyz-only PointCloud2 from Nx3 numpy array — zero-copy binary."""
        n = len(pts)
        if n == 0:
            return
        buf = np.ascontiguousarray(pts[:, :3].astype(np.float32))
        m = PointCloud2()
        m.header = header
        m.height = 1
        m.width = n
        m.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        m.is_bigendian = False
        m.point_step = 12
        m.row_step = 12 * n
        m.data = buf.tobytes()
        m.is_dense = True
        publisher.publish(m)
    
    def non_ground_callback(self, msg):
        """Process non-ground points to detect obstacles (optimised for Jetson Orin)."""
        try:
            # Fast binary decode — zero Python-level per-point loops
            points = self._fast_read_xyz(msg)
            if points is None or len(points) == 0:
                return

            # Filter in-range points (vectorised)
            mask = (
                (points[:, 0] > self.min_distance_threshold) &
                (points[:, 0] < self.max_detection_dist) &
                (np.abs(points[:, 1]) < self.road_width * 0.5) &
                (points[:, 2] > 0.3) & (points[:, 2] < 2.5)
            )
            filtered_points = points[mask]
            if len(filtered_points) == 0:
                return

            # ── Voxel downsample before DBSCAN (huge speedup on Orin) ──
            vox = 0.20  # 20 cm voxel
            vox_idx = np.floor(filtered_points / vox).astype(np.int32)
            keys = vox_idx[:, 0] * 2_000_000 + vox_idx[:, 1] * 2_000 + vox_idx[:, 2]
            _, uniq_idx = np.unique(keys, return_index=True)
            ds_points = filtered_points[uniq_idx]

            # ── Cluster with DBSCAN on downsampled set ──────────────
            if len(ds_points) < 8:
                return
            db = DBSCAN(eps=0.5, min_samples=5, algorithm='ball_tree').fit(ds_points[:, :3])
            labels = db.labels_
            
            # Process each cluster
            obstacles = []
            is_path_blocked = False
            
            unique_labels = set(labels)
            unique_labels.discard(-1)
            close_obstacle_found = False
            red_zone_obstacle = None
            yellow_zone_obstacle = None
            for label in unique_labels:
                cluster = ds_points[labels == label]
                
                center = np.mean(cluster, axis=0)
                min_point = np.min(cluster, axis=0)
                max_point = np.max(cluster, axis=0)
                dimensions = max_point - min_point
                
                # Skip tiny clusters or too close to sensor
                if (dimensions[0] < 0.2 and dimensions[1] < 0.2 and dimensions[2] < 0.2) or \
                   (center[0] <= self.min_distance_threshold):
                    continue
                    
                object_class = self.classify_object(dimensions, cluster)
                
                half_path = self.path_width * 0.5
                in_path = (abs(center[1]) < half_path)
                
                angle_rad = np.arctan2(center[1], center[0])
                angle_deg = float(np.degrees(angle_rad))
                side_str = self.get_side_from_angle(angle_deg)
                
                # Determine if object is dangerous
                is_dangerous = object_class in ['person', 'vehicle', 'wall'] or \
                            (in_path and center[0] < self.danger_zone)
                # === NEW: Check for close-range obstacle (0 < x < 3m) ===
                is_close_obstacle = (center[0] > 0 and center[0] < self.danger_zone)
                # === NEW: Check for red/yellow zone obstacles ===
                in_red_zone = (center[0] > 0 and center[0] < self.danger_zone)
                in_yellow_zone = (center[0] >= self.danger_zone and center[0] < self.warning_zone)
                if in_red_zone:
                    if (red_zone_obstacle is None) or (center[0] < red_zone_obstacle['distance']):
                        red_zone_obstacle = {'distance': center[0], 'in_path': in_path}
                if in_yellow_zone:
                    if (yellow_zone_obstacle is None) or (center[0] < yellow_zone_obstacle['distance']):
                        yellow_zone_obstacle = {'distance': center[0], 'in_path': in_path}
                obstacles.append({
                    'center': center,
                    'dimensions': dimensions,
                    'min_point': min_point,
                    'max_point': max_point,
                    'points': cluster,
                    'in_path': in_path,
                    'distance': float(center[0]),
                    'object_class': object_class,
                    'is_dangerous': is_dangerous,
                    'is_close_obstacle': is_close_obstacle,
                    'angle_deg': angle_deg,
                    'side': side_str,
                })
                
                if in_path and center[0] < self.danger_zone:
                    is_path_blocked = True
            
            # Publish visualisation
            self.publish_obstacle_markers(obstacles, msg.header)
            self.publish_path_markers(msg.header, is_path_blocked)
            
            current_obstacle_count = len(obstacles)
            if current_obstacle_count == 0 and self.has_published_obstacles:
                self.publish_obstacle_markers([], msg.header)
                
            self.last_obstacle_count = current_obstacle_count
            if current_obstacle_count > 0:
                self.has_published_obstacles = True
            
            # Publish obstacle point cloud (zero-copy binary)
            if len(obstacles) > 0:
                all_obstacle_points = np.vstack([obs['points'] for obs in obstacles])
                self._fast_pub_xyz(msg.header, all_obstacle_points, self.obstacle_pub)

            # Publish zone status
            red_zone_status = bool(red_zone_obstacle is not None and red_zone_obstacle['in_path'])
            yellow_zone_status = bool(yellow_zone_obstacle is not None and yellow_zone_obstacle['in_path'])
            red_zone_dist = float(red_zone_obstacle['distance']) if red_zone_obstacle else -1.0
            yellow_zone_dist = float(yellow_zone_obstacle['distance']) if yellow_zone_obstacle else -1.0

            zone_msg = ZoneObstacleStatus()
            zone_msg.red_zone = red_zone_status
            zone_msg.red_zone_distance = red_zone_dist
            zone_msg.yellow_zone = yellow_zone_status
            zone_msg.yellow_zone_distance = yellow_zone_dist
            closest_obs = None
            min_dist = float('inf')
            for obs in obstacles:
                if obs['in_path'] and obs['distance'] < min_dist:
                    closest_obs = obs
                    min_dist = obs['distance']
            if closest_obs is not None and closest_obs['distance'] < self.warning_zone:
                zone_msg.obstacle_angle_deg = float(closest_obs['angle_deg'])
                zone_msg.obstacle_side = str(closest_obs['side'])
            else:
                zone_msg.obstacle_angle_deg = float('nan')
                zone_msg.obstacle_side = ''
            self.zone_status_pub.publish(zone_msg)

        except Exception as e:
            self.get_logger().error(f"Error in non_ground_callback: {e}")

    def classify_object(self, dimensions, cluster):
        """Enhanced object classification based on dimensions, point distribution and density"""
        width = dimensions[1]   # Y dimension
        height = dimensions[2]  # Z dimension
        depth = dimensions[0]   # X dimension
        points = len(cluster)
        
        # Calculate volume and point density
        volume = width * height * depth
        density = points / volume if volume > 0 else 0
        
        # Calculate aspect ratios
        hw_ratio = height / width if width > 0 else 0  # height-to-width ratio
        hd_ratio = height / depth if depth > 0 else 0  # height-to-depth ratio
        wd_ratio = width / depth if depth > 0 else 0   # width-to-depth ratio
        
        # Calculate standard deviation of points in each dimension
        if points > 5:
            std_x = np.std(cluster[:, 0])
            std_y = np.std(cluster[:, 1])
            std_z = np.std(cluster[:, 2])
        else:
            std_x, std_y, std_z = 0, 0, 0
        
        # Person detection - tall, narrow, relatively uniform point distribution vertically
        if (height > 1.2 and height < 2.0 and  # Height range for typical people
            max(width, depth) < 0.9 and        # Not too wide/deep
            min(width, depth) > 0.2 and        # Not too thin
            points > 20 and                    # Enough points
            hw_ratio > 2.0):                   # Much taller than wide
            return 'person'
            
        # Vehicle detection - balanced dimensions, medium height, high point count
        elif ((width > 1.0 or depth > 1.5) and      # Wide or long but not excessive
              height > 0.5 and height < 2.2 and     # Height range for vehicles
              points > 50 and                       # Substantial point count
              (width * depth) > 2.0 and             # Large footprint area
              min(wd_ratio, 1/wd_ratio) > 0.3 and   # More balanced width/depth ratio than walls
              (std_x > 0.15 and std_y > 0.15)):     # Not too uniform in any dimension (unlike walls)
            return 'vehicle'
            
        # Wall/barrier detection - large in one or two dimensions, flat in one dimension
        elif ((width > 1.5 or depth > 1.5) and      # Large in at least one dimension
              height > 0.5 and                      # Tall enough to be a barrier
              (std_z < 0.2 or                       # Flat vertically OR
               std_x < 0.2 or std_y < 0.2) and      # Flat in another dimension
              min(wd_ratio, 1/wd_ratio) < 0.5):     # Significantly longer in one horizontal dimension
            return 'wall'
            
        # Ground objects - low height, significant width/depth
        elif (height < 0.6 and                      # Low height
              max(width, depth) > 0.3 and           # Large enough footprint
              points > 10):                         # Enough points to be real
            return 'ground_object'
            
        # Default classification
        else:
            # Try to make a best guess for smaller clusters
            if points < 20 and height < 0.7:
                return 'ground_object'
            elif height > 1.3 and max(width, depth) < 1.0:
                return 'person'
            else:
                return 'unknown'
    
    def publish_obstacle_markers(self, obstacles, header):
        """Publish markers for detected obstacles with labels and distance info"""
        marker_array = MarkerArray()
        
        # Delete all previous markers first by creating DELETEALL markers for each namespace
        for namespace in self.all_marker_namespaces:
            delete_marker = Marker()
            delete_marker.header = header
            delete_marker.ns = namespace
            delete_marker.id = 0
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)
        
        # Publish deletion markers first to ensure clean slate
        self.markers_pub.publish(marker_array)
        
        # Create new markers
        marker_array = MarkerArray()
        
        for i, obs in enumerate(obstacles):
            # Skip visualization if too far away (reduce clutter)
            if obs['distance'] > self.max_detection_dist:
                continue
                
            # Determine if this object should be highlighted
            is_critical = False
            if 'object_class' in obs and obs['object_class'] in ['person', 'vehicle', 'wall']:
                is_critical = True
            if obs['in_path'] and obs['distance'] < self.danger_zone:
                is_critical = True
            
            # Create visually appealing marker based on object type
            marker = Marker()
            marker.header = header
            marker.ns = self.markers_ns
            marker.id = i
            
            # Position on ground — cast to Python float for ROS 2 msg setters
            x_pos = float(obs['center'][0])
            y_pos = float(obs['center'][1])
            z_bottom = 0.05  # Slightly above ground to prevent z-fighting
            
            # Get dimensions with minimum sizes for visibility
            width = float(max(obs['dimensions'][1], 0.4))
            depth = float(max(obs['dimensions'][0], 0.4))
            height = float(max(obs['dimensions'][2], 0.5))
            
            # Set different marker colors based on classification
            # All obstacles now use CUBE markers (bounding boxes) with actual dimensions
            obj_class = obs.get('object_class', 'unknown')
            
            # Choose marker color based on object classification
            if obj_class == 'person':
                # Use box for people (instead of cylinder)
                marker.type = Marker.CUBE
                marker.pose.position.x = x_pos
                marker.pose.position.y = y_pos
                marker.pose.position.z = height/2 + z_bottom
                marker.scale.x = max(depth, 0.4)  # Ensure minimum width for visibility
                marker.scale.y = max(width, 0.4)  # Ensure minimum width for visibility
                marker.scale.z = height
                
                # Color from helper method
                marker.color = self.get_object_color('person', alpha=0.9)
                
            elif obj_class == 'vehicle':
                # Use more detailed shape for vehicles
                marker.type = Marker.CUBE
                marker.pose.position.x = x_pos
                marker.pose.position.y = y_pos
                marker.pose.position.z = height/2 + z_bottom
                marker.scale.x = depth
                marker.scale.y = width
                marker.scale.z = height
                
                # Color from helper method
                marker.color = self.get_object_color('vehicle', alpha=0.8)
                
            elif obj_class == 'wall':
                # Use flat shape for walls and barriers
                marker.type = Marker.CUBE
                marker.pose.position.x = x_pos
                marker.pose.position.y = y_pos
                marker.pose.position.z = height/2 + z_bottom
                marker.scale.x = depth
                marker.scale.y = width
                marker.scale.z = height
                
                # Color from helper method
                marker.color = self.get_object_color('wall', alpha=0.8)
                
            elif obj_class == 'ground_object':
                # Use cube for small ground objects (instead of sphere)
                marker.type = Marker.CUBE
                marker.pose.position.x = x_pos
                marker.pose.position.y = y_pos
                marker.pose.position.z = height/2 + z_bottom
                marker.scale.x = depth
                marker.scale.y = width
                marker.scale.z = height
                
                # Color from helper method
                marker.color = self.get_object_color('ground_object', alpha=0.7)
                
            else:
                # Default: use cube with actual dimensions (instead of cylinder)
                marker.type = Marker.CUBE
                marker.pose.position.x = x_pos
                marker.pose.position.y = y_pos
                marker.pose.position.z = height/2 + z_bottom
                marker.scale.x = depth
                marker.scale.y = width
                marker.scale.z = height
                
                # Color from helper method
                marker.color = self.get_object_color('unknown', is_critical, obs['in_path'], 
                                                   obs['distance'], alpha=0.7)
            
            # Common properties
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            
            marker_array.markers.append(marker)
            
            # For important objects, add text label with classification and distance
            if is_critical or (obs['in_path'] and obs['distance'] < self.warning_zone):
                # First add a white background text for better visibility
                bg_text = Marker()
                bg_text.header = header
                bg_text.ns = self.markers_ns + "_text_bg"
                bg_text.id = i
                bg_text.type = Marker.TEXT_VIEW_FACING
                bg_text.action = Marker.ADD
                
                bg_text.pose.position.x = x_pos
                bg_text.pose.position.y = y_pos
                bg_text.pose.position.z = float(obs['dimensions'][2]) + 0.3  # Position text above the obstacle
                
                # Prepare enhanced label text with more information
                obj_class = obs.get('object_class', 'object')
                distance = obs['distance']
                
                # Make class names more presentable
                class_display = {
                    'person': '👤 PERSON',
                    'vehicle': '🚗 VEHICLE',
                    'wall': '🧱 WALL',
                    'ground_object': 'OBJECT',
                    'unknown': 'OBSTACLE'
                }.get(obj_class, obj_class.upper())
                
                # Add side info for important obstacles
                if is_critical and 'side' in obs:
                    side_emoji = {
                        'left': '⬅️',  # Left arrow for objects on the left
                        'right': '➡️',  # Right arrow for objects on the right
                        'center': '⬆️'  # Up arrow for objects ahead
                    }.get(obs['side'], '')
                    label = f"{side_emoji} {class_display} ({distance:.1f}m)"
                else:
                    label = f"{class_display} ({distance:.1f}m)"
                
                # Create glow effect with multiple layers for better visibility in demos
                # First add black outline (largest)
                bg_text = Marker()
                bg_text.header = header
                bg_text.ns = self.markers_ns + "_text_bg"
                bg_text.id = i
                bg_text.type = Marker.TEXT_VIEW_FACING
                bg_text.action = Marker.ADD
                
                # Position text higher above object for better visibility
                bg_text.pose.position.x = x_pos + 0.04
                bg_text.pose.position.y = y_pos + 0.04
                bg_text.pose.position.z = float(obs['dimensions'][2]) + 0.4  # Higher above obstacle
                
                bg_text.text = label
                bg_text.scale.z = 0.44  # Slightly larger for background
                bg_text.color = _rgba(0.0, 0.0, 0.0, 0.7)  # Black
                
                marker_array.markers.append(bg_text)
                
                # Add the main text (in front)
                text_marker = Marker()
                text_marker.header = header
                text_marker.ns = self.markers_ns + "_text"
                text_marker.id = i
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                
                text_marker.pose.position.x = x_pos
                text_marker.pose.position.y = y_pos
                # Position text higher above object for better visibility
                text_marker.pose.position.z = float(obs['dimensions'][2]) + 0.4  # Higher above obstacle
                
                text_marker.text = label
                text_marker.scale.z = 0.42  # Text height
                
                # Set text color based on object class using helper method (always full opacity for text)
                text_marker.color = self.get_object_color(obj_class, is_critical=is_critical, 
                                                        in_path=obs['in_path'], 
                                                        distance=obs['distance'], 
                                                        alpha=1.0)  # Full opacity for text
                
                marker_array.markers.append(text_marker)
            
            # Add wireframe bounding box for important obstacles (better visualization)
            if is_critical or (obj_class in ['person', 'vehicle']):
                # Create wireframe box outline
                outline = Marker()
                outline.header = header
                outline.ns = self.markers_ns + "_outline"
                outline.id = i
                outline.type = Marker.LINE_LIST
                outline.action = Marker.ADD
                
                # Set position and orientation (same as the main marker)
                outline.pose.position.x = x_pos
                outline.pose.position.y = y_pos
                outline.pose.position.z = height/2 + z_bottom
                outline.pose.orientation.w = 1.0
                
                # Line thickness
                outline.scale.x = 0.03  # Line width
                
                # Set color based on object type using helper method (always full opacity for outlines)
                outline.color = self.get_object_color(obj_class, is_critical=is_critical, 
                                                    in_path=obs['in_path'], 
                                                    distance=obs['distance'], 
                                                    alpha=1.0)  # Full opacity for outlines
                
                # Calculate half dimensions
                hx = depth/2
                hy = width/2
                hz = height/2
                
                # Create points for the 12 lines of a 3D box
                pts = []
                # Bottom rectangle
                pts.extend([_point(-hx, -hy, -hz), _point(hx, -hy, -hz)])
                pts.extend([_point(hx, -hy, -hz), _point(hx, hy, -hz)])
                pts.extend([_point(hx, hy, -hz), _point(-hx, hy, -hz)])
                pts.extend([_point(-hx, hy, -hz), _point(-hx, -hy, -hz)])
                # Top rectangle
                pts.extend([_point(-hx, -hy, hz), _point(hx, -hy, hz)])
                pts.extend([_point(hx, -hy, hz), _point(hx, hy, hz)])
                pts.extend([_point(hx, hy, hz), _point(-hx, hy, hz)])
                pts.extend([_point(-hx, hy, hz), _point(-hx, -hy, hz)])
                # Connecting verticals
                pts.extend([_point(-hx, -hy, -hz), _point(-hx, -hy, hz)])
                pts.extend([_point(hx, -hy, -hz), _point(hx, -hy, hz)])
                pts.extend([_point(hx, hy, -hz), _point(hx, hy, hz)])
                pts.extend([_point(-hx, hy, -hz), _point(-hx, hy, hz)])
                
                outline.points = pts
                marker_array.markers.append(outline)
        
        # Publish marker array
        self.markers_pub.publish(marker_array)

    def publish_vehicle_box(self, header):
        """Publish a lightweight golf-cart representation optimised for Jetson Orin.

        Uses only CUBE and LINE_LIST markers (no CYLINDER/SPHERE rotations that
        cause rendering issues on Orin's RViz2).  The MarkerArray is built once
        and cached; subsequent calls only update the timestamp.
        """
        if self._vehicle_markers_cache is not None:
            # Fast path — just update stamp and re-publish
            for m in self._vehicle_markers_cache.markers:
                m.header.stamp = header.stamp
            self.vehicle_marker_pub.publish(self._vehicle_markers_cache)
            return

        # ── Build the array once ────────────────────────────────────
        marker_array = MarkerArray()

        vehicle_header = Header()
        vehicle_header.stamp = header.stamp
        vehicle_header.frame_id = "base_link"

        mid = 0

        L = self.vehicle_length   # ~3.2 m
        W = self.vehicle_width    # ~1.73 m
        H = self.vehicle_height   # 1.8 m
        half_l = L / 2.0
        half_w = W / 2.0

        def _cube(x, y, z, sx, sy, sz, r, g, b, a):
            nonlocal mid
            m = Marker()
            m.header = vehicle_header
            m.ns = "vehicle"
            m.id = mid; mid += 1
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position = _point(x, y, z)
            m.pose.orientation.w = 1.0
            m.scale.x = float(sx)
            m.scale.y = float(sy)
            m.scale.z = float(sz)
            m.color = _rgba(r, g, b, a)
            marker_array.markers.append(m)

        # 1. Chassis / floor pan
        _cube(0, 0, 0.15,   L, W, 0.30,   0.18, 0.20, 0.22, 0.95)

        # 2. Front hood
        hood_l = L * 0.22
        _cube(half_l - hood_l/2, 0, 0.425,  hood_l, W*0.90, 0.25,  0.85, 0.88, 0.90, 0.85)

        # 3. Rear body
        rear_l = L * 0.15
        _cube(-half_l + rear_l/2, 0, 0.425,  rear_l, W*0.90, 0.25,  0.85, 0.88, 0.90, 0.85)

        # 4. Canopy / roof
        roof_l = L * 0.52
        _cube(-0.05, 0, H - 0.04,  roof_l, W*0.95, 0.08,  0.20, 0.22, 0.25, 0.55)

        # 5. Canopy pillars (4 thin cubes — avoids CYLINDER rotation issues)
        px_front = -0.05 + roof_l/2 - 0.05
        px_rear  = -0.05 - roof_l/2 + 0.05
        pillar_h = H - 0.55
        for px in [px_front, px_rear]:
            for py in [half_w * 0.85, -half_w * 0.85]:
                _cube(px, py, 0.55 + pillar_h/2,
                      0.06, 0.06, pillar_h,  0.30, 0.32, 0.35, 0.95)

        # 6. Front seat
        _cube(0.25, 0, 0.65,  0.45, W*0.70, 0.30,  0.15, 0.15, 0.15, 0.90)
        # Front backrest
        _cube(0.0, 0, 0.92,  0.08, W*0.70, 0.45,   0.15, 0.15, 0.15, 0.90)

        # 7. Rear seat
        _cube(-0.45, 0, 0.65,  0.45, W*0.70, 0.30,  0.15, 0.15, 0.15, 0.90)
        # Rear backrest
        _cube(-0.70, 0, 0.92,  0.08, W*0.70, 0.45,  0.15, 0.15, 0.15, 0.90)

        # 8. Wheels (cubes instead of rotated cylinders — Orin-safe)
        for wx, wy in [(half_l-0.35, half_w+0.06), (half_l-0.35, -half_w-0.06),
                        (-half_l+0.35, half_w+0.06), (-half_l+0.35, -half_w-0.06)]:
            _cube(wx, wy, 0.25,  0.25, 0.12, 0.50,  0.12, 0.12, 0.12, 1.0)

        # 9. Headlights (small bright cubes)
        for sy in [half_w*0.55, -half_w*0.55]:
            _cube(half_l-0.02, sy, 0.50,  0.06, 0.14, 0.10,  1.0, 0.98, 0.80, 0.95)

        # 10. Taillights (small red cubes)
        for sy in [half_w*0.55, -half_w*0.55]:
            _cube(-half_l+0.02, sy, 0.50,  0.06, 0.14, 0.10,  0.9, 0.05, 0.05, 0.95)

        # 11. LiDAR puck (flat cube)
        _cube(1.73, 0, 1.0,  0.22, 0.22, 0.10,  0.85, 0.10, 0.10, 1.0)

        # 12. Direction arrow (flat elongated cube on ground)
        _cube(0.0, 0, 0.08,  L*0.60, 0.12, 0.04,  1.0, 0.55, 0.0, 0.75)

        # 13. Label
        lbl = Marker()
        lbl.header = vehicle_header
        lbl.ns = "vehicle"
        lbl.id = mid; mid += 1
        lbl.type = Marker.TEXT_VIEW_FACING
        lbl.action = Marker.ADD
        lbl.pose.position = _point(0, 0, H + 0.35)
        lbl.pose.orientation.w = 1.0
        lbl.scale.z = 0.25
        lbl.color = _rgba(1.0, 1.0, 1.0, 0.85)
        lbl.text = "GOLF CART"
        marker_array.markers.append(lbl)

        # 14. Wireframe outline (single LINE_LIST — very cheap)
        outline = Marker()
        outline.header = vehicle_header
        outline.ns = "vehicle"
        outline.id = mid; mid += 1
        outline.type = Marker.LINE_LIST
        outline.action = Marker.ADD
        outline.pose.orientation.w = 1.0
        outline.scale.x = 0.03
        outline.color = _rgba(0.5, 0.5, 0.5, 0.35)
        l, w, h = half_l, half_w, H
        pts = []
        for z0, z1 in [(0, 0), (h, h)]:
            pts += [_point(-l,-w,z0), _point(l,-w,z1),
                    _point(l,-w,z0), _point(l,w,z1),
                    _point(l,w,z0), _point(-l,w,z1),
                    _point(-l,w,z0), _point(-l,-w,z1)]
        for cx, cy in [(-l,-w),(l,-w),(l,w),(-l,w)]:
            pts += [_point(cx,cy,0), _point(cx,cy,h)]
        outline.points = pts
        marker_array.markers.append(outline)

        # ── Cache and publish ───────────────────────────────────────
        self._vehicle_markers_cache = marker_array
        self.vehicle_marker_pub.publish(marker_array)
    
    def publish_path_markers(self, header, is_blocked):
        """Publish markers showing the vehicle's projected path and safety zones"""
        marker_array = MarkerArray()
        
        # Delete all previous path markers first
        delete_path = Marker()
        delete_path.header = header
        delete_path.ns = "path"
        delete_path.action = Marker.DELETEALL
        marker_array.markers.append(delete_path)
        
        # Delete all previous safety zone markers
        delete_zones = Marker()
        delete_zones.header = header
        delete_zones.ns = "safety_zones"
        delete_zones.action = Marker.DELETEALL
        marker_array.markers.append(delete_zones)
        
        # Delete all previous path text markers
        delete_path_text = Marker()
        delete_path_text.header = header
        delete_path_text.ns = "path_text"
        delete_path_text.action = Marker.DELETEALL
        marker_array.markers.append(delete_path_text)
        
        # Publish deletion markers first to ensure clean slate
        self.path_marker_pub.publish(marker_array)
        
        # Create new marker array
        marker_array = MarkerArray()
        
        # Path polygon
        path = Marker()
        path.header = header
        path.ns = "path"
        path.id = 0
        path.type = Marker.LINE_STRIP
        path.action = Marker.ADD
        path.pose.orientation.w = 1.0
        path.scale.x = 0.1  # Line width
        
        # Path color depends on blockage
        if is_blocked:
            path.color = _rgba(1.0, 0.0, 0.0, 0.8)  # Red if blocked
        else:
            path.color = _rgba(0.0, 1.0, 0.0, 0.8)  # Green if clear
        
        # Define path boundaries
        half_width = self.path_width / 2
        
        # Add visual indicators for left/right sides
        # Add a "LEFT" text marker on the left side
        left_text = Marker()
        left_text.header = header
        left_text.ns = "path_text"
        left_text.id = 0
        left_text.type = Marker.TEXT_VIEW_FACING
        left_text.action = Marker.ADD
        left_text.pose.position.x = 2.0  # 2m ahead
        left_text.pose.position.y = half_width * 0.7  # Left side
        left_text.pose.position.z = 0.5
        left_text.pose.orientation.w = 1.0
        left_text.text = "LEFT"
        left_text.scale.z = 0.4  # Text height
        left_text.color = _rgba(0.2, 0.8, 0.2, 0.8)  # Green
        marker_array.markers.append(left_text)
        
        # Add a "RIGHT" text marker on the right side
        right_text = Marker()
        right_text.header = header
        right_text.ns = "path_text"
        right_text.id = 1
        right_text.type = Marker.TEXT_VIEW_FACING
        right_text.action = Marker.ADD
        right_text.pose.position.x = 2.0  # 2m ahead
        right_text.pose.position.y = -half_width * 0.7  # Right side
        right_text.pose.position.z = 0.5
        right_text.pose.orientation.w = 1.0
        right_text.text = "RIGHT"
        right_text.scale.z = 0.4  # Text height
        right_text.color = _rgba(0.2, 0.8, 0.2, 0.8)  # Green
        marker_array.markers.append(right_text)
        
        # Create a box extending from LiDAR forward to max detection distance
        path.points = [
            _point(0, -half_width, 0.1),
            _point(self.max_detection_dist, -half_width, 0.1),
            _point(self.max_detection_dist, half_width, 0.1),
            _point(0, half_width, 0.1),
            _point(0, -half_width, 0.1),
        ]
        marker_array.markers.append(path)

        # === NEW: Add red and yellow zone rectangles in front of vehicle ===
        # Red zone (danger)
        red_zone = Marker()
        red_zone.header = header
        red_zone.ns = "safety_zones"
        red_zone.id = 100
        red_zone.type = Marker.CUBE
        red_zone.action = Marker.ADD
        red_zone.pose.position.x = self.danger_zone / 2.0
        red_zone.pose.position.y = 0.0
        red_zone.pose.position.z = 0.05
        red_zone.scale.x = self.danger_zone
        red_zone.scale.y = self.path_width
        red_zone.scale.z = 0.1
        red_zone.color = _rgba(1.0, 0.0, 0.0, 0.25)  # Red, semi-transparent
        marker_array.markers.append(red_zone)
        # Yellow zone (warning)
        yellow_zone = Marker()
        yellow_zone.header = header
        yellow_zone.ns = "safety_zones"
        yellow_zone.id = 101
        yellow_zone.type = Marker.CUBE
        yellow_zone.action = Marker.ADD
        yellow_zone.pose.position.x = (self.danger_zone + self.warning_zone) / 2.0
        yellow_zone.pose.position.y = 0.0
        yellow_zone.pose.position.z = 0.05
        yellow_zone.scale.x = self.warning_zone - self.danger_zone
        yellow_zone.scale.y = self.path_width
        yellow_zone.scale.z = 0.1
        yellow_zone.color = _rgba(1.0, 1.0, 0.0, 0.18)  # Yellow, more transparent
        marker_array.markers.append(yellow_zone)

        # Add zone markers (red, yellow, green)
        for i, (dist, color) in enumerate([
                (self.danger_zone, _rgba(1.0, 0.0, 0.0, 0.5)),  # Red zone
                (self.warning_zone, _rgba(1.0, 1.0, 0.0, 0.5)),  # Yellow zone
                (self.safe_zone, _rgba(0.0, 1.0, 0.0, 0.5))      # Green zone
            ]):
            
            zone = Marker()
            zone.header = header
            zone.ns = "safety_zones"
            zone.id = i
            zone.type = Marker.LINE_STRIP
            zone.action = Marker.ADD
            zone.pose.orientation.w = 1.0
            zone.scale.x = 0.1  # Line width
            zone.color = color
            
            # Create arc at the zone distance
            zone.points = [
                _point(dist, -half_width, 0.15),
                _point(dist, half_width, 0.15)
            ]
            
            marker_array.markers.append(zone)
            
            # Add text marker for the zone
            text = Marker()
            text.header = header
            text.ns = "safety_zones_text"
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = dist
            text.pose.position.y = 0.0
            text.pose.position.z = 0.5
            text.pose.orientation.w = 1.0
            text.text = f"{dist}m"
            text.scale.z = 0.5  # Text height
            text.color = _rgba(1.0, 1.0, 1.0, 0.8)  # White
            
            marker_array.markers.append(text)
        
        self.path_marker_pub.publish(marker_array)

    def get_side_from_angle(self, angle_deg):
        """Determine which side an object is on based on its angle
        
        Args:
            angle_deg: Angle in degrees from forward (x-axis)
            
        Returns:
            str: 'left', 'right', or 'center'
        """
        # Using standard ROS coordinate system:
        # - Positive angles (positive Y) = left side
        # - Negative angles (negative Y) = right side
        # - Near-zero angles = center/front
        if angle_deg > 5:
            return 'left'
        elif angle_deg < -5:
            return 'right'
        else:
            return 'center'
            
    def update_parameters(self):
        """Update parameters from ROS 2 parameter server (for dynamic reconfigure support)"""
        # Get current parameters
        new_min_distance = self.get_parameter('min_distance_threshold').value
        
        # Check if parameters have changed
        if new_min_distance != self.min_distance_threshold:
            self.min_distance_threshold = new_min_distance
            self.get_logger().info(f"Updated minimum distance threshold: {self.min_distance_threshold}m")
            
    def get_object_color(self, obj_class, is_critical=False, in_path=False, distance=float('inf'), alpha=0.8):
        """Get consistent color for an object based on its class and properties
        
        Args:
            obj_class: Classification of the object ('person', 'vehicle', etc)
            is_critical: Whether the object is critical (important to highlight)
            in_path: Whether the object is in the vehicle's path
            distance: Distance to the object (for color grading by distance)
            alpha: Transparency value (0.0-1.0)
            
        Returns:
            ColorRGBA: A color for visualization
        """
        # Set color based on object classification and importance
        if obj_class == 'person':
            # Bright red for people - highest priority
            return _rgba(1.0, 0.0, 0.0, alpha)
            
        elif obj_class == 'vehicle':
            # Blue for vehicles
            return _rgba(0.0, 0.3, 1.0, alpha)
            
        elif obj_class == 'wall':
            # Gray for walls
            return _rgba(0.7, 0.7, 0.7, alpha)
            
        elif obj_class == 'ground_object':
            # Orange for ground objects
            return _rgba(1.0, 0.5, 0.0, alpha)
            
        else:
            # Default colors based on criticality and distance
            if is_critical:
                # Purple for critical unknown objects
                return _rgba(0.8, 0.0, 0.8, alpha)
            elif in_path:
                if distance < self.warning_zone:
                    # Yellow for objects in warning zone
                    return _rgba(1.0, 1.0, 0.0, alpha)
                else:
                    # Green for objects in path but at safe distance
                    return _rgba(0.0, 1.0, 0.0, alpha)
            else:
                # Muted blue for objects outside path
                return _rgba(0.0, 0.5, 1.0, alpha * 0.6)  # More transparent

    def publish_vehicle_timer_callback(self):
        """Timer callback to continuously publish vehicle visualization markers"""
        # Create a header for the current time
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "base_link"  # Vehicle markers are in base_link frame
        
        # Publish vehicle visualization markers
        self.publish_vehicle_box(header)

def main(args=None):
    rclpy.init(args=args)
    detector = ObstacleDetector()
    
    # Set up a timer to periodically check for parameter updates (1Hz)
    update_timer = detector.create_timer(1.0, detector.update_parameters)
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()