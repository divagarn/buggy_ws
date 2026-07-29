#!/usr/bin/env python3
"""
Main ROS2 Node for Driver Monitoring System using MediaPipe (No Calibration)
"""

import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String

from dms_adas_mediapipe.scripts.utils.drowsiness_detector import DrowsinessDetector
from dms_adas_mediapipe.scripts.utils.head_pose_estimator import HeadPoseEstimator
from dms_adas_mediapipe.scripts.utils.iris_gaze_detector import IrisGazeDetector
from dms_adas_mediapipe.scripts.utils.mediapipe_processor import MediaPipeProcessor
from dms_adas_mediapipe.scripts.utils.phone_detector import PhoneDetector


class DMSNode(Node):
    def __init__(self):
        super().__init__('dms_node')
        self.get_logger().info('Initializing DMS with MediaPipe (No Calibration)...')
        self.load_parameters()

        self.bridge = CvBridge()
        
        self.mediapipe_processor = MediaPipeProcessor(
            static_image_mode=False,
            max_num_faces=self.max_num_faces,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # Store MediaPipe drawing utilities for enhanced visualization
        import mediapipe as mp
        self.mp_drawing = mp.solutions.drawing_utils

        self.drowsiness_detector = DrowsinessDetector(
            ear_threshold=self.ear_threshold,
            eye_closed_frames=self.eye_closed_frames,
            perclos_window=self.perclos_window_frames,
            perclos_threshold=self.perclos_threshold,
            mar_threshold=self.mar_threshold,
            yawn_frames=self.yawn_frames,
            smoothing_alpha=self.smoothing_alpha,
            use_smoothed_mar=getattr(self, 'use_smoothed_mar', True)
        )

        self.phone_detector = PhoneDetector(confidence_threshold=self.phone_confidence)
        self.head_pose_estimator = HeadPoseEstimator()
        self.iris_gaze_detector = IrisGazeDetector(smoothing_alpha=0.3)

        self.phone_detected_counter = 0
        self.last_alert_time = 0
        self.phone_popup_start_time = None  # Track when to show phone popup
        
        # Multi-level drowsiness tracking
        self.eyes_closed_start_time = None
        self.face_missing_start_time = None
        self.gaze_off_road_start_time = None
        self.perclos_alert_start_time = None
        self.head_nod_detected = False
        
        # Alert level tracking
        self.current_alert_level = 0  # 0=None, 1=Mild, 2=Moderate, 3=Severe
        self.alert_level_start_time = None
        
        # Blink tracking
        self.last_blink_duration = 0
        self.long_blink_detected = False
        
        # FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0

        self.alert_pub = self.create_publisher(String, '/dms/alert', 10)
        self.alert_level_pub = self.create_publisher(Float32, '/dms/alert_level', 10)
        self.drowsy_pub = self.create_publisher(Bool, '/dms/drowsy', 10)
        self.yawn_pub = self.create_publisher(Bool, '/dms/yawning', 10)
        self.phone_pub = self.create_publisher(Bool, '/dms/phone_detected', 10)
        self.ear_pub = self.create_publisher(Float32, '/dms/eye_aspect_ratio', 10)
        self.image_pub = self.create_publisher(Image, '/dms/visualization', 10)
        self.face_detected_pub = self.create_publisher(Bool, '/dms/face_detected', 10)

        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            1,
        )

        self.mar_pub = self.create_publisher(Float32, '/dms/mouth_aspect_ratio', 10)
        self.yawn_counter_pub = self.create_publisher(Float32, '/dms/yawn_counter', 10)

        # Frame counter used for periodic ROS2 parameter reload to allow tuning without restart
        self._frame_count = 0
        # check params every ~300 frames (approx 10s @30fps)
        self._param_reload_interval = getattr(self, 'param_reload_interval_frames', 300)

        self.get_logger().info(f'DMS Node initialized and subscribed to {self.camera_topic}')

    def load_parameters(self):
        defaults = {
            'camera_topic': '/camera/image_raw',
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.5,
            'max_num_faces': 1,
            'ear_threshold': 0.25,
            'eye_closed_frames': 30,
            'perclos_window_frames': 60,
            'perclos_threshold': 0.2,
            'mar_threshold': 0.3,
            'yawn_frames': 12,
            'phone_confidence': 0.4,
            'head_yaw_threshold': 25.0,
            'head_pitch_threshold': 20.0,
            'head_roll_threshold': 15.0,
            'enable_audio_alert': True,
            'alert_cooldown': 3.0,
            'show_visualization': True,
            'display_fps': True,
            'draw_landmarks': True,
            'smoothing_alpha': 0.35,
            'use_smoothed_mar': True,
            'param_reload_interval_frames': 300,
            'camera_yaw_offset': 45.0,
        }

        for key, value in defaults.items():
            self.declare_parameter(key, value)

        self.camera_topic = self.get_parameter('camera_topic').value
        self.min_detection_confidence = self.get_parameter('min_detection_confidence').value
        self.min_tracking_confidence = self.get_parameter('min_tracking_confidence').value
        self.max_num_faces = self.get_parameter('max_num_faces').value
        self.ear_threshold = self.get_parameter('ear_threshold').value
        self.eye_closed_frames = self.get_parameter('eye_closed_frames').value
        self.perclos_window_frames = self.get_parameter('perclos_window_frames').value
        self.perclos_threshold = self.get_parameter('perclos_threshold').value
        self.mar_threshold = self.get_parameter('mar_threshold').value
        self.yawn_frames = self.get_parameter('yawn_frames').value
        self.phone_confidence = self.get_parameter('phone_confidence').value
        self.head_yaw_threshold = self.get_parameter('head_yaw_threshold').value
        self.head_pitch_threshold = self.get_parameter('head_pitch_threshold').value
        self.head_roll_threshold = self.get_parameter('head_roll_threshold').value
        self.enable_audio_alert = self.get_parameter('enable_audio_alert').value
        self.alert_cooldown = self.get_parameter('alert_cooldown').value
        self.show_visualization = self.get_parameter('show_visualization').value
        self.display_fps = self.get_parameter('display_fps').value
        self.draw_landmarks = self.get_parameter('draw_landmarks').value
        self.smoothing_alpha = self.get_parameter('smoothing_alpha').value
        self.use_smoothed_mar = self.get_parameter('use_smoothed_mar').value
        self.param_reload_interval_frames = self.get_parameter('param_reload_interval_frames').value
        self.camera_yaw_offset = self.get_parameter('camera_yaw_offset').value

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return
        self.process_frame(frame)

    def process_frame(self, frame):
        start_time = time.time()
        
        # Calculate FPS
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed > 1.0:  # Update FPS every second
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = time.time()

        # Periodically reload params so thresholds can be tuned via ROS2 parameters without restarting
        self._frame_count += 1
        if self._frame_count >= self._param_reload_interval:
            try:
                self.reload_parameters()
            finally:
                self._frame_count = 0

        detection_results = self.mediapipe_processor.detect_faces_and_landmarks(frame)

        drowsy_results = {}
        pose_results = None
        gaze_results = None

        if detection_results['face_landmarks'] and detection_results['all_landmarks']:
            # Prepare dictionary landmark data for drowsiness detector
            landmarks = detection_results['all_landmarks'][0]
            mp_proc = self.mediapipe_processor
            face_landmarks_dict = {
                'left_eye': landmarks[mp_proc.LEFT_EYE],
                'right_eye': landmarks[mp_proc.RIGHT_EYE],
                'mouth': landmarks[mp_proc.MOUTH],
                'nose': landmarks[mp_proc.NOSE],
                'head_pose_points': landmarks[mp_proc.HEAD_POSE_POINTS],
                'left_iris': landmarks[mp_proc.LEFT_IRIS],
                'right_iris': landmarks[mp_proc.RIGHT_IRIS]
            }

            drowsy_results = self.drowsiness_detector.detect(face_landmarks_dict)
            
            # Debug logging for ear/perclos detection
            if drowsy_results.get('ear') is not None:
                ear_val = drowsy_results.get('ear')
                perclos = drowsy_results.get('perclos', 0.0)
                self.get_logger().debug(f'Drowsiness - EAR: {ear_val:.3f}, PERCLOS: {perclos:.1%}, Counter: {self.drowsiness_detector.eye_closed_counter}')

            pose_results = self.head_pose_estimator.estimate_pose(
                landmarks,
                frame.shape
            )
            
            # Detect gaze using iris landmarks
            gaze_results = self.iris_gaze_detector.detect_gaze(face_landmarks_dict)
        else:
            # Reset gaze smoothing when face is lost
            self.iris_gaze_detector.reset_smoothing()

        phone_results = self.phone_detector.detect(frame)

        alert_message = self.check_alerts(drowsy_results, phone_results, pose_results, gaze_results)

        if self.show_visualization:
            vis_frame = self.draw_visualization(
                frame.copy(),
                detection_results,
                drowsy_results,
                phone_results,
                pose_results,
                gaze_results
            )

            if self.display_fps:
                fps = 1.0 / max((time.time() - start_time), 1e-6)
                cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            try:
                vis_msg = self.bridge.cv2_to_imgmsg(vis_frame, "bgr8")
                self.image_pub.publish(vis_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'CV Bridge Error: {e}')

        if 'ear' in drowsy_results:
            self.ear_pub.publish(Float32(data=float(drowsy_results['ear'])))
            # publish smoothed ear/mar and yawn counter when available
            mar_val = drowsy_results.get('mar_smoothed', drowsy_results.get('mar', 0.0))
            self.mar_pub.publish(Float32(data=float(mar_val)))
            self.yawn_counter_pub.publish(Float32(data=float(drowsy_results.get('yawn_counter', 0))))

    def check_alerts(self, drowsy_results, phone_results, pose_results, gaze_results):
        """
        Multi-level drowsiness and inattention detection system
        
        Levels:
        - Level 1: Mild Drowsiness (Blink > 400ms or PERCLOS > 25% for 2s)
        - Level 2: Moderate Drowsiness (Eyes closed > 1.5s or head nod > 20°)
        - Level 3: Severe Drowsiness (Eyes closed > 2.5s or PERCLOS > 50% for 5s)
        - Inattention: Gaze off road > 5s (reduced sensitivity) - uses iris gaze if available
        - No Face: Face missing > 2s
        """
        alert_message = ""
        alert_level = 0
        now = time.time()
        
        # Check for face detection
        face_detected = drowsy_results.get('ear', None) is not None
        self.face_detected_pub.publish(Bool(data=bool(face_detected)))
        
        if not face_detected:
            if self.face_missing_start_time is None:
                self.face_missing_start_time = now
            elif (now - self.face_missing_start_time) > 2.0:  # Face missing > 2s
                alert_level = max(alert_level, 4)  # Special level for no face
                alert_message = "DRIVER NOT DETECTED"
                self.get_logger().warning('ALERT: Driver face not detected for > 2s')
        else:
            self.face_missing_start_time = None
        
        # Get current metrics
        ear_val = drowsy_results.get('ear', 1.0)
        perclos = drowsy_results.get('perclos', 0.0)
        eyes_closed = ear_val < self.ear_threshold
        
        # Track blink duration (400ms threshold for Level 1)
        if eyes_closed:
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = now
            else:
                eyes_closed_duration = now - self.eyes_closed_start_time
                
                # Level 1: Blink duration > 400ms (0.4s)
                if eyes_closed_duration > 0.4 and not self.long_blink_detected:
                    self.long_blink_detected = True
                    alert_level = max(alert_level, 1)
                    alert_message = "MILD DROWSINESS" if not alert_message else alert_message
                    self.get_logger().info('Level 1 Alert: Long blink detected (> 400ms)')
                
                # Level 2: Eyes closed > 1.5s
                elif eyes_closed_duration > 1.5:
                    alert_level = max(alert_level, 2)
                    alert_message = "MODERATE DROWSINESS"
                    self.get_logger().warning('Level 2 Alert: Eyes closed > 1.5s')
                    self.drowsy_pub.publish(Bool(data=True))
                
                # Level 3: Eyes closed > 2.5s
                if eyes_closed_duration > 2.5:
                    alert_level = max(alert_level, 3)
                    alert_message = "SEVERE DROWSINESS"
                    self.get_logger().error('Level 3 Alert: Eyes closed > 2.5s')
                    self.drowsy_pub.publish(Bool(data=True))
        else:
            # Eyes opened - reset tracking
            if self.eyes_closed_start_time is not None:
                self.last_blink_duration = now - self.eyes_closed_start_time
                self.eyes_closed_start_time = None
                self.long_blink_detected = False
            self.drowsy_pub.publish(Bool(data=False))
        
        # Level 1: PERCLOS > 25% for 2 seconds
        if perclos > 0.25:
            if self.perclos_alert_start_time is None:
                self.perclos_alert_start_time = now
            elif (now - self.perclos_alert_start_time) > 2.0:
                alert_level = max(alert_level, 1)
                if not alert_message:
                    alert_message = "MILD DROWSINESS"
                self.get_logger().info(f'Level 1 Alert: PERCLOS > 25% for 2s (current: {perclos:.1%})')
        else:
            self.perclos_alert_start_time = None
        
        # Level 3: PERCLOS > 50% for 5 seconds
        if perclos > 0.50:
            if self.perclos_alert_start_time is not None:
                perclos_duration = now - self.perclos_alert_start_time
                if perclos_duration > 5.0:
                    alert_level = max(alert_level, 3)
                    alert_message = "SEVERE DROWSINESS"
                    self.get_logger().error(f'Level 3 Alert: PERCLOS > 50% for 5s (current: {perclos:.1%})')
        
        # Level 2: Head nod detection (pitch > 20°)
        if pose_results:
            pitch = pose_results.get('pitch', 0)
            yaw = pose_results.get('yaw', 0)
            
            # Head nod (looking down)
            if pitch < -20:  # Negative pitch = looking down
                if not self.head_nod_detected:
                    self.head_nod_detected = True
                    alert_level = max(alert_level, 2)
                    if not alert_message:
                        alert_message = "MODERATE DROWSINESS"
                    self.get_logger().warning(f'Level 2 Alert: Head nod detected (pitch: {pitch:.1f}°)')
            else:
                self.head_nod_detected = False
            
            # Inattention: Gaze off road detection
            # Prioritize iris-based gaze detection if available, otherwise use head pose
            gaze_off_road = False
            
            if gaze_results and gaze_results.get('gaze_detected', False):
                # Use iris-based gaze detection (more accurate for eye direction)
                looking_forward = gaze_results.get('looking_forward', True)
                gaze_off_road = not looking_forward
                
                if gaze_off_road:
                    h_dir = gaze_results.get('horizontal_direction', 'CENTER')
                    v_dir = gaze_results.get('vertical_direction', 'CENTER')
                    self.get_logger().debug(f'Iris gaze off road: H={h_dir}, V={v_dir}')
            else:
                # Fallback to head pose-based detection (compensate for camera angle)
                # Adjust yaw for camera offset: if camera is at 45° to the right,
                # driver looking forward will appear to look 45° to the left in camera view
                compensated_yaw = yaw + self.camera_yaw_offset
                
                # Normalize compensated yaw to -180 to 180 range
                if compensated_yaw > 180:
                    compensated_yaw -= 360
                elif compensated_yaw < -180:
                    compensated_yaw += 360
                
                gaze_off_road = (abs(compensated_yaw) > self.head_yaw_threshold or 
                               abs(pitch) > self.head_pitch_threshold)
            
            if gaze_off_road:
                if self.gaze_off_road_start_time is None:
                    self.gaze_off_road_start_time = now
                    # Immediate gentle warning when gaze changes away
                    alert_level = max(alert_level, 1)
                    if not alert_message:
                        alert_message = "STAY ALERT - KEEP EYE ON THE ROAD"
                elif (now - self.gaze_off_road_start_time) > 5.0:  # 5s - reduced sensitivity
                    # More serious alert after sustained gaze off road
                    alert_level = max(alert_level, 2)
                    if not alert_message or alert_level < 3:
                        alert_message = "INATTENTION DETECTED"
                    
                    # Log with appropriate gaze info
                    if gaze_results and gaze_results.get('gaze_detected', False):
                        h_dir = gaze_results.get('horizontal_direction', 'CENTER')
                        v_dir = gaze_results.get('vertical_direction', 'CENTER')
                        self.get_logger().warning(f'Inattention Alert: Iris gaze off road > 5s (H={h_dir}, V={v_dir})')
                    else:
                        compensated_yaw = yaw + self.camera_yaw_offset
                        if compensated_yaw > 180:
                            compensated_yaw -= 360
                        elif compensated_yaw < -180:
                            compensated_yaw += 360
                        self.get_logger().warning(f'Inattention Alert: Gaze off road > 5s (compensated yaw: {compensated_yaw:.1f}°, pitch: {pitch:.1f}°)')
            else:
                self.gaze_off_road_start_time = None
        
        # Yawning detection
        if drowsy_results.get('yawning', False):
            alert_level = max(alert_level, 1)
            if not alert_message:
                alert_message = "YAWNING DETECTED"
            self.yawn_pub.publish(Bool(data=True))
            self.get_logger().info('Alert: Excessive yawning detected')
        else:
            self.yawn_pub.publish(Bool(data=False))
        
        # Phone detection
        if phone_results.get('phone_detected', False):
            self.phone_detected_counter += 1
            if self.phone_detected_counter >= 10:
                alert_level = max(alert_level, 2)
                if not alert_message:
                    alert_message = "PHONE USAGE DETECTED"
                self.phone_pub.publish(Bool(data=True))
                self.get_logger().warning('Alert: Phone usage detected')
                # Trigger phone popup
                if self.phone_popup_start_time is None:
                    self.phone_popup_start_time = now
        else:
            self.phone_detected_counter = 0
            self.phone_pub.publish(Bool(data=False))
            self.phone_popup_start_time = None  # Reset popup when no phone
        
        # Publish alert level
        self.alert_level_pub.publish(Float32(data=float(alert_level)))
        
        # Update current alert level tracking
        if alert_level != self.current_alert_level:
            self.current_alert_level = alert_level
            self.alert_level_start_time = now
        
        # Publish alert message with cooldown
        if alert_message and (now - self.last_alert_time) > self.alert_cooldown:
            self.alert_pub.publish(String(data=alert_message))
            self.last_alert_time = now
        
        return alert_message

    def reload_parameters(self):
        """Reload relevant parameters from ROS2 parameter server and apply to runtime objects."""
        try:
            new_ear = self.get_parameter('ear_threshold').value
            new_mar = self.get_parameter('mar_threshold').value
            new_yawn_frames = self.get_parameter('yawn_frames').value
            new_smoothing = self.get_parameter('smoothing_alpha').value

            changed = False
            if new_ear != self.ear_threshold:
                self.get_logger().info(f'Updating ear_threshold: {self.ear_threshold} -> {new_ear}')
                self.ear_threshold = new_ear
                self.drowsiness_detector.ear_threshold = new_ear
                changed = True
            if new_mar != self.mar_threshold:
                self.get_logger().info(f'Updating mar_threshold: {self.mar_threshold} -> {new_mar}')
                self.mar_threshold = new_mar
                self.drowsiness_detector.mar_threshold = new_mar
                changed = True
            if new_yawn_frames != self.yawn_frames:
                self.get_logger().info(f'Updating yawn_frames: {self.yawn_frames} -> {new_yawn_frames}')
                self.yawn_frames = new_yawn_frames
                self.drowsiness_detector.yawn_frames = new_yawn_frames
                changed = True
            if new_smoothing != getattr(self.drowsiness_detector, 'smoothing_alpha', None):
                self.get_logger().info(
                    f"Updating smoothing_alpha: {getattr(self.drowsiness_detector, 'smoothing_alpha', None)} -> {new_smoothing}"
                )
                self.drowsiness_detector.smoothing_alpha = new_smoothing
                changed = True

            if changed:
                self.get_logger().info('Drowsiness detector parameters reloaded from ROS2 parameters.')
        except Exception as e:
            self.get_logger().debug(f'Param reload failed: {e}')

    def draw_visualization(self, frame, detection_results, drowsy_results, phone_results, pose_results, gaze_results):
        """
        Professional dashboard visualization for eBuggy DMS
        Combines driver feed, metrics panel, gaze vectors, and alert system
        """
        h, w = frame.shape[:2]
        
        # Calculate drowsiness index (0.0 - 1.0)
        drowsiness_index = self.calculate_drowsiness_index(drowsy_results, pose_results)
        
        # Determine overall driver state
        driver_state, state_color = self.get_driver_state(drowsiness_index, self.current_alert_level, gaze_results)
        
        # ═══════════════════════════════════════════════════════════
        # MAIN VISUAL LAYER: Driver Feed with Overlays
        # ═══════════════════════════════════════════════════════════
        
        face_detected = detection_results.get('face_landmarks') is not None
        
        # Face bounding box removed per user request
        # Draw complete green face mask with smooth curves
        if face_detected and detection_results.get('all_landmarks'):
            landmarks = detection_results['all_landmarks'][0]
            
            # Prepare landmarks dict for iris/gaze drawing
            mp_proc = self.mediapipe_processor
            face_landmarks_dict = {
                'left_eye': landmarks[mp_proc.LEFT_EYE],
                'right_eye': landmarks[mp_proc.RIGHT_EYE],
                'left_iris': landmarks[mp_proc.LEFT_IRIS],
                'right_iris': landmarks[mp_proc.RIGHT_IRIS]
            }
            
            # Draw unified green face mask with smooth curves
            frame = self.draw_green_face_mask(frame, landmarks, mp_proc)
            
            # Draw iris landmarks (cyan/yellow dots) - kept for gaze tracking
            frame = self.iris_gaze_detector.draw_iris_points(frame, face_landmarks_dict)
            
            # Draw iris-based gaze vector
            if gaze_results and gaze_results.get('gaze_detected', False):
                frame = self.iris_gaze_detector.draw_gaze_vector(
                    frame, face_landmarks_dict, gaze_results, arrow_length=80
                )
        
        # Draw MediaPipe mesh with enhanced visibility (unified green)
        if self.draw_landmarks and detection_results.get('mesh_results'):
            frame = self.draw_enhanced_face_mesh(frame, detection_results['mesh_results'])
        
        # ═══════════════════════════════════════════════════════════
        # TOP HEADER: Compact system branding
        # ═══════════════════════════════════════════════════════════
        
        header_height = 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_height), (15, 15, 18), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        
        cv2.putText(frame, "eBuggy DMS", (10, 20),
                   cv2.FONT_HERSHEY_DUPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
        
        sys_info = "Jetson Orin + 3 MP Cam"
        cv2.putText(frame, sys_info, (w - 205, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
        
        # ═══════════════════════════════════════════════════════════
        # METRICS PANEL: Compact right-side analytics (reduced overlap with face area)
        # ═══════════════════════════════════════════════════════════
        
        panel_width = min(300, max(260, int(w * 0.28)))
        panel_x = w - panel_width - 8
        panel_y = header_height + 8
        panel_height = min(max(260, int(h * 0.56)), h - panel_y - 52)
        
        # Dark transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (18, 18, 20), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        
        # Panel border
        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (95, 95, 100), 1)
        
        cv2.putText(frame, "ANALYTICS", (panel_x + 10, panel_y + 20),
                   cv2.FONT_HERSHEY_DUPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
        
        cv2.line(frame, (panel_x + 10, panel_y + 27),
                (panel_x + panel_width - 10, panel_y + 27), (80, 80, 84), 1)
        
        metric_y = panel_y + 42
        line_spacing = 26
        
        # Driver State
        self.draw_metric_row(frame, panel_x, metric_y, "Driver State", 
                            driver_state, state_color, is_status=True)
        metric_y += line_spacing
        
        # Drowsiness Index with gradient bar
        di_color = self.get_gradient_color(drowsiness_index)
        self.draw_metric_row(frame, panel_x, metric_y, "Drowsiness Index", 
                            f"{drowsiness_index:.2f}", di_color)
        self.draw_metric_bar(frame, panel_x + 12, metric_y + 15, 
                           panel_width - 24, drowsiness_index, di_color)
        metric_y += line_spacing + 3
        
        # PERCLOS
        perclos = drowsy_results.get('perclos', 0.0)
        perclos_color = (0, 255, 0) if perclos < 0.25 else ((0, 255, 255) if perclos < 0.5 else (0, 0, 255))
        self.draw_metric_row(frame, panel_x, metric_y, "PERCLOS", 
                            f"{perclos*100:.0f}%", perclos_color)
        metric_y += line_spacing
        
        # Yawning Status
        yawning = drowsy_results.get('yawning', False)
        yawn_text = "YES" if yawning else "NO"
        yawn_color = (0, 255, 255) if yawning else (0, 255, 0)
        self.draw_metric_row(frame, panel_x, metric_y, "Yawning", 
                            yawn_text, yawn_color)
        metric_y += line_spacing
        
        # Head Tilt (Roll)
        roll = pose_results.get('roll', 0) if pose_results else 0
        roll_color = (0, 255, 0) if abs(roll) < 15 else (0, 165, 255)
        self.draw_metric_row(frame, panel_x, metric_y, "Head Tilt", 
                            f"{roll:.0f}°", roll_color)
        metric_y += line_spacing
        
        # Head Pitch
        pitch = pose_results.get('pitch', 0) if pose_results else 0
        pitch_color = (0, 255, 0) if abs(pitch) < 20 else (0, 165, 255)
        self.draw_metric_row(frame, panel_x, metric_y, "Head Pitch", 
                            f"{pitch:.0f}°", pitch_color)
        metric_y += line_spacing
        
        # Iris-based Gaze Direction (if available)
        if gaze_results and gaze_results.get('gaze_detected', False):
            h_dir = gaze_results.get('horizontal_direction', 'CENTER')
            v_dir = gaze_results.get('vertical_direction', 'CENTER')
            gaze_dir_text = f"{h_dir}"
            if v_dir != 'CENTER':
                gaze_dir_text += f"-{v_dir}"
            
            gaze_dir_color = (0, 255, 0) if gaze_results.get('looking_forward', False) else (0, 165, 255)
            self.draw_metric_row(frame, panel_x, metric_y, "Gaze Direction", 
                                gaze_dir_text, gaze_dir_color, is_status=True)
            metric_y += line_spacing
        
        # Phone Usage
        phone_detected = phone_results.get('phone_detected', False)
        phone_text = "YES" if phone_detected else "NO"
        phone_color = (0, 0, 255) if phone_detected else (0, 255, 0)
        self.draw_metric_row(frame, panel_x, metric_y, "Phone Usage", 
                            phone_text, phone_color, is_status=phone_detected)
        metric_y += line_spacing
        
        # Alert Level
        alert_text = self.get_alert_level_text(self.current_alert_level)
        alert_color = self.get_alert_level_color(self.current_alert_level)
        self.draw_metric_row(frame, panel_x, metric_y, "Alert Level", 
                            alert_text, alert_color, is_status=True)
        metric_y += line_spacing
        
        # System FPS (real-time)
        fps_color = (0, 255, 0) if self.current_fps > 25 else (0, 255, 255)
        self.draw_metric_row(frame, panel_x, metric_y, "System FPS", 
                            f"{self.current_fps:.1f}", fps_color)
        
        # ═══════════════════════════════════════════════════════════
        # STATUS BAR: Slim bottom summary
        # ═══════════════════════════════════════════════════════════
        
        status_y = h - 38
        status_height = 38
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, status_y), (w, h), (15, 15, 18), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        
        # Separator line
        cv2.line(frame, (0, status_y), (w, status_y), (80, 80, 84), 1)
        
        # Quick summary line (smaller font) - Updated with gaze metrics
        gaze_info = ""
        if gaze_results and gaze_results.get('gaze_detected', False):
            h_dir = gaze_results.get('horizontal_direction', 'CENTER')
            v_dir = gaze_results.get('vertical_direction', 'CENTER')
            gaze_dir = f"{h_dir}" + (f"-{v_dir}" if v_dir != 'CENTER' else "")
            gaze_info = f"  |  Gaze: {gaze_dir}"
        
        summary = (f"Driver: {driver_state}  |  DI: {drowsiness_index:.2f}  |  "
                  f"PERCLOS: {perclos*100:.0f}%  |  Yawn: {yawn_text}{gaze_info}")
        
        summary_compact = (f"{summary}  |  Phone: {phone_text}  |  "
                           f"Roll: {roll:.0f}°  Pitch: {pitch:.0f}°  |  "
                           f"FPS: {self.current_fps:.1f}  |  Alert: {alert_text}")
        cv2.putText(frame, summary_compact, (12, status_y + 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.40, (205, 205, 205), 1, cv2.LINE_AA)
        
        # Timestamp on right (smaller)
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (w - 84, status_y + 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
        
        # ═══════════════════════════════════════════════════════════
        # ALERT BANNER: Center (if active or phone detected)
        # ═══════════════════════════════════════════════════════════
        
        phone_detected = phone_results.get('phone_detected', False) and self.phone_detected_counter >= 10
        if self.current_alert_level > 0 or phone_detected:
            self.draw_alert_banner(frame, drowsy_results, pose_results, phone_results)
        
        return frame
    
    # ═══════════════════════════════════════════════════════════════════
    # HELPER METHODS FOR ENHANCED VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════
    
    def calculate_drowsiness_index(self, drowsy_results, pose_results):
        """
        Calculate overall drowsiness index (0.0 = alert, 1.0 = severely drowsy)
        Combines multiple factors: PERCLOS, EAR, head pose, yawning
        """
        di = 0.0
        
        # PERCLOS contribution (35%)
        perclos = drowsy_results.get('perclos', 0.0)
        di += perclos * 0.35
        
        # EAR contribution (40%) - increased weight since eye closure is most reliable indicator
        ear_val = drowsy_results.get('ear', None)
        if ear_val is not None:
            # Direct EAR-based calculation: when EAR < threshold (0.25), increase DI
            ear_factor = max(0, 1.0 - (ear_val / 0.35))  # More sensitive: 0.35 baseline for alert eyes
            di += ear_factor * 0.40
            
            # Add bonus when eyes are closed (below threshold)
            if ear_val < self.ear_threshold:
                di += 0.10  # Extra 10% when eyes are clearly closed
        
        # Head pose contribution (15%)
        if pose_results:
            pitch = abs(pose_results.get('pitch', 0))
            yaw = abs(pose_results.get('yaw', 0))
            pose_factor = min(1.0, (pitch / 30.0 + yaw / 40.0) / 2.0)
            di += pose_factor * 0.15
        
        # Yawning contribution (10%)
        if drowsy_results.get('yawning', False):
            di += 0.10
        
        return min(1.0, di)

    def draw_enhanced_face_mesh(self, frame, mesh_results):
        """Draw enhanced face mesh with better visibility"""
        try:
            if hasattr(mesh_results, 'multi_face_landmarks') and mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    # Draw with thicker lines and improved color
                    self.mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        self.mediapipe_processor.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(100, 200, 100),  # Bright green
                            thickness=1, 
                            circle_radius=0
                        ),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(80, 180, 80),  # Darker green for connections
                            thickness=1
                        )
                    )
                    # Draw tesselation with lighter color for additional face structure
                    self.mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        self.mediapipe_processor.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,  # Don't draw landmark circles
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(50, 100, 50),  # Very light green
                            thickness=0.5
                        )
                    )
        except Exception as e:
            self.get_logger().debug(f"Error drawing enhanced mesh: {e}")
        
        return frame

    def draw_green_face_mask(self, frame, landmarks, mp_proc):
        """
        Draw a unified green face mask with smooth curves connecting all facial landmarks.
        Creates a complete face representation with all points connected smoothly.
        Transparent and thin lines for subtle visualization.
        """
        # Create transparent overlay for the face mask
        overlay = frame.copy()
        
        # Get all 468 landmarks and convert to 2D points
        all_pts = np.array([(int(x), int(y)) for (x, y, z) in landmarks], dtype=np.int32)
        
        # Draw all landmark points as green circles (very small and transparent)
        for pt in all_pts:
            cv2.circle(overlay, tuple(pt), 1, (0, 255, 0), -1)  # Small green dots
        
        # Draw key facial features with thin smooth curves
        green_color = (0, 200, 0)  # Primary green
        dark_green = (0, 150, 0)   # Darker green for connections
        
        # Draw face outline using left/right contours with smooth curves
        # Left face contour (cheek, jaw, chin on left side)
        left_face_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 
                             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162]
        self._draw_smooth_contour(overlay, landmarks[left_face_indices], green_color, thickness=1, label="L-Face")
        
        # Right face contour
        right_face_indices = [21, 54, 103, 67, 109, 33, 7, 163, 144, 145, 153, 154, 155, 157, 158, 159, 
                              160, 161, 246, 161, 160, 159, 158]
        self._draw_smooth_contour(overlay, landmarks[right_face_indices], green_color, thickness=1, label="R-Face")
        
        # Draw eyes with smooth curves
        left_eye_indices = mp_proc.LEFT_EYE
        right_eye_indices = mp_proc.RIGHT_EYE
        self._draw_smooth_contour(overlay, landmarks[left_eye_indices], green_color, thickness=1, label="L-Eye")
        self._draw_smooth_contour(overlay, landmarks[right_eye_indices], green_color, thickness=1, label="R-Eye")
        
        # Draw eyebrows with smooth curves
        left_eyebrow_indices = [70, 63, 105, 66, 107]
        right_eyebrow_indices = [336, 296, 334, 293, 300]
        self._draw_smooth_contour(overlay, landmarks[left_eyebrow_indices], dark_green, thickness=1, label="L-Brow")
        self._draw_smooth_contour(overlay, landmarks[right_eyebrow_indices], dark_green, thickness=1, label="R-Brow")
        
        # Draw nose with smooth curves
        nose_indices = mp_proc.NOSE
        self._draw_smooth_contour(overlay, landmarks[nose_indices], green_color, thickness=1, label="Nose")
        
        # Draw mouth with smooth curves
        mouth_indices = mp_proc.MOUTH
        self._draw_smooth_contour(overlay, landmarks[mouth_indices], green_color, thickness=1, label="Mouth")
        
        # Blend overlay with original frame for transparency (30% opacity for the mask)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        return frame
    
    def _draw_smooth_contour(self, frame, pts, color, thickness=1, label=""):
        """
        Draw a smooth contour connecting landmark points using spline interpolation.
        Creates curved connections instead of straight lines for a more natural look.
        Uses thin lines for subtle visualization.
        """
        if len(pts) < 3:
            return
        
        try:
            # Filter points to 2D (x, y)
            curve_pts = []
            for pt in pts:
                if len(pt) >= 2:
                    curve_pts.append([int(pt[0]), int(pt[1])])
            
            if len(curve_pts) < 3:
                # If too few points, just draw straight lines
                pts_array = np.array(curve_pts, dtype=np.int32)
                cv2.polylines(frame, [pts_array], True, color, thickness, cv2.LINE_AA)
                return
            
            # Build smooth curve using OpenCV approximation
            pts_array = np.array(curve_pts, dtype=np.float32)
            
            # Use thin polylines with closed curve
            cv2.polylines(frame, [pts_array.astype(np.int32)], True, color, thickness, cv2.LINE_AA)
            
            # Add smooth curves between points using minimal intermediate points
            for i in range(len(curve_pts)):
                p1 = np.array(curve_pts[i], dtype=np.float32)
                p2 = np.array(curve_pts[(i + 1) % len(curve_pts)], dtype=np.float32)
                
                # Reduced interpolation for thinner, subtler curves
                for t in np.linspace(0, 1, 4):
                    pt = p1 * (1 - t) + p2 * t
                    if i == 0 or t > 0:
                        cv2.circle(frame, tuple(pt.astype(int)), 1, color, -1)
        
        except Exception as e:
            self.get_logger().debug(f"Error drawing smooth contour {label}: {e}")

    def get_driver_state(self, drowsiness_index, alert_level, gaze_results=None):
        """Determine driver state and color based on DI and alert level (no emojis)"""
        if alert_level == 4:
            return "NO DRIVER", (0, 0, 255)
        # Check iris-based gaze inattention
        gaze_off_road = False
        if gaze_results and gaze_results.get('gaze_detected', False):
            gaze_off_road = not gaze_results.get('looking_forward', True)
            
        if drowsiness_index > 0.65 or alert_level >= 3:
            return "DROWSY", (0, 0, 255)
        elif drowsiness_index > 0.40 or alert_level >= 1 or gaze_off_road:
            return "CAUTION", (0, 255, 255)
        else:
            return "ATTENTIVE", (0, 255, 0)
    
    def get_gradient_color(self, value):
        """Get gradient color: green -> yellow -> red based on value (0.0-1.0)"""
        if value < 0.3:
            return (0, 255, 0)  # Green
        elif value < 0.6:
            # Blend green to yellow
            ratio = (value - 0.3) / 0.3
            return (0, int(255), int(255 - 255 * ratio))
        else:
            # Blend yellow to red
            ratio = (value - 0.6) / 0.4
            return (0, int(255 * (1 - ratio)), 255)
    
    def draw_metric_row(self, frame, x, y, label, value, color, is_status=False):
        """Draw a compact metric row with refined typography."""
        cv2.putText(frame, label, (x + 12, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.40, (176, 176, 176), 1, cv2.LINE_AA)
        
        font_scale = 0.46 if is_status else 0.44
        thickness = 1
        value_x = x + 142
        cv2.putText(frame, str(value), (value_x, y),
                   cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA)
    
    def draw_metric_bar(self, frame, x, y, width, value, color):
        """Draw a thin progress bar for normalized metrics."""
        bar_height = 8
        
        # Background bar
        cv2.rectangle(frame, (x, y), (x + width, y + bar_height), (45, 45, 48), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + bar_height), (75, 75, 78), 1)
        
        # Filled portion
        fill_width = int(width * value)
        if fill_width > 0:
            cv2.rectangle(frame, (x, y), (x + fill_width, y + bar_height), color, -1)
    
    def get_alert_level_text(self, level):
        """Get alert level description"""
        if level == 0:
            return "0 - None"
        elif level == 1:
            return "1 - Visual"
        elif level == 2:
            return "2 - Visual + Audio"
        elif level == 3:
            return "3 - V + A + Haptic"
        else:
            return "! - No Driver"
    
    def get_alert_level_color(self, level):
        """Get color for alert level"""
        if level == 0:
            return (0, 255, 0)
        elif level == 1:
            return (0, 255, 255)
        elif level == 2:
            return (0, 165, 255)
        else:
            return (0, 0, 255)
    
    def draw_gaze_vector(self, frame, landmarks, pose_results):
        """Draw gaze direction arrow from eye midpoint (eye-based gaze estimation)"""
        if not pose_results:
            return
        
        # Calculate eye centers
        mp_proc = self.mediapipe_processor
        left_eye = landmarks[mp_proc.LEFT_EYE]
        right_eye = landmarks[mp_proc.RIGHT_EYE]
        
        eye_center_x = int((left_eye[:, 0].mean() + right_eye[:, 0].mean()) / 2)
        eye_center_y = int((left_eye[:, 1].mean() + right_eye[:, 1].mean()) / 2)
        
        # Calculate eye aspect ratio for both eyes to estimate gaze
        def eye_aspect_ratio(eye_points):
            # Vertical distances
            v1 = np.linalg.norm(eye_points[1] - eye_points[5])
            v2 = np.linalg.norm(eye_points[2] - eye_points[4])
            # Horizontal distance
            h = np.linalg.norm(eye_points[0] - eye_points[3])
            ear = (v1 + v2) / (2.0 * h)
            return ear
        
        # Calculate horizontal gaze direction based on eye position relative to iris
        # Use the inner and outer eye corners to estimate gaze
        left_inner = left_eye[0]  # Inner corner
        left_outer = left_eye[3]  # Outer corner
        right_inner = right_eye[3]  # Inner corner
        right_outer = right_eye[0]  # Outer corner
        
        # Calculate eye width and iris center position
        left_width = np.linalg.norm(left_outer - left_inner)
        right_width = np.linalg.norm(right_outer - right_inner)
        
        # Estimate gaze based on eye geometry and head pose
        yaw = pose_results.get('yaw', 0)
        pitch = pose_results.get('pitch', 0)
        
        # Use eye openness to refine gaze estimation
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Calculate arrow endpoint with camera angle compensation
        arrow_length = 80
        # Compensate for camera angle: adjust yaw based on camera position
        compensated_yaw = yaw + self.camera_yaw_offset
        
        # Normalize compensated yaw to -180 to 180 range
        if compensated_yaw > 180:
            compensated_yaw -= 360
        elif compensated_yaw < -180:
            compensated_yaw += 360
        
        # Use compensated angles for gaze direction
        gaze_yaw = compensated_yaw
        gaze_pitch = pitch
        
        end_x = int(eye_center_x + arrow_length * np.sin(np.radians(gaze_yaw)))
        end_y = int(eye_center_y - arrow_length * np.sin(np.radians(gaze_pitch)))
        
        # Color: green if on-road (eyes looking forward), red if off-road
        # Consider both head pose and eye openness
        gaze_on_road = abs(gaze_yaw) < self.head_yaw_threshold and abs(gaze_pitch) < self.head_pitch_threshold and avg_ear > 0.15
        arrow_color = (0, 255, 0) if gaze_on_road else (0, 0, 255)
        
        # Draw arrow with shadow (thinner)
        cv2.arrowedLine(frame, (eye_center_x+1, eye_center_y+1), (end_x+1, end_y+1),
                       (0, 0, 0), 3, tipLength=0.3)
        cv2.arrowedLine(frame, (eye_center_x, eye_center_y), (end_x, end_y),
                       arrow_color, 2, tipLength=0.3)
        
        # Add label (smaller)
        label = "ON ROAD" if gaze_on_road else "OFF ROAD"
        label_x = end_x + 8
        label_y = end_y
        cv2.putText(frame, label, (label_x+1, label_y+1),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, arrow_color, 1, cv2.LINE_AA)
    
    def draw_head_pose_axis_enhanced(self, frame, rotation_vector, translation_vector, landmarks):
        """Draw head pose axis with enhanced visibility"""
        if self.head_pose_estimator.camera_matrix is None:
            return
        
        # Get nose tip as origin
        nose_tip = landmarks[1]  # Nose tip landmark
        origin = (int(nose_tip[0]), int(nose_tip[1]))
        
        # 3D axis points
        axis_length = 100
        axis_points_3d = np.array([
            [0, 0, 0],
            [axis_length, 0, 0],  # X-axis (red)
            [0, axis_length, 0],  # Y-axis (green)
            [0, 0, axis_length]   # Z-axis (blue)
        ], dtype=np.float64)
        
        # Project to 2D
        points_2d, _ = cv2.projectPoints(
            axis_points_3d,
            rotation_vector,
            translation_vector,
            self.head_pose_estimator.camera_matrix,
            self.head_pose_estimator.dist_coeffs
        )
        
        points_2d = points_2d.reshape(-1, 2).astype(int)
        
        # Draw axes with shadows and labels (thinner, smaller labels)
        # X-axis (Red)
        cv2.line(frame, (origin[0]+1, origin[1]+1), (points_2d[1][0]+1, points_2d[1][1]+1),
                (0, 0, 0), 3)
        cv2.line(frame, origin, tuple(points_2d[1]), (0, 0, 255), 2)
        cv2.putText(frame, "X", tuple(points_2d[1] + [4, 4]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Y-axis (Green)
        cv2.line(frame, (origin[0]+1, origin[1]+1), (points_2d[2][0]+1, points_2d[2][1]+1),
                (0, 0, 0), 3)
        cv2.line(frame, origin, tuple(points_2d[2]), (0, 255, 0), 2)
        cv2.putText(frame, "Y", tuple(points_2d[2] + [4, 4]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Z-axis (Blue)
        cv2.line(frame, (origin[0]+1, origin[1]+1), (points_2d[3][0]+1, points_2d[3][1]+1),
                (0, 0, 0), 3)
        cv2.line(frame, origin, tuple(points_2d[3]), (255, 100, 0), 2)
        cv2.putText(frame, "Z", tuple(points_2d[3] + [4, 4]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1, cv2.LINE_AA)
    
    def draw_alert_banner(self, frame, drowsy_results, pose_results, phone_results):
        """Draw alert banner with current alert information (compact, non-overlapping).

        Banner is centered but constrained to avoid overlapping the right metrics panel.
        It will shrink horizontally to fit available space; text font is reduced if needed.
        """
        h, w = frame.shape[:2]

        # Recompute panel geometry used in draw_visualization so banner won't overlap it
        panel_width = min(300, max(260, int(w * 0.28)))
        panel_x = w - panel_width - 8

        # Banner geometry: moved to left side, avoiding the right panel
        max_banner_width = max(220, w - panel_width - 32)
        banner_width = min(560, max_banner_width)
        # Position banner more to the left (instead of centered)
        banner_left = 20  # Left aligned with some margin
        banner_right = banner_left + banner_width

        alert_level = self.current_alert_level
        banner_y = 45
        banner_height = 60

        # Check for phone usage first (highest priority alert)
        phone_detected = phone_results.get('phone_detected', False) and self.phone_detected_counter >= 10
        
        # Determine alert details (no emojis)
        if phone_detected:
            # Phone usage alert overrides other alerts
            alert_text = "ALERT: PHONE USAGE DETECTED - Put Down Phone Immediately!"
            bg_color = (0, 0, 220)  # Bright red for phone
            flash_rate = 4  # Fast flashing
        elif alert_level == 4:
            alert_text = "ALERT: Driver Not Detected - Please Return to Seat"
            bg_color = (0, 0, 200)
            flash_rate = 4
        elif alert_level == 3:
            duration = ""
            if self.eyes_closed_start_time:
                duration = f" {time.time() - self.eyes_closed_start_time:.1f}s"
            alert_text = f"ALERT: Severe Drowsiness - Eyes Closed{duration} | Take a Break!"
            bg_color = (0, 0, 180)
            flash_rate = 4
        elif alert_level == 2:
            details = []
            if self.eyes_closed_start_time and (time.time() - self.eyes_closed_start_time) > 1.5:
                details.append("Eyes Closed")
            if self.head_nod_detected:
                details.append("Head Nodding")
            if self.gaze_off_road_start_time and (time.time() - self.gaze_off_road_start_time) > 5.0:
                details.append("Gaze Off Road")
            detail_str = " | ".join(details) if details else "Moderate Drowsiness"
            alert_text = f"ALERT: {detail_str} - Stay Alert"
            bg_color = (0, 100, 200)
            flash_rate = 2
        else:  # Level 1
            alert_text = "WARNING: Mild Drowsiness - Consider a Break"
            bg_color = (0, 200, 200)
            flash_rate = 1

        alpha = 0.85 if int(time.time() * flash_rate) % 2 == 0 else 0.70

        # Draw banner restricted to banner_left..banner_right
        overlay = frame.copy()
        cv2.rectangle(overlay, (banner_left, banner_y), (banner_right, banner_y + banner_height),
                     bg_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Border (thinner)
        border_color = (0, 0, 255) if alert_level >= 3 else (0, 165, 255)
        cv2.rectangle(frame, (banner_left, banner_y), (banner_right, banner_y + banner_height),
                     border_color, 2)

        # Adjust font scale to fit text inside banner
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(alert_text, font, font_scale, thickness)
        if text_w > (banner_width - 24):
            # scale down proportionally but not below 0.35
            scale_factor = max(0.35, (banner_width - 24) / float(text_w) * font_scale)
            font_scale = scale_factor
            (text_w, text_h), _ = cv2.getTextSize(alert_text, font, font_scale, thickness)

        text_x = banner_left + (banner_width - text_w) // 2
        text_y = banner_y + (banner_height // 2) + (text_h // 2)

        # Draw shadow + text
        cv2.putText(frame, alert_text, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, alert_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    def draw_phone_popup(self, frame):
        """Draw a prominent popup message for phone usage detection"""
        if self.phone_popup_start_time is None:
            return
        
        h, w = frame.shape[:2]
        
        # Popup dimensions - center of screen
        popup_width = 500
        popup_height = 150
        popup_x = (w - popup_width) // 2
        popup_y = (h - popup_height) // 2
        
        # Flashing effect
        elapsed = time.time() - self.phone_popup_start_time
        alpha = 0.95 if int(elapsed * 3) % 2 == 0 else 0.85
        
        # Draw semi-transparent red background
        overlay = frame.copy()
        cv2.rectangle(overlay, (popup_x, popup_y), 
                     (popup_x + popup_width, popup_y + popup_height),
                     (0, 0, 200), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw thick border
        cv2.rectangle(frame, (popup_x, popup_y), 
                     (popup_x + popup_width, popup_y + popup_height),
                     (0, 0, 255), 4)
        
        # Draw icon circle (phone symbol area)
        icon_center_x = popup_x + 60
        icon_center_y = popup_y + popup_height // 2
        cv2.circle(frame, (icon_center_x, icon_center_y), 35, (255, 255, 255), 3)
        cv2.line(frame, (icon_center_x - 15, icon_center_y - 20), 
                (icon_center_x - 15, icon_center_y + 20), (255, 255, 255), 3)
        cv2.line(frame, (icon_center_x + 15, icon_center_y - 20), 
                (icon_center_x + 15, icon_center_y + 20), (255, 255, 255), 3)
        cv2.line(frame, (icon_center_x - 15, icon_center_y - 20), 
                (icon_center_x + 15, icon_center_y - 20), (255, 255, 255), 3)
        cv2.line(frame, (icon_center_x - 15, icon_center_y + 20), 
                (icon_center_x + 15, icon_center_y + 20), (255, 255, 255), 3)
        
        # Main warning text
        main_text = "PHONE DETECTED!"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.2
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(main_text, font, font_scale, thickness)
        text_x = popup_x + 120
        text_y = popup_y + 55
        
        # Draw text with shadow
        cv2.putText(frame, main_text, (text_x + 2, text_y + 2), font, font_scale, 
                   (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, main_text, (text_x, text_y), font, font_scale, 
                   (255, 255, 255), thickness, cv2.LINE_AA)
        
        # Secondary warning text
        sub_text = "Put down the phone immediately!"
        font_scale_sub = 0.7
        thickness_sub = 1
        (text_w_sub, text_h_sub), _ = cv2.getTextSize(sub_text, font, font_scale_sub, thickness_sub)
        text_x_sub = popup_x + (popup_width - text_w_sub) // 2
        text_y_sub = popup_y + 105
        
        cv2.putText(frame, sub_text, (text_x_sub + 1, text_y_sub + 1), font, font_scale_sub, 
                   (0, 0, 0), thickness_sub + 1, cv2.LINE_AA)
        cv2.putText(frame, sub_text, (text_x_sub, text_y_sub), font, font_scale_sub, 
                   (255, 255, 0), thickness_sub, cv2.LINE_AA)

    def image_callback(self, msg):
        """Process incoming image messages from camera"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return

        self.process_frame(frame)

    def run(self):
        self.get_logger().info('DMS Node is running...')


def main(args=None):
    rclpy.init(args=args)
    node = DMSNode()
    try:
        node.run()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('DMS Node terminated.')
    except Exception as e:
        node.get_logger().error(f'Unexpected error: {e}')
        import traceback

        traceback.print_exc()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
