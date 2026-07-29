#!/usr/bin/env python3
"""
Iris-based Gaze Detection Module
Uses MediaPipe iris landmarks to estimate gaze direction
"""

import numpy as np
import cv2


class IrisGazeDetector:
    def __init__(self, smoothing_alpha=0.3, mirror_view=True):
        """
        Initialize iris-based gaze detector
        
        Args:
            smoothing_alpha: Smoothing factor for gaze direction (0.0 = no smoothing, 1.0 = max smoothing)
            mirror_view: If True, invert left/right only in visualization to match mirrored camera view
        """
        self.smoothing_alpha = smoothing_alpha
        self.mirror_view = mirror_view
        
        # Smoothed gaze ratios
        self.smoothed_left_gaze_ratio = None
        self.smoothed_right_gaze_ratio = None
        self.smoothed_left_vertical_ratio = None
        self.smoothed_right_vertical_ratio = None
        
        # Calibration parameters (can be tuned based on camera setup)
        self.horizontal_threshold = 0.5  # Center threshold for horizontal gaze
        self.vertical_threshold = 0.5    # Center threshold for vertical gaze
        
        # Gaze direction labels
        self.GAZE_LEFT = "LEFT"
        self.GAZE_RIGHT = "RIGHT"
        self.GAZE_CENTER = "CENTER"
        self.GAZE_UP = "UP"
        self.GAZE_DOWN = "DOWN"
    
    def calculate_gaze_ratio(self, eye_points, iris_center):
        """
        Calculate gaze ratio based on iris position relative to eye corners
        
        Args:
            eye_points: Eye corner landmarks (numpy array)
            iris_center: Iris center point (x, y)
        
        Returns:
            horizontal_ratio: Horizontal gaze ratio (0.0 = left, 1.0 = right)
            vertical_ratio: Vertical gaze ratio (0.0 = up, 1.0 = down)
        """
        # Find leftmost and rightmost points by x-coordinate (not by index)
        x_coords = eye_points[:, 0]
        y_coords = eye_points[:, 1]
        
        eye_left = eye_points[np.argmin(x_coords)]  # Leftmost point
        eye_right = eye_points[np.argmax(x_coords)] # Rightmost point
        
        # Calculate eye width and height
        eye_width = np.linalg.norm(eye_right[:2] - eye_left[:2])
        
        # Get top and bottom of eye by y-coordinate
        eye_top_idx = np.argmin(y_coords)
        eye_bottom_idx = np.argmax(y_coords)
        eye_top = eye_points[eye_top_idx, 1]
        eye_bottom = eye_points[eye_bottom_idx, 1]
        eye_height = eye_bottom - eye_top
        
        # Calculate horizontal ratio
        # Distance from left corner to iris center
        iris_x = iris_center[0]
        iris_y = iris_center[1]
        
        left_distance = iris_x - eye_left[0]
        horizontal_ratio = left_distance / eye_width if eye_width > 0 else 0.5
        
        # Calculate vertical ratio
        vertical_distance = iris_y - eye_top
        vertical_ratio = vertical_distance / eye_height if eye_height > 0 else 0.5
        
        # Clamp ratios to [0, 1]
        horizontal_ratio = np.clip(horizontal_ratio, 0.0, 1.0)
        vertical_ratio = np.clip(vertical_ratio, 0.0, 1.0)
        
        return horizontal_ratio, vertical_ratio
    
    def smooth_gaze_ratio(self, new_ratio, smoothed_ratio):
        """Apply exponential smoothing to gaze ratio"""
        if smoothed_ratio is None:
            return new_ratio
        
        return self.smoothing_alpha * smoothed_ratio + (1 - self.smoothing_alpha) * new_ratio
    
    def get_gaze_direction(self, horizontal_ratio, vertical_ratio):
        """
        Determine gaze direction from ratios
        
        Args:
            horizontal_ratio: 0.0 (left) to 1.0 (right)
            vertical_ratio: 0.0 (up) to 1.0 (down)
        
        Returns:
            horizontal_direction: "LEFT", "CENTER", or "RIGHT"
            vertical_direction: "UP", "CENTER", or "DOWN"
        """
        # Horizontal direction
        if horizontal_ratio < (self.horizontal_threshold - 0.1):
            horizontal = self.GAZE_LEFT
        elif horizontal_ratio > (self.horizontal_threshold + 0.1):
            horizontal = self.GAZE_RIGHT
        else:
            horizontal = self.GAZE_CENTER
        
        # Vertical direction
        if vertical_ratio < (self.vertical_threshold - 0.1):
            vertical = self.GAZE_UP
        elif vertical_ratio > (self.vertical_threshold + 0.1):
            vertical = self.GAZE_DOWN
        else:
            vertical = self.GAZE_CENTER
        
        return horizontal, vertical
    
    def detect_gaze(self, landmarks_dict):
        """
        Detect gaze direction using iris landmarks
        
        Args:
            landmarks_dict: Dictionary containing:
                - 'left_eye': Left eye landmarks
                - 'right_eye': Right eye landmarks
                - 'left_iris': Left iris landmarks (including center at index 0)
                - 'right_iris': Right iris landmarks (including center at index 0)
        
        Returns:
            Dictionary with gaze information:
                - 'left_gaze_ratio': (horizontal, vertical) for left eye
                - 'right_gaze_ratio': (horizontal, vertical) for right eye
                - 'combined_gaze_ratio': Average of both eyes
                - 'horizontal_direction': "LEFT", "CENTER", or "RIGHT"
                - 'vertical_direction': "UP", "CENTER", or "DOWN"
                - 'looking_forward': True if gaze is centered
                - 'gaze_angle': Estimated gaze angle in degrees (yaw)
        """
        results = {
            'left_gaze_ratio': None,
            'right_gaze_ratio': None,
            'combined_gaze_ratio': None,
            'horizontal_direction': self.GAZE_CENTER,
            'vertical_direction': self.GAZE_CENTER,
            'looking_forward': True,
            'gaze_angle': 0.0,
            'gaze_detected': False
        }
        
        left_eye = landmarks_dict.get('left_eye')
        right_eye = landmarks_dict.get('right_eye')
        left_iris = landmarks_dict.get('left_iris')
        right_iris = landmarks_dict.get('right_iris')
        
        if left_eye is None or right_eye is None or left_iris is None or right_iris is None:
            return results
        
        # Get iris centers (center is typically the first landmark or compute average)
        # MediaPipe iris: [center, right, top, left, bottom]
        left_iris_center = left_iris[0][:2]  # Center point (x, y)
        right_iris_center = right_iris[0][:2]
        
        # Calculate gaze ratios for each eye
        left_h_ratio, left_v_ratio = self.calculate_gaze_ratio(left_eye, left_iris_center)
        right_h_ratio, right_v_ratio = self.calculate_gaze_ratio(right_eye, right_iris_center)
        
        # Apply smoothing
        left_h_ratio = self.smooth_gaze_ratio(left_h_ratio, self.smoothed_left_gaze_ratio)
        right_h_ratio = self.smooth_gaze_ratio(right_h_ratio, self.smoothed_right_gaze_ratio)
        left_v_ratio = self.smooth_gaze_ratio(left_v_ratio, self.smoothed_left_vertical_ratio)
        right_v_ratio = self.smooth_gaze_ratio(right_v_ratio, self.smoothed_right_vertical_ratio)
        
        # Update smoothed values
        self.smoothed_left_gaze_ratio = left_h_ratio
        self.smoothed_right_gaze_ratio = right_h_ratio
        self.smoothed_left_vertical_ratio = left_v_ratio
        self.smoothed_right_vertical_ratio = right_v_ratio
        
        # Store individual eye ratios
        results['left_gaze_ratio'] = (left_h_ratio, left_v_ratio)
        results['right_gaze_ratio'] = (right_h_ratio, right_v_ratio)
        
        # Calculate combined gaze ratio (average of both eyes)
        combined_h_ratio = (left_h_ratio + right_h_ratio) / 2.0
        combined_v_ratio = (left_v_ratio + right_v_ratio) / 2.0
        results['combined_gaze_ratio'] = (combined_h_ratio, combined_v_ratio)
        
        # Determine gaze direction
        horizontal_dir, vertical_dir = self.get_gaze_direction(combined_h_ratio, combined_v_ratio)
        results['horizontal_direction'] = horizontal_dir
        results['vertical_direction'] = vertical_dir
        
        # Check if looking forward (both horizontal and vertical centered)
        results['looking_forward'] = (horizontal_dir == self.GAZE_CENTER and 
                                     vertical_dir == self.GAZE_CENTER)
        
        # Estimate gaze angle in degrees (approximate)
        # Map ratio to angle: 0.0 -> -30°, 0.5 -> 0°, 1.0 -> +30°
        results['gaze_angle'] = (combined_h_ratio - 0.5) * 60.0
        
        results['gaze_detected'] = True
        
        return results
    
    def draw_gaze_vector(self, frame, landmarks_dict, gaze_results, arrow_length=100):
        """
        Draw gaze direction arrow on frame
        
        Args:
            frame: Image frame
            landmarks_dict: Dictionary with eye and iris landmarks
            gaze_results: Results from detect_gaze()
            arrow_length: Length of gaze arrow
        
        Returns:
            Modified frame with gaze vector drawn
        """
        if not gaze_results.get('gaze_detected', False):
            return frame
        
        left_eye = landmarks_dict.get('left_eye')
        right_eye = landmarks_dict.get('right_eye')
        
        if left_eye is None or right_eye is None:
            return frame
        
        # Calculate eye center (midpoint between eyes)
        left_center = np.mean(left_eye[:, :2], axis=0)
        right_center = np.mean(right_eye[:, :2], axis=0)
        eye_center = ((left_center + right_center) / 2.0).astype(int)
        
        # Get gaze ratio
        h_ratio, v_ratio = gaze_results['combined_gaze_ratio']
        
        # Calculate arrow endpoint
        # Map ratio to direction: 0.5 = center, <0.5 = left/up, >0.5 = right/down
        dx = (h_ratio - 0.5) * arrow_length * 2
        if self.mirror_view:
            dx = -dx
        dy = (v_ratio - 0.5) * arrow_length * 2
        
        end_point = (int(eye_center[0] + dx), int(eye_center[1] + dy))
        
        # Determine arrow color based on gaze direction
        if gaze_results['looking_forward']:
            arrow_color = (0, 255, 0)  # Green = on road
        else:
            arrow_color = (0, 0, 255)  # Red = off road
        
        # Draw arrow with shadow
        cv2.arrowedLine(frame, tuple(eye_center + 1), (end_point[0] + 1, end_point[1] + 1),
                       (0, 0, 0), 3, tipLength=0.3)
        cv2.arrowedLine(frame, tuple(eye_center), end_point,
                       arrow_color, 2, tipLength=0.3)
        
        # Add direction label
        horizontal_label = gaze_results['horizontal_direction']
        if self.mirror_view:
            if horizontal_label == self.GAZE_LEFT:
                horizontal_label = self.GAZE_RIGHT
            elif horizontal_label == self.GAZE_RIGHT:
                horizontal_label = self.GAZE_LEFT
        label = f"{horizontal_label}"
        if gaze_results['vertical_direction'] != self.GAZE_CENTER:
            label += f"-{gaze_results['vertical_direction']}"
        
        label_pos = (end_point[0] + 10, end_point[1])
        cv2.putText(frame, label, (label_pos[0] + 1, label_pos[1] + 1),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, label, label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, arrow_color, 1, cv2.LINE_AA)
        
        return frame
    
    def draw_iris_points(self, frame, landmarks_dict):
        """
        Draw iris landmarks on frame
        
        Args:
            frame: Image frame
            landmarks_dict: Dictionary with iris landmarks
        
        Returns:
            Modified frame with iris points drawn
        """
        left_iris = landmarks_dict.get('left_iris')
        right_iris = landmarks_dict.get('right_iris')
        
        if left_iris is not None:
            for point in left_iris:
                x, y = int(point[0]), int(point[1])
                # Draw iris points in cyan
                cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
                cv2.circle(frame, (x, y), 2, (255, 255, 255), 1)
        
        if right_iris is not None:
            for point in right_iris:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
                cv2.circle(frame, (x, y), 2, (255, 255, 255), 1)
        
        return frame
    
    def reset_smoothing(self):
        """Reset smoothed values (useful when face is lost)"""
        self.smoothed_left_gaze_ratio = None
        self.smoothed_right_gaze_ratio = None
        self.smoothed_left_vertical_ratio = None
        self.smoothed_right_vertical_ratio = None
