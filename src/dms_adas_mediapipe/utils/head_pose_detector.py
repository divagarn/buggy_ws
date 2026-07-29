#!/usr/bin/env python3
"""
Head Pose Estimation using MediaPipe Landmarks and solvePnP
"""

import cv2
import numpy as np

class HeadPoseEstimator:
    def __init__(self):
        """Initialize head pose estimator"""
        # 3D model points (generic human face model in mm)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
    
    def initialize_camera(self, frame_width, frame_height):
        """Initialize camera matrix based on frame size"""
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def estimate_pose(self, landmarks, frame_shape):
        """
        Estimate head pose from facial landmarks
        
        Args:
            landmarks: All facial landmarks (468 points)
            frame_shape: Shape of the frame
            
        Returns:
            Dictionary with pose angles
        """
        if landmarks is None or len(landmarks) < 468:
            return None
        
        if self.camera_matrix is None:
            self.initialize_camera(frame_shape[1], frame_shape[0])
        
        # 2D image points (using MediaPipe landmark indices)
        # 1: Nose tip, 152: Chin, 33: Left eye, 263: Right eye
        # 61: Left mouth, 291: Right mouth
        image_points = np.array([
            landmarks[1, :2],      # Nose
            landmarks[152, :2],    # Chin
            landmarks[33, :2],     # Left eye
            landmarks[263, :2],    # Right eye
            landmarks[61, :2],     # Left mouth
            landmarks[291, :2]     # Right mouth
        ], dtype=np.float64)
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Calculate Euler angles
        pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
        
        pitch = float(euler_angles[0])
        yaw = float(euler_angles[1])
        roll = float(euler_angles[2])
        
        return {
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'rotation_vector': rotation_vector,
            'translation_vector': translation_vector
        }
    
    def draw_pose_axes(self, frame, rotation_vector, translation_vector):
        """Draw 3D axes for head pose visualization"""
        if self.camera_matrix is None:
            return frame
        
        axis_points = np.array([
            (0, 0, 0),
            (300, 0, 0),
            (0, 300, 0),
            (0, 0, 300)
        ], dtype=np.float64)
        
        points_2d, _ = cv2.projectPoints(
            axis_points,
            rotation_vector,
            translation_vector,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        points_2d = points_2d.reshape(-1, 2).astype(int)
        origin = tuple(points_2d[0])
        
        cv2.line(frame, origin, tuple(points_2d[1]), (0, 0, 255), 3)  # X-axis (red)
        cv2.line(frame, origin, tuple(points_2d[2]), (0, 255, 0), 3)  # Y-axis (green)
        cv2.line(frame, origin, tuple(points_2d[3]), (255, 0, 0), 3)  # Z-axis (blue)
        
        return frame
