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
            (0.0, 0.0, 0.0),             # Nose tip
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
            landmarks: Nx3 numpy array of all landmarks
            frame_shape: (height, width, channels)
            
        Returns:
            Dictionary with pose angles in degrees
        """
        if landmarks is None or len(landmarks) < 468:
            return None

        height, width, _ = frame_shape

        if self.camera_matrix is None:
            self.initialize_camera(width, height)

        # 2D image points (using MediaPipe landmark indices)
        image_points = np.array([
            landmarks[1, :2],    # Nose tip
            landmarks[152, :2],  # Chin
            landmarks[33, :2],   # Left eye left corner
            landmarks[263, :2],  # Right eye right corner
            landmarks[61, :2],   # Left mouth corner
            landmarks[291, :2]   # Right mouth corner
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

        # Get Euler angles from rotation matrix
        # Reference: https://stackoverflow.com/questions/1556260
        sy = np.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            pitch = np.arctan2(-rotation_matrix[2,1], rotation_matrix[2,2])
            yaw = np.arctan2(rotation_matrix[2,0], sy)
            roll = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            pitch = np.arctan2(-rotation_matrix[2,1], rotation_matrix[2,2])
            yaw = np.arctan2(-rotation_matrix[2,0], sy)
            roll = 0

        # Convert radians to degrees
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        roll_deg = np.degrees(roll)

        return {
            'yaw': yaw_deg,
            'pitch': pitch_deg,
            'roll': roll_deg,
            'rotation_vector': rotation_vector,
            'translation_vector': translation_vector
        }

    def draw_pose_axes(self, frame, rotation_vector, translation_vector):
        """Draw 3D axes on the face for visual feedback"""
        if self.camera_matrix is None:
            return frame
        axis_length = 100  # length of axes in 3D space
        axis_points = np.array([
            [0, 0, 0],  # origin
            [axis_length, 0, 0],   # X (red)
            [0, axis_length, 0],   # Y (green)
            [0, 0, axis_length]    # Z (blue)
        ], dtype=np.float64)

        imgpts, _ = cv2.projectPoints(
            axis_points,
            rotation_vector,
            translation_vector,
            self.camera_matrix,
            self.dist_coeffs
        )

        origin = tuple(imgpts[0].ravel().astype(int))
        x_axis = tuple(imgpts[1].ravel().astype(int))
        y_axis = tuple(imgpts[2].ravel().astype(int))
        z_axis = tuple(imgpts[3].ravel().astype(int))

        # Draw axes
        cv2.line(frame, origin, x_axis, (0, 0, 255), 2)  # X axis - Red
        cv2.line(frame, origin, y_axis, (0, 255, 0), 2)  # Y axis - Green
        cv2.line(frame, origin, z_axis, (255, 0, 0), 2)  # Z axis - Blue

        return frame

