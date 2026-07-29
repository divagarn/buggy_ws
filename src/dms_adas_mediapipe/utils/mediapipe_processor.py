#!/usr/bin/env python3
"""
MediaPipe Processor for Driver Monitoring System
Detects and extracts facial landmarks from the driver's face
"""

import cv2
import mediapipe as mp
import numpy as np

class MediaPipeProcessor:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize Face Mesh for single face (driver)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Landmark indices for key regions
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375]
        self.NOSE = [1, 2, 98, 327, 331]
        self.HEAD_POSE_POINTS = [33, 263, 1, 61, 291, 199]

    def detect_driver_face(self, frame):
        """
        Detect driver's face and extract landmarks

        Args:
            frame: BGR image frame

        Returns:
            Dictionary with facial region landmarks and full mesh
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        mesh_results = self.face_mesh.process(rgb_frame)

        if not mesh_results.multi_face_landmarks:
            return None  # No face detected

        face_landmarks = mesh_results.multi_face_landmarks[0]
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z]
            for lm in face_landmarks.landmark
        ], dtype=np.float32)

        return {
            'landmarks': landmarks,
            'left_eye': landmarks[self.LEFT_EYE],
            'right_eye': landmarks[self.RIGHT_EYE],
            'mouth': landmarks[self.MOUTH],
            'nose': landmarks[self.NOSE],
            'head_pose_points': landmarks[self.HEAD_POSE_POINTS],
            'mesh': face_landmarks
        }

    def draw_driver_face(self, frame, mesh):
        """
        Draw facial mesh on driver's face

        Args:
            frame: BGR image frame
            mesh: MediaPipe face mesh object

        Returns:
            Annotated frame
        """
        self.mp_drawing.draw_landmarks(
            frame,
            mesh,
            self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 0),
                thickness=1,
                circle_radius=1
            )
        )
        return frame

    def release(self):
        """Release MediaPipe resources"""
        self.face_mesh.close()

