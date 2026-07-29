#!/usr/bin/env python3
"""
MediaPipe Face Detection and Landmark Detection Processor
Handles facial landmark extraction from video frames
"""

import cv2
import mediapipe as mp
import numpy as np

class MediaPipeProcessor:
    def __init__(self, static_image_mode=False, max_num_faces=1,
                 min_detection_confidence=0.25, min_tracking_confidence=0.5):
        """Initialize MediaPipe Face Detection and Face Mesh"""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Landmark indices for facial features
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        # Iris landmarks (5 points each including center)
        self.RIGHT_IRIS = [468, 469, 470, 471, 472]  # Right eye iris: [center, right, top, left, bottom]
        self.LEFT_IRIS = [473, 474, 475, 476, 477]   # Left eye iris: [center, right, top, left, bottom]
        # Include both outer and inner lip landmarks to better capture mouth opening
        self.MOUTH = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,  # outer lip
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324    # inner lip / additional points
        ]
        self.NOSE = [1, 2, 98, 327, 331]
        self.HEAD_POSE_POINTS = [33, 263, 1, 61, 291, 199]

    def detect_faces_and_landmarks(self, frame):
        """Detect faces and extract facial landmarks from frame"""
        results = {
            'faces_detected': 0,
            'face_landmarks': [],  # List of face_landmarks objects
            'all_landmarks': [],   # List of numpy arrays
            'face_bboxes': [],
            'mesh_results': None
        }

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Detect faces using face detection
        face_detection_results = self.face_detection.process(rgb_frame)
        
        if face_detection_results.detections:
            results['faces_detected'] = len(face_detection_results.detections)
            for detection in face_detection_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * w)
                y_min = int(bbox.ymin * h)
                x_max = int((bbox.xmin + bbox.width) * w)
                y_max = int((bbox.ymin + bbox.height) * h)
                
                results['face_bboxes'].append({
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max,
                    'confidence': detection.score[0] if detection.score else None
                })

        # Detect face mesh landmarks
        mesh_results = self.face_mesh.process(rgb_frame)
        results['mesh_results'] = mesh_results

        # Process detected faces
        if mesh_results and mesh_results.multi_face_landmarks:
            # mesh_results.multi_face_landmarks is a LIST
            # Each element in this list is a face_landmarks object (NOT a list)
            for face_landmarks_obj in mesh_results.multi_face_landmarks:
                # NOW face_landmarks_obj is a SINGLE object with .landmark attribute
                # Convert to numpy array
                landmarks_array = np.array([
                    [lm.x * w, lm.y * h, lm.z]
                    for lm in face_landmarks_obj.landmark
                ], dtype=np.float32)

                # Store the numpy array
                results['all_landmarks'].append(landmarks_array)
                
                # Store the face_landmarks object itself
                results['face_landmarks'].append(face_landmarks_obj)

        return results

    def draw_landmarks(self, frame, mesh_results):
        """Draw face mesh landmarks on frame"""
        if mesh_results is None:
            return frame

        try:
            if hasattr(mesh_results, 'multi_face_landmarks') and mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    # Draw only the connections (no landmark circles) to reduce visual clutter
                    self.mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 0), 
                            thickness=1, 
                            circle_radius=0
                        ),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 0),
                            thickness=1
                        )
                    )
            elif hasattr(mesh_results, 'landmark'):
                self.mp_drawing.draw_landmarks(
                    frame,
                    mesh_results,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), 
                        thickness=1, 
                        circle_radius=0
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0),
                        thickness=1
                    )
                )
        except Exception as e:
            print(f"Error drawing landmarks: {e}")

        return frame

    def get_eye_region(self, frame, landmarks, eye_indices):
        """Extract eye region from frame"""
        eye_coords = landmarks[eye_indices]
        x_min = int(np.min(eye_coords[:, 0]))
        x_max = int(np.max(eye_coords[:, 0]))
        y_min = int(np.min(eye_coords[:, 1]))
        y_max = int(np.max(eye_coords[:, 1]))

        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)

        return {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'coords': eye_coords
        }

    def get_mouth_region(self, frame, landmarks, mouth_indices):
        """Extract mouth region from frame"""
        mouth_coords = landmarks[mouth_indices]
        

        # bounding box of mouth landmarks
        mx_min = float(np.min(mouth_coords[:, 0]))
        mx_max = float(np.max(mouth_coords[:, 0]))
        my_min = float(np.min(mouth_coords[:, 1]))
        my_max = float(np.max(mouth_coords[:, 1]))

        mouth_w = max(1.0, mx_max - mx_min)
        mouth_h = max(1.0, my_max - my_min)

        # Asymmetric expansion: bias the region downward to capture lower-lip / jaw movement
        # Compute symmetric width expansion but asymmetric vertical padding
        width_factor = 2.0
        region_w = int(max(mouth_w * width_factor, mouth_w + 40))

        # Top padding: small fraction of mouth height
        top_pad = max(6, int(0.35 * mouth_h))
        # Bottom padding: larger to follow lower-lip/jaw extension
        bottom_pad = max(20, int(1.2 * mouth_h))

        # Center x unchanged, build x_min/x_max from region_w
        cx = int((mx_min + mx_max) / 2.0)
        x_min = int(cx - region_w // 2)
        x_max = int(cx + region_w // 2)

        y_min = int(my_min - top_pad)
        y_max = int(my_max + bottom_pad)

        # Clamp to image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)

        return {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'coords': mouth_coords
        }

    def release(self):
        """Release MediaPipe resources"""
        self.face_detection.close()
        self.face_mesh.close()