#!/usr/bin/env python3
"""
Drowsiness Detection using MediaPipe Landmarks
Uses Eye Aspect Ratio (EAR) and PERCLOS
"""

import numpy as np
from scipy.spatial import distance
import logging

class DrowsinessDetector:
    def __init__(self, ear_threshold=0.25, eye_closed_frames=30,
                 perclos_window=60, perclos_threshold=0.2,
                 mar_threshold=0.3, yawn_frames=12,
                 smoothing_alpha=0.35, single_eye_frames=8,
                 use_smoothed_mar=True):
        """
        Initialize drowsiness detector
        """
        self.ear_threshold = ear_threshold
        self.eye_closed_frames = eye_closed_frames
        self.perclos_window = perclos_window
        self.perclos_threshold = perclos_threshold
        self.mar_threshold = mar_threshold
        self.yawn_frames = yawn_frames

        # Tracking variables
        self.eye_closed_counter = 0
        # track left/right eye closure separately to detect single-eye closure
        self.left_eye_closed_counter = 0
        self.right_eye_closed_counter = 0
        self.blink_counter = 0
        self.perclos_history = []
        # Yawn tracking
        self.yawn_counter = 0
        # Exponential moving average smoothing for EAR/MAR to reduce jitter
        self.smoothing_alpha = smoothing_alpha
        self.ear_ema = None
        self.mar_ema = None
        # threshold for single-eye closure detection (consecutive frames)
        self.single_eye_frames = single_eye_frames
        # whether to use EMA-smoothed MAR for detection (reduces jitter)
        self.use_smoothed_mar = use_smoothed_mar

    def eye_aspect_ratio(self, eye_points):
        """
        Calculate Eye Aspect Ratio (EAR).
        Args:
            eye_points: Nx2 or Nx3 array of eye landmark points.
        Returns:
            EAR as float.
        """
        # Robust EAR: compute vertical separation between upper and lower eyelid
        # and normalize by horizontal eye width. Works regardless of specific
        # landmark ordering as long as they cover the eye region.
        if len(eye_points) < 4:
            return 0.0

        pts = np.array(eye_points)
        xs = pts[:, 0]
        ys = pts[:, 1]

        # horizontal width: distance between leftmost and rightmost points
        horiz = float(np.max(xs) - np.min(xs))
        if horiz <= 1e-6:
            return 0.0

        # vertical: difference between mean of bottom-most y's and top-most y's
        top_idx = np.argsort(ys)[:2]
        bottom_idx = np.argsort(ys)[-2:]
        top_mean = np.mean(ys[top_idx])
        bottom_mean = np.mean(ys[bottom_idx])

        vertical = float(bottom_mean - top_mean)
        ear = vertical / horiz

        # Sanity clamp
        if ear < 0:
            ear = 0.0
        return float(ear)

    def mouth_aspect_ratio(self, mouth_points):
        """
        Calculate Mouth Aspect Ratio (MAR).
        Args:
            mouth_points: Nx2 or Nx3 array of mouth landmark points.
        Returns:
            MAR as float.
        """
        # Robust MAR: measure vertical opening of mouth normalized by width.
        if len(mouth_points) < 6:
            return 0.0

        pts = np.array(mouth_points)
        xs = pts[:, 0]
        ys = pts[:, 1]

        horiz = float(np.max(xs) - np.min(xs))
        if horiz <= 1e-6:
            return 0.0

        # Use the top-most and bottom-most points to estimate opening
        top_mean = np.mean(ys[np.argsort(ys)[:3]])
        bottom_mean = np.mean(ys[np.argsort(ys)[-3:]])
        vertical = float(bottom_mean - top_mean)
        mar = vertical / horiz
        if mar < 0:
            mar = 0.0
        return float(mar)

    def detect(self, face_landmarks):
        """
        Detect drowsiness indicators
        Args:
            face_landmarks: dict with 'left_eye', 'right_eye', 'mouth' numpy arrays
        Returns:
            dict with detection results
        """
        results = {
            'drowsy': False,
            'yawning': False,
            'left_eye_closed': False,
            'right_eye_closed': False,
            'ear': 0.0,
            'mar': 0.0,
            'left_ear': 0.0,
            'right_ear': 0.0,
            'perclos': 0.0,
            'blink_count': self.blink_counter
        }

        if face_landmarks is None:
            return results

        # Extract landmarks
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        mouth = face_landmarks['mouth']

        if len(left_eye) < 6 or len(right_eye) < 6 or len(mouth) < 10:
            return results

        # Calculate EARs
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Initialize or update EAR EMA
        if self.ear_ema is None:
            self.ear_ema = ear
        else:
            self.ear_ema = (self.smoothing_alpha * ear) + ((1.0 - self.smoothing_alpha) * self.ear_ema)

        results['left_ear'] = left_ear
        results['right_ear'] = right_ear
        results['ear'] = ear
        results['ear_smoothed'] = float(self.ear_ema)

        # Debug logging when values are near thresholds
        try:
            if ear < (self.ear_threshold * 1.3) or ear < (self.ear_threshold + 0.05):
                logging.debug(f"EAR L:{left_ear:.3f} R:{right_ear:.3f} AVG:{ear:.3f} thr:{self.ear_threshold}")
        except Exception:
            pass

        # Track eye closure (both eyes) and per-eye counters
        if ear < self.ear_threshold:
            self.eye_closed_counter += 1
        else:
            if self.eye_closed_counter > 0:
                self.blink_counter += 1
            self.eye_closed_counter = 0

        # left/right eye closure counters
        if left_ear < self.ear_threshold:
            self.left_eye_closed_counter += 1
        else:
            self.left_eye_closed_counter = 0

        if right_ear < self.ear_threshold:
            self.right_eye_closed_counter += 1
        else:
            self.right_eye_closed_counter = 0

        # expose single-eye closure in results
        if self.left_eye_closed_counter >= self.single_eye_frames:
            results['left_eye_closed'] = True
            results['drowsy'] = True
        if self.right_eye_closed_counter >= self.single_eye_frames:
            results['right_eye_closed'] = True
            results['drowsy'] = True

        # PERCLOS window update
        eyes_closed = 1 if ear < self.ear_threshold else 0
        self.perclos_history.append(eyes_closed)
        if len(self.perclos_history) > self.perclos_window:
            self.perclos_history.pop(0)

        # Compute PERCLOS
        if self.perclos_history:
            perclos = sum(self.perclos_history) / len(self.perclos_history)
            results['perclos'] = perclos
            if perclos > self.perclos_threshold:
                results['drowsy'] = True

        # Check for prolonged eye closure
        if self.eye_closed_counter >= self.eye_closed_frames:
            results['drowsy'] = True

        # Yawn detection (configurable threshold + consecutive-frame detection)
        mar = self.mouth_aspect_ratio(mouth)
        # Initialize or update MAR EMA
        if self.mar_ema is None:
            self.mar_ema = mar
        else:
            self.mar_ema = (self.smoothing_alpha * mar) + ((1.0 - self.smoothing_alpha) * self.mar_ema)
        results['mar'] = mar
        results['mar_smoothed'] = float(self.mar_ema)
        try:
            if mar > (self.mar_threshold * 0.8) or mar > (self.mar_threshold - 0.2):
                logging.debug(f"MAR:{mar:.3f} thr:{self.mar_threshold}")
        except Exception:
            pass

        # Use smoothed MAR for detection optionally
        mar_for_detection = float(self.mar_ema) if (self.use_smoothed_mar and self.mar_ema is not None) else mar

        if mar_for_detection > (0.8 * self.mar_threshold):
            logging.info(f"MAR near threshold: {mar_for_detection:.3f} thr:{self.mar_threshold}")

        if mar_for_detection > self.mar_threshold:
            self.yawn_counter += 1
            if self.yawn_counter >= self.yawn_frames:
                results['yawning'] = True
                logging.info(f"Yawn detected: mar={mar_for_detection:.3f} counter={self.yawn_counter}")
        else:
            self.yawn_counter = 0

        # expose yawn counter for publishing/inspection
        results['yawn_counter'] = int(self.yawn_counter)

        return results

    def reset(self):
        """Reset internal counters and history"""
        self.eye_closed_counter = 0
        self.perclos_history = []
        self.yawn_counter = 0
        self.ear_ema = None
        self.mar_ema = None

