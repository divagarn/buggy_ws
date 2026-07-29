#!/usr/bin/env python3
"""
Drowsiness Detection using MediaPipe Landmarks
Uses Eye Aspect Ratio (EAR) and PERCLOS
"""

import numpy as np
from scipy.spatial import distance
import time

class DrowsinessDetector:
    def __init__(self, ear_threshold=0.25, eye_closed_frames=30,
                 perclos_window=60, perclos_threshold=0.2):
        """
        Initialize drowsiness detector
        
        Args:
            ear_threshold: Eye Aspect Ratio threshold
            eye_closed_frames: Consecutive closed eye frames for alert
            perclos_window: Window size for PERCLOS calculation (frames)
            perclos_threshold: PERCLOS threshold (% eye closure)
        """
        self.ear_threshold = ear_threshold
        self.eye_closed_frames = eye_closed_frames
        self.perclos_window = perclos_window
        self.perclos_threshold = perclos_threshold
        
        # Tracking variables
        self.eye_closed_counter = 0
        self.blink_counter = 0
        self.perclos_history = []
        
    def eye_aspect_ratio(self, eye_points):
        """
        Calculate Eye Aspect Ratio (EAR)
        MediaPipe uses 6 points per eye: top, bottom, left_inner, left_outer, right_inner, right_outer
        
        Args:
            eye_points: Array of 6 eye landmark points
            
        Returns:
            EAR value
        """
        # For MediaPipe 6-point eye model:
        # Points: [33/362, 160/385, 158/387, 133/263, 153/373, 144/380]
        # Top: 160/385, Bottom: 133/263, Left: 158/387, Right: 153/373
        
        # Vertical distances
        vertical_1 = distance.euclidean(eye_points, eye_points)
        vertical_2 = distance.euclidean(eye_points, eye_points)
        
        # Horizontal distance
        horizontal = distance.euclidean(eye_points, eye_points)
        
        # EAR formula
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def mouth_aspect_ratio(self, mouth_points):
        """
        Calculate Mouth Aspect Ratio (MAR) for yawn detection
        
        Args:
            mouth_points: Array of mouth landmark points
            
        Returns:
            MAR value
        """
        # Vertical distances
        vertical_1 = distance.euclidean(mouth_points, mouth_points)
        vertical_2 = distance.euclidean(mouth_points, mouth_points)
        vertical_3 = distance.euclidean(mouth_points, mouth_points)
        
        # Horizontal distance
        horizontal = distance.euclidean(mouth_points, mouth_points)
        
        # MAR formula
        mar = (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)
        return mar
    
    def detect(self, face_landmarks):
        """
        Detect drowsiness indicators
        
        Args:
            face_landmarks: Dictionary with eye and mouth landmarks
            
        Returns:
            Dictionary with detection results
        """
        results = {
            'drowsy': False,
            'yawning': False,
            'ear': 0.0,
            'mar': 0.0,
            'left_ear': 0.0,
            'right_ear': 0.0,
            'perclos': 0.0,
            'blink_count': self.blink_counter
        }
        
        if face_landmarks is None:
            return results
        
        # Extract eye and mouth points
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        mouth = face_landmarks['mouth']
        
        if len(left_eye) < 6 or len(right_eye) < 6 or len(mouth) < 10:
            return results
        
        # Calculate EAR for both eyes
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        results['left_ear'] = left_ear
        results['right_ear'] = right_ear
        results['ear'] = ear
        
        # Track eye closure
        if ear < self.ear_threshold:
            self.eye_closed_counter += 1
        else:
            if self.eye_closed_counter > 0:
                self.blink_counter += 1
            self.eye_closed_counter = 0
        
        # Add to PERCLOS history
        eyes_closed = 1 if ear < self.ear_threshold else 0
        self.perclos_history.append(eyes_closed)
        
        # Keep history window
        if len(self.perclos_history) > self.perclos_window:
            self.perclos_history.pop(0)
        
        # Calculate PERCLOS
        if len(self.perclos_history) > 0:
            perclos = sum(self.perclos_history) / len(self.perclos_history)
            results['perclos'] = perclos
            
            # Check PERCLOS threshold
            if perclos > self.perclos_threshold:
                results['drowsy'] = True
        
        # Check consecutive eye closure
        if self.eye_closed_counter >= self.eye_closed_frames:
            results['drowsy'] = True
        
        # Detect yawning
        mar = self.mouth_aspect_ratio(mouth)
        results['mar'] = mar
        
        if mar > 0.6:  # Yawn threshold
            results['yawning'] = True
        
        return results
    
    def reset(self):
        """Reset detection state"""
        self.eye_closed_counter = 0
        self.perclos_history = []
