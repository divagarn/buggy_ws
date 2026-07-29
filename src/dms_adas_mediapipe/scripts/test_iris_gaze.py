#!/usr/bin/env python3
"""
Test script for iris-based gaze detection
Tests the IrisGazeDetector with sample landmark data
"""

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import numpy as np
from utils.iris_gaze_detector import IrisGazeDetector


def create_test_landmarks():
    """Create synthetic test landmarks for left eye, right eye, and iris"""
    # Simulate left eye (6 points) - positioned at (100, 100)
    left_eye = np.array([
        [80, 100, 0],   # Left corner
        [85, 95, 0],    # Top left
        [90, 93, 0],    # Top middle
        [100, 100, 0],  # Right corner
        [90, 107, 0],   # Bottom middle
        [85, 105, 0],   # Bottom left
    ], dtype=np.float32)
    
    # Simulate right eye (6 points) - positioned at (200, 100)
    right_eye = np.array([
        [180, 100, 0],  # Left corner
        [185, 95, 0],   # Top left
        [190, 93, 0],   # Top middle
        [200, 100, 0],  # Right corner
        [190, 107, 0],  # Bottom middle
        [185, 105, 0],  # Bottom left
    ], dtype=np.float32)
    
    # Test different iris positions
    test_cases = {
        'center': {
            'left_iris': np.array([
                [90, 100, 0],   # Center
                [95, 100, 0],   # Right
                [90, 95, 0],    # Top
                [85, 100, 0],   # Left
                [90, 105, 0],   # Bottom
            ], dtype=np.float32),
            'right_iris': np.array([
                [190, 100, 0],  # Center
                [195, 100, 0],  # Right
                [190, 95, 0],   # Top
                [185, 100, 0],  # Left
                [190, 105, 0],  # Bottom
            ], dtype=np.float32),
        },
        'looking_left': {
            'left_iris': np.array([
                [85, 100, 0],   # Center (shifted left)
                [90, 100, 0],   # Right
                [85, 95, 0],    # Top
                [80, 100, 0],   # Left
                [85, 105, 0],   # Bottom
            ], dtype=np.float32),
            'right_iris': np.array([
                [185, 100, 0],  # Center (shifted left)
                [190, 100, 0],  # Right
                [185, 95, 0],   # Top
                [180, 100, 0],  # Left
                [185, 105, 0],  # Bottom
            ], dtype=np.float32),
        },
        'looking_right': {
            'left_iris': np.array([
                [95, 100, 0],   # Center (shifted right)
                [100, 100, 0],  # Right
                [95, 95, 0],    # Top
                [90, 100, 0],   # Left
                [95, 105, 0],   # Bottom
            ], dtype=np.float32),
            'right_iris': np.array([
                [195, 100, 0],  # Center (shifted right)
                [200, 100, 0],  # Right
                [195, 95, 0],   # Top
                [190, 100, 0],  # Left
                [195, 105, 0],  # Bottom
            ], dtype=np.float32),
        },
        'looking_up': {
            'left_iris': np.array([
                [90, 97, 0],    # Center (shifted up)
                [95, 97, 0],    # Right
                [90, 92, 0],    # Top
                [85, 97, 0],    # Left
                [90, 102, 0],   # Bottom
            ], dtype=np.float32),
            'right_iris': np.array([
                [190, 97, 0],   # Center (shifted up)
                [195, 97, 0],   # Right
                [190, 92, 0],   # Top
                [185, 97, 0],   # Left
                [190, 102, 0],  # Bottom
            ], dtype=np.float32),
        },
        'looking_down': {
            'left_iris': np.array([
                [90, 103, 0],   # Center (shifted down)
                [95, 103, 0],   # Right
                [90, 98, 0],    # Top
                [85, 103, 0],   # Left
                [90, 108, 0],   # Bottom
            ], dtype=np.float32),
            'right_iris': np.array([
                [190, 103, 0],  # Center (shifted down)
                [195, 103, 0],  # Right
                [190, 98, 0],   # Top
                [185, 103, 0],  # Left
                [190, 108, 0],  # Bottom
            ], dtype=np.float32),
        },
    }
    
    return left_eye, right_eye, test_cases


def test_gaze_detector():
    """Test the iris gaze detector with various eye positions"""
    print("=" * 60)
    print("Testing Iris-Based Gaze Detector")
    print("=" * 60)
    
    # Initialize detector
    detector = IrisGazeDetector(smoothing_alpha=0.0)  # No smoothing for testing
    
    # Create test data
    left_eye, right_eye, test_cases = create_test_landmarks()
    
    # Test each case
    for case_name, iris_data in test_cases.items():
        print(f"\n--- Test Case: {case_name.upper().replace('_', ' ')} ---")
        
        landmarks_dict = {
            'left_eye': left_eye,
            'right_eye': right_eye,
            'left_iris': iris_data['left_iris'],
            'right_iris': iris_data['right_iris'],
        }
        
        # Detect gaze
        results = detector.detect_gaze(landmarks_dict)
        
        # Print results
        print(f"Gaze Detected: {results['gaze_detected']}")
        print(f"Looking Forward: {results['looking_forward']}")
        print(f"Horizontal Direction: {results['horizontal_direction']}")
        print(f"Vertical Direction: {results['vertical_direction']}")
        print(f"Gaze Angle: {results['gaze_angle']:.1f}°")
        
        if results['left_gaze_ratio']:
            h_ratio, v_ratio = results['left_gaze_ratio']
            print(f"Left Eye - H: {h_ratio:.3f}, V: {v_ratio:.3f}")
        
        if results['right_gaze_ratio']:
            h_ratio, v_ratio = results['right_gaze_ratio']
            print(f"Right Eye - H: {h_ratio:.3f}, V: {v_ratio:.3f}")
        
        if results['combined_gaze_ratio']:
            h_ratio, v_ratio = results['combined_gaze_ratio']
            print(f"Combined - H: {h_ratio:.3f}, V: {v_ratio:.3f}")
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)


if __name__ == '__main__':
    test_gaze_detector()
