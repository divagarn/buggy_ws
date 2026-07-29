#!/usr/bin/env python3
"""
Phone Detection using YOLOv8 with Tracking
"""

import cv2
import os
import numpy as np
import urllib.request

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class PhoneDetector:
    def __init__(self, model_path='/home/orin/demo_ws/src/dms_adas_mediapipe/models/yolov8n.engine', confidence_threshold=0.4):
        """
        Initialize phone detector with tracking
        
        Args:
            model_path: Path to YOLOv8 model (supports .pt, .engine/.trt for TensorRT)
            confidence_threshold: Minimum confidence for detection
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_path = model_path
        
        # Tracking variables
        self.tracker = None
        self.tracking_active = False
        self.tracked_bbox = None
        self.frames_since_detection = 0
        self.max_frames_without_detection = 15  # Stop tracking after 15 frames without detection
        self.detection_history = []  # Store recent detections for smoothing
        self.history_size = 5  # Number of frames to keep in history
        self.iou_threshold = 0.5  # IoU threshold for matching detections
        
        # Detection stability counters
        self.consecutive_detections = 0
        self.required_detections = 3  # Require 3 consecutive detections before confirming
        
        # Check if the model exists, if not, download it
        if not os.path.exists(self.model_path):
            print(f"Model file not found at {self.model_path}. Attempting to download...")
            self.download_model()
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
    
    def download_model(self):
        """
        Download the YOLOv8 model if not available locally.
        For TensorRT models, you need to export them yourself.
        This will download the .pt model which can then be exported.
        """
        try:
            # If looking for .engine file but not found, try to download .pt version
            if self.model_path.endswith('.engine') or self.model_path.endswith('.trt'):
                pt_path = self.model_path.replace('.engine', '.pt').replace('.trt', '.pt')
                if not os.path.exists(pt_path):
                    print(f"Downloading PyTorch model to {pt_path}...")
                    # Ultralytics will auto-download when you instantiate with a model name
                    temp_model = YOLO('yolov8n.pt')
                    print(f"Model downloaded. Please export to TensorRT using:")
                    print(f"  yolo export model={pt_path} format=engine device=0")
                else:
                    print(f"PyTorch model found at {pt_path}. Please export to TensorRT using:")
                    print(f"  yolo export model={pt_path} format=engine device=0")
            else:
                # For .pt files, ultralytics will auto-download
                print(f"Downloading model...")
                self.model_path = 'yolov8n.pt'  # Let ultralytics handle the download
                
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please manually download or export the model.")
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def smooth_bbox(self, new_bbox):
        """Smooth bounding box using history of recent detections"""
        self.detection_history.append(new_bbox)
        
        # Keep only recent history
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # Average the bounding boxes
        if len(self.detection_history) == 0:
            return new_bbox
        
        avg_x1 = int(np.mean([bbox[0] for bbox in self.detection_history]))
        avg_y1 = int(np.mean([bbox[1] for bbox in self.detection_history]))
        avg_x2 = int(np.mean([bbox[2] for bbox in self.detection_history]))
        avg_y2 = int(np.mean([bbox[3] for bbox in self.detection_history]))
        
        return (avg_x1, avg_y1, avg_x2, avg_y2)
    
    def init_tracker(self, frame, bbox):
        """Initialize OpenCV tracker with detected phone bbox"""
        try:
            # Use CSRT tracker (accurate but slower) or KCF (faster but less accurate)
            # For Jetson, KCF might be better for performance
            self.tracker = cv2.TrackerKCF_create()
            
            # Convert bbox format from (x1, y1, x2, y2) to (x, y, w, h)
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            self.tracker.init(frame, (x, y, w, h))
            self.tracking_active = True
            self.tracked_bbox = bbox
            return True
        except Exception as e:
            print(f"Error initializing tracker: {e}")
            self.tracking_active = False
            return False
    
    def update_tracker(self, frame):
        """Update tracker with new frame"""
        if not self.tracking_active or self.tracker is None:
            return None
        
        try:
            success, bbox_xywh = self.tracker.update(frame)
            
            if success:
                # Convert from (x, y, w, h) to (x1, y1, x2, y2)
                x, y, w, h = bbox_xywh
                tracked_bbox = (int(x), int(y), int(x + w), int(y + h))
                self.tracked_bbox = tracked_bbox
                return tracked_bbox
            else:
                return None
        except Exception as e:
            print(f"Error updating tracker: {e}")
            return None
    
    def reset_tracker(self):
        """Reset tracker state"""
        self.tracker = None
        self.tracking_active = False
        self.tracked_bbox = None
        self.frames_since_detection = 0
        self.detection_history = []
        self.consecutive_detections = 0
    
    def detect(self, frame):
        """
        Detect phone in frame with tracking support
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary with detection results including tracking info
        """
        results = {
            'phone_detected': False,
            'confidence': 0.0,
            'bbox': None,
            'detections': [],
            'tracking': False,
            'stable': False  # True only after consecutive detections
        }
        
        if self.model is None:
            return results
        
        detected_in_frame = False
        best_detection = None
        best_confidence = 0.0
        
        try:
            # Run YOLO inference
            detections = self.model(frame, verbose=False)
            
            # Process results
            for detection in detections:
                boxes = detection.boxes
                
                for box in boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    
                    # Class 67 is 'cell phone' in COCO
                    if class_id == 67 and confidence >= self.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        
                        detected_in_frame = True
                        
                        # Keep track of best detection
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_detection = {
                                'class_id': class_id,
                                'confidence': confidence,
                                'bbox': bbox
                            }
                        
                        results['detections'].append({
                            'class_id': class_id,
                            'confidence': confidence,
                            'bbox': bbox
                        })
            
            # Handle detection and tracking logic
            if detected_in_frame and best_detection:
                # Reset frames counter
                self.frames_since_detection = 0
                
                # Check if this detection is consistent with tracking
                if self.tracking_active and self.tracked_bbox:
                    iou = self.calculate_iou(best_detection['bbox'], self.tracked_bbox)
                    
                    if iou > self.iou_threshold:
                        # Good match with tracker - update tracker
                        self.consecutive_detections += 1
                        smoothed_bbox = self.smooth_bbox(best_detection['bbox'])
                        self.init_tracker(frame, smoothed_bbox)
                        
                        results['phone_detected'] = True
                        results['confidence'] = best_confidence
                        results['bbox'] = smoothed_bbox
                        results['tracking'] = True
                        results['stable'] = self.consecutive_detections >= self.required_detections
                    else:
                        # Detection doesn't match tracker - reset and start new tracking
                        self.reset_tracker()
                        self.consecutive_detections = 1
                        self.init_tracker(frame, best_detection['bbox'])
                        
                        # Don't report as detected until stable
                        results['phone_detected'] = False
                        results['tracking'] = True
                else:
                    # No active tracking - start new tracking
                    self.consecutive_detections += 1
                    self.init_tracker(frame, best_detection['bbox'])
                    
                    if self.consecutive_detections >= self.required_detections:
                        smoothed_bbox = self.smooth_bbox(best_detection['bbox'])
                        results['phone_detected'] = True
                        results['confidence'] = best_confidence
                        results['bbox'] = smoothed_bbox
                        results['tracking'] = True
                        results['stable'] = True
                    else:
                        # Not stable yet
                        results['phone_detected'] = False
                        results['tracking'] = True
                        results['stable'] = False
            
            else:
                # No detection in this frame
                self.frames_since_detection += 1
                
                # Try to use tracker if still active
                if self.tracking_active and self.frames_since_detection < self.max_frames_without_detection:
                    tracked_bbox = self.update_tracker(frame)
                    
                    if tracked_bbox:
                        # Tracker still working - continue reporting detection
                        results['phone_detected'] = True
                        results['confidence'] = best_confidence if best_confidence > 0 else 0.5
                        results['bbox'] = tracked_bbox
                        results['tracking'] = True
                        results['stable'] = True
                    else:
                        # Tracker lost - reset
                        self.reset_tracker()
                else:
                    # Too many frames without detection or no active tracking
                    self.reset_tracker()
                    
        except Exception as e:
            print(f"Error in phone detection: {e}")
            self.reset_tracker()
        
        return results
    
    def draw_detections(self, frame, results):
        """Draw detection boxes on frame with tracking status"""
        if not results['phone_detected'] and not results['tracking']:
            return frame
        
        # Draw the main tracked/detected phone
        if results['bbox']:
            bbox = results['bbox']
            confidence = results['confidence']
            
            # Color based on tracking status
            if results['stable']:
                color = (0, 255, 0)  # Green for stable tracking
                status = "TRACKED"
            elif results['tracking']:
                color = (0, 255, 255)  # Yellow for unstable tracking
                status = "DETECTING..."
            else:
                color = (0, 0, 255)  # Red for new detection
                status = "DETECTED"
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         color, 2)
            
            # Draw label with confidence and status
            label = f"Phone {status}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Background for text
            cv2.rectangle(frame, 
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]),
                         color, -1)
            
            cv2.putText(frame, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw center point
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            cv2.circle(frame, (center_x, center_y), 4, color, -1)
        
        return frame
