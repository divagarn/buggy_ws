#!/usr/bin/env python3
"""
Phone Detection using YOLOv8
"""

import cv2
import numpy as np
import traceback
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class PhoneDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.4):
        """
        Initialize phone detector
        
        Args:
            model_path: Path to YOLOv8 model
            confidence_threshold: Minimum confidence for detection
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
    
    def detect(self, frame):
        """
        Detect phone in frame
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary with detection results
        """
        results = {
            'phone_detected': False,
            'confidence': 0.0,
            'bbox': None,
            'detections': []
        }
        
        if self.model is None:
            return results
        
        try:
            # Run inference
            detections = self.model(frame, verbose=False)

            # Process results
            for detection in detections:
                boxes = getattr(detection, 'boxes', [])

                for box in boxes:
                    try:
                        # box.cls and box.conf may be tensors or scalars
                        class_id = int(getattr(box, 'cls', 0))
                        confidence = float(getattr(box, 'conf', 0.0))

                        # Class 67 is 'cell phone' in COCO
                        if class_id == 67 and confidence >= self.confidence_threshold:
                            coords = getattr(box, 'xyxy', None)
                            if coords is None:
                                continue

                            # Convert to numpy and flatten if needed
                            try:
                                arr = coords.cpu().numpy()
                            except Exception:
                                try:
                                    arr = np.array(coords)
                                except Exception:
                                    arr = None

                            if arr is None:
                                continue

                            if arr.ndim == 2 and arr.shape[0] == 1:
                                vals = arr[0]
                            else:
                                vals = arr.ravel()

                            if vals.size < 4:
                                continue

                            x1, y1, x2, y2 = vals[:4]
                            bbox = (int(x1), int(y1), int(x2), int(y2))

                            results['phone_detected'] = True
                            results['confidence'] = confidence
                            results['bbox'] = bbox
                            results['detections'].append({
                                'class_id': class_id,
                                'confidence': confidence,
                                'bbox': bbox
                            })
                    except Exception:
                        # protect inner loop from bad boxes
                        traceback.print_exc()
                        continue
        except Exception as e:
            print(f"Error in phone detection: {e}")
            traceback.print_exc()
        
        return results
    
    def draw_detections(self, frame, results):
        """Draw detection boxes on frame"""
        if not results['phone_detected']:
            return frame
        
        for detection in results['detections']:
            bbox = detection['bbox']
            confidence = detection['confidence']
            try:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"Phone: {confidence:.2f}"
                cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            except Exception:
                # fallback: ignore bad bbox
                continue
        
        return frame
