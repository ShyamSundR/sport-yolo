import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class SportsDetector:
    """YOLO-based sports object detector optimized for football/soccer"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize the sports detector
        
        Args:
            model_path: Path to YOLO model weights
        """
        try:
            # Load YOLO model
            self.model = YOLO(model_path)
            
            # Sports-relevant classes from COCO dataset
            self.sports_classes = {
                0: 'person',      # Players, referees, coaches
                32: 'sports ball', # Football/soccer ball
                36: 'skis',       # Sometimes detected as sports equipment
                37: 'snowboard',  # Sometimes confused with other sports gear
            }
            
            # Football-specific confidence thresholds
            self.confidence_threshold = 0.5
            self.iou_threshold = 0.45
            
            # Track performance metrics
            self.total_inferences = 0
            self.total_inference_time = 0.0
            
            logger.info(f"YOLO model loaded: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_frame(self, frame: np.ndarray) -> Dict:
        """
        Detect objects in a single frame
        
        Args:
            frame: Input image as numpy array
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        try:
            # Run YOLO inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Extract detections
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        # Filter for sports-relevant classes
                        if cls_id in self.sports_classes or cls_id == 0:  # Include all persons
                            x1, y1, x2, y2 = box
                            
                            detection = {
                                'id': i,
                                'class': self.model.names[cls_id],
                                'class_id': int(cls_id),
                                'confidence': float(conf),
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                                'area': float((x2 - x1) * (y2 - y1))
                            }
                            
                            # Add sports-specific metadata
                            if cls_id == 0:  # Person
                                detection['sports_role'] = self._classify_person_role(detection)
                            elif cls_id == 32:  # Sports ball
                                detection['sports_role'] = 'ball'
                                
                            detections.append(detection)
            
            inference_time = time.time() - start_time
            
            # Update metrics
            self.total_inferences += 1
            self.total_inference_time += inference_time
            
            return {
                'detections': detections,
                'inference_time': inference_time,
                'frame_shape': frame.shape,
                'model_confidence': self.confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {
                'detections': [],
                'inference_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _classify_person_role(self, detection: Dict) -> str:
        """
        Classify person role based on detection characteristics
        
        Args:
            detection: Person detection dictionary
            
        Returns:
            Estimated role (player, referee, coach, spectator)
        """
        # Simple heuristics for role classification
        # In a real system, you'd use more sophisticated methods
        
        bbox = detection['bbox']
        area = detection['area']
        
        # Assume larger persons in central field positions are players
        # This is a simplified approach - real systems use jersey colors, position tracking, etc.
        
        if area > 5000:  # Large bounding box
            return 'player'
        elif area > 2000:
            return 'referee_or_coach'
        else:
            return 'spectator_or_other'
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            
        Returns:
            Frame with drawn detections
        """
        annotated_frame = frame.copy()
        
        # Define colors for different classes
        colors = {
            'person': (0, 255, 0),      # Green for persons
            'sports ball': (0, 0, 255), # Red for ball
            'default': (255, 255, 0)    # Yellow for others
        }
        
        for det in detections:
            bbox = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            color = colors.get(class_name, colors['default'])
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            if 'sports_role' in det:
                label += f" ({det['sports_role']})"
            
            # Calculate label position
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_y = max(y1 - 10, label_size[1])
            
            # Draw label background
            cv2.rectangle(
                annotated_frame,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + 5),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return annotated_frame
    
    def get_model_info(self) -> Dict:
        """Get model information and performance metrics"""
        avg_inference_time = (
            self.total_inference_time / self.total_inferences 
            if self.total_inferences > 0 else 0
        )
        
        return {
            'model_type': 'YOLOv8',
            'total_inferences': self.total_inferences,
            'average_inference_time': avg_inference_time,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'sports_classes': self.sports_classes,
            'fps_estimate': 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        }
