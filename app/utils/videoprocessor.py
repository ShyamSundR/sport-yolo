import cv2
import numpy as np
import asyncio
import time
from typing import Dict, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Process videos for sports object detection"""
    
    def __init__(self, detector):
        """
        Initialize video processor
        
        Args:
            detector: SportsDetector instance
        """
        self.detector = detector
        self.frame_skip = 1  # Process every frame (set to 2 for every other frame)
        
    async def process_video(self, input_path: str, output_path: str) -> Dict:
        """
        Process entire video file
        
        Args:
            input_path: Path to input video
            output_path: Path to save processed video
            
        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        
        try:
            # Open video
            cap = cv2.VideoCapture(input_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Processing metrics
            processed_frames = 0
            total_detections = 0
            frame_times = []
            
            # Process frames
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed (for performance)
                if frame_num % self.frame_skip != 0:
                    out.write(frame)
                    frame_num += 1
                    continue
                
                # Process frame
                frame_start = time.time()
                results = self.detector.detect_frame(frame)
                frame_time = time.time() - frame_start
                
                # Draw detections
                annotated_frame = self.detector.draw_detections(frame, results['detections'])
                
                # Add frame info overlay
                annotated_frame = self._add_frame_overlay(
                    annotated_frame, 
                    frame_num, 
                    len(results['detections']),
                    frame_time
                )
                
                # Write frame
                out.write(annotated_frame)
                
                # Update metrics
                processed_frames += 1
                total_detections += len(results['detections'])
                frame_times.append(frame_time)
                
                # Log progress every 100 frames
                if processed_frames % 100 == 0:
                    progress = (frame_num / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_num}/{total_frames})")
                
                frame_num += 1
                
                # Allow other tasks to run
                if processed_frames % 10 == 0:
                    await asyncio.sleep(0.001)
            
            # Cleanup
            cap.release()
            out.release()
            
            processing_time = time.time() - start_time
            
            # Calculate final metrics
            avg_frame_time = np.mean(frame_times) if frame_times else 0
            avg_detections = total_detections / processed_frames if processed_frames > 0 else 0
            
            results = {
                'input_path': input_path,
                'output_path': output_path,
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'processing_time': processing_time,
                'avg_frame_time': avg_frame_time,
                'fps_processed': processed_frames / processing_time if processing_time > 0 else 0,
                'total_detections': total_detections,
                'avg_detections_per_frame': avg_detections,
                'video_properties': {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'duration_seconds': total_frames / fps if fps > 0 else 0
                }
            }
            
            logger.info(
                "Video processing completed",
                **{k: v for k, v in results.items() if k not in ['input_path', 'output_path']}
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise
    
    def _add_frame_overlay(self, frame: np.ndarray, frame_num: int, 
                          detection_count: int, frame_time: float) -> np.ndarray:
        """
        Add information overlay to frame
        
        Args:
            frame: Input frame
            frame_num: Current frame number
            detection_count: Number of detections in frame
            frame_time: Processing time for this frame
            
        Returns:
            Frame with overlay
        """
        overlay_frame = frame.copy()
        
        # Overlay background
        overlay_height = 80
        overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)
        overlay[:] = (0, 0, 0)  # Black background
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)  # White text
        thickness = 2
        
        # Frame info
        text1 = f"Frame: {frame_num}"
        text2 = f"Detections: {detection_count}"
        text3 = f"Process Time: {frame_time:.3f}s"
        text4 = f"FPS: {1.0/frame_time:.1f}" if frame_time > 0 else "FPS: --"
        
        # Position text
        cv2.putText(overlay, text1, (10, 20), font, font_scale, color, thickness)
        cv2.putText(overlay, text2, (10, 45), font, font_scale, color, thickness)
        cv2.putText(overlay, text3, (250, 20), font, font_scale, color, thickness)
        cv2.putText(overlay, text4, (250, 45), font, font_scale, color, thickness)
        
        # Blend overlay with frame
        overlay_frame[:overlay_height] = cv2.addWeighted(
            overlay_frame[:overlay_height], 0.7, overlay, 0.3, 0
        )
        
        return overlay_frame
    
    async def process_webcam_stream(self, camera_id: int = 0):
        """
        Process live webcam stream (for testing)
        
        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = self.detector.detect_frame(frame)
                
                # Draw detections
                annotated_frame = self.detector.draw_detections(frame, results['detections'])
                
                # Add overlay
                annotated_frame = self._add_frame_overlay(
                    annotated_frame, 
                    0,  # Frame counter for live stream
                    len(results['detections']),
                    results['inference_time']
                )
                
                # Display frame (for local testing)
                cv2.imshow('Sports Detection', annotated_frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                await asyncio.sleep(0.01)  # Small delay
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
