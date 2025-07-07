#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path

def create_realistic_test_video():
    output_path = Path("app/static/sample_videos/realistic_test.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    width, height = 854, 480
    fps = 30
    duration = 5
    total_frames = fps * duration
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    try:
        for frame_num in range(total_frames):
            # Green field
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (34, 139, 34)
            
            # Field markings
            cv2.rectangle(frame, (50, 50), (width-50, height-50), (255, 255, 255), 2)
            cv2.line(frame, (width//2, 50), (width//2, height-50), (255, 255, 255), 2)
            cv2.circle(frame, (width//2, height//2), 50, (255, 255, 255), 2)
            
            time_factor = frame_num / total_frames
            
            # Person-like shape 1
            p1_x = int(150 + 300 * np.sin(time_factor * 2 * np.pi))
            p1_y = int(height//2 + 80 * np.cos(time_factor * np.pi))
            
            # Head, body, legs (more person-like)
            cv2.circle(frame, (p1_x, p1_y - 40), 15, (255, 180, 120), -1)
            cv2.rectangle(frame, (p1_x-20, p1_y-25), (p1_x+20, p1_y+30), (0, 0, 255), -1)
            cv2.rectangle(frame, (p1_x-15, p1_y+30), (p1_x-5, p1_y+60), (0, 0, 128), -1)
            cv2.rectangle(frame, (p1_x+5, p1_y+30), (p1_x+15, p1_y+60), (0, 0, 128), -1)
            cv2.rectangle(frame, (p1_x-35, p1_y-15), (p1_x-20, p1_y+15), (255, 180, 120), -1)
            cv2.rectangle(frame, (p1_x+20, p1_y-15), (p1_x+35, p1_y+15), (255, 180, 120), -1)
            
            # Person 2
            p2_x = int(width - 200 - 200 * np.sin(time_factor * 1.5 * np.pi))
            p2_y = int(height//2 + 60 * np.sin(time_factor * 3 * np.pi))
            
            cv2.circle(frame, (p2_x, p2_y - 40), 15, (255, 180, 120), -1)
            cv2.rectangle(frame, (p2_x-20, p2_y-25), (p2_x+20, p2_y+30), (255, 255, 0), -1)
            cv2.rectangle(frame, (p2_x-15, p2_y+30), (p2_x-5, p2_y+60), (0, 0, 128), -1)
            cv2.rectangle(frame, (p2_x+5, p2_y+30), (p2_x+15, p2_y+60), (0, 0, 128), -1)
            cv2.rectangle(frame, (p2_x-35, p2_y-15), (p2_x-20, p2_y+15), (255, 180, 120), -1)
            cv2.rectangle(frame, (p2_x+20, p2_y-15), (p2_x+35, p2_y+15), (255, 180, 120), -1)
            
            # Ball
            ball_x = int(width//2 + 150 * np.sin(time_factor * 4 * np.pi))
            ball_y = int(height//2 + 30 * np.cos(time_factor * 4 * np.pi))
            cv2.circle(frame, (ball_x, ball_y), 12, (255, 255, 255), -1)
            cv2.circle(frame, (ball_x, ball_y), 12, (0, 0, 0), 2)
            
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"✅ Realistic test video created: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"✗ Failed to create video: {e}")
        return None

if __name__ == "__main__":
    create_realistic_test_video()
