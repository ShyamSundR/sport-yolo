#!/usr/bin/env python3
"""
Download and setup YOLO models for sports detection
"""

import os
import sys
import requests
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

def download_yolo_models():
    """Download YOLO models"""
    print("🤖 Downloading YOLO models...")
    
    models = [
        "yolov8n.pt",  # Nano - fastest
        "yolov8s.pt",  # Small - balanced
    ]
    
    for model_name in models:
        try:
            print(f"  Downloading {model_name}...")
            model = YOLO(model_name)
            print(f"  ✓ {model_name} downloaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to download {model_name}: {e}")

def download_sample_videos():
    """Download sample sports videos for testing"""
    print("📹 Downloading sample sports videos...")
    
    # Create sample videos directory
    sample_dir = Path("app/static/sample_videos")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample video URLs (Creative Commons / Public Domain)
    videos = [
        {
            "name": "football_sample.mp4",
            "url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
            "description": "Sample football video for testing"
        }
    ]
    
    for video in videos:
        video_path = sample_dir / video["name"]
        
        if video_path.exists():
            print(f"  ✓ {video['name']} already exists")
            continue
            
        try:
            print(f"  Downloading {video['name']}...")
            response = requests.get(video["url"], stream=True)
            response.raise_for_status()
            
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"  ✓ {video['name']} downloaded successfully")
            
        except Exception as e:
            print(f"  ✗ Failed to download {video['name']}: {e}")

def create_synthetic_football_video():
    """Create a simple synthetic football video for testing"""
    print("⚽ Creating synthetic football video...")
    
    output_path = Path("app/static/sample_videos/synthetic_football.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Video parameters
    width, height = 854, 480
    fps = 30
    duration = 10  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    try:
        for frame_num in range(total_frames):
            # Create green football field background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (34, 139, 34)  # Forest green for field
            
            # Draw field lines
            cv2.rectangle(frame, (50, 50), (width-50, height-50), (255, 255, 255), 2)
            cv2.line(frame, (width//2, 50), (width//2, height-50), (255, 255, 255), 2)
            cv2.circle(frame, (width//2, height//2), 50, (255, 255, 255), 2)
            
            # Animate players (simple moving rectangles)
            time_factor = frame_num / total_frames
            
            # Player 1 (red team)
            player1_x = int(100 + 200 * np.sin(time_factor * 4 * np.pi))
            player1_y = int(height//2 + 50 * np.cos(time_factor * 2 * np.pi))
            cv2.rectangle(frame, (player1_x-15, player1_y-25), (player1_x+15, player1_y+25), (0, 0, 255), -1)
            
            # Player 2 (blue team)  
            player2_x = int(width - 150 - 150 * np.sin(time_factor * 3 * np.pi))
            player2_y = int(height//2 + 30 * np.sin(time_factor * 5 * np.pi))
            cv2.rectangle(frame, (player2_x-15, player2_y-25), (player2_x+15, player2_y+25), (255, 0, 0), -1)
            
            # Ball (moving between players)
            ball_x = int(width//2 + 100 * np.sin(time_factor * 6 * np.pi))
            ball_y = int(height//2 + 20 * np.cos(time_factor * 6 * np.pi))
            cv2.circle(frame, (ball_x, ball_y), 8, (255, 255, 255), -1)
            cv2.circle(frame, (ball_x, ball_y), 8, (0, 0, 0), 2)
            
            # Add frame number
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"  ✓ Synthetic football video created: {output_path}")
        
    except Exception as e:
        print(f"  ✗ Failed to create synthetic video: {e}")
        if out:
            out.release()

def test_model_loading():
    """Test that models can be loaded properly"""
    print("🧪 Testing model loading...")
    
    try:
        from app.models.yolo_detector import SportsDetector
        
        # Test model initialization
        detector = SportsDetector()
        print("  ✓ SportsDetector initialized successfully")
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test detection
        results = detector.detect_frame(test_image)
        print(f"  ✓ Detection test completed: {len(results['detections'])} detections")
        
        # Test model info
        info = detector.get_model_info()
        print(f"  ✓ Model info retrieved: {info['model_type']}")
        
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False
    
    return True

def setup_environment():
    """Setup environment and directories"""
    print("📁 Setting up directories...")
    
    directories = [
        "app/static/uploads",
        "app/static/outputs", 
        "app/static/sample_videos",
        "models",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")

def main():
    """Main setup function"""
    print("🚀 Sports Video MLOps - Model Setup")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Download models
    download_yolo_models()
    
    # Create sample content
    create_synthetic_football_video()
    download_sample_videos()
    
    # Test everything works
    if test_model_loading():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: docker-compose up")
        print("2. Test API: curl http://localhost:8000/health")
        print("3. Upload test video via API")
    else:
        print("\n⚠️  Setup completed with issues")
        print("Some components may not work correctly")
        sys.exit(1)

if __name__ == "__main__":
    main()
