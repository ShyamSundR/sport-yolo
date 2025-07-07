import pytest
import asyncio
from fastapi.testclient import TestClient
import numpy as np
import cv2
import io
from PIL import Image

from app.main import app

client = TestClient(app)

class TestSportsVideoAPI:
    """Test suite for Sports Video API"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Sports Video Analytics API" in data["message"]
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data
        assert len(data["endpoints"]) > 0
    
    def create_test_image(self, width=640, height=480):
        """Create a test image for detection"""
        # Create a simple test image with some shapes
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some colored rectangles (simulating people)
        cv2.rectangle(image, (100, 100), (200, 300), (0, 255, 0), -1)  # Green rectangle
        cv2.rectangle(image, (400, 150), (500, 350), (255, 0, 0), -1)  # Blue rectangle
        
        # Add a circle (simulating ball)
        cv2.circle(image, (320, 240), 20, (255, 255, 255), -1)
        
        return image
    
    def image_to_bytes(self, image):
        """Convert numpy image to bytes for upload"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        byte_arr = io.BytesIO()
        pil_image.save(byte_arr, format='JPEG')
        byte_arr.seek(0)
        
        return byte_arr.getvalue()
    
    def test_image_detection(self):
        """Test image detection endpoint"""
        # Create test image
        test_image = self.create_test_image()
        image_bytes = self.image_to_bytes(test_image)
        
        # Upload image for detection
        files = {"file": ("test_image.jpg", image_bytes, "image/jpeg")}
        response = client.post("/detect/image", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "filename" in data
        assert "inference_time" in data
        assert "detections" in data
        assert "detection_count" in data
        assert isinstance(data["detections"], list)
        assert isinstance(data["detection_count"], int)
        assert data["detection_count"] >= 0
    
    def test_image_detection_invalid_file(self):
        """Test image detection with invalid file"""
        # Upload invalid file
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/detect/image", files=files)
        
        # Should still process but may have different results
        # The endpoint should handle this gracefully
        assert response.status_code in [200, 422, 500]
    
    def create_test_video(self, width=640, height=480, fps=30, duration=2):
        """Create a simple test video"""
        frames = fps * duration
        
        # Create temporary video file
        output_path = "/tmp/test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i in range(frames):
            # Create frame with moving objects
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Moving rectangle (person)
            x = int(100 + 200 * (i / frames))
            cv2.rectangle(frame, (x, 200), (x+50, 300), (0, 255, 0), -1)
            
            # Moving circle (ball)
            ball_x = int(300 + 100 * np.sin(i * 0.2))
            cv2.circle(frame, (ball_x, 240), 15, (255, 255, 255), -1)
            
            out.write(frame)
        
        out.release()
        return output_path
    
    def test_video_upload(self):
        """Test video upload endpoint"""
        # Create test video
        video_path = self.create_test_video()
        
        try:
            # Read video file
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            
            # Upload video
            files = {"file": ("test_video.mp4", video_bytes, "video/mp4")}
            response = client.post("/detect/video", files=files)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "message" in data
            assert "job_id" in data
            assert "status_url" in data
            assert isinstance(data["job_id"], int)
            
            # Test status endpoint
            job_id = data["job_id"]
            status_response = client.get(f"/status/{job_id}")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            assert "job_id" in status_data
            assert "status" in status_data
            assert status_data["job_id"] == job_id
            
        finally:
            # Cleanup
            import os
            if os.path.exists(video_path):
                os.remove(video_path)
    
    def test_video_upload_invalid_format(self):
        """Test video upload with invalid format"""
        # Upload invalid video file
        files = {"file": ("test.txt", b"not a video", "text/plain")}
        response = client.post("/detect/video", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Invalid video format" in data["detail"]
    
    def test_status_nonexistent_job(self):
        """Test status endpoint with non-existent job ID"""
        response = client.get("/status/99999")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "processing"  # Should indicate not found/processing
    
    def test_download_nonexistent_job(self):
        """Test download endpoint with non-existent job ID"""
        response = client.get("/download/99999")
        assert response.status_code == 404

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test concurrent API requests"""
    async def make_request():
        response = client.get("/health")
        return response.status_code == 200
    
    # Make multiple concurrent requests
    tasks = [make_request() for _ in range(10)]
    results = await asyncio.gather(*tasks)
    
    # All requests should succeed
    assert all(results)

def test_api_performance():
    """Basic performance test"""
    import time
    
    start_time = time.time()
    response = client.get("/health")
    end_time = time.time()
    
    assert response.status_code == 200
    assert (end_time - start_time) < 1.0  # Should respond within 1 second

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
