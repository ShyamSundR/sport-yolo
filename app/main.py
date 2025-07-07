from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import time
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.models.yolo_detector import SportsDetector
from app.utils.logger import setup_logger
from app.utils.video_processor import VideoProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Sports Video Analytics API",
    description="Real-time football player detection and tracking",
    version="1.0.0"
)

# Setup logging
logger = setup_logger()

# Global detector instance
detector = None
video_processor = None

# Create directories
os.makedirs("app/static/uploads", exist_ok=True)
os.makedirs("app/static/outputs", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global detector, video_processor
    try:
        logger.info("Loading YOLO model...")
        detector = SportsDetector()
        video_processor = VideoProcessor(detector)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Sports Video Analytics API",
        "status": "running",
        "model_loaded": detector is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_status": "loaded" if detector else "not_loaded",
        "timestamp": time.time()
    }

@app.post("/detect/image")
async def detect_objects_image(file: UploadFile = File(...)):
    """Detect objects in a single image"""
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run detection
        start_time = time.time()
        results = detector.detect_frame(image)
        inference_time = time.time() - start_time
        
        # Log metrics
        logger.info(
            "Image detection completed",
            inference_time=inference_time,
            detections_count=len(results['detections'])
        )
        
        return {
            "filename": file.filename,
            "inference_time": inference_time,
            "detections": results['detections'],
            "detection_count": len(results['detections']),
            "classes_detected": list(set([d['class'] for d in results['detections']]))
        }
        
    except Exception as e:
        logger.error(f"Image detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/video")
async def detect_objects_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Process video for object detection"""
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    try:
        # Save uploaded file
        timestamp = int(time.time())
        input_path = f"app/static/uploads/{timestamp}_{file.filename}"
        output_path = f"app/static/outputs/{timestamp}_processed_{file.filename}"
        
        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process video in background
        background_tasks.add_task(
            process_video_background,
            input_path,
            output_path,
            timestamp
        )
        
        return {
            "message": "Video processing started",
            "job_id": timestamp,
            "status_url": f"/status/{timestamp}",
            "input_file": file.filename
        }
        
    except Exception as e:
        logger.error(f"Video upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}")
async def get_processing_status(job_id: int):
    """Check video processing status"""
    output_path = f"app/static/outputs/{job_id}_processed_"
    
    # Check if any processed file exists
    output_dir = Path("app/static/outputs")
    processed_files = list(output_dir.glob(f"{job_id}_processed_*"))
    
    if processed_files:
        return {
            "job_id": job_id,
            "status": "completed",
            "download_url": f"/download/{job_id}",
            "processed_file": processed_files[0].name
        }
    else:
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Video is being processed..."
        }

@app.get("/download/{job_id}")
async def download_processed_video(job_id: int):
    """Download processed video"""
    output_dir = Path("app/static/outputs")
    processed_files = list(output_dir.glob(f"{job_id}_processed_*"))
    
    if not processed_files:
        raise HTTPException(status_code=404, detail="Processed video not found")
    
    return FileResponse(
        processed_files[0],
        media_type="video/mp4",
        filename=f"processed_{job_id}.mp4"
    )

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "model_info": detector.get_model_info() if detector else None,
        "uptime": time.time(),
        "endpoints": [
            "/detect/image",
            "/detect/video", 
            "/status/{job_id}",
            "/download/{job_id}"
        ]
    }

async def process_video_background(input_path: str, output_path: str, job_id: int):
    """Background task for video processing"""
    try:
        logger.info(f"Starting video processing for job {job_id}")
        
        # Process video
        results = await video_processor.process_video(input_path, output_path)
        
        logger.info(
            "Video processing completed",
            job_id=job_id,
            total_frames=results.get('total_frames', 0),
            processing_time=results.get('processing_time', 0),
            avg_detections=results.get('avg_detections_per_frame', 0)
        )
        
        # Clean up input file
        os.remove(input_path)
        
    except Exception as e:
        logger.error(f"Video processing failed for job {job_id}: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
