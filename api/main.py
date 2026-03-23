#!/usr/bin/env python3
"""
FastAPI Backend for Skin Disease Classification

Production-ready API with:
- Trained model loading
- Image upload and processing
- Disease prediction with confidence scores
- Grad-CAM visualization
- Production optimization

Author: AI Assistant
Date: 2025-03-22
"""

import os
import logging
import io
import base64
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Import model and visualization components
from efficientnet_skin_classifier import EfficientNetB4Classifier
from cnn_vit_hybrid import CNNViTHybrid
from groupdro_training import GroupDROTrainer
from gradcam_visualization import MedicalGradCAM, GradCAMConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Skin Disease Classification API",
    description="Production-ready API for skin disease classification with Grad-CAM visualization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total predictions', ['class', 'status'])
prediction_duration = Histogram('prediction_duration_seconds', 'Time spent on predictions')
upload_counter = Counter('uploads_total', 'Total uploads', ['status'])


# Pydantic models for API
class PredictionResponse(BaseModel):
    """Response model for disease prediction"""
    class_label: str = Field(..., description="Predicted disease class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Prediction timestamp")


class GradCAMResponse(BaseModel):
    """Response model for Grad-CAM visualization"""
    gradcam_image: str = Field(..., description="Base64 encoded Grad-CAM image")
    heatmap_image: str = Field(..., description="Base64 encoded heatmap")
    overlay_image: str = Field(..., description="Base64 encoded overlay")
    class_label: str = Field(..., description="Predicted disease class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="API health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used")
    uptime: float = Field(..., description="API uptime in seconds")
    timestamp: str = Field(..., description="Health check timestamp")


class ModelManager:
    """Manages model loading and inference"""
    
    def __init__(self):
        self.model = None
        self.gradcam = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = None
        self.start_time = time.time()
        self.class_names = ['acne', 'psoriasis', 'eczema']
        self.class_descriptions = {
            'acne': 'Inflammatory skin condition with comedones, papules, and pustules',
            'psoriasis': 'Autoimmune condition causing scaly, red patches',
            'eczema': 'Inflammatory condition causing dry, itchy, inflamed skin'
        }
        
        # Model paths to try
        self.model_paths = [
            'models/best_model.pth',
            'models_hybrid/best_hybrid_model.pth',
            'models_groupdro/best_groupdro_model.pth',
            'models_segmented/best_model.pth',
            'models_augmented/best_model.pth'
        ]
        
        self.load_model()
    
    def load_model(self):
        """Load the best available model"""
        logger.info("Loading model...")
        
        # Find available models
        available_models = [path for path in self.model_paths if os.path.exists(path)]
        
        if not available_models:
            logger.error("No trained models found!")
            raise HTTPException(status_code=500, detail="No trained models available")
        
        # Select best model (prioritize GroupDRO > Hybrid > Standard)
        model_path = None
        if 'models_groupdro/best_groupdro_model.pth' in available_models:
            model_path = 'models_groupdro/best_groupdro_model.pth'
            self.model_type = 'groupdro'
        elif 'models_hybrid/best_hybrid_model.pth' in available_models:
            model_path = 'models_hybrid/best_hybrid_model.pth'
            self.model_type = 'hybrid'
        else:
            model_path = available_models[0]
            self.model_type = 'standard'
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize appropriate model
            if self.model_type == 'hybrid':
                self.model = CNNViTHybrid()
            else:
                self.model = EfficientNetB4Classifier()
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize Grad-CAM
            gradcam_config = GradCAMConfig()
            self.gradcam = MedicalGradCAM(self.model, gradcam_config)
            
            logger.info(f"Model loaded successfully: {model_path}")
            logger.info(f"Model type: {self.model_type}")
            logger.info(f"Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Validate image
            if image.size[0] < 100 or image.size[1] < 100:
                raise ValueError("Image too small")
            
            # Preprocess
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0)
            return input_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")
    
    def predict(self, image_bytes: bytes) -> Dict:
        """Make prediction on image"""
        start_time = time.time()
        
        try:
            # Preprocess
            input_tensor = self.preprocess_image(image_bytes)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0, predicted_class_idx].item()
            
            # Create response
            class_probabilities = {
                class_name: float(probabilities[0, i].item())
                for i, class_name in enumerate(self.class_names)
            }
            
            processing_time = time.time() - start_time
            
            result = {
                'class_label': self.class_names[predicted_class_idx],
                'confidence': confidence,
                'probabilities': class_probabilities,
                'processing_time': processing_time,
                'timestamp': datetime.utcnow().isoformat(),
                'model_type': self.model_type
            }
            
            # Update metrics
            prediction_counter.labels(
                class=self.class_names[predicted_class_idx],
                status='success'
            ).inc()
            prediction_duration.observe(processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            prediction_counter.labels(class='error', status='failed').inc()
            raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
    
    def generate_gradcam(self, image_bytes: bytes, prediction_result: Dict) -> Dict:
        """Generate Grad-CAM visualization"""
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image_bytes)
            
            # Generate Grad-CAM
            results = self.gradcam.generate_single_visualization(image_bytes, 'api_request')
            
            # Convert images to base64
            def image_to_base64(image_array):
                _, buffer = cv2.imencode('.jpg', image_array)
                return base64.b64encode(buffer).decode('utf-8')
            
            gradcam_b64 = image_to_base64(results['overlay'])
            heatmap_b64 = image_to_base64(results['heatmap'])
            overlay_b64 = image_to_base64(results['overlay'])
            
            return {
                'gradcam_image': gradcam_b64,
                'heatmap_image': heatmap_b64,
                'overlay_image': overlay_b64,
                'class_label': prediction_result['class_label'],
                'confidence': prediction_result['confidence'],
                'class_description': self.class_descriptions[prediction_result['class_label']]
            }
            
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating Grad-CAM: {str(e)}")


# Initialize model manager
model_manager = ModelManager()


# API Endpoints
@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Skin Disease Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - model_manager.start_time
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.model is not None,
        device=str(model_manager.device),
        uptime=uptime,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Predict skin disease from uploaded image
    
    - **file**: Image file (JPG, PNG, etc.)
    
    Returns:
    - class_label: Predicted disease class
    - confidence: Prediction confidence (0-1)
    - probabilities: Class probabilities for all classes
    - processing_time: Time taken for prediction
    - timestamp: Prediction timestamp
    """
    
    start_time = time.time()
    
    # Validate file
    if not file.content_type.startswith('image/'):
        upload_counter.labels(status='invalid_type').inc()
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if file.size > 10 * 1024 * 1024:  # 10MB limit
        upload_counter.labels(status='too_large').inc()
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    try:
        # Read file
        image_bytes = await file.read()
        
        # Make prediction
        prediction_result = model_manager.predict(image_bytes)
        
        # Log prediction
        background_tasks.add_task(
            log_prediction,
            prediction_result['class_label'],
            prediction_result['confidence'],
            file.filename
        )
        
        upload_counter.labels(status='success').inc()
        
        return PredictionResponse(**prediction_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        upload_counter.labels(status='error').inc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/predict-with-gradcam", response_model=GradCAMResponse)
async def predict_with_gradcam(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Predict skin disease and generate Grad-CAM visualization
    
    - **file**: Image file (JPG, PNG, etc.)
    
    Returns:
    - gradcam_image: Base64 encoded Grad-CAM visualization
    - heatmap_image: Base64 encoded heatmap
    - overlay_image: Base64 encoded overlay
    - class_label: Predicted disease class
    - confidence: Prediction confidence
    """
    
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if file.size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    try:
        # Read file
        image_bytes = await file.read()
        
        # Make prediction
        prediction_result = model_manager.predict(image_bytes)
        
        # Generate Grad-CAM
        gradcam_result = model_manager.generate_gradcam(image_bytes, prediction_result)
        
        # Log prediction
        background_tasks.add_task(
            log_prediction,
            prediction_result['class_label'],
            prediction_result['confidence'],
            file.filename,
            with_gradcam=True
        )
        
        return GradCAMResponse(**gradcam_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Grad-CAM endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_type": model_manager.model_type,
        "device": str(model_manager.device),
        "class_names": model_manager.class_names,
        "class_descriptions": model_manager.class_descriptions,
        "input_size": (224, 224),
        "model_loaded": model_manager.model is not None
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        io.StringIO(generate_latest()),
        media_type=CONTENT_TYPE_LATEST
    )


# Background tasks
async def log_prediction(class_label: str, confidence: float, filename: str, with_gradcam: bool = False):
    """Log prediction for monitoring"""
    logger.info(f"Prediction: {class_label} (confidence: {confidence:.3f}) - {filename} (Grad-CAM: {with_gradcam})")


# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("Skin Disease Classification API starting up...")
    logger.info(f"Model loaded: {model_manager.model is not None}")
    logger.info(f"Device: {model_manager.device}")
    logger.info(f"Model type: {model_manager.model_type}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    logger.info("Skin Disease Classification API shutting down...")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Skin Disease Classification API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    print("🚀 Starting Skin Disease Classification API")
    print(f"📊 Model: {model_manager.model_type}")
    print(f"🔧 Device: {model_manager.device}")
    print(f"🌐 Host: {args.host}")
    print(f"📌 Port: {args.port}")
    print(f"📚 Docs: http://{args.host}:{args.port}/docs")
    
    # Run server
    if args.reload:
        # Development mode
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    else:
        # Production mode
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level="info"
        )
