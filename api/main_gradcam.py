#!/usr/bin/env python3
"""
FastAPI Backend with Real Grad-CAM Visualization

This FastAPI server provides endpoints for skin disease classification
with real Grad-CAM visualization using actual model gradients.

Author: AI Assistant
Date: 2025-03-23
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import torchvision.transforms as transforms
import uvicorn
import os
import sys
import base64
import numpy as np
import cv2
from typing import Dict, Any

# Add parent directory to path to import model and Grad-CAM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradcam_real import SkinDiseaseGradCAM
from simple_train import SimpleSkinClassifier

app = FastAPI(title="Skin Disease Classification API with Real Grad-CAM", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
gradcam_generator = None
class_names = ['acne', 'psoriasis', 'eczema']

# Setup transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    """Load the trained model and initialize Grad-CAM"""
    global model, gradcam_generator
    
    print("Loading model...")
    
    # Create model
    model = SimpleSkinClassifier(num_classes=3)
    
    # Load trained weights
    model_path = "../models/simple_best_model.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {model_path}")
        print(f"Best validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    else:
        print(f"⚠️  No trained model found at {model_path}")
        print("Using randomly initialized model for demonstration")
    
    model.to(device)
    model.eval()
    
    # Initialize Grad-CAM
    try:
        gradcam_generator = SkinDiseaseGradCAM(model)
        print("✅ Grad-CAM initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Grad-CAM: {e}")
        gradcam_generator = None

def predict_disease(image: Image.Image) -> Dict[str, Any]:
    """
    Predict disease from image
    
    Args:
        image: PIL Image
        
    Returns:
        Prediction dictionary
    """
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item()
    
    # Create response
    class_probabilities = {
        class_names[i]: float(probabilities[0, i].item())
        for i in range(len(class_names))
    }
    
    return {
        'class_label': class_names[predicted_class_idx],
        'confidence': confidence,
        'probabilities': class_probabilities,
        'model_type': 'Simple CNN with Grad-CAM',
        'device': str(device)
    }

def generate_gradcam_visualization(image: Image.Image, predicted_class: str) -> Dict[str, Any]:
    """
    Generate Grad-CAM visualization
    
    Args:
        image: PIL Image
        predicted_class: Predicted class name
        
    Returns:
        Visualization dictionary
    """
    if gradcam_generator is None:
        raise HTTPException(status_code=500, detail="Grad-CAM not available")
    
    try:
        # Generate visualization
        visualization = gradcam_generator.generate_visualization(image, predicted_class)
        
        return {
            'gradcam_image': visualization['overlay_image'],
            'heatmap_image': visualization['heatmap_image'],
            'original_image': visualization['original_image'],
            'class_label': predicted_class,
            'target_layer': visualization['target_layer'],
            'class_index': visualization['class_index']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize model and Grad-CAM on startup"""
    load_model()

@app.get("/")
async def root():
    return {
        "message": "Skin Disease Classification API with Real Grad-CAM",
        "version": "1.0.0",
        "features": ["Classification", "Real Grad-CAM Visualization"],
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict-with-gradcam": "/predict-with-gradcam",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gradcam_available": gradcam_generator is not None,
        "device": str(device),
        "classes": class_names
    }

@app.post("/predict")
async def predict_disease_endpoint(file: UploadFile = File(...)):
    """
    Predict skin disease from uploaded image
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Predict
        prediction = predict_disease(image)
        prediction['filename'] = file.filename
        
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict-with-gradcam")
async def predict_with_gradcam_endpoint(file: UploadFile = File(...)):
    """
    Predict skin disease and generate Grad-CAM visualization
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Predict disease
        prediction = predict_disease(image)
        predicted_class = prediction['class_label']
        
        # Generate Grad-CAM visualization
        try:
            gradcam_viz = generate_gradcam_visualization(image, predicted_class)
            
            # Combine prediction and visualization
            response = {
                **prediction,
                **gradcam_viz,
                'filename': file.filename,
                'visualization_type': 'real_gradcam',
                'description': f"Real Grad-CAM visualization showing regions that contributed to the '{predicted_class}' classification"
            }
            
            return response
            
        except Exception as e:
            # If Grad-CAM fails, still return prediction
            print(f"Grad-CAM failed: {e}")
            return {
                **prediction,
                'filename': file.filename,
                'gradcam_error': f"Grad-CAM visualization failed: {str(e)}",
                'visualization_type': 'prediction_only'
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get Grad-CAM info
    gradcam_info = {}
    if gradcam_generator:
        gradcam_info = {
            'target_layer': gradcam_generator.target_layer,
            'class_names': gradcam_generator.class_names,
            'status': 'available'
        }
    else:
        gradcam_info = {'status': 'unavailable'}
    
    return {
        'model_type': 'Simple CNN',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'input_size': (224, 224),
        'num_classes': len(class_names),
        'class_names': class_names,
        'device': str(device),
        'gradcam': gradcam_info
    }

@app.post("/test-gradcam")
async def test_gradcam_endpoint():
    """Test Grad-CAM with a synthetic image"""
    if gradcam_generator is None:
        raise HTTPException(status_code=500, detail="Grad-CAM not available")
    
    try:
        # Create synthetic image
        synthetic_image = Image.new('RGB', (224, 224), color='blue')
        
        # Generate visualization
        result = gradcam_generator.generate_visualization(synthetic_image, 'acne')
        
        return {
            'status': 'success',
            'message': 'Grad-CAM test completed successfully',
            'test_image_size': synthetic_image.size,
            'target_layer': result['target_layer'],
            'class_name': result['class_name'],
            'images_generated': len([k for k in result.keys() if 'image' in k])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM test failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting FastAPI server with Real Grad-CAM...")
    print("📊 API will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔥 Grad-CAM endpoint: http://localhost:8000/predict-with-gradcam")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
