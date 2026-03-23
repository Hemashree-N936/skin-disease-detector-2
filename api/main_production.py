#!/usr/bin/env python3
"""
Production FastAPI Backend with Real PyTorch Model

This FastAPI server provides endpoints for skin disease classification
with real trained model loading and inference.

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
import logging
from typing import Dict, Any, Optional
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import model architecture
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model architecture
try:
    from ham10000_training_complete import SkinDiseaseClassifier
except ImportError:
    # Fallback if training script not available
    import torchvision.models as models
    
    class SkinDiseaseClassifier(nn.Module):
        """Skin disease classification model using EfficientNet-B0"""
        
        def __init__(self, num_classes: int = 3, pretrained: bool = False):
            super().__init__()
            
            # Load EfficientNet-B0 backbone
            self.backbone = models.efficientnet_b0(weights=None)
            
            # Replace classifier (must match training exactly)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
            
            self.num_classes = num_classes
            
        def forward(self, x):
            return self.backbone(x)

# Global variables
app = FastAPI(
    title="Skin Disease Classification API",
    description="Production API for skin disease classification using trained PyTorch model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
class_names = None
device = None
transform = None

# Setup transforms (same as training validation transforms)
def setup_transforms():
    """Setup image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_model_at_startup():
    """Load trained model at startup"""
    global model, class_names, device, transform
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Model path
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "models", "best_model.pth")
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found at: {model_path}")
            logger.error("Please train the model first using ham10000_training_complete.py")
            return False
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract metadata
        class_names = checkpoint.get('class_names', ['benign', 'malignant'])
        num_classes = len(class_names)
        
        logger.info(f"Model metadata:")
        logger.info(f"  Classes: {class_names}")
        logger.info(f"  Number of classes: {num_classes}")
        logger.info(f"  Best validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        
        # Create model with same architecture as training
        model = SkinDiseaseClassifier(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Setup transforms
        transform = setup_transforms()
        
        logger.info("✅ Model loaded successfully!")
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for inference
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed tensor
    """
    if transform is None:
        raise RuntimeError("Transforms not initialized. Model may not be loaded.")
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    tensor = tensor.to(device)
    
    return tensor

def run_inference(image: Image.Image) -> Dict[str, Any]:
    """
    Run inference on preprocessed image
    
    Args:
        image: PIL Image
        
    Returns:
        Dictionary with prediction results
    """
    if model is None:
        raise RuntimeError("Model not loaded. Call load_model_at_startup() first.")
    
    try:
        # Preprocess image
        input_tensor = preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()
        
        # Get class name
        predicted_class = class_names[predicted_class_idx]
        
        # Build result
        result = {
            'class': predicted_class,
            'confidence': confidence,
            'class_idx': predicted_class_idx,
            'all_probabilities': {
                class_names[i]: probabilities[0, i].item()
                for i in range(len(class_names))
            }
        }
        
        logger.info(f"Inference successful: {predicted_class} (confidence: {confidence:.3f})")
        return result
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize model and transforms at startup"""
    logger.info("🚀 Starting FastAPI server...")
    logger.info("📁 Loading trained model...")
    
    success = load_model_at_startup()
    
    if not success:
        logger.error("❌ Failed to load model. Server will start but predictions will fail.")
    else:
        logger.info("✅ Server startup completed successfully!")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Skin Disease Classification API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "class_names": class_names if class_names else None,
        "model_parameters": sum(p.numel() for p in model.parameters()) if model else 0
    }

@app.get("/model/info")
async def model_info():
    """Get detailed model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_type": "SkinDiseaseClassifier",
        "architecture": "EfficientNet-B0",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "input_size": (224, 224),
        "input_channels": 3,
        "num_classes": len(class_names) if class_names else 0,
        "class_names": class_names,
        "device": str(device),
        "pretrained": False,  # Set to False for inference
        "model_loaded": True,
        "status": "ready"
    }

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict skin disease from uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with prediction results
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    # Validate file
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Expected image, got: {file.content_type}"
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Validate image
        if image.size[0] < 32 or image.size[1] < 32:
            raise HTTPException(
                status_code=400, 
                detail="Image too small. Minimum size: 32x32 pixels"
            )
        
        # Run inference
        result = run_inference(image)
        
        # Format response (as requested)
        response = {
            "class": result['class'],
            "confidence": result['confidence']
        }
        
        logger.info(f"Prediction for {file.filename}: {result['class']} ({result['confidence']:.3f})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/predict-detailed")
async def predict_disease_detailed(file: UploadFile = File(...)):
    """
    Predict skin disease with detailed information
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with detailed prediction results
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    # Validate file
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Expected image, got: {file.content_type}"
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Validate image
        if image.size[0] < 32 or image.size[1] < 32:
            raise HTTPException(
                status_code=400, 
                detail="Image too small. Minimum size: 32x32 pixels"
            )
        
        # Run inference
        result = run_inference(image)
        
        # Format detailed response
        response = {
            "predicted_class": result['class'],
            "confidence": result['confidence'],
            "class_index": result['class_idx'],
            "all_probabilities": result['all_probabilities'],
            "image_info": {
                "filename": file.filename,
                "size": image.size,
                "mode": image.mode
            },
            "model_info": {
                "classes": class_names,
                "architecture": "EfficientNet-B0"
            }
        }
        
        logger.info(f"Detailed prediction for {file.filename}: {result['class']} ({result['confidence']:.3f})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict skin diseases for multiple images
    
    Args:
        files: List of uploaded image files
        
    Returns:
        JSON response with batch prediction results
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400, 
            detail="Batch size too large. Maximum 10 files per request."
        )
    
    results = []
    
    for file in files:
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            results.append({
                "filename": file.filename,
                "error": f"Invalid file type: {file.content_type}"
            })
            continue
        
        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Validate image
            if image.size[0] < 32 or image.size[1] < 32:
                results.append({
                    "filename": file.filename,
                    "error": "Image too small. Minimum size: 32x32 pixels"
                })
                continue
            
            # Run inference
            prediction = run_inference(image)
            
            results.append({
                "filename": file.filename,
                "class": prediction['class'],
                "confidence": prediction['confidence'],
                "success": True
            })
            
        except Exception as e:
            logger.error(f"Error processing image {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    return {
        "batch_size": len(files),
        "successful_predictions": sum(1 for r in results if r.get("success", False)),
        "results": results
    }

if __name__ == "__main__":
    # Print startup message
    print("🏥 Skin Disease Classification API - Production")
    print("=" * 50)
    print("🚀 Starting FastAPI server...")
    print("📁 Loading trained PyTorch model...")
    
    # Load model before starting server
    if not load_model_at_startup():
        print("❌ Failed to load model!")
        print("Please train the model first:")
        print("   python ham10000_training_complete.py")
        print("Then ensure the model is saved as: models/best_model.pth")
        exit(1)
    
    print("✅ Model loaded successfully!")
    print("🌐 API will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔥 Prediction endpoint: http://localhost:8000/predict")
    print("🔍 Health check: http://localhost:8000/health")
    print("📊 Model info: http://localhost:8000/model/info")
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000)
