#!/usr/bin/env python3
"""
Production FastAPI Backend with Real Grad-CAM Integration

This FastAPI server provides endpoints for skin disease classification
with real Grad-CAM visualization using the trained PyTorch model.

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
import torchvision.models as models
import uvicorn
import os
import sys
import logging
import cv2
import numpy as np
import base64
from typing import Dict, Any, Optional, Tuple
import traceback
from collections import defaultdict

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

class GradCAM:
    """
    Grad-CAM implementation for generating class activation maps
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        """
        Initialize Grad-CAM
        
        Args:
            model: PyTorch model
            target_layer: Target layer name for Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        # Find target layer
        target_module = self._find_target_layer()
        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found")
        
        # Register forward hook
        self.forward_hook = target_module.register_forward_hook(
            self._save_activation
        )
        
        # Register backward hook
        self.backward_hook = target_module.register_backward_hook(
            self._save_gradient
        )
        
        self.hooks = [self.forward_hook, self.backward_hook]
    
    def _find_target_layer(self) -> nn.Module:
        """Find the target layer in the model"""
        # Try different possible layer names for EfficientNet-B0
        possible_layers = [
            'backbone.features.8.0',  # Last conv block in EfficientNet-B0
            'backbone.features.7.0',  # Second to last conv block
            'backbone.features.6.0',  # Third to last conv block
            'features.8.0',          # Alternative naming
            'features.7.0',          # Alternative naming
        ]
        
        for layer_name in possible_layers:
            try:
                module = self._get_module_by_name(layer_name)
                if module is not None:
                    logger.info(f"Found target layer: {layer_name}")
                    self.target_layer = layer_name
                    return module
            except Exception:
                continue
        
        raise ValueError(f"Could not find any suitable target layer. Tried: {possible_layers}")
    
    def _get_module_by_name(self, name: str) -> nn.Module:
        """Get module by name"""
        modules = name.split('.')
        current = self.model
        
        for module_name in modules:
            current = getattr(current, module_name)
        
        return current
    
    def _save_activation(self, module, input, output):
        """Save activation from forward pass"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Save gradient from backward pass"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate class activation map
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            class_idx: Target class index
            
        Returns:
            Class activation map as numpy array
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)
        
        # Get gradients and activations
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured")
        
        # Global average pooling of gradients
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate weights
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)  # (H, W)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Convert to numpy
        cam = cam.cpu().numpy()
        
        return cam
    
    def cleanup(self):
        """Clean up hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

class GradCAMGenerator:
    """
    Grad-CAM generator with image processing utilities
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize Grad-CAM generator
        
        Args:
            model: PyTorch model
            device: Device (cpu/cuda)
        """
        self.model = model
        self.device = device
        self.gradcam = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def initialize_gradcam(self):
        """Initialize Grad-CAM"""
        try:
            self.gradcam = GradCAM(self.model, 'backbone.features.8.0')
            logger.info("Grad-CAM initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Grad-CAM: {e}")
            return False
    
    def preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocess image for Grad-CAM
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (preprocessed tensor, original image)
        """
        # Store original image
        original_image = image.copy()
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        input_tensor = self.transform(image)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(self.device)
        
        return input_tensor, original_image
    
    def generate_gradcam(self, image: Image.Image, predicted_class_idx: int) -> Dict[str, Any]:
        """
        Generate Grad-CAM visualization
        
        Args:
            image: PIL Image
            predicted_class_idx: Predicted class index
            
        Returns:
            Dictionary with Grad-CAM results
        """
        if self.gradcam is None:
            if not self.initialize_gradcam():
                raise RuntimeError("Failed to initialize Grad-CAM")
        
        try:
            # Preprocess image
            input_tensor, original_image = self.preprocess_image(image)
            
            # Generate CAM
            cam = self.gradcam.generate_cam(input_tensor, predicted_class_idx)
            
            # Resize CAM to original image size
            cam_resized = cv2.resize(cam, original_image.size)
            
            # Apply colormap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            
            # Convert original image to numpy
            original_np = np.array(original_image)
            
            # Overlay heatmap on original image
            alpha = 0.4  # Transparency for heatmap
            overlay = cv2.addWeighted(original_np, 1 - alpha, heatmap, alpha, 0)
            
            # Convert to PIL
            overlay_image = Image.fromarray(overlay)
            heatmap_image = Image.fromarray(heatmap)
            
            # Convert to base64
            overlay_base64 = self._image_to_base64(overlay_image)
            heatmap_base64 = self._image_to_base64(heatmap_image)
            original_base64 = self._image_to_base64(original_image)
            
            return {
                'overlay_image': overlay_base64,
                'heatmap_image': heatmap_base64,
                'original_image': original_base64,
                'cam_array': cam_resized,
                'target_layer': self.gradcam.target_layer,
                'class_index': predicted_class_idx
            }
            
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {e}")
            raise
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

# Global variables
app = FastAPI(
    title="Skin Disease Classification API with Grad-CAM",
    description="Production API for skin disease classification with Grad-CAM visualization",
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
gradcam_generator = None

def load_model_at_startup():
    """Load trained model at startup"""
    global model, class_names, device, transform, gradcam_generator
    
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
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize Grad-CAM generator
        gradcam_generator = GradCAMGenerator(model, device)
        if not gradcam_generator.initialize_gradcam():
            logger.error("Failed to initialize Grad-CAM")
            return False
        
        logger.info("✅ Model and Grad-CAM loaded successfully!")
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  Grad-CAM target layer: {gradcam_generator.gradcam.target_layer}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

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
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        input_tensor = transform(image)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(device)
        
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
            'predicted_class': predicted_class,
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
    """Initialize model and Grad-CAM at startup"""
    logger.info("🚀 Starting FastAPI server with Grad-CAM...")
    logger.info("📁 Loading trained model and Grad-CAM...")
    
    success = load_model_at_startup()
    
    if not success:
        logger.error("❌ Failed to load model or Grad-CAM. Server will start but predictions will fail.")
    else:
        logger.info("✅ Server startup completed successfully!")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Skin Disease Classification API with Grad-CAM",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "gradcam_available": gradcam_generator is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict-gradcam": "/predict-gradcam",
            "model/info": "/model/info",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gradcam_available": gradcam_generator is not None,
        "device": str(device) if device else "unknown",
        "class_names": class_names if class_names else None,
        "model_parameters": sum(p.numel() for p in model.parameters()) if model else 0
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
        
        # Format response
        response = {
            "class": result['predicted_class'],
            "confidence": result['confidence']
        }
        
        logger.info(f"Prediction for {file.filename}: {result['predicted_class']} ({result['confidence']:.3f})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/predict-gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    """
    Predict skin disease with Grad-CAM visualization
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with prediction and Grad-CAM visualization
    """
    if model is None or gradcam_generator is None:
        raise HTTPException(
            status_code=503, 
            detail="Model or Grad-CAM not loaded. Please check server logs."
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
        prediction = run_inference(image)
        predicted_class_idx = prediction['class_idx']
        
        # Generate Grad-CAM
        gradcam_result = gradcam_generator.generate_gradcam(image, predicted_class_idx)
        
        # Format response
        response = {
            "prediction": {
                "class": prediction['predicted_class'],
                "confidence": prediction['confidence'],
                "class_idx": prediction['class_idx']
            },
            "gradcam": {
                "overlay_image": gradcam_result['overlay_image'],
                "heatmap_image": gradcam_result['heatmap_image'],
                "original_image": gradcam_result['original_image'],
                "target_layer": gradcam_result['target_layer']
            }
        }
        
        logger.info(f"Grad-CAM prediction for {file.filename}: {prediction['predicted_class']} ({prediction['confidence']:.3f})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )

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
        "gradcam_available": gradcam_generator is not None,
        "gradcam_target_layer": gradcam_generator.gradcam.target_layer if gradcam_generator else None,
        "model_loaded": True,
        "status": "ready"
    }

if __name__ == "__main__":
    # Print startup message
    print("🏥 Skin Disease Classification API with Grad-CAM")
    print("=" * 60)
    print("🚀 Starting FastAPI server...")
    print("📁 Loading trained PyTorch model...")
    print("🔥 Initializing Grad-CAM...")
    
    # Load model before starting server
    if not load_model_at_startup():
        print("❌ Failed to load model or Grad-CAM!")
        print("Please train the model first:")
        print("   python ham10000_training_complete.py")
        print("Then ensure the model is saved as: models/best_model.pth")
        exit(1)
    
    print("✅ Model and Grad-CAM loaded successfully!")
    print("🌐 API will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔥 Prediction endpoint: http://localhost:8000/predict")
    print("🔥 Grad-CAM endpoint: http://localhost:8000/predict-gradcam")
    print("🔍 Health check: http://localhost:8000/health")
    print("📊 Model info: http://localhost:8000/model/info")
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000)
