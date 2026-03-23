#!/usr/bin/env python3
"""
Production-Ready Inference Function

Clean, production-ready function for skin disease classification inference.
Loads trained model and provides predictions with confidence scores.

Author: AI Assistant
Date: 2025-03-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import json
import os
from typing import Dict, Union, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinDiseaseClassifier(nn.Module):
    """
    Skin disease classification model using EfficientNet-B0
    Must match the training architecture exactly
    """
    
    def __init__(self, num_classes: int = 3, pretrained: bool = False):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights (False for inference)
        """
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

class ModelLoader:
    """
    Model loading utility with error handling and validation
    """
    
    def __init__(self, model_path: str = "models/best_model.pth"):
        """
        Initialize model loader
        
        Args:
            model_path: Path to trained model file
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = None
        self.transform = None
        
    def load_model(self) -> bool:
        """
        Load trained model and metadata
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load checkpoint
            logger.info(f"Loading model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract metadata
            self.class_names = checkpoint.get('class_names', ['benign', 'malignant'])
            num_classes = checkpoint.get('num_classes', len(self.class_names))
            model_config = checkpoint.get('model_config', {})
            
            logger.info(f"Model loaded successfully:")
            logger.info(f"  Classes: {self.class_names}")
            logger.info(f"  Architecture: {model_config.get('architecture', 'efficientnet_b0')}")
            logger.info(f"  Device: {self.device}")
            
            # Create model with same architecture as training
            self.model = SkinDiseaseClassifier(num_classes=num_classes, pretrained=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Create preprocessing transforms (same as validation during training)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("Model preprocessing pipeline ready")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded successfully"""
        return self.model is not None and self.class_names is not None and self.transform is not None

# Global model loader instance
_model_loader = ModelLoader()

def load_model(model_path: str = "models/best_model.pth") -> bool:
    """
    Load the trained model for inference
    
    Args:
        model_path: Path to the trained model file
        
    Returns:
        True if model loaded successfully, False otherwise
    """
    global _model_loader
    _model_loader = ModelLoader(model_path)
    return _model_loader.load_model()

def preprocess_image(image: Union[Image.Image, np.ndarray, str]) -> torch.Tensor:
    """
    Preprocess image for inference
    
    Args:
        image: Input image (PIL Image, numpy array, or file path)
        
    Returns:
        Preprocessed tensor ready for model input
        
    Raises:
        ValueError: If image format is not supported
        RuntimeError: If model is not loaded
    """
    if not _model_loader.is_loaded():
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    try:
        # Handle different input types
        if isinstance(image, str):
            # Load from file path
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = Image.open(image).convert('RGB')
            
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
            
        elif isinstance(image, Image.Image):
            # Use PIL Image directly
            pass
            
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        tensor = _model_loader.transform(image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        tensor = tensor.to(_model_loader.device)
        
        return tensor
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def predict(image: Union[Image.Image, np.ndarray, str], 
            return_probabilities: bool = False) -> Dict[str, Union[str, float, Dict]]:
    """
    Predict skin disease class from image
    
    Args:
        image: Input image (PIL Image, numpy array, or file path)
        return_probabilities: Whether to return class probabilities
        
    Returns:
        Dictionary containing:
        - 'predicted_class': Predicted class name
        - 'confidence': Confidence score (0-1)
        - 'class_probabilities': Dict of class probabilities (if return_probabilities=True)
        
    Raises:
        RuntimeError: If model is not loaded
        ValueError: If image preprocessing fails
    """
    if not _model_loader.is_loaded():
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    try:
        # Preprocess image
        input_tensor = preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = _model_loader.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()
        
        # Get class name
        predicted_class = _model_loader.class_names[predicted_class_idx]
        
        # Build result
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'predicted_class_idx': predicted_class_idx
        }
        
        # Add probabilities if requested
        if return_probabilities:
            class_probabilities = {
                _model_loader.class_names[i]: probabilities[0, i].item()
                for i in range(len(_model_loader.class_names))
            }
            result['class_probabilities'] = class_probabilities
        
        logger.info(f"Prediction successful: {predicted_class} (confidence: {confidence:.3f})")
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

def predict_batch(images: list, return_probabilities: bool = False) -> list:
    """
    Predict skin disease classes for a batch of images
    
    Args:
        images: List of images (PIL Images, numpy arrays, or file paths)
        return_probabilities: Whether to return class probabilities
        
    Returns:
        List of prediction dictionaries
    """
    if not _model_loader.is_loaded():
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    try:
        # Preprocess batch
        batch_tensors = []
        for image in images:
            tensor = preprocess_image(image)
            batch_tensors.append(tensor)
        
        # Stack batch
        batch_tensor = torch.cat(batch_tensors, dim=0)
        
        # Run inference
        with torch.no_grad():
            outputs = _model_loader.model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1).values
        
        # Build results
        results = []
        for i in range(len(images)):
            predicted_class_idx = predicted_classes[i].item()
            predicted_class = _model_loader.class_names[predicted_class_idx]
            confidence = confidences[i].item()
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'predicted_class_idx': predicted_class_idx
            }
            
            if return_probabilities:
                class_probabilities = {
                    _model_loader.class_names[j]: probabilities[i, j].item()
                    for j in range(len(_model_loader.class_names))
                }
                result['class_probabilities'] = class_probabilities
            
            results.append(result)
        
        logger.info(f"Batch prediction successful: {len(images)} images processed")
        return results
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise

def get_model_info() -> Dict[str, Union[str, int, list]]:
    """
    Get information about the loaded model
    
    Returns:
        Dictionary containing model information
    """
    if not _model_loader.is_loaded():
        return {'error': 'Model not loaded'}
    
    return {
        'model_path': _model_loader.model_path,
        'device': str(_model_loader.device),
        'num_classes': len(_model_loader.class_names),
        'class_names': _model_loader.class_names,
        'architecture': 'efficientnet_b0',
        'input_size': (224, 224),
        'input_channels': 3
    }

# Convenience function for one-line usage
def predict_skin_disease(image_path: str) -> Dict[str, Union[str, float]]:
    """
    Convenience function for quick prediction from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Prediction dictionary with class and confidence
    """
    # Load model if not already loaded
    if not _model_loader.is_loaded():
        if not load_model():
            raise RuntimeError("Failed to load model")
    
    # Make prediction
    return predict(image_path, return_probabilities=True)

# Example usage and testing
def test_inference_function():
    """
    Test the inference function with sample data
    """
    print("🧪 Testing Inference Function")
    print("=" * 40)
    
    # Load model
    if not load_model():
        print("❌ Failed to load model. Please run training first.")
        return
    
    # Test model info
    print("\n📊 Model Information:")
    info = get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with synthetic image
    print("\n🎨 Testing with synthetic image...")
    synthetic_image = Image.new('RGB', (224, 224), color='lightblue')
    
    try:
        # Test single prediction
        result = predict(synthetic_image, return_probabilities=True)
        print("✅ Single prediction successful:")
        print(f"  Predicted: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Probabilities: {result['class_probabilities']}")
        
        # Test batch prediction
        print("\n🔄 Testing batch prediction...")
        test_images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='green'),
            Image.new('RGB', (224, 224), color='blue')
        ]
        
        batch_results = predict_batch(test_images, return_probabilities=True)
        print("✅ Batch prediction successful:")
        for i, result in enumerate(batch_results):
            print(f"  Image {i+1}: {result['predicted_class']} ({result['confidence']:.3f})")
        
        # Test convenience function
        print("\n⚡ Testing convenience function...")
        # Save synthetic image for testing
        synthetic_image.save('test_image.jpg')
        convenience_result = predict_skin_disease('test_image.jpg')
        print("✅ Convenience function successful:")
        print(f"  Predicted: {convenience_result['predicted_class']}")
        print(f"  Confidence: {convenience_result['confidence']:.3f}")
        
        # Clean up
        if os.path.exists('test_image.jpg'):
            os.remove('test_image.jpg')
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Run tests
    test_inference_function()
