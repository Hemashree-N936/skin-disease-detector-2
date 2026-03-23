#!/usr/bin/env python3
"""
Simple Inference Function - Clean and Minimal

Production-ready inference function with minimal dependencies
and clean interface for skin disease classification.

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
import os
from typing import Dict, Union

# Global variables for model
_model = None
_class_names = None
_device = None
_transform = None

class SkinDiseaseClassifier(nn.Module):
    """Skin disease classification model - must match training exactly"""
    
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def load_model(model_path: str = "models/best_model.pth") -> bool:
    """
    Load trained model for inference
    
    Args:
        model_path: Path to trained model
        
    Returns:
        True if successful, False otherwise
    """
    global _model, _class_names, _device, _transform
    
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return False
        
        # Set device
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=_device)
        
        # Extract metadata
        _class_names = checkpoint.get('class_names', ['benign', 'malignant'])
        num_classes = len(_class_names)
        
        # Create model
        _model = SkinDiseaseClassifier(num_classes=num_classes)
        _model.load_state_dict(checkpoint['model_state_dict'])
        _model.to(_device)
        _model.eval()
        
        # Create preprocessing transforms
        _transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Model loaded successfully")
        print(f"   Classes: {_class_names}")
        print(f"   Device: {_device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def preprocess_image(image: Union[Image.Image, np.ndarray, str]) -> torch.Tensor:
    """
    Preprocess image for inference
    
    Args:
        image: PIL Image, numpy array, or file path
        
    Returns:
        Preprocessed tensor
    """
    global _transform, _device
    
    if _transform is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Handle different input types
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Apply transforms
    tensor = _transform(image)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    tensor = tensor.to(_device)
    
    return tensor

def predict(image: Union[Image.Image, np.ndarray, str], 
            return_probabilities: bool = False) -> Dict[str, Union[str, float, Dict]]:
    """
    Predict skin disease class from image
    
    Args:
        image: Input image (PIL Image, numpy array, or file path)
        return_probabilities: Whether to return class probabilities
        
    Returns:
        Dictionary with prediction results
    """
    global _model, _class_names
    
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Preprocess
    input_tensor = preprocess_image(image)
    
    # Inference
    with torch.no_grad():
        outputs = _model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item()
    
    # Build result
    predicted_class = _class_names[predicted_class_idx]
    
    result = {
        'predicted_class': predicted_class,
        'confidence': confidence
    }
    
    if return_probabilities:
        class_probabilities = {
            _class_names[i]: probabilities[0, i].item()
            for i in range(len(_class_names))
        }
        result['class_probabilities'] = class_probabilities
    
    return result

def predict_from_file(image_path: str, return_probabilities: bool = False) -> Dict[str, Union[str, float, Dict]]:
    """
    Convenience function to predict from file path
    
    Args:
        image_path: Path to image file
        return_probabilities: Whether to return class probabilities
        
    Returns:
        Prediction dictionary
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    return predict(image_path, return_probabilities)

def get_model_info() -> Dict[str, Union[str, int, list]]:
    """Get information about loaded model"""
    global _class_names, _device
    
    if _model is None:
        return {'error': 'Model not loaded'}
    
    return {
        'classes': _class_names,
        'num_classes': len(_class_names) if _class_names else 0,
        'device': str(_device) if _device else 'unknown',
        'architecture': 'efficientnet_b0',
        'input_size': (224, 224)
    }

# Example usage
if __name__ == "__main__":
    print("🏥 Simple Skin Disease Classification Inference")
    print("=" * 50)
    
    # Load model
    if not load_model():
        print("❌ Please run training first to create models/best_model.pth")
        exit(1)
    
    # Test with synthetic image
    print("\n🎨 Testing with synthetic image...")
    
    # Create test image
    test_image = Image.new('RGB', (224, 224), color='lightblue')
    
    # Make prediction
    try:
        result = predict(test_image, return_probabilities=True)
        
        print("📊 Prediction Results:")
        print(f"   Predicted Class: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        
        if 'class_probabilities' in result:
            print("   Class Probabilities:")
            for class_name, prob in result['class_probabilities'].items():
                print(f"      {class_name}: {prob:.3f}")
        
        print("\n✅ Inference test successful!")
        
        # Show model info
        print("\n📊 Model Info:")
        info = get_model_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
    
    print("\n🎉 Ready for production use!")
    print("\n📚 Usage Examples:")
    print("   # Load model once")
    print("   load_model('models/best_model.pth')")
    print("   ")
    print("   # Predict from file")
    print("   result = predict_from_file('image.jpg')")
    print("   ")
    print("   # Predict from PIL Image")
    print("   result = predict(pil_image, return_probabilities=True)")
