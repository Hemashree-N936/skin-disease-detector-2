#!/usr/bin/env python3
"""
Inference-Compatible Model Loading and Testing

This script demonstrates how to load the trained model for inference
and ensures compatibility with the training architecture.

Author: AI Assistant
Date: 2025-03-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import os

class SkinDiseaseClassifier(nn.Module):
    """
    Skin disease classification model using EfficientNet-B0
    Must match the training architecture exactly
    """
    
    def __init__(self, num_classes=3, pretrained=False):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights (False for inference)
        """
        super().__init__()
        
        # Load EfficientNet-B0 backbone
        import torchvision.models as models
        self.backbone = models.efficientnet_b0(weights=None)  # No pretrained weights for inference
        
        # Replace classifier (must match training)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x):
        return self.backbone(x)

class InferenceModel:
    """
    Inference model wrapper with preprocessing and prediction
    """
    
    def __init__(self, model_path="models/best_model.pth"):
        """
        Initialize inference model
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = None
        self.transform = None
        
        self.load_model()
        
    def load_model(self):
        """Load trained model and metadata"""
        print(f"📁 Loading model from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract metadata
        self.class_names = checkpoint.get('class_names', ['benign', 'malignant'])
        num_classes = checkpoint.get('num_classes', len(self.class_names))
        model_config = checkpoint.get('model_config', {})
        
        print(f"📊 Model metadata:")
        print(f"   Classes: {self.class_names}")
        print(f"   Number of classes: {num_classes}")
        print(f"   Architecture: {model_config.get('architecture', 'efficientnet_b0')}")
        print(f"   Best validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        
        # Create model with same architecture as training
        self.model = SkinDiseaseClassifier(num_classes=num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Create preprocessing transforms (same as validation)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ Model loaded successfully!")
        
    def preprocess_image(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be PIL Image or numpy array")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
    
    def predict(self, image, return_probabilities=False):
        """
        Make prediction on image
        
        Args:
            image: PIL Image or numpy array
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()
        
        # Create results
        predicted_class = self.class_names[predicted_class_idx]
        
        results = {
            'predicted_class': predicted_class,
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence,
            'class_names': self.class_names
        }
        
        if return_probabilities:
            class_probabilities = {
                self.class_names[i]: probabilities[0, i].item()
                for i in range(len(self.class_names))
            }
            results['class_probabilities'] = class_probabilities
        
        return results
    
    def predict_batch(self, images):
        """
        Make predictions on batch of images
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            List of prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess batch
        batch_tensors = []
        for image in images:
            tensor = self.preprocess_image(image)
            batch_tensors.append(tensor)
        
        batch_tensor = torch.cat(batch_tensors, dim=0)
        
        # Batch inference
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1).values
        
        # Create results
        results = []
        for i in range(len(images)):
            predicted_class_idx = predicted_classes[i].item()
            predicted_class = self.class_names[predicted_class_idx]
            confidence = confidences[i].item()
            
            class_probabilities = {
                self.class_names[j]: probabilities[i, j].item()
                for j in range(len(self.class_names))
            }
            
            result = {
                'predicted_class': predicted_class,
                'predicted_class_idx': predicted_class_idx,
                'confidence': confidence,
                'class_probabilities': class_probabilities,
                'class_names': self.class_names
            }
            results.append(result)
        
        return results

def test_inference():
    """
    Test inference with sample images
    """
    print("🧪 Testing Inference Model")
    print("=" * 40)
    
    # Create inference model
    try:
        inference_model = InferenceModel("models/best_model.pth")
    except FileNotFoundError:
        print("❌ Model not found. Please run training first.")
        return
    
    # Test with synthetic image
    print("\n🎨 Testing with synthetic image...")
    
    # Create a synthetic image
    synthetic_image = Image.new('RGB', (224, 224), color='lightblue')
    
    # Make prediction
    results = inference_model.predict(synthetic_image, return_probabilities=True)
    
    print("📊 Prediction Results:")
    print(f"   Predicted Class: {results['predicted_class']}")
    print(f"   Confidence: {results['confidence']:.3f}")
    print(f"   Class Probabilities:")
    for class_name, prob in results['class_probabilities'].items():
        print(f"      {class_name}: {prob:.3f}")
    
    # Test batch prediction
    print("\n🔄 Testing batch prediction...")
    
    # Create multiple synthetic images
    test_images = [
        Image.new('RGB', (224, 224), color='red'),
        Image.new('RGB', (224, 224), color='green'),
        Image.new('RGB', (224, 224), color='blue')
    ]
    
    batch_results = inference_model.predict_batch(test_images)
    
    print("📊 Batch Prediction Results:")
    for i, result in enumerate(batch_results):
        print(f"   Image {i+1}: {result['predicted_class']} ({result['confidence']:.3f})")
    
    print("\n✅ Inference testing completed!")

def validate_model_compatibility():
    """
    Validate that the inference model is compatible with training
    """
    print("🔍 Validating Model Compatibility")
    print("=" * 40)
    
    # Check if model file exists
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print("❌ Model file not found. Run training first.")
        return False
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Validate checkpoint structure
        required_keys = ['model_state_dict', 'class_names', 'num_classes']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            print(f"❌ Missing checkpoint keys: {missing_keys}")
            return False
        
        # Validate model state dict
        state_dict = checkpoint['model_state_dict']
        class_names = checkpoint['class_names']
        num_classes = checkpoint['num_classes']
        
        print(f"✅ Checkpoint validation passed:")
        print(f"   Number of parameters: {len(state_dict)}")
        print(f"   Classes: {class_names}")
        print(f"   Number of classes: {num_classes}")
        
        # Try to recreate model
        model = SkinDiseaseClassifier(num_classes=num_classes, pretrained=False)
        model.load_state_dict(state_dict)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        expected_shape = (1, num_classes)
        if output.shape != expected_shape:
            print(f"❌ Output shape mismatch: {output.shape} != {expected_shape}")
            return False
        
        print(f"✅ Model architecture validation passed:")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected shape: {expected_shape}")
        
        # Test inference model
        inference_model = InferenceModel(model_path)
        
        # Test with dummy image
        dummy_image = Image.new('RGB', (224, 224), color='white')
        results = inference_model.predict(dummy_image)
        
        print(f"✅ Inference validation passed:")
        print(f"   Prediction: {results['predicted_class']}")
        print(f"   Confidence: {results['confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def create_inference_script():
    """
    Create a simple inference script for deployment
    """
    script_content = '''#!/usr/bin/env python3
"""
Simple Inference Script for Skin Disease Classification

Usage:
    python inference.py path/to/image.jpg
"""

import sys
import os
from PIL import Image
from inference_compatible import InferenceModel

def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)
    
    # Load model
    model = InferenceModel("models/best_model.pth")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Make prediction
    results = model.predict(image, return_probabilities=True)
    
    # Print results
    print(f"Image: {image_path}")
    print(f"Predicted Class: {results['predicted_class']}")
    print(f"Confidence: {results['confidence']:.3f}")
    print("Class Probabilities:")
    for class_name, prob in results['class_probabilities'].items():
        print(f"  {class_name}: {prob:.3f}")

if __name__ == "__main__":
    main()
'''
    
    with open('inference.py', 'w') as f:
        f.write(script_content)
    
    print("📝 Created inference.py script")

if __name__ == "__main__":
    print("🏥 Skin Disease Classification - Inference Compatibility")
    print("=" * 60)
    
    # Validate model compatibility
    if validate_model_compatibility():
        print("\n✅ Model is compatible with inference!")
        
        # Test inference
        test_inference()
        
        # Create inference script
        create_inference_script()
        
        print("\n🎉 All tests passed!")
        print("📁 Files ready:")
        print("   - models/best_model.pth (trained model)")
        print("   - inference.py (simple inference script)")
        print("   - inference_compatible.py (inference utilities)")
        
    else:
        print("\n❌ Model compatibility issues found!")
        print("Please run training first to create a compatible model.")
