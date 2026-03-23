#!/usr/bin/env python3
"""
Complete Example: Grad-CAM Integration with FastAPI

This script demonstrates the complete integration of real Grad-CAM
visualization with the FastAPI backend for skin disease classification.

Author: AI Assistant
Date: 2025-03-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
import requests
import json

# Import our Grad-CAM implementation
from gradcam_real import SkinDiseaseGradCAM

class SimpleCNN(nn.Module):
    """Simple CNN model for demonstration"""
    
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth conv block (target for Grad-CAM)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def create_sample_image():
    """Create a sample skin disease image for testing"""
    # Create a synthetic image that looks like skin with some patterns
    image = np.random.randint(180, 255, (224, 224, 3), dtype=np.uint8)
    
    # Add some patterns to simulate skin disease
    # Red patches (like acne or rash)
    for _ in range(10):
        x, y = np.random.randint(20, 204, 2)
        cv2.circle(image, (x, y), 5, (200, 50, 50), -1)
    
    # Some darker patches
    for _ in range(5):
        x, y = np.random.randint(20, 204, 2)
        cv2.circle(image, (x, y), 8, (150, 100, 50), -1)
    
    return Image.fromarray(image)

def test_gradcam_standalone():
    """Test Grad-CAM implementation standalone"""
    print("🧪 Testing Grad-CAM Implementation...")
    
    # Create model
    model = SimpleCNN(num_classes=3)
    model.eval()
    
    # Initialize Grad-CAM
    try:
        gradcam = SkinDiseaseGradCAM(model)
        print("✅ Grad-CAM initialized successfully")
        
        # Create sample image
        sample_image = create_sample_image()
        print("📷 Created sample image")
        
        # Generate visualization
        result = gradcam.generate_visualization(sample_image, 'acne')
        
        print("🔥 Grad-CAM visualization generated!")
        print(f"   Target layer: {result['target_layer']}")
        print(f"   Class: {result['class_name']}")
        print(f"   Images generated: {len([k for k in result.keys() if 'image' in k])}")
        
        # Save sample visualization
        overlay_data = base64.b64decode(result['overlay_image'])
        overlay_image = Image.open(BytesIO(overlay_data))
        overlay_image.save('gradcam_sample.jpg')
        print("💾 Sample visualization saved as 'gradcam_sample.jpg'")
        
        gradcam.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Grad-CAM test failed: {e}")
        return False

def test_api_integration():
    """Test API integration"""
    print("\n🌐 Testing API Integration...")
    
    # API base URL
    base_url = "http://localhost:8000"
    
    try:
        # Check if API is running
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
            health_data = response.json()
            print(f"   Model loaded: {health_data['model_loaded']}")
            print(f"   Grad-CAM available: {health_data['gradcam_available']}")
        else:
            print("❌ API is not responding")
            return False
        
        # Test Grad-CAM endpoint
        print("\n🔥 Testing Grad-CAM endpoint...")
        
        # Create sample image
        sample_image = create_sample_image()
        
        # Convert to bytes
        buffer = BytesIO()
        sample_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        # Send to API
        files = {'file': ('test_image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(f"{base_url}/predict-with-gradcam", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Grad-CAM API test successful!")
            print(f"   Predicted class: {result['class_label']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Target layer: {result['target_layer']}")
            print(f"   Visualization type: {result['visualization_type']}")
            
            # Save API result
            if 'overlay_image' in result:
                overlay_data = base64.b64decode(result['overlay_image'])
                overlay_image = Image.open(BytesIO(overlay_data))
                overlay_image.save('api_gradcam_result.jpg')
                print("💾 API result saved as 'api_gradcam_result.jpg'")
            
            return True
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API connection failed: {e}")
        print("   Make sure the API is running: python api/main_gradcam.py")
        return False

def demonstrate_gradcam_workflow():
    """Demonstrate complete Grad-CAM workflow"""
    print("\n🎯 Complete Grad-CAM Workflow Demonstration")
    print("=" * 50)
    
    # Step 1: Test standalone Grad-CAM
    if not test_gradcam_standalone():
        print("❌ Standalone Grad-CAM test failed")
        return
    
    # Step 2: Test API integration
    if not test_api_integration():
        print("❌ API integration test failed")
        return
    
    print("\n🎉 All tests passed! Grad-CAM is working correctly.")
    print("\n📋 Summary:")
    print("   ✅ Grad-CAM implementation working")
    print("   ✅ Real gradients from model")
    print("   ✅ Proper layer hooking")
    print("   ✅ Heatmap generation")
    print("   ✅ Image overlay")
    print("   ✅ API integration")
    print("   ✅ Base64 encoding for web")

def show_api_usage_examples():
    """Show examples of how to use the API"""
    print("\n📚 API Usage Examples")
    print("=" * 30)
    
    print("\n1. Health Check:")
    print("GET http://localhost:8000/health")
    
    print("\n2. Simple Prediction:")
    print("POST http://localhost:8000/predict")
    print("Content-Type: multipart/form-data")
    print("Body: file (image)")
    
    print("\n3. Prediction with Grad-CAM:")
    print("POST http://localhost:8000/predict-with-gradcam")
    print("Content-Type: multipart/form-data")
    print("Body: file (image)")
    
    print("\n4. Model Information:")
    print("GET http://localhost:8000/model/info")
    
    print("\n5. Test Grad-CAM:")
    print("POST http://localhost:8000/test-gradcam")
    
    print("\n📄 API Documentation:")
    print("http://localhost:8000/docs")

def explain_gradcam_technical_details():
    """Explain technical details of the Grad-CAM implementation"""
    print("\n🔬 Technical Details")
    print("=" * 20)
    
    print("\n1. Layer Hooking:")
    print("   - Forward hook captures activations")
    print("   - Backward hook captures gradients")
    print("   - Hooks automatically find target layer")
    
    print("\n2. Gradient Computation:")
    print("   - Backward pass on target class")
    print("   - Gradients flow back to conv layer")
    print("   - Global average pooling of gradients")
    
    print("\n3. CAM Generation:")
    print("   - Weighted sum of activations")
    print("   - ReLU to keep positive contributions")
    print("   - Normalization to [0, 1] range")
    
    print("\n4. Visualization:")
    print("   - Resize CAM to original image size")
    print("   - Apply JET colormap for heatmap")
    print("   - Overlay with transparency")
    
    print("\n5. Model Compatibility:")
    print("   - Auto-detects target layer")
    print("   - Works with CNN architectures")
    print("   - Handles different layer naming")

if __name__ == "__main__":
    print("🏥 Skin Disease Classification - Real Grad-CAM Integration")
    print("=" * 60)
    
    # Run demonstration
    demonstrate_gradcam_workflow()
    
    # Show usage examples
    show_api_usage_examples()
    
    # Explain technical details
    explain_gradcam_technical_details()
    
    print("\n🚀 Ready to use!")
    print("Start the API server: python api/main_gradcam.py")
    print("Open browser: http://localhost:8000/docs")
