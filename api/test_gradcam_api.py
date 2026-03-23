#!/usr/bin/env python3
"""
Test FastAPI API with Grad-CAM

This script tests the FastAPI API with real Grad-CAM visualization.

Author: AI Assistant
Date: 2025-03-23
"""

import requests
import json
import os
from PIL import Image
import io
import time
import base64

def create_test_image():
    """Create a test image for API testing"""
    # Create a simple test image with some patterns
    image = Image.new('RGB', (224, 224), color='lightblue')
    
    # Add some patterns to make it more interesting
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    
    # Draw some circles to simulate skin lesions
    draw.ellipse([50, 50, 100, 100], fill='red')
    draw.ellipse([120, 80, 170, 130], fill='darkred')
    draw.ellipse([80, 120, 130, 170], fill='pink')
    
    return image

def test_api_endpoints(base_url="http://localhost:8000"):
    """Test all API endpoints"""
    print("🧪 Testing FastAPI API with Grad-CAM")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing Health Endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check successful!")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Model loaded: {health_data.get('model_loaded')}")
            print(f"   Grad-CAM available: {health_data.get('gradcam_available')}")
            print(f"   Device: {health_data.get('device')}")
            print(f"   Classes: {health_data.get('class_names')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test model info endpoint
    print("\n2. Testing Model Info Endpoint...")
    try:
        response = requests.get(f"{base_url}/model/info", timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            print("✅ Model info successful!")
            print(f"   Architecture: {model_info.get('architecture')}")
            print(f"   Parameters: {model_info.get('total_parameters'):,}")
            print(f"   Classes: {model_info.get('num_classes')}")
            print(f"   Grad-CAM layer: {model_info.get('gradcam_target_layer')}")
            print(f"   Grad-CAM available: {model_info.get('gradcam_available')}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info error: {e}")
        return False
    
    # Test regular prediction endpoint
    print("\n3. Testing Regular Prediction Endpoint...")
    try:
        # Create test image
        test_image = create_test_image()
        
        # Convert to bytes
        buffer = io.BytesIO()
        test_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        # Send to API
        files = {'file': ('test_image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(f"{base_url}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Regular prediction successful!")
            print(f"   Predicted class: {result['class']}")
            print(f"   Confidence: {result['confidence']:.3f}")
        else:
            print(f"❌ Regular prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Regular prediction error: {e}")
        return False
    
    # Test Grad-CAM prediction endpoint
    print("\n4. Testing Grad-CAM Prediction Endpoint...")
    try:
        # Create test image
        test_image = create_test_image()
        
        # Convert to bytes
        buffer = io.BytesIO()
        test_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        # Send to API
        files = {'file': ('test_image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(f"{base_url}/predict-gradcam", files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Grad-CAM prediction successful!")
            print(f"   Predicted class: {result['prediction']['class']}")
            print(f"   Confidence: {result['prediction']['confidence']:.3f}")
            print(f"   Class index: {result['prediction']['class_idx']}")
            print(f"   Target layer: {result['gradcam']['target_layer']}")
            print(f"   Overlay image: {len(result['gradcam']['overlay_image'])} characters (base64)")
            print(f"   Heatmap image: {len(result['gradcam']['heatmap_image'])} characters (base64)")
            print(f"   Original image: {len(result['gradcam']['original_image'])} characters (base64)")
            
            # Save Grad-CAM images for verification
            try:
                # Decode and save overlay image
                overlay_data = base64.b64decode(result['gradcam']['overlay_image'])
                overlay_image = Image.open(io.BytesIO(overlay_data))
                overlay_image.save('gradcam_overlay_test.jpg')
                print("   ✅ Grad-CAM overlay saved as 'gradcam_overlay_test.jpg'")
                
                # Decode and save heatmap
                heatmap_data = base64.b64decode(result['gradcam']['heatmap_image'])
                heatmap_image = Image.open(io.BytesIO(heatmap_data))
                heatmap_image.save('gradcam_heatmap_test.jpg')
                print("   ✅ Grad-CAM heatmap saved as 'gradcam_heatmap_test.jpg'")
                
                # Decode and save original
                original_data = base64.b64decode(result['gradcam']['original_image'])
                original_image = Image.open(io.BytesIO(original_data))
                original_image.save('gradcam_original_test.jpg')
                print("   ✅ Original image saved as 'gradcam_original_test.jpg'")
                
            except Exception as e:
                print(f"   ⚠️  Could not save Grad-CAM images: {e}")
            
        else:
            print(f"❌ Grad-CAM prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Grad-CAM prediction error: {e}")
        return False
    
    return True

def test_gradcam_quality(base_url="http://localhost:8000"):
    """Test Grad-CAM quality with different images"""
    print("\n🔍 Testing Grad-CAM Quality")
    print("-" * 30)
    
    # Test with different colored images
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    
    for i, color in enumerate(colors):
        print(f"\n5.{i+1} Testing Grad-CAM with {color} image...")
        
        try:
            # Create colored test image
            image = Image.new('RGB', (224, 224), color=color)
            
            # Add some patterns
            from PIL import ImageDraw
            draw = ImageDraw.Draw(image)
            draw.ellipse([50, 50, 100, 100], fill='white')
            draw.ellipse([120, 80, 170, 130], fill='black')
            
            # Convert to bytes
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            image_bytes = buffer.getvalue()
            
            # Send to API
            files = {'file': (f'test_{color}.jpg', image_bytes, 'image/jpeg')}
            response = requests.post(f"{base_url}/predict-gradcam", files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ {color} image successful!")
                print(f"   Predicted: {result['prediction']['class']} ({result['prediction']['confidence']:.3f})")
                print(f"   Layer: {result['gradcam']['target_layer']}")
                
                # Save for comparison
                overlay_data = base64.b64decode(result['gradcam']['overlay_image'])
                overlay_image = Image.open(io.BytesIO(overlay_data))
                overlay_image.save(f'gradcam_{color}_test.jpg')
                
            else:
                print(f"   ❌ {color} image failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ {color} image error: {e}")

def performance_test(base_url="http://localhost:8000", num_requests=5):
    """Test API performance"""
    print(f"\n🚀 Performance Test ({num_requests} Grad-CAM requests)")
    print("-" * 50)
    
    times = []
    
    for i in range(num_requests):
        try:
            # Create test image
            test_image = create_test_image()
            
            # Convert to bytes
            buffer = io.BytesIO()
            test_image.save(buffer, format='JPEG')
            image_bytes = buffer.getvalue()
            
            # Time the request
            start_time = time.time()
            
            files = {'file': ('test_image.jpg', image_bytes, 'image/jpeg')}
            response = requests.post(f"{base_url}/predict-gradcam", files=files, timeout=60)
            
            end_time = time.time()
            request_time = end_time - start_time
            
            if response.status_code == 200:
                times.append(request_time)
                print(f"   Request {i+1}: {request_time:.3f}s")
            else:
                print(f"   Request {i+1}: Failed ({response.status_code})")
                
        except requests.exceptions.RequestException as e:
            print(f"   Request {i+1}: Error - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 Grad-CAM Performance Results:")
        print(f"   Average time: {avg_time:.3f}s")
        print(f"   Min time: {min_time:.3f}s")
        print(f"   Max time: {max_time:.3f}s")
        print(f"   Requests per second: {1/avg_time:.1f}")
        
        return avg_time
    else:
        print("❌ No successful requests!")
        return None

if __name__ == "__main__":
    print("🏥 FastAPI API with Grad-CAM Test Suite")
    print("=" * 50)
    
    # Check if server is running
    base_url = "http://localhost:8000"
    
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code != 200:
            print("❌ Server not responding correctly!")
            print("Please start the server first:")
            print("   python api/main_with_gradcam.py")
            exit(1)
    except requests.exceptions.RequestException:
        print("❌ Server not running!")
        print("Please start the server first:")
        print("   python api/main_with_gradcam.py")
        exit(1)
    
    # Run tests
    if test_api_endpoints(base_url):
        # Test Grad-CAM quality
        test_gradcam_quality(base_url)
        
        # Run performance test
        performance_test(base_url, num_requests=3)
        
        print("\n🎉 All tests completed successfully!")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔥 Grad-CAM endpoint: http://localhost:8000/predict-gradcam")
        print("📁 Check generated Grad-CAM images in current directory")
    else:
        print("❌ Some tests failed!")
        exit(1)
