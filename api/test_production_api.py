#!/usr/bin/env python3
"""
Test Production FastAPI API

This script tests the production FastAPI API with real model inference.

Author: AI Assistant
Date: 2025-03-23
"""

import requests
import json
import os
from PIL import Image
import io
import time

def create_test_image():
    """Create a test image for API testing"""
    # Create a simple test image
    image = Image.new('RGB', (224, 224), color='lightblue')
    return image

def test_api_endpoints(base_url="http://localhost:8000"):
    """Test all API endpoints"""
    print("🧪 Testing Production FastAPI API")
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
            print(f"   Input size: {model_info.get('input_size')}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info error: {e}")
        return False
    
    # Test prediction endpoint
    print("\n3. Testing Prediction Endpoint...")
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
            print("✅ Prediction successful!")
            print(f"   Predicted class: {result['class']}")
            print(f"   Confidence: {result['confidence']:.3f}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction error: {e}")
        return False
    
    # Test detailed prediction endpoint
    print("\n4. Testing Detailed Prediction Endpoint...")
    try:
        # Create test image
        test_image = create_test_image()
        
        # Convert to bytes
        buffer = io.BytesIO()
        test_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        # Send to API
        files = {'file': ('test_image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(f"{base_url}/predict-detailed", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Detailed prediction successful!")
            print(f"   Predicted class: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Class index: {result['class_index']}")
            print(f"   All probabilities: {result['all_probabilities']}")
            print(f"   Image size: {result['image_info']['size']}")
        else:
            print(f"❌ Detailed prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Detailed prediction error: {e}")
        return False
    
    # Test batch prediction endpoint
    print("\n5. Testing Batch Prediction Endpoint...")
    try:
        # Create multiple test images
        test_images = []
        for i, color in enumerate(['red', 'green', 'blue']):
            image = Image.new('RGB', (224, 224), color=color)
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            test_images.append(('test_image_{}.jpg'.format(i), buffer.getvalue(), 'image/jpeg'))
        
        # Send batch to API
        files = [('files', img) for img in test_images]
        response = requests.post(f"{base_url}/predict-batch", files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch prediction successful!")
            print(f"   Batch size: {result['batch_size']}")
            print(f"   Successful predictions: {result['successful_predictions']}")
            for i, prediction in enumerate(result['results']):
                if prediction.get('success', False):
                    print(f"   Image {i+1}: {prediction['class']} ({prediction['confidence']:.3f})")
                else:
                    print(f"   Image {i+1}: Error - {prediction.get('error', 'Unknown error')}")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Batch prediction error: {e}")
        return False
    
    # Test error handling
    print("\n6. Testing Error Handling...")
    
    # Test invalid file type
    try:
        files = {'file': ('test.txt', b'this is not an image', 'text/plain')}
        response = requests.post(f"{base_url}/predict", files=files, timeout=10)
        
        if response.status_code == 400:
            print("✅ Invalid file type handled correctly!")
        else:
            print(f"❌ Invalid file type not handled: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error handling test error: {e}")
        return False
    
    # Test too small image
    try:
        small_image = Image.new('RGB', (16, 16), color='red')
        buffer = io.BytesIO()
        small_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        files = {'file': ('small_image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(f"{base_url}/predict", files=files, timeout=10)
        
        if response.status_code == 400:
            print("✅ Small image handled correctly!")
        else:
            print(f"❌ Small image not handled: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Small image test error: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    return True

def performance_test(base_url="http://localhost:8000", num_requests=10):
    """Test API performance"""
    print(f"\n🚀 Performance Test ({num_requests} requests)")
    print("-" * 40)
    
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
            response = requests.post(f"{base_url}/predict", files=files, timeout=30)
            
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
        
        print(f"\n📊 Performance Results:")
        print(f"   Average time: {avg_time:.3f}s")
        print(f"   Min time: {min_time:.3f}s")
        print(f"   Max time: {max_time:.3f}s")
        print(f"   Requests per second: {1/avg_time:.1f}")
        
        return avg_time
    else:
        print("❌ No successful requests!")
        return None

if __name__ == "__main__":
    print("🏥 Production FastAPI API Test Suite")
    print("=" * 50)
    
    # Check if server is running
    base_url = "http://localhost:8000"
    
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code != 200:
            print("❌ Server not responding correctly!")
            print("Please start the server first:")
            print("   python api/main_production.py")
            exit(1)
    except requests.exceptions.RequestException:
        print("❌ Server not running!")
        print("Please start the server first:")
        print("   python api/main_production.py")
        exit(1)
    
    # Run tests
    if test_api_endpoints(base_url):
        # Run performance test
        performance_test(base_url, num_requests=5)
        
        print("\n🎉 All tests completed successfully!")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔥 Prediction endpoint: http://localhost:8000/predict")
    else:
        print("❌ Some tests failed!")
        exit(1)
