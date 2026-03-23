#!/usr/bin/env python3
"""
Test script for Skin Disease Classification API

Tests all API endpoints with sample data and validates responses.

Author: AI Assistant
Date: 2025-03-22
"""

import os
import io
import json
import base64
import time
import requests
from PIL import Image
import numpy as np
from typing import Dict, Any


class APITester:
    """Test suite for the Skin Disease Classification API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Create test images
        self.test_images = self.create_test_images()
        
        print(f"API Tester initialized for {base_url}")
    
    def create_test_images(self) -> Dict[str, bytes]:
        """Create test images for testing"""
        test_images = {}
        
        # Create synthetic skin disease images
        for disease in ['acne', 'psoriasis', 'eczema']:
            # Create a simple synthetic image
            image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            
            # Add disease-specific patterns
            if disease == 'acne':
                # Add red dots for acne
                for _ in range(20):
                    x, y = np.random.randint(0, 224, 2)
                    image[y-2:y+3, x-2:x+3] = [255, 0, 0]  # Red spots
            
            elif disease == 'psoriasis':
                # Add patches for psoriasis
                for _ in range(5):
                    x, y = np.random.randint(0, 200, 2)
                    image[y:y+25, x:x+25] = [255, 165, 0]  # Orange patches
            
            elif disease == 'eczema':
                # Add irregular patches for eczema
                for _ in range(8):
                    x, y = np.random.randint(0, 200, 2)
                    mask = np.zeros((25, 25), dtype=bool)
                    for i in range(25):
                        for j in range(25):
                            if (i-12)**2 + (j-12)**2 <= 144:  # Circle
                                mask[i, j] = True
                    image[y:y+25, x:x+25][mask] = [255, 192, 203]  # Pink patches
            
            # Convert to bytes
            pil_image = Image.fromarray(image)
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            test_images[disease] = img_bytes.getvalue()
        
        return test_images
    
    def test_health_endpoint(self) -> bool:
        """Test health check endpoint"""
        print("Testing health endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed")
                print(f"   Status: {data.get('status')}")
                print(f"   Model loaded: {data.get('model_loaded')}")
                print(f"   Device: {data.get('device')}")
                print(f"   Uptime: {data.get('uptime', 0):.2f}s")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            return False
    
    def test_model_info(self) -> bool:
        """Test model info endpoint"""
        print("\nTesting model info endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model info retrieved successfully")
                print(f"   Model type: {data.get('model_type')}")
                print(f"   Device: {data.get('device')}")
                print(f"   Classes: {data.get('class_names')}")
                print(f"   Input size: {data.get('input_size')}")
                print(f"   Model loaded: {data.get('model_loaded')}")
                return True
            else:
                print(f"❌ Model info failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Model info error: {str(e)}")
            return False
    
    def test_prediction_endpoint(self) -> bool:
        """Test prediction endpoint"""
        print("\nTesting prediction endpoint...")
        
        success_count = 0
        total_count = 0
        
        for disease, image_bytes in self.test_images.items():
            total_count += 1
            
            try:
                # Create file-like object
                files = {'file': (f'{disease}_test.jpg', image_bytes, 'image/jpeg')}
                
                start_time = time.time()
                response = self.session.post(f"{self.base_url}/predict", files=files)
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ {disease} prediction successful")
                    print(f"   Predicted: {data.get('class_label')}")
                    print(f"   Confidence: {data.get('confidence', 0):.3f}")
                    print(f"   Processing time: {end_time - start_time:.3f}s")
                    print(f"   Probabilities: {data.get('probabilities', {})}")
                    success_count += 1
                else:
                    print(f"❌ {disease} prediction failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    
            except Exception as e:
                print(f"❌ {disease} prediction error: {str(e)}")
        
        print(f"\nPrediction test results: {success_count}/{total_count} passed")
        return success_count == total_count
    
    def test_gradcam_endpoint(self) -> bool:
        """Test Grad-CAM endpoint"""
        print("\nTesting Grad-CAM endpoint...")
        
        success_count = 0
        total_count = 0
        
        for disease, image_bytes in self.test_images.items():
            total_count += 1
            
            try:
                # Create file-like object
                files = {'file': (f'{disease}_test.jpg', image_bytes, 'image/jpeg')}
                
                start_time = time.time()
                response = self.session.post(f"{self.base_url}/predict-with-gradcam", files=files)
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ {disease} Grad-CAM successful")
                    print(f"   Predicted: {data.get('class_label')}")
                    print(f"   Confidence: {data.get('confidence', 0):.3f}")
                    print(f"   Processing time: {end_time - start_time:.3f}s")
                    print(f"   Grad-CAM image size: {len(data.get('gradcam_image', ''))} chars")
                    print(f"   Heatmap image size: {len(data.get('heatmap_image', ''))} chars")
                    print(f"   Overlay image size: {len(data.get('overlay_image', ''))} chars")
                    
                    # Validate base64 images
                    for img_type in ['gradcam_image', 'heatmap_image', 'overlay_image']:
                        if img_type in data:
                            try:
                                base64.b64decode(data[img_type])
                                print(f"   {img_type}: Valid base64")
                            except:
                                print(f"   {img_type}: Invalid base64")
                    
                    success_count += 1
                else:
                    print(f"❌ {disease} Grad-CAM failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    
            except Exception as e:
                print(f"❌ {disease} Grad-CAM error: {str(e)}")
        
        print(f"\nGrad-CAM test results: {success_count}/{total_count} passed")
        return success_count == total_count
    
    def test_error_handling(self) -> bool:
        """Test error handling"""
        print("\nTesting error handling...")
        
        success_count = 0
        total_tests = 0
        
        # Test 1: Invalid file type
        total_tests += 1
        try:
            files = {'file': ('test.txt', b'text content', 'text/plain')}
            response = self.session.post(f"{self.base_url}/predict", files=files)
            
            if response.status_code == 400:
                print(f"✅ Invalid file type handled correctly")
                success_count += 1
            else:
                print(f"❌ Invalid file type not handled: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Invalid file type test error: {str(e)}")
        
        # Test 2: Large file
        total_tests += 1
        try:
            large_file = b'x' * (11 * 1024 * 1024)  # 11MB file
            files = {'file': ('large.jpg', large_file, 'image/jpeg')}
            response = self.session.post(f"{self.base_url}/predict", files=files)
            
            if response.status_code == 400:
                print(f"✅ Large file handled correctly")
                success_count += 1
            else:
                print(f"❌ Large file not handled: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Large file test error: {str(e)}")
        
        # Test 3: Invalid endpoint
        total_tests += 1
        try:
            response = self.session.get(f"{self.base_url}/invalid_endpoint")
            
            if response.status_code == 404:
                print(f"✅ Invalid endpoint handled correctly")
                success_count += 1
            else:
                print(f"❌ Invalid endpoint not handled: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Invalid endpoint test error: {str(e)}")
        
        print(f"\nError handling test results: {success_count}/{total_tests} passed")
        return success_count == total_tests
    
    def test_performance(self) -> Dict[str, float]:
        """Test API performance"""
        print("\nTesting performance...")
        
        # Test prediction performance
        image_bytes = self.test_images['acne']
        files = {'file': ('performance_test.jpg', image_bytes, 'image/jpeg')}
        
        times = []
        for i in range(10):
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/predict", files=files)
            end_time = time.time()
            
            if response.status_code == 200:
                times.append(end_time - start_time)
        
        if times:
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            print(f"✅ Performance test completed")
            print(f"   Average time: {avg_time:.3f}s")
            print(f"   Min time: {min_time:.3f}s")
            print(f"   Max time: {max_time:.3f}s")
            print(f"   Requests/second: {1/avg_time:.1f}")
            
            return {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'rps': 1/avg_time
            }
        else:
            print("❌ Performance test failed")
            return {}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results"""
        print("="*80)
        print("SKIN DISEASE CLASSIFICATION API TEST SUITE")
        print("="*80)
        
        results = {
            'health_check': self.test_health_endpoint(),
            'model_info': self.test_model_info(),
            'prediction': self.test_prediction_endpoint(),
            'gradcam': self.test_gradcam_endpoint(),
            'error_handling': self.test_error_handling(),
            'performance': self.test_performance()
        }
        
        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result is True)
        
        print("\n" + "="*80)
        print("TEST SUITE SUMMARY")
        print("="*80)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result is True else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! API is ready for production.")
        else:
            print("⚠️  Some tests failed. Please check the API.")
        
        print("="*80)
        
        return results


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Skin Disease Classification API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--test", choices=["all", "health", "predict", "gradcam", "errors", "performance"], 
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    print("Skin Disease Classification API Tester")
    print(f"Testing API at: {args.url}")
    print()
    
    # Create tester
    tester = APITester(args.url)
    
    # Run tests
    if args.test == "all":
        results = tester.run_all_tests()
    elif args.test == "health":
        tester.test_health_endpoint()
    elif args.test == "predict":
        tester.test_prediction_endpoint()
    elif args.test == "gradcam":
        tester.test_gradcam_endpoint()
    elif args.test == "errors":
        tester.test_error_handling()
    elif args.test == "performance":
        tester.test_performance()
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()
