#!/usr/bin/env python3
"""
Grad-CAM Demo and Testing Script

This script demonstrates the Grad-CAM visualization system with sample images
and provides testing functionality for the visualization pipeline.

Author: AI Assistant
Date: 2025-03-22
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import Grad-CAM components
from gradcam_visualization import MedicalGradCAM, GradCAMConfig
from efficientnet_skin_classifier import EfficientNetB4Classifier


def create_sample_images():
    """Create sample images for testing"""
    print("Creating sample images for testing...")
    
    # Create output directory
    sample_dir = 'sample_images'
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create synthetic skin disease images
    for class_name in ['acne', 'psoriasis', 'eczema']:
        for i in range(3):
            # Create a simple synthetic image
            image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            
            # Add some disease-specific patterns
            if class_name == 'acne':
                # Add red dots for acne
                for _ in range(20):
                    x, y = np.random.randint(0, 224, 2)
                    cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
            
            elif class_name == 'psoriasis':
                # Add patches for psoriasis
                for _ in range(5):
                    x, y = np.random.randint(0, 200, 2)
                    cv2.rectangle(image, (x, y), (x+24, y+24), (255, 165, 0), -1)
            
            elif class_name == 'eczema':
                # Add irregular patches for eczema
                for _ in range(8):
                    x, y = np.random.randint(0, 200, 2)
                    cv2.ellipse(image, (x+12, y+12), (15, 10), 0, 0, 360, (255, 192, 203), -1)
            
            # Save image
            image_path = os.path.join(sample_dir, f'{class_name}_{i+1}.jpg')
            Image.fromarray(image).save(image_path)
    
    print(f"Sample images created in {sample_dir}")
    return sample_dir


def test_gradcam_with_sample_model():
    """Test Grad-CAM with a sample model"""
    print("Testing Grad-CAM with sample model...")
    
    # Create a simple model
    model = EfficientNetB4Classifier()
    model.eval()
    
    # Create Grad-CAM instance
    config = GradCAMConfig()
    gradcam = MedicalGradCAM(model, config)
    
    # Create sample images
    sample_dir = create_sample_images()
    
    # Test with one image
    sample_image = os.path.join(sample_dir, 'acne_1.jpg')
    
    if os.path.exists(sample_image):
        print(f"Testing with {sample_image}")
        
        # Generate visualization
        results = gradcam.generate_single_visualization(sample_image, 'test_sample')
        
        print(f"✅ Test successful!")
        print(f"  Predicted class: {results['class_name']}")
        print(f"  Confidence: {results['confidence']:.3f}")
        print(f"  CAM shape: {results['cam'].shape}")
        print(f"  Heatmap shape: {results['heatmap'].shape}")
        
        return True
    else:
        print(f"❌ Sample image not found: {sample_image}")
        return False


def demo_gradcam_features():
    """Demonstrate Grad-CAM features"""
    print("\n" + "="*60)
    print("GRAD-CAM FEATURES DEMONSTRATION")
    print("="*60)
    
    # Test with sample model
    if test_gradcam_with_sample_model():
        print("\n✅ Grad-CAM implementation working correctly!")
        
        print("\n🔥 GRAD-CAM FEATURES:")
        print("  • Class activation map generation")
        print("  • Heatmap visualization")
        print("  • Overlay creation")
        print("  • Medical analysis")
        print("  • Disease-specific insights")
        print("  • Batch processing")
        print("  • Comprehensive reporting")
        
        print("\n🏥 MEDICAL INTERPRETABILITY:")
        print("  • Disease-specific activation patterns")
        print("  • Clinical feature highlighting")
        print("  • Confidence-based analysis")
        print("  • Medical annotation")
        print("  • Research-ready outputs")
        
        print("\n📊 OUTPUT FORMATS:")
        print("  • Original images")
        print("  • Activation heatmaps")
        print("  • Overlay visualizations")
        print("  • Combined analysis plots")
        print("  • Batch summary reports")
        print("  • Comprehensive JSON reports")
        
    else:
        print("\n❌ Grad-CAM test failed!")


def run_interactive_demo():
    """Run interactive demo"""
    print("\n" + "="*60)
    print("INTERACTIVE GRAD-CAM DEMO")
    print("="*60)
    
    # Check for trained model
    model_paths = [
        'models/best_model.pth',
        'models_hybrid/best_hybrid_model.pth',
        'models_groupdro/best_groupdro_model.pth'
    ]
    
    available_models = [path for path in model_paths if os.path.exists(path)]
    
    if not available_models:
        print("❌ No trained models found!")
        print("Please train a model first.")
        return
    
    print("Available models:")
    for i, path in enumerate(available_models):
        print(f"  {i+1}. {path}")
    
    try:
        choice = int(input("\nSelect model (enter number): ")) - 1
        if 0 <= choice < len(available_models):
            model_path = available_models[choice]
        else:
            model_path = available_models[0]
    except (ValueError, IndexError):
        model_path = available_models[0]
    
    print(f"Using model: {model_path}")
    
    # Load model
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Determine model type
        if 'hybrid' in model_path:
            from cnn_vit_hybrid import CNNViTHybrid
            model = CNNViTHybrid()
        else:
            model = EfficientNetB4Classifier()
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Model loaded successfully")
        
        # Initialize Grad-CAM
        config = GradCAMConfig()
        gradcam = MedicalGradCAM(model, config)
        
        # Check for test images
        test_dataset_path = 'dataset/test'
        if os.path.exists(test_dataset_path):
            print("\n📁 Found test dataset")
            
            # Get sample images
            sample_images = []
            for class_name in ['acne', 'psoriasis', 'eczema']:
                class_dir = os.path.join(test_dataset_path, class_name)
                if os.path.exists(class_dir):
                    image_files = [f for f in os.listdir(class_dir) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:2]
                    sample_images.extend([os.path.join(class_dir, f) for f in image_files])
            
            if sample_images:
                print(f"Found {len(sample_images)} sample images")
                
                # Process a few images
                for i, image_path in enumerate(sample_images[:3]):
                    print(f"\nProcessing image {i+1}: {os.path.basename(image_path)}")
                    
                    results = gradcam.generate_single_visualization(image_path, f'demo_{i+1}')
                    
                    print(f"  Predicted: {results['class_name']}")
                    print(f"  Confidence: {results['confidence']:.3f}")
                    print(f"  Saved to: {config.OUTPUT_DIR}/demo_{i+1}_*.jpg")
                
                print(f"\n✅ Demo completed! Check {config.OUTPUT_DIR}/ for results")
                
            else:
                print("❌ No images found in test dataset")
        else:
            print("❌ Test dataset not found")
            print("Creating sample images for demo...")
            sample_dir = create_sample_images()
            
            # Process sample images
            sample_images = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) 
                           if f.endswith('.jpg')][:3]
            
            for i, image_path in enumerate(sample_images):
                print(f"\nProcessing sample image {i+1}: {os.path.basename(image_path)}")
                results = gradcam.generate_single_visualization(image_path, f'demo_{i+1}')
                print(f"  Predicted: {results['class_name']}")
                print(f"  Confidence: {results['confidence']:.3f}")
            
            print(f"\n✅ Demo completed! Check {config.OUTPUT_DIR}/ for results")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def main():
    """Main function for Grad-CAM demo"""
    print("="*80)
    print("GRAD-CAM VISUALIZATION DEMO AND TESTING")
    print("="*80)
    print("Testing and demonstrating Grad-CAM for skin disease classification")
    print("="*80)
    
    print("Choose an option:")
    print("1. Test Grad-CAM with sample model")
    print("2. Run interactive demo with trained model")
    print("3. Show features overview")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            test_gradcam_with_sample_model()
        elif choice == '2':
            run_interactive_demo()
        elif choice == '3':
            demo_gradcam_features()
        else:
            print("Invalid choice. Running feature demo...")
            demo_gradcam_features()
    
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
    except Exception as e:
        print(f"\nError: {str(e)}")
    
    print("\n" + "="*80)
    print("GRAD-CAM DEMO COMPLETED")
    print("="*80)
    print("For full visualization, run:")
    print("python gradcam_visualization.py")
    print("\nThis will:")
    print("• Load your trained model")
    print("• Process test dataset images")
    print("• Generate comprehensive visualizations")
    print("• Create medical analysis reports")
    print("• Provide research-ready outputs")


if __name__ == "__main__":
    main()
