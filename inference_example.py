#!/usr/bin/env python3
"""
Inference Example for EfficientNetB4 Skin Disease Classifier

Example script demonstrating how to use the trained model for inference
on new skin disease images.

Author: AI Assistant
Date: 2025-03-22
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import json
from typing import Dict, List, Tuple, Optional

from efficientnet_skin_classifier import EfficientNetB4Classifier, ModelConfig


class SkinDiseaseInference:
    """Inference class for skin disease classification"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize inference class
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.class_names = ['acne', 'psoriasis', 'eczema']
        
        # Setup preprocessing transforms
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup device for inference"""
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        print(f"Using device: {device}")
        return device
    
    def _load_model(self, model_path: str) -> EfficientNetB4Classifier:
        """Load trained model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract config if available
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            config = ModelConfig()
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        else:
            config = ModelConfig()
        
        # Create model
        model = EfficientNetB4Classifier(
            num_classes=config.NUM_CLASSES,
            dropout_rate=config.DROPOUT_RATE,
            pretrained=False  # Don't use pretrained weights for inference
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Model epoch: {checkpoint.get('epoch', 'Unknown')}")
        
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"Model validation accuracy: {metrics.get('accuracy', 'Unknown'):.4f}")
        
        return model
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed tensor
        """
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Read image using OpenCV (BGR)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(self, image_path: str, return_probabilities: bool = True) -> Dict:
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to the input image
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        # Prepare results
        results = {
            'predicted_class': self.class_names[predicted_class_idx],
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence,
            'image_path': image_path
        }
        
        if return_probabilities:
            class_probabilities = {}
            for i, class_name in enumerate(self.class_names):
                class_probabilities[class_name] = probabilities[0][i].item()
            
            results['class_probabilities'] = class_probabilities
        
        return results
    
    def predict_batch(self, image_paths: List[str], return_probabilities: bool = True) -> List[Dict]:
        """
        Make predictions on multiple images
        
        Args:
            image_paths: List of image paths
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.predict(image_path, return_probabilities)
                results.append(result)
                print(f"✓ Processed {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"✗ Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, image_path: str, save_path: Optional[str] = None):
        """
        Visualize prediction with confidence scores
        
        Args:
            image_path: Path to the input image
            save_path: Path to save the visualization (optional)
        """
        import matplotlib.pyplot as plt
        
        # Make prediction
        result = self.predict(image_path, return_probabilities=True)
        
        # Load original image for visualization
        image = Image.open(image_path).convert('RGB')
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display image
        ax1.imshow(image)
        ax1.set_title(f'Input Image\nPredicted: {result["predicted_class"]}')
        ax1.axis('off')
        
        # Display confidence scores
        classes = list(result['class_probabilities'].keys())
        confidences = list(result['class_probabilities'].values())
        colors = ['green' if cls == result['predicted_class'] else 'lightblue' for cls in classes]
        
        bars = ax2.barh(classes, confidences, color=colors)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Confidence')
        ax2.set_title('Class Probabilities')
        ax2.grid(True, alpha=0.3)
        
        # Add confidence values on bars
        for bar, conf in zip(bars, confidences):
            ax2.text(conf + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{conf:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main function demonstrating inference usage"""
    
    print("="*80)
    print("SKIN DISEASE CLASSIFICATION - INFERENCE EXAMPLE")
    print("="*80)
    
    # Check if model exists
    model_path = 'models/best_model.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using train_model.py")
        return
    
    # Initialize inference class
    try:
        inference = SkinDiseaseInference(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return
    
    # Example usage
    print("\n" + "="*50)
    print("SINGLE IMAGE PREDICTION")
    print("="*50)
    
    # Replace with your image path
    example_image_path = "dataset/test/acne/acne_000001.jpg"
    
    if os.path.exists(example_image_path):
        try:
            result = inference.predict(example_image_path)
            
            print(f"Image: {os.path.basename(example_image_path)}")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Class Probabilities:")
            for class_name, prob in result['class_probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")
            
            # Visualize prediction
            print("\nGenerating visualization...")
            inference.visualize_prediction(example_image_path, 'prediction_example.png')
            
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
    else:
        print(f"⚠️  Example image not found at {example_image_path}")
        print("Please provide a valid image path")
    
    # Batch prediction example
    print("\n" + "="*50)
    print("BATCH PREDICTION EXAMPLE")
    print("="*50)
    
    # Get some test images (modify as needed)
    test_images = []
    test_dir = "dataset/test"
    
    if os.path.exists(test_dir):
        for class_name in ['acne', 'psoriasis', 'eczema']:
            class_dir = os.path.join(test_dir, class_name)
            if os.path.exists(class_dir):
                images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                         if f.endswith('.jpg')][:3]  # Take first 3 images per class
                test_images.extend(images)
    
    if test_images:
        print(f"Processing {len(test_images)} test images...")
        batch_results = inference.predict_batch(test_images)
        
        # Summary statistics
        successful_predictions = [r for r in batch_results if 'error' not in r]
        failed_predictions = len(batch_results) - len(successful_predictions)
        
        print(f"\nBatch Prediction Summary:")
        print(f"  Total images: {len(batch_results)}")
        print(f"  Successful: {len(successful_predictions)}")
        print(f"  Failed: {failed_predictions}")
        
        if successful_predictions:
            # Class distribution
            class_counts = {}
            for result in successful_predictions:
                pred_class = result['predicted_class']
                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
            
            print(f"\nPredicted Class Distribution:")
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count}")
            
            # Average confidence
            avg_confidence = np.mean([r['confidence'] for r in successful_predictions])
            print(f"\nAverage Confidence: {avg_confidence:.4f}")
        
        # Save batch results
        with open('batch_predictions.json', 'w') as f:
            json.dump(batch_results, f, indent=2)
        print(f"\nBatch results saved to batch_predictions.json")
    
    else:
        print("⚠️  No test images found for batch prediction")
    
    print("\n" + "="*80)
    print("INFERENCE EXAMPLE COMPLETED")
    print("="*80)
    print("Usage tips:")
    print("1. Replace example_image_path with your actual image path")
    print("2. Use the SkinDiseaseInference class in your own applications")
    print("3. Batch processing is efficient for multiple images")
    print("4. Visualizations help understand model confidence")


if __name__ == "__main__":
    main()
