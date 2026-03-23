#!/usr/bin/env python3
"""
Real Grad-CAM Implementation for Skin Disease Classification

This script implements actual Grad-CAM visualization using real gradients
from the PyTorch model, with proper layer hooking and heatmap generation.

Author: AI Assistant
Date: 2025-03-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class GradCAM:
    """
    Real Grad-CAM implementation that uses actual gradients from the model
    to generate class activation maps.
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        """
        Initialize Grad-CAM
        
        Args:
            model: PyTorch model
            target_layer: Name of the target convolutional layer
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients"""
        
        def forward_hook(module, input, output):
            """Capture forward activations"""
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            """Capture backward gradients"""
            self.gradients = grad_output[0].detach()
        
        # Find and hook the target layer
        target_module = self._find_layer_by_name(self.target_layer)
        if target_module is None:
            raise ValueError(f"Layer '{self.target_layer}' not found in model")
        
        # Register hooks
        forward_handle = target_module.register_forward_hook(forward_hook)
        backward_handle = target_module.register_backward_hook(backward_hook)
        
        self.hooks.extend([forward_handle, backward_handle])
    
    def _find_layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        """Find a layer in the model by its name"""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        return None
    
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
        self.model.eval()
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backward pass
        target = output[0, class_idx]
        
        # Backward pass to get gradients
        target.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # Shape: (1, C, H, W)
        activations = self.activations  # Shape: (1, C, H, W)
        
        if gradients is None or activations is None:
            raise RuntimeError("Failed to capture gradients or activations")
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # Shape: (1, C, 1, 1)
        
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * activations, dim=1)  # Shape: (1, H, W)
        
        # Apply ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Convert to numpy and resize
        cam = cam.squeeze().cpu().numpy()  # Shape: (H, W)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def generate_heatmap(self, cam: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Convert CAM to heatmap
        
        Args:
            cam: Class activation map
            original_size: Original image size (height, width)
            
        Returns:
            Heatmap as numpy array
        """
        # Resize CAM to original image size
        cam_resized = cv2.resize(cam, original_size)
        
        # Convert to heatmap (apply colormap)
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), 
            cv2.COLORMAP_JET
        )
        
        return heatmap
    
    def overlay_heatmap(self, original_image: np.ndarray, heatmap: np.ndarray, 
                        alpha: float = 0.4) -> np.ndarray:
        """
        Overlay heatmap on original image
        
        Args:
            original_image: Original image as numpy array
            heatmap: Heatmap as numpy array
            alpha: Transparency factor for heatmap
            
        Returns:
            Overlaid image as numpy array
        """
        # Ensure both images have the same size
        if original_image.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
        
        return overlay
    
    def cleanup(self):
        """Remove hooks to prevent memory leaks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class SkinDiseaseGradCAM:
    """
    Specialized Grad-CAM for skin disease classification models
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize Grad-CAM for skin disease model
        
        Args:
            model: Trained skin disease classification model
        """
        self.model = model
        self.grad_cam = None
        self.class_names = ['acne', 'psoriasis', 'eczema']
        
        # Auto-detect target layer based on model architecture
        self.target_layer = self._find_target_layer()
        
        # Initialize Grad-CAM
        self.grad_cam = GradCAM(model, self.target_layer)
    
    def _find_target_layer(self) -> str:
        """
        Automatically find the best target layer for Grad-CAM
        
        Returns:
            Name of the target convolutional layer
        """
        # Common layer names for different architectures
        possible_layers = [
            # EfficientNet
            'features.8.0',
            'features.7.0', 
            'features.6.0',
            # ResNet
            'layer4.2.conv3',
            'layer4.1.conv2',
            'layer3.3.conv3',
            # Custom CNN
            'features.8',
            'features.7',
            'features.6'
        ]
        
        # Find the first existing layer
        for layer_name in possible_layers:
            if self._find_layer_by_name(layer_name) is not None:
                print(f"Found target layer: {layer_name}")
                return layer_name
        
        # Fallback: find any convolutional layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                print(f"Using fallback layer: {name}")
                return name
        
        raise ValueError("No suitable convolutional layer found")
    
    def _find_layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        """Find a layer in the model by its name"""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def generate_visualization(self, image: Image.Image, predicted_class: str) -> Dict[str, Any]:
        """
        Generate complete Grad-CAM visualization
        
        Args:
            image: Input PIL image
            predicted_class: Predicted class name
            
        Returns:
            Dictionary containing all visualization components
        """
        # Get class index
        class_idx = self.class_names.index(predicted_class)
        
        # Preprocess image
        input_tensor = self._preprocess_image(image)
        
        # Generate CAM
        cam = self.grad_cam.generate_cam(input_tensor, class_idx)
        
        # Convert original image to numpy
        original_np = np.array(image)
        
        # Generate heatmap
        heatmap = self.grad_cam.generate_heatmap(cam, (original_np.shape[0], original_np.shape[1]))
        
        # Create overlay
        overlay = self.grad_cam.overlay_heatmap(original_np, heatmap)
        
        # Convert to base64 for API response
        import base64
        from io import BytesIO
        
        def image_to_base64(img_array):
            """Convert numpy array to base64 string"""
            img_pil = Image.fromarray(img_array)
            buffer = BytesIO()
            img_pil.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return img_str
        
        return {
            'original_image': image_to_base64(original_np),
            'heatmap_image': image_to_base64(heatmap),
            'overlay_image': image_to_base64(overlay),
            'class_name': predicted_class,
            'class_index': class_idx,
            'target_layer': self.target_layer
        }
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: PIL image
            
        Returns:
            Preprocessed tensor
        """
        # Define transforms (same as training)
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        input_tensor = transform(image).unsqueeze(0)
        
        return input_tensor
    
    def cleanup(self):
        """Clean up resources"""
        if self.grad_cam:
            self.grad_cam.cleanup()

# Test function
def test_gradcam():
    """Test Grad-CAM implementation with a sample model"""
    print("Testing Grad-CAM implementation...")
    
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(128, num_classes)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # Create model and Grad-CAM
    model = SimpleModel()
    gradcam = SkinDiseaseGradCAM(model)
    
    # Create sample image
    sample_image = Image.new('RGB', (224, 224), color='red')
    
    # Generate visualization
    try:
        result = gradcam.generate_visualization(sample_image, 'acne')
        print("✅ Grad-CAM test successful!")
        print(f"Generated visualization for class: {result['class_name']}")
        print(f"Target layer: {result['target_layer']}")
        print(f"Images generated: {len([k for k in result.keys() if 'image' in k])}")
    except Exception as e:
        print(f"❌ Grad-CAM test failed: {e}")
    finally:
        gradcam.cleanup()

if __name__ == "__main__":
    test_gradcam()
