#!/usr/bin/env python3
"""
Example Usage Script for Skin Disease Dataset Preparation with Data Augmentation

This script demonstrates how to use the enhanced dataset preparation script
with Albumentations-based data augmentation optimized for dark skin tones.

Author: AI Assistant
Date: 2025-03-22
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from skin_disease_dataset_preparation import DatasetPreparer, Config

def visualize_augmented_samples(train_loader, class_names, num_samples=6):
    """
    Visualize augmented samples from the training dataset
    
    Args:
        train_loader: PyTorch DataLoader for training data
        class_names: List of class names
        num_samples: Number of samples to visualize
    """
    # Get a batch of data
    images, labels = next(iter(train_loader))
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # Create subplot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        # Convert tensor to numpy and transpose
        img = images_denorm[i].permute(1, 2, 0).numpy()
        label = class_names[labels[i].item()]
        
        axes[i].imshow(img)
        axes[i].set_title(f'Class: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmented_samples.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Augmented samples visualization saved as 'augmented_samples.png'")

def test_data_loaders(train_loader, val_loader, test_loader):
    """
    Test the created DataLoaders
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
    """
    print("\n" + "="*50)
    print("DATALOADER TESTING")
    print("="*50)
    
    # Test train loader
    print(f"\nTraining DataLoader:")
    print(f"  Number of batches: {len(train_loader)}")
    print(f"  Batch size: {train_loader.batch_size}")
    
    # Get a sample batch
    images, labels = next(iter(train_loader))
    print(f"  Image tensor shape: {images.shape}")
    print(f"  Label tensor shape: {labels.shape}")
    print(f"  Image dtype: {images.dtype}")
    print(f"  Label dtype: {labels.dtype}")
    print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Test val loader
    print(f"\nValidation DataLoader:")
    print(f"  Number of batches: {len(val_loader)}")
    
    # Test test loader
    print(f"\nTest DataLoader:")
    print(f"  Number of batches: {len(test_loader)}")
    
    print("\nAll DataLoaders are working correctly!")

def main():
    """Main function demonstrating usage"""
    print("Skin Disease Dataset Preparation with Advanced Data Augmentation")
    print("="*70)
    
    # Create custom configuration (optional)
    config = Config()
    
    # You can modify configuration here if needed
    config.BATCH_SIZE = 16  # Smaller batch size for demonstration
    config.NUM_WORKERS = 2  # Reduce workers for demonstration
    
    print(f"Configuration:")
    print(f"  Image size: {config.IMAGE_SIZE}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Target classes: {config.TARGET_CLASSES}")
    print(f"  Augmentation probability: {config.AUGMENTATION_PROB}")
    print(f"  Horizontal flip probability: {config.HORIZONTAL_FLIP_PROB}")
    print(f"  Rotation range: ±{config.ROTATION_RANGE}°")
    print(f"  Hue shift limit: ±{config.HUE_SHIFT_LIMIT} (important for skin tones)")
    
    # Initialize dataset preparer
    preparer = DatasetPreparer(config)
    
    try:
        # Run dataset preparation and get DataLoaders
        print("\nStarting dataset preparation...")
        train_loader, val_loader, test_loader = preparer.run()
        
        # Test the DataLoaders
        test_data_loaders(train_loader, val_loader, test_loader)
        
        # Visualize augmented samples
        print("\nVisualizing augmented training samples...")
        visualize_augmented_samples(train_loader, config.TARGET_CLASSES)
        
        # Print class distribution information
        print("\n" + "="*50)
        print("CLASS DISTRIBUTION SUMMARY")
        print("="*50)
        
        # Count samples in each dataset
        train_samples = len(train_loader.dataset)
        val_samples = len(val_loader.dataset)
        test_samples = len(test_loader.dataset)
        total_samples = train_samples + val_samples + test_samples
        
        print(f"Total samples: {total_samples}")
        print(f"Training: {train_samples} ({train_samples/total_samples*100:.1f}%)")
        print(f"Validation: {val_samples} ({val_samples/total_samples*100:.1f}%)")
        print(f"Test: {test_samples} ({test_samples/total_samples*100:.1f}%)")
        
        print("\nDataset preparation completed successfully!")
        print("You can now use the DataLoaders for training your CNN-GAN-ViT hybrid model.")
        
        # Example of how to use the DataLoaders in training loop
        print("\n" + "="*50)
        print("EXAMPLE TRAINING LOOP USAGE")
        print("="*50)
        print("""
# Example training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Calculate validation metrics
""")
        
    except Exception as e:
        print(f"Error during dataset preparation: {str(e)}")
        print("Please check your dataset paths and ensure all required directories exist.")
        return

if __name__ == "__main__":
    main()
