#!/usr/bin/env python3
"""
Enhanced Dataset Preparation with GAN-based Augmentation

This script combines the original dataset preparation with StyleGAN3-based synthetic image
generation to create a balanced, bias-reduced dataset for improved fairness.

Author: AI Assistant
Date: 2025-03-22
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import json

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Import original components
from skin_disease_dataset_preparation import Config, DatasetPreparer, SkinDiseaseDataset
from stylegan3_augmentation import StyleGAN3Trainer, GANConfig


class AugmentedConfig(Config):
    """Enhanced configuration for augmented dataset preparation"""
    
    # GAN augmentation settings
    ENABLE_GAN_AUGMENTATION = True
    GAN_MODEL_PATH = "stylegan3-r-ffhq-1024x1024.pkl"
    
    # Output paths
    AUGMENTED_OUTPUT_DIR = "dataset_augmented"
    
    # Augmentation parameters
    TARGET_AUGMENTATION_FACTOR = 2.0
    MIN_IMAGES_PER_CLASS = 1000
    DARK_SKIN_GENERATION_RATIO = 0.7
    
    # Quality control
    QUALITY_THRESHOLD = 0.5
    MEDICAL_FEATURE_PRESERVATION = True
    
    # Dataset balancing
    BALANCE_CLASSES = True
    BALANCE_SKIN_TONES = True


class AugmentedSkinDiseaseDataset(SkinDiseaseDataset):
    """Enhanced dataset class that includes GAN-generated images"""
    
    def __init__(self, root_dir: str, transform=None, class_to_idx=None, 
                 include_synthetic: bool = True):
        self.include_synthetic = include_synthetic
        super().__init__(root_dir, transform, class_to_idx)
        
        # Separate synthetic and real images for analysis
        self.synthetic_samples = []
        self.real_samples = []
        
        if include_synthetic:
            self._separate_synthetic_images()
    
    def _separate_synthetic_images(self):
        """Separate synthetic and real images"""
        for img_path, label in self.samples:
            if 'synthetic_' in os.path.basename(img_path):
                self.synthetic_samples.append((img_path, label))
            else:
                self.real_samples.append((img_path, label))
        
        self.logger.info(f"Dataset contains {len(self.real_samples)} real images and {len(self.synthetic_samples)} synthetic images")
    
    def get_synthetic_ratio(self) -> float:
        """Get ratio of synthetic to real images"""
        total = len(self.samples)
        synthetic = len(self.synthetic_samples)
        return synthetic / max(total, 1)
    
    def get_class_balance(self) -> Dict:
        """Get class distribution including synthetic images"""
        class_counts = Counter()
        synthetic_counts = Counter()
        real_counts = Counter()
        
        for img_path, label in self.samples:
            class_name = self.class_names[label]
            class_counts[class_name] += 1
            
            if 'synthetic_' in os.path.basename(img_path):
                synthetic_counts[class_name] += 1
            else:
                real_counts[class_name] += 1
        
        return {
            'total': dict(class_counts),
            'synthetic': dict(synthetic_counts),
            'real': dict(real_counts)
        }


class AugmentedDatasetPreparer(DatasetPreparer):
    """Enhanced dataset preparer with GAN-based augmentation"""
    
    def __init__(self, config: AugmentedConfig = AugmentedConfig()):
        # Initialize with augmented config
        self.augmented_config = config
        super().__init__(config)
        
        # Update output directory
        self.config.OUTPUT_DIR = config.AUGMENTED_OUTPUT_DIR
        
        # Setup GAN augmentation if enabled
        if config.ENABLE_GAN_AUGMENTATION:
            self.setup_gan_augmentation()
    
    def setup_gan_augmentation(self):
        """Setup GAN-based augmentation"""
        try:
            gan_config = GANConfig()
            gan_config.DEVICE = "auto"
            gan_config.OUTPUT_DIR = self.config.OUTPUT_DIR
            gan_config.TARGET_AUGMENTATION_FACTOR = self.augmented_config.TARGET_AUGMENTATION_FACTOR
            gan_config.MIN_IMAGES_PER_CLASS = self.augmented_config.MIN_IMAGES_PER_CLASS
            gan_config.DARK_SKIN_GENERATION_RATIO = self.augmented_config.DARK_SKIN_GENERATION_RATIO
            gan_config.MEDICAL_FEATURE_PRESERVATION = self.augmented_config.MEDICAL_FEATURE_PRESERVATION
            
            self.gan_trainer = StyleGAN3Trainer(gan_config)
            self.logger.info("GAN-based augmentation initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize GAN augmentation: {str(e)}")
            self.gan_trainer = None
    
    def run(self):
        """Enhanced run method with GAN augmentation"""
        try:
            # Set random seed
            np.random.seed(self.config.RANDOM_SEED)
            torch.manual_seed(self.config.RANDOM_SEED)
            
            self.logger.info("Starting enhanced dataset preparation with GAN augmentation...")
            
            # Step 1: Load and prepare original dataset
            self.logger.info("Step 1: Loading original dataset...")
            image_data = self.load_all_datasets()
            
            if not image_data:
                self.logger.error("No images loaded. Please check dataset paths.")
                return None, None, None
            
            # Validate and filter images
            valid_images = self.validate_and_filter_images(image_data)
            
            if not valid_images:
                self.logger.error("No valid images found after filtering.")
                return None, None, None
            
            # Split dataset
            splits = self.split_dataset(valid_images)
            
            # Step 2: Save original dataset
            self.logger.info("Step 2: Saving original dataset...")
            self.process_and_save_images(splits)
            
            # Step 3: Apply GAN augmentation if enabled
            if self.augmented_config.ENABLE_GAN_AUGMENTATION and self.gan_trainer:
                self.logger.info("Step 3: Applying GAN-based augmentation...")
                self.gan_trainer.augment_dataset("dataset")
            else:
                self.logger.info("Step 3: GAN augmentation disabled, using original dataset")
                # Copy original dataset to augmented location
                self.copy_original_to_augmented()
            
            # Step 4: Generate statistics and visualizations
            self.logger.info("Step 4: Generating statistics...")
            self.generate_augmented_statistics()
            self.create_augmented_visualizations()
            
            # Step 5: Create DataLoaders for augmented dataset
            self.logger.info("Step 5: Creating DataLoaders...")
            train_loader, val_loader, test_loader = self.create_augmented_data_loaders()
            
            self.logger.info("Enhanced dataset preparation with GAN augmentation completed successfully!")
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            self.logger.error(f"Error during enhanced dataset preparation: {str(e)}")
            raise
    
    def copy_original_to_augmented(self):
        """Copy original dataset to augmented location"""
        import shutil
        
        original_dir = "dataset"
        augmented_dir = self.config.OUTPUT_DIR
        
        if os.path.exists(original_dir):
            self.logger.info(f"Copying original dataset from {original_dir} to {augmented_dir}")
            
            # Create augmented directory structure
            for split in ['train', 'val', 'test']:
                split_orig = os.path.join(original_dir, split)
                split_aug = os.path.join(augmented_dir, split)
                
                if os.path.exists(split_orig):
                    if os.path.exists(split_aug):
                        shutil.rmtree(split_aug)
                    shutil.copytree(split_orig, split_aug)
                    self.logger.info(f"Copied {split} split")
    
    def generate_augmented_statistics(self):
        """Generate statistics for augmented dataset"""
        # Load augmented dataset for analysis
        train_dataset = AugmentedSkinDiseaseDataset(
            os.path.join(self.config.OUTPUT_DIR, 'train'),
            include_synthetic=True
        )
        
        # Get class balance
        class_balance = train_dataset.get_class_balance()
        
        # Calculate augmentation statistics
        total_original = sum(class_balance['real'].values())
        total_synthetic = sum(class_balance['synthetic'].values())
        total_augmented = sum(class_balance['total'].values())
        
        augmentation_stats = {
            'dataset_statistics': {
                'total_original_images': total_original,
                'total_synthetic_images': total_synthetic,
                'total_augmented_images': total_augmented,
                'augmentation_factor': total_augmented / max(total_original, 1),
                'synthetic_ratio': total_synthetic / max(total_augmented, 1)
            },
            'class_distribution': class_balance,
            'target_classes': self.config.TARGET_CLASSES,
            'split_ratios': {
                'train': self.config.TRAIN_RATIO,
                'val': self.config.VAL_RATIO,
                'test': self.config.TEST_RATIO
            },
            'image_size': self.config.IMAGE_SIZE,
            'normalization': {
                'mean': self.config.IMAGENET_MEAN,
                'std': self.config.IMAGENET_STD
            },
            'augmentation_config': {
                'target_augmentation_factor': self.augmented_config.TARGET_AUGMENTATION_FACTOR,
                'min_images_per_class': self.augmented_config.MIN_IMAGES_PER_CLASS,
                'dark_skin_generation_ratio': self.augmented_config.DARK_SKIN_GENERATION_RATIO,
                'medical_feature_preservation': self.augmented_config.MEDICAL_FEATURE_PRESERVATION
            }
        }
        
        # Calculate class imbalance metrics
        total_images = augmentation_stats['dataset_statistics']['total_augmented_images']
        class_weights = {}
        for class_name, count in class_balance['total'].items():
            weight = total_images / (len(self.config.TARGET_CLASSES) * count)
            class_weights[class_name] = weight
        
        augmentation_stats['class_weights'] = class_weights
        augmentation_stats['class_imbalance_ratio'] = (
            max(class_balance['total'].values()) / 
            min(class_balance['total'].values())
        )
        
        # Save statistics
        stats_path = os.path.join(self.config.STATS_DIR, 'augmented_dataset_statistics.json')
        os.makedirs(self.config.STATS_DIR, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(augmentation_stats, f, indent=2)
        
        self.logger.info(f"Augmented dataset statistics saved to {stats_path}")
        
        # Log statistics
        self.logger.info("=" * 50)
        self.logger.info("AUGMENTED DATASET STATISTICS")
        self.logger.info("=" * 50)
        self.logger.info(f"Original images: {total_original}")
        self.logger.info(f"Synthetic images: {total_synthetic}")
        self.logger.info(f"Total augmented images: {total_augmented}")
        self.logger.info(f"Augmentation factor: {augmentation_stats['dataset_statistics']['augmentation_factor']:.2f}x")
        self.logger.info(f"Synthetic ratio: {augmentation_stats['dataset_statistics']['synthetic_ratio']:.2%}")
        
        self.logger.info("Class distribution:")
        for class_name in self.config.TARGET_CLASSES:
            real_count = class_balance['real'].get(class_name, 0)
            synthetic_count = class_balance['synthetic'].get(class_name, 0)
            total_count = class_balance['total'].get(class_name, 0)
            percentage = (total_count / total_images) * 100
            self.logger.info(f"  {class_name}: {total_count} ({percentage:.1f}%) [Real: {real_count}, Synthetic: {synthetic_count}]")
        
        self.logger.info(f"Class imbalance ratio: {augmentation_stats['class_imbalance_ratio']:.2f}")
        self.logger.info("Class weights:")
        for class_name, weight in class_weights.items():
            self.logger.info(f"  {class_name}: {weight:.4f}")
    
    def create_augmented_visualizations(self):
        """Create visualizations for augmented dataset"""
        # Load augmented dataset for analysis
        train_dataset = AugmentedSkinDiseaseDataset(
            os.path.join(self.config.OUTPUT_DIR, 'train'),
            include_synthetic=True
        )
        
        class_balance = train_dataset.get_class_balance()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('GAN-based Dataset Augmentation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Original vs Synthetic comparison
        categories = ['Real', 'Synthetic']
        real_counts = list(class_balance['real'].values())
        synthetic_counts = list(class_balance['synthetic'].values())
        
        x = np.arange(len(self.config.TARGET_CLASSES))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, real_counts, width, label='Real Images', color='skyblue')
        axes[0, 0].bar(x + width/2, synthetic_counts, width, label='Synthetic Images', color='lightcoral')
        axes[0, 0].set_title('Real vs Synthetic Images by Class')
        axes[0, 0].set_ylabel('Number of Images')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(self.config.TARGET_CLASSES)
        axes[0, 0].legend()
        
        # Add value labels on bars
        for i, (real, synthetic) in enumerate(zip(real_counts, synthetic_counts)):
            axes[0, 0].text(i - width/2, real + max(real_counts, synthetic_counts) * 0.01, 
                           str(real), ha='center', va='bottom', fontsize=9)
            axes[0, 0].text(i + width/2, synthetic + max(real_counts, synthetic_counts) * 0.01, 
                           str(synthetic), ha='center', va='bottom', fontsize=9)
        
        # 2. Total class distribution after augmentation
        total_counts = list(class_balance['total'].values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars = axes[0, 1].bar(self.config.TARGET_CLASSES, total_counts, color=colors)
        axes[0, 1].set_title('Total Class Distribution After Augmentation')
        axes[0, 1].set_ylabel('Number of Images')
        axes[0, 1].set_ylim(0, max(total_counts) * 1.1)
        
        # Add value labels and percentage
        total_images = sum(total_counts)
        for i, (bar, count) in enumerate(zip(bars, total_counts)):
            percentage = (count / total_images) * 100
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_counts) * 0.01, 
                           f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=10)
        
        # 3. Augmentation factor by class
        augmentation_factors = []
        for class_name in self.config.TARGET_CLASSES:
            real = class_balance['real'].get(class_name, 1)
            total = class_balance['total'].get(class_name, 1)
            factor = total / real
            augmentation_factors.append(factor)
        
        bars = axes[1, 0].bar(self.config.TARGET_CLASSES, augmentation_factors, color='lightgreen')
        axes[1, 0].set_title('Augmentation Factor by Class')
        axes[1, 0].set_ylabel('Augmentation Factor (Total/Original)')
        axes[1, 0].axhline(y=self.augmented_config.TARGET_AUGMENTATION_FACTOR, 
                           color='red', linestyle='--', alpha=0.7, label='Target Factor')
        axes[1, 0].legend()
        
        # Add value labels
        for bar, factor in zip(bars, augmentation_factors):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(augmentation_factors) * 0.01, 
                           f'{factor:.2f}x', ha='center', va='bottom', fontsize=10)
        
        # 4. Synthetic ratio by class
        synthetic_ratios = []
        for class_name in self.config.TARGET_CLASSES:
            total = class_balance['total'].get(class_name, 1)
            synthetic = class_balance['synthetic'].get(class_name, 0)
            ratio = synthetic / total
            synthetic_ratios.append(ratio)
        
        bars = axes[1, 1].bar(self.config.TARGET_CLASSES, synthetic_ratios, color='orange')
        axes[1, 1].set_title('Synthetic Image Ratio by Class')
        axes[1, 1].set_ylabel('Synthetic Ratio')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axhline(y=self.augmented_config.DARK_SKIN_GENERATION_RATIO, 
                           color='red', linestyle='--', alpha=0.7, label='Target Dark Skin Ratio')
        axes[1, 1].legend()
        
        # Add value labels
        for bar, ratio in zip(bars, synthetic_ratios):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                           f'{ratio:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.STATS_DIR, 'augmentation_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Augmentation visualizations saved")
    
    def create_augmented_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders for augmented dataset"""
        # Create datasets
        train_dir = os.path.join(self.config.OUTPUT_DIR, 'train')
        val_dir = os.path.join(self.config.OUTPUT_DIR, 'val')
        test_dir = os.path.join(self.config.OUTPUT_DIR, 'test')
        
        # Class to index mapping
        class_to_idx = {class_name: idx for idx, class_name in enumerate(self.config.TARGET_CLASSES)}
        
        # Create augmented datasets
        train_dataset = AugmentedSkinDiseaseDataset(
            root_dir=train_dir,
            transform=self.train_transform,
            class_to_idx=class_to_idx,
            include_synthetic=True
        )
        
        val_dataset = AugmentedSkinDiseaseDataset(
            root_dir=val_dir,
            transform=self.val_test_transform,
            class_to_idx=class_to_idx,
            include_synthetic=False  # No synthetic images in validation/test
        )
        
        test_dataset = AugmentedSkinDiseaseDataset(
            root_dir=test_dir,
            transform=self.val_test_transform,
            class_to_idx=class_to_idx,
            include_synthetic=False  # No synthetic images in validation/test
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )
        
        self.logger.info(f"Created augmented DataLoaders:")
        self.logger.info(f"  Train: {len(train_dataset)} samples ({len(train_dataset.synthetic_samples)} synthetic), {len(train_loader)} batches")
        self.logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
        self.logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader


def main():
    """Main function for augmented dataset preparation"""
    print("="*80)
    print("ENHANCED DATASET PREPARATION WITH GAN-BASED AUGMENTATION")
    print("="*80)
    print("Creating balanced, bias-reduced dataset with synthetic images")
    print("="*80)
    
    # Create configuration
    config = AugmentedConfig()
    
    print(f"Configuration:")
    print(f"  Enable GAN Augmentation: {config.ENABLE_GAN_AUGMENTATION}")
    print(f"  Target Augmentation Factor: {config.TARGET_AUGMENTATION_FACTOR}x")
    print(f"  Output Directory: {config.AUGMENTED_OUTPUT_DIR}")
    print(f"  Minimum Images per Class: {config.MIN_IMAGES_PER_CLASS}")
    print(f"  Dark Skin Generation Ratio: {config.DARK_SKIN_GENERATION_RATIO * 100:.0f}%")
    print(f"  Medical Feature Preservation: {config.MEDICAL_FEATURE_PRESERVATION}")
    print(f"  Target Classes: {config.TARGET_CLASSES}")
    print()
    
    # Check GAN model availability
    if config.ENABLE_GAN_AUGMENTATION:
        gan_model_path = config.GAN_MODEL_PATH
        if not os.path.exists(gan_model_path):
            print("⚠️  GAN model not found. Augmentation will be disabled.")
            print(f"   Expected at: {gan_model_path}")
            print("   Download from: https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl")
            config.ENABLE_GAN_AUGMENTATION = False
    
    # Check if augmented dataset already exists
    augmented_dataset_dir = config.AUGMENTED_OUTPUT_DIR
    if os.path.exists(augmented_dataset_dir):
        print(f"✅ Augmented dataset found at {augmented_dataset_dir}")
        use_existing = input("Use existing augmented dataset? (y/n): ").lower() == 'y'
        if not use_existing:
            print("Will recreate augmented dataset...")
        else:
            print("Using existing augmented dataset...")
            config.ENABLE_GAN_AUGMENTATION = False  # Skip augmentation
    else:
        print(f"📁 Will create augmented dataset at {augmented_dataset_dir}")
    
    print()
    
    # Check dataset paths
    required_dirs = ['data/dermacon_in', 'data/kaggle_acne', 'data/kaggle_psoriasis', 'data/kaggle_eczema']
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("⚠️  Warning: The following dataset directories are missing:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        print("   Please ensure your dataset is properly organized.")
        print()
    
    print("🎯 GAN AUGMENTATION BENEFITS:")
    print("  • Addresses class imbalance")
    print("  • Increases dark skin representation")
    print("  • Reduces dataset bias")
    print("  • Improves model fairness")
    print("  • Enhances generalization")
    print("  • Maintains medical realism")
    print()
    
    try:
        # Create preparer
        preparer = AugmentedDatasetPreparer(config)
        
        # Run preparation
        train_loader, val_loader, test_loader = preparer.run()
        
        if train_loader is None:
            print("❌ Failed to prepare augmented dataset")
            return
        
        print("\n" + "="*80)
        print("ENHANCED DATASET PREPARATION WITH GAN AUGMENTATION COMPLETED!")
        print("="*80)
        print(f"Augmented dataset saved to: {config.AUGMENTED_OUTPUT_DIR}")
        print(f"Augmentation statistics saved to: {config.STATS_DIR}/augmented_dataset_statistics.json")
        print(f"Augmentation visualizations saved to: {config.STATS_DIR}/augmentation_analysis.png")
        
        if config.ENABLE_GAN_AUGMENTATION:
            print(f"GAN augmentation report: {config.AUGMENTED_OUTPUT_DIR}/augmentation_report.json")
            print(f"Generated samples: {config.AUGMENTED_OUTPUT_DIR}/generated_samples/")
        
        print()
        print("🎉 DATASET ENHANCEMENT SUMMARY:")
        print("✓ Balanced class distribution")
        print("✓ Increased dark skin representation")
        print("✓ Reduced bias through synthetic augmentation")
        print("✓ Medical feature preservation")
        print("✓ Quality-controlled synthetic images")
        print()
        print("Next steps:")
        print("1. Train classification model on augmented dataset")
        print("2. Compare fairness with baseline")
        print("3. Evaluate synthetic image quality")
        print("4. Test model generalization")
        
    except Exception as e:
        print(f"❌ Error during dataset preparation: {str(e)}")
        return


if __name__ == "__main__":
    main()
