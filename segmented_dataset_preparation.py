#!/usr/bin/env python3
"""
Enhanced Dataset Preparation with SAM-based Lesion Segmentation

This script combines the original dataset preparation with SAM-based lesion segmentation
to create a focused dataset that emphasizes lesion areas for better classification performance.

Author: AI Assistant
Date: 2025-03-22
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
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
from sam_lesion_segmentation import LesionSegmenter, SAMConfig


class SegmentedConfig(Config):
    """Enhanced configuration for segmented dataset preparation"""
    
    # Segmentation settings
    ENABLE_SEGMENTATION = True
    SAM_MODEL_PATH = "sam_vit_h_4b8939.pth"
    
    # Output paths
    SEGMENTED_OUTPUT_DIR = "dataset_segmented"
    
    # Segmentation parameters
    CONFIDENCE_THRESHOLD = 0.5
    MIN_MASK_AREA = 100
    MAX_MASK_AREA = 50000
    
    # Dark skin optimization
    DARK_SKIN_ENHANCEMENT = True
    CONTRAST_ENHANCEMENT_FACTOR = 1.2
    
    # Fallback to original if segmentation fails
    USE_ORIGINAL_ON_FAILURE = True


class SegmentedSkinDiseaseDataset(SkinDiseaseDataset):
    """Enhanced dataset class that uses segmented images"""
    
    def __init__(self, root_dir: str, transform=None, class_to_idx=None, 
                 fallback_original: bool = True):
        self.fallback_original = fallback_original
        self.original_root_dir = root_dir.replace("dataset_segmented", "dataset")
        super().__init__(root_dir, transform, class_to_idx)
        
        # Check for missing segmented images and prepare fallback
        self._prepare_fallback_images()
    
    def _prepare_fallback_images(self):
        """Prepare fallback to original images if segmented ones are missing"""
        if not self.fallback_original:
            return
        
        missing_images = []
        
        for img_path, label in self.samples:
            if not os.path.exists(img_path):
                # Try to find original image
                original_path = img_path.replace("dataset_segmented", "dataset")
                if os.path.exists(original_path):
                    missing_images.append((original_path, label, img_path))
        
        if missing_images:
            print(f"Found {len(missing_images)} missing segmented images, using originals as fallback")
            
            # Update samples with original paths
            for original_path, label, segmented_path in missing_images:
                # Replace segmented path with original path
                for i, (sample_path, sample_label) in enumerate(self.samples):
                    if sample_path == segmented_path:
                        self.samples[i] = (original_path, sample_label)
                        break
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label


class SegmentedDatasetPreparer(DatasetPreparer):
    """Enhanced dataset preparer with SAM-based segmentation"""
    
    def __init__(self, config: SegmentedConfig = SegmentedConfig()):
        # Initialize with segmented config
        self.segmented_config = config
        super().__init__(config)
        
        # Update output directory
        self.config.OUTPUT_DIR = config.SEGMENTED_OUTPUT_DIR
        
        # Setup segmentation if enabled
        if config.ENABLE_SEGMENTATION:
            self.setup_segmentation()
    
    def setup_segmentation(self):
        """Setup SAM-based segmentation"""
        try:
            sam_config = SAMConfig()
            sam_config.DEVICE = "auto"
            sam_config.OUTPUT_DIR = self.config.OUTPUT_DIR
            sam_config.DARK_SKIN_ENHANCEMENT = self.segmented_config.DARK_SKIN_ENHANCEMENT
            sam_config.CONFIDENCE_THRESHOLD = self.segmented_config.CONFIDENCE_THRESHOLD
            
            self.segmenter = LesionSegmenter(sam_config)
            self.logger.info("SAM-based lesion segmentation initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize segmentation: {str(e)}")
            self.segmenter = None
    
    def run(self):
        """Enhanced run method with segmentation"""
        try:
            # Set random seed
            np.random.seed(self.config.RANDOM_SEED)
            torch.manual_seed(self.config.RANDOM_SEED)
            
            self.logger.info("Starting enhanced dataset preparation with segmentation...")
            
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
            
            # Step 2: Save original dataset (for fallback)
            self.logger.info("Step 2: Saving original dataset...")
            self.process_and_save_images(splits)
            
            # Step 3: Apply segmentation if enabled
            if self.segmented_config.ENABLE_SEGMENTATION and self.segmenter:
                self.logger.info("Step 3: Applying SAM-based lesion segmentation...")
                self.apply_segmentation()
            else:
                self.logger.info("Step 3: Segmentation disabled, using original dataset")
                # Copy original dataset to segmented location
                self.copy_original_to_segmented()
            
            # Step 4: Generate statistics and visualizations
            self.logger.info("Step 4: Generating statistics...")
            self.generate_statistics()
            self.create_visualizations()
            
            # Step 5: Create DataLoaders for segmented dataset
            self.logger.info("Step 5: Creating DataLoaders...")
            train_loader, val_loader, test_loader = self.create_segmented_data_loaders()
            
            self.logger.info("Enhanced dataset preparation with segmentation completed successfully!")
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            self.logger.error(f"Error during enhanced dataset preparation: {str(e)}")
            raise
    
    def apply_segmentation(self):
        """Apply SAM-based segmentation to the dataset"""
        if not self.segmenter:
            self.logger.error("Segmenter not initialized")
            return
        
        # Process all dataset splits
        self.segmenter.process_all_datasets()
        
        # Save segmentation report
        self.segmenter.save_segmentation_report()
    
    def copy_original_to_segmented(self):
        """Copy original dataset to segmented location"""
        import shutil
        
        original_dir = "dataset"
        segmented_dir = self.config.OUTPUT_DIR
        
        if os.path.exists(original_dir):
            self.logger.info(f"Copying original dataset from {original_dir} to {segmented_dir}")
            
            # Create segmented directory structure
            for split in ['train', 'val', 'test']:
                split_orig = os.path.join(original_dir, split)
                split_seg = os.path.join(segmented_dir, split)
                
                if os.path.exists(split_orig):
                    if os.path.exists(split_seg):
                        shutil.rmtree(split_seg)
                    shutil.copytree(split_orig, split_seg)
                    self.logger.info(f"Copied {split} split")
    
    def create_segmented_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders for segmented dataset"""
        # Create datasets
        train_dir = os.path.join(self.config.OUTPUT_DIR, 'train')
        val_dir = os.path.join(self.config.OUTPUT_DIR, 'val')
        test_dir = os.path.join(self.config.OUTPUT_DIR, 'test')
        
        # Class to index mapping
        class_to_idx = {class_name: idx for idx, class_name in enumerate(self.config.TARGET_CLASSES)}
        
        # Create segmented datasets
        train_dataset = SegmentedSkinDiseaseDataset(
            root_dir=train_dir,
            transform=self.train_transform,
            class_to_idx=class_to_idx,
            fallback_original=self.segmented_config.USE_ORIGINAL_ON_FAILURE
        )
        
        val_dataset = SegmentedSkinDiseaseDataset(
            root_dir=val_dir,
            transform=self.val_test_transform,
            class_to_idx=class_to_idx,
            fallback_original=self.segmented_config.USE_ORIGINAL_ON_FAILURE
        )
        
        test_dataset = SegmentedSkinDiseaseDataset(
            root_dir=test_dir,
            transform=self.val_test_transform,
            class_to_idx=class_to_idx,
            fallback_original=self.segmented_config.USE_ORIGINAL_ON_FAILURE
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
        
        self.logger.info(f"Created segmented DataLoaders:")
        self.logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        self.logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
        self.logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader


def main():
    """Main function for segmented dataset preparation"""
    print("="*80)
    print("ENHANCED DATASET PREPARATION WITH SAM-BASED LESION SEGMENTATION")
    print("="*80)
    print("Creating focused dataset with lesion area emphasis")
    print("="*80)
    
    # Create configuration
    config = SegmentedConfig()
    
    print(f"Configuration:")
    print(f"  Enable Segmentation: {config.ENABLE_SEGMENTATION}")
    print(f"  Output Directory: {config.SEGMENTED_OUTPUT_DIR}")
    print(f"  Dark Skin Enhancement: {config.DARK_SKIN_ENHANCEMENT}")
    print(f"  Use Original on Failure: {config.USE_ORIGINAL_ON_FAILURE}")
    print(f"  Target Classes: {config.TARGET_CLASSES}")
    print()
    
    # Check SAM model availability
    if config.ENABLE_SEGMENTATION:
        sam_model_path = config.SAM_MODEL_PATH
        if not os.path.exists(sam_model_path):
            print("⚠️  SAM model not found. Segmentation will be disabled.")
            print(f"   Expected at: {sam_model_path}")
            print("   Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            config.ENABLE_SEGMENTATION = False
    
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
    
    try:
        # Create preparer
        preparer = SegmentedDatasetPreparer(config)
        
        # Run preparation
        train_loader, val_loader, test_loader = preparer.run()
        
        print("\n" + "="*80)
        print("ENHANCED DATASET PREPARATION COMPLETED!")
        print("="*80)
        print(f"Segmented dataset saved to: {config.SEGMENTED_OUTPUT_DIR}")
        print(f"Original dataset available at: dataset/")
        
        if config.ENABLE_SEGMENTATION:
            print(f"Segmentation report: {config.SEGMENTED_OUTPUT_DIR}/segmentation_report.json")
            print(f"Segmentation visualizations: {config.SEGMENTED_OUTPUT_DIR}/visualizations/")
        
        print(f"Dataset statistics: {config.STATS_DIR}/dataset_statistics.json")
        print()
        print("Dataset ready for training with:")
        print("✓ Focused lesion areas")
        print("✓ Dark skin tone optimization")
        print("✓ Fallback to original images if segmentation fails")
        print("✓ Enhanced data augmentation")
        print()
        print("Next steps:")
        print("1. Train the classification model using the segmented dataset")
        print("2. Compare performance with and without segmentation")
        print("3. Evaluate fairness on the segmented dataset")
        
    except Exception as e:
        print(f"❌ Error during dataset preparation: {str(e)}")
        return


if __name__ == "__main__":
    main()
