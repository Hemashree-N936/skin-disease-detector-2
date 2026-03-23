#!/usr/bin/env python3
"""
Bias-Mitigated CNN-GAN-ViT Hybrid for Multi-Class Detection of Acne, Psoriasis, and Eczema on Indian Dark Skin Tones

Dataset Preparation Script

This script loads image datasets from multiple sources, filters for specific classes,
preprocesses images, and creates a structured dataset for training deep learning models.

Author: AI Assistant
Date: 2025-03-22
"""

import os
import shutil
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


# Configuration
class Config:
    """Configuration class for dataset preparation"""
    
    # Dataset paths
    DERMACON_IN_PATH = "data/dermacon_in"
    KAGGLE_ACNE_PATH = "data/kaggle_acne"
    KAGGLE_PSORIASIS_PATH = "data/kaggle_psoriasis"
    KAGGLE_ECZEMA_PATH = "data/kaggle_eczema"
    
    # Target classes (only these will be kept)
    TARGET_CLASSES = ["acne", "psoriasis", "eczema"]
    
    # Image preprocessing
    IMAGE_SIZE = (224, 224)
    
    # ImageNet normalization parameters
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Dataset split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Output paths
    OUTPUT_DIR = "dataset"
    LOGS_DIR = "logs"
    STATS_DIR = "stats"
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Data augmentation parameters
    AUGMENTATION_PROB = 0.5
    HORIZONTAL_FLIP_PROB = 0.5
    ROTATION_RANGE = 15  # degrees
    BRIGHTNESS_LIMIT = 0.2
    CONTRAST_LIMIT = 0.2
    HUE_SHIFT_LIMIT = 20  # important for skin tone variation
    SATURATION_LIMIT = 0.3
    RANDOM_CROP_SCALE = (0.8, 1.0)
    
    # DataLoader parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    PIN_MEMORY = True


class SkinDiseaseDataset(Dataset):
    """Custom PyTorch Dataset for skin disease classification with augmentation"""
    
    def __init__(self, root_dir: str, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = class_to_idx or {}
        
        # Load image paths and labels
        self._load_samples()
    
    def _load_samples(self):
        """Load image paths and corresponding labels"""
        if not os.path.exists(self.root_dir):
            return
            
        for class_name in sorted(os.listdir(self.root_dir)):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)
                
                class_idx = self.class_to_idx[class_name]
                
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label


class DatasetPreparer:
    """Main class for dataset preparation"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        
        # Statistics tracking
        self.class_counts = Counter()
        self.dataset_stats = {}
        
        # Image transformations for different splits
        self.setup_transforms()
        
        # For visualization (without normalization)
        self.viz_transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor()
        ])
    
    def setup_transforms(self):
        """Setup augmentation transforms for different dataset splits"""
        
        # Training augmentations - optimized for skin tone variation
        self.train_transform = A.Compose([
            # Geometric augmentations
            A.HorizontalFlip(p=self.config.HORIZONTAL_FLIP_PROB),
            A.Rotate(limit=self.config.ROTATION_RANGE, p=self.config.AUGMENTATION_PROB, border_mode=cv2.BORDER_REFLECT),
            A.RandomResizedCrop(
                height=self.config.IMAGE_SIZE[0], 
                width=self.config.IMAGE_SIZE[1],
                scale=self.config.RANDOM_CROP_SCALE,
                ratio=(0.9, 1.1),
                p=self.config.AUGMENTATION_PROB
            ),
            
            # Color augmentations - important for dark skin tones
            A.RandomBrightnessContrast(
                brightness_limit=self.config.BRIGHTNESS_LIMIT,
                contrast_limit=self.config.CONTRAST_LIMIT,
                p=self.config.AUGMENTATION_PROB
            ),
            A.HueSaturationValue(
                hue_shift_limit=self.config.HUE_SHIFT_LIMIT,
                sat_shift_limit=int(self.config.SATURATION_LIMIT * 255),
                val_shift_limit=20,
                p=self.config.AUGMENTATION_PROB
            ),
            
            # Medical image specific augmentations
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                fill_value=0,
                p=0.3
            ),
            
            # Normalization
            A.Normalize(
                mean=self.config.IMAGENET_MEAN,
                std=self.config.IMAGENET_STD
            ),
            ToTensorV2()
        ])
        
        # Validation/Test transforms - no augmentation, only normalization
        self.val_test_transform = A.Compose([
            A.Resize(height=self.config.IMAGE_SIZE[0], width=self.config.IMAGE_SIZE[1]),
            A.Normalize(
                mean=self.config.IMAGENET_MEAN,
                std=self.config.IMAGENET_STD
            ),
            ToTensorV2()
        ])
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for train, validation, and test sets
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dir = os.path.join(self.config.OUTPUT_DIR, 'train')
        val_dir = os.path.join(self.config.OUTPUT_DIR, 'val')
        test_dir = os.path.join(self.config.OUTPUT_DIR, 'test')
        
        # Class to index mapping
        class_to_idx = {class_name: idx for idx, class_name in enumerate(self.config.TARGET_CLASSES)}
        
        # Create datasets
        train_dataset = SkinDiseaseDataset(
            root_dir=train_dir,
            transform=self.train_transform,
            class_to_idx=class_to_idx
        )
        
        val_dataset = SkinDiseaseDataset(
            root_dir=val_dir,
            transform=self.val_test_transform,
            class_to_idx=class_to_idx
        )
        
        test_dataset = SkinDiseaseDataset(
            root_dir=test_dir,
            transform=self.val_test_transform,
            class_to_idx=class_to_idx
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
        
        self.logger.info(f"Created DataLoaders:")
        self.logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        self.logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
        self.logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader

    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.config.LOGS_DIR, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.LOGS_DIR, 'dataset_preparation.log')),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Dataset preparation started")
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.OUTPUT_DIR,
            self.config.STATS_DIR,
            os.path.join(self.config.OUTPUT_DIR, 'train'),
            os.path.join(self.config.OUTPUT_DIR, 'val'),
            os.path.join(self.config.OUTPUT_DIR, 'test')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Create class-specific directories
        for split in ['train', 'val', 'test']:
            for class_name in self.config.TARGET_CLASSES:
                os.makedirs(
                    os.path.join(self.config.OUTPUT_DIR, split, class_name),
                    exist_ok=True
                )
    
    def load_dataset_from_path(self, dataset_path: str, dataset_name: str) -> List[Tuple[str, str]]:
        """
        Load images from a dataset path and return list of (image_path, class_name) tuples
        
        Args:
            dataset_path: Path to the dataset
            dataset_name: Name of the dataset for logging
            
        Returns:
            List of (image_path, class_name) tuples
        """
        if not os.path.exists(dataset_path):
            self.logger.warning(f"Dataset path does not exist: {dataset_path}")
            return []
        
        image_data = []
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(tuple(supported_extensions)):
                    image_path = os.path.join(root, file)
                    
                    # Extract class name from folder structure
                    # This assumes class name is in the parent folder name
                    class_name = os.path.basename(root).lower()
                    
                    # Map various possible class names to target classes
                    mapped_class = self.map_class_name(class_name)
                    
                    if mapped_class in self.config.TARGET_CLASSES:
                        image_data.append((image_path, mapped_class))
                        self.class_counts[mapped_class] += 1
        
        self.logger.info(f"Loaded {len(image_data)} images from {dataset_name}")
        return image_data
    
    def map_class_name(self, class_name: str) -> str:
        """
        Map various class name variations to target classes
        
        Args:
            class_name: Original class name from dataset
            
        Returns:
            Mapped class name
        """
        class_mapping = {
            # Acne variations
            'acne': 'acne',
            'acne_vulgaris': 'acne',
            'pimple': 'acne',
            'pimples': 'acne',
            
            # Psoriasis variations
            'psoriasis': 'psoriasis',
            'psoriatic': 'psoriasis',
            'plaque_psoriasis': 'psoriasis',
            
            # Eczema variations
            'eczema': 'eczema',
            'dermatitis': 'eczema',
            'atopic_dermatitis': 'eczema',
            'contact_dermatitis': 'eczema'
        }
        
        return class_mapping.get(class_name.lower(), class_name.lower())
    
    def load_all_datasets(self) -> List[Tuple[str, str]]:
        """
        Load images from all dataset sources
        
        Returns:
            Combined list of (image_path, class_name) tuples
        """
        all_images = []
        
        # Load from DermaCon-IN dataset
        dermacon_images = self.load_dataset_from_path(
            self.config.DERMACON_IN_PATH, "DermaCon-IN"
        )
        all_images.extend(dermacon_images)
        
        # Load from Kaggle datasets
        kaggle_paths = [
            (self.config.KAGGLE_ACNE_PATH, "Kaggle Acne"),
            (self.config.KAGGLE_PSORIASIS_PATH, "Kaggle Psoriasis"),
            (self.config.KAGGLE_ECZEMA_PATH, "Kaggle Eczema")
        ]
        
        for path, name in kaggle_paths:
            kaggle_images = self.load_dataset_from_path(path, name)
            all_images.extend(kaggle_images)
        
        self.logger.info(f"Total images loaded: {len(all_images)}")
        return all_images
    
    def validate_and_filter_images(self, image_data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Validate images and filter for target classes only
        
        Args:
            image_data: List of (image_path, class_name) tuples
            
        Returns:
            Filtered and validated list of (image_path, class_name) tuples
        """
        valid_images = []
        corrupted_count = 0
        
        for image_path, class_name in image_data:
            if class_name not in self.config.TARGET_CLASSES:
                continue
            
            try:
                # Try to open and validate the image
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Check image size (skip very small images)
                    if img.size[0] < 32 or img.size[1] < 32:
                        continue
                    
                    valid_images.append((image_path, class_name))
                    
            except Exception as e:
                corrupted_count += 1
                self.logger.warning(f"Corrupted image: {image_path} - {str(e)}")
        
        self.logger.info(f"Valid images: {len(valid_images)}, Corrupted: {corrupted_count}")
        return valid_images
    
    def split_dataset(self, image_data: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            image_data: List of (image_path, class_name) tuples
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys containing image lists
        """
        # Group images by class
        class_images = {}
        for class_name in self.config.TARGET_CLASSES:
            class_images[class_name] = [
                (path, cls) for path, cls in image_data if cls == class_name
            ]
        
        # Split each class separately to maintain class distribution
        splits = {'train': [], 'val': [], 'test': []}
        
        for class_name, images in class_images.items():
            np.random.shuffle(images)
            
            total_images = len(images)
            train_size = int(total_images * self.config.TRAIN_RATIO)
            val_size = int(total_images * self.config.VAL_RATIO)
            test_size = total_images - train_size - val_size
            
            splits['train'].extend(images[:train_size])
            splits['val'].extend(images[train_size:train_size + val_size])
            splits['test'].extend(images[train_size + val_size:])
            
            self.logger.info(
                f"Class {class_name}: Train={train_size}, Val={val_size}, Test={test_size}"
            )
        
        return splits
    
    def process_and_save_images(self, splits: Dict[str, List[Tuple[str, str]]]):
        """
        Process images (resize, normalize) and save to output directories
        
        Args:
            splits: Dictionary with train/val/test splits
        """
        processed_count = 0
        
        for split_name, image_list in splits.items():
            self.logger.info(f"Processing {split_name} split: {len(image_list)} images")
            
            for image_path, class_name in image_list:
                try:
                    # Open and process image
                    with Image.open(image_path) as img:
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Resize image
                        img_resized = img.resize(self.config.IMAGE_SIZE, Image.LANCZOS)
                        
                        # Generate output filename
                        filename = f"{class_name}_{processed_count:06d}.jpg"
                        output_path = os.path.join(
                            self.config.OUTPUT_DIR, split_name, class_name, filename
                        )
                        
                        # Save processed image
                        img_resized.save(output_path, 'JPEG', quality=95)
                        processed_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing image {image_path}: {str(e)}")
        
        self.logger.info(f"Total processed images: {processed_count}")
    
    def generate_statistics(self):
        """Generate and save dataset statistics"""
        stats = {
            'total_images': sum(self.class_counts.values()),
            'class_distribution': dict(self.class_counts),
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
            }
        }
        
        # Calculate class imbalance metrics
        total_images = stats['total_images']
        class_weights = {}
        for class_name, count in stats['class_distribution'].items():
            weight = total_images / (len(self.config.TARGET_CLASSES) * count)
            class_weights[class_name] = weight
        
        stats['class_weights'] = class_weights
        stats['class_imbalance_ratio'] = (
            max(stats['class_distribution'].values()) / 
            min(stats['class_distribution'].values())
        )
        
        # Save statistics
        stats_path = os.path.join(self.config.STATS_DIR, 'dataset_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Dataset statistics saved to {stats_path}")
        
        # Log statistics
        self.logger.info("=" * 50)
        self.logger.info("DATASET STATISTICS")
        self.logger.info("=" * 50)
        self.logger.info(f"Total images: {stats['total_images']}")
        self.logger.info("Class distribution:")
        for class_name, count in stats['class_distribution'].items():
            percentage = (count / total_images) * 100
            self.logger.info(f"  {class_name}: {count} ({percentage:.2f}%)")
        
        self.logger.info(f"Class imbalance ratio: {stats['class_imbalance_ratio']:.2f}")
        self.logger.info("Class weights:")
        for class_name, weight in stats['class_weights'].items():
            self.logger.info(f"  {class_name}: {weight:.4f}")
    
    def create_visualizations(self):
        """Create visualization plots for dataset analysis"""
        # Class distribution plot
        plt.figure(figsize=(10, 6))
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        
        plt.bar(classes, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Disease Class')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for i, count in enumerate(counts):
            plt.text(i, count + max(counts) * 0.01, str(count), 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.STATS_DIR, 'class_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Class distribution plot saved")
    
    def run(self):
        """Main execution method"""
        try:
            # Set random seed
            np.random.seed(self.config.RANDOM_SEED)
            torch.manual_seed(self.config.RANDOM_SEED)
            
            self.logger.info("Starting dataset preparation...")
            
            # Load all datasets
            image_data = self.load_all_datasets()
            
            if not image_data:
                self.logger.error("No images loaded. Please check dataset paths.")
                return
            
            # Validate and filter images
            valid_images = self.validate_and_filter_images(image_data)
            
            if not valid_images:
                self.logger.error("No valid images found after filtering.")
                return
            
            # Split dataset
            splits = self.split_dataset(valid_images)
            
            # Process and save images
            self.process_and_save_images(splits)
            
            # Generate statistics and visualizations
            self.generate_statistics()
            self.create_visualizations()
            
            # Create DataLoaders
            train_loader, val_loader, test_loader = self.create_data_loaders()
            
            self.logger.info("Dataset preparation completed successfully!")
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            self.logger.error(f"Error during dataset preparation: {str(e)}")
            raise


def main():
    """Main function"""
    config = Config()
    preparer = DatasetPreparer(config)
    preparer.run()


if __name__ == "__main__":
    main()
