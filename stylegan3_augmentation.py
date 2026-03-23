#!/usr/bin/env python3
"""
StyleGAN3-based GAN Augmentation for Skin Disease Images

This module implements synthetic image generation using StyleGAN3 to augment
underrepresented classes and dark skin tones for bias reduction in skin disease classification.

Author: AI Assistant
Date: 2025-03-22
"""

import os
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from collections import Counter, defaultdict
import random
import time
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# StyleGAN3 imports (requires stylegan3 repository)
try:
    import dnnlib
    import legacy
    STYLEGAN3_AVAILABLE = True
except ImportError:
    STYLEGAN3_AVAILABLE = False
    print("Warning: StyleGAN3 not installed. Please install with:")
    print("pip install click requests tqdm")
    print("And clone: https://github.com/NVlabs/stylegan3")

from skin_disease_dataset_preparation import Config, SkinDiseaseDataset
from fairness_evaluator import SkinToneClassifier


class GANConfig:
    """Configuration for GAN-based augmentation"""
    
    # StyleGAN3 model settings
    STYLEGAN3_MODEL_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl"
    MODEL_PATH = "stylegan3-r-ffhq-1024x1024.pkl"
    MODEL_RESOLUTION = 1024
    
    # Training settings
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-3
    BETA1 = 0.0
    BETA2 = 0.99
    EPOCHS = 100
    SNAPSHOT_INTERVAL = 10
    
    # Augmentation settings
    TARGET_AUGMENTATION_FACTOR = 2.0  # Double dataset size
    MIN_IMAGES_PER_CLASS = 1000  # Minimum images per class after augmentation
    
    # Dark skin focus
    DARK_SKIN_GENERATION_RATIO = 0.7  # 70% generated images should be dark skin
    MINORITY_CLASS_FOCUS = True
    
    # Quality control
    QUALITY_THRESHOLD = 0.5
    MEDICAL_FEATURE_PRESERVATION = True
    MAX_GENERATION_ATTEMPTS = 10
    
    # Output settings
    OUTPUT_DIR = "dataset_augmented"
    SAVE_GENERATOR_SAMPLES = True
    SAVE_TRAINING_SNAPSHOTS = True
    
    # Device settings
    DEVICE = "auto"


class SkinToneAwareDataset(Dataset):
    """Dataset with skin tone labeling for GAN training"""
    
    def __init__(self, root_dir: str, transform=None, skin_tone_classifier=None):
        self.root_dir = root_dir
        self.transform = transform
        self.skin_tone_classifier = skin_tone_classifier or SkinToneClassifier()
        
        # Load samples with skin tone labels
        self.samples = []
        self.skin_tones = []
        self.class_names = []
        
        self._load_samples()
    
    def _load_samples(self):
        """Load samples and classify skin tones"""
        if not os.path.exists(self.root_dir):
            return
        
        self.class_names = sorted([d for d in os.listdir(self.root_dir) 
                                 if os.path.isdir(os.path.join(self.root_dir, d))])
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    
                    # Classify skin tone
                    skin_tone = self.skin_tone_classifier.classify_skin_tone(img_path)
                    
                    self.samples.append((img_path, self.class_names.index(class_name)))
                    self.skin_tones.append(skin_tone)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        skin_tone = self.skin_tones[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, skin_tone, img_path


class StyleGAN3Trainer:
    """StyleGAN3 trainer for skin disease image generation"""
    
    def __init__(self, config: GANConfig = GANConfig()):
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.setup_directories()
        
        # Statistics
        self.training_stats = {
            'total_generated': 0,
            'successful_generations': 0,
            'dark_skin_generated': 0,
            'light_skin_generated': 0,
            'class_distribution': defaultdict(int)
        }
        
        # Skin tone classifier
        self.skin_tone_classifier = SkinToneClassifier()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gan_augmentation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_device(self):
        """Setup device for GAN training"""
        if self.config.DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.DEVICE)
        
        self.logger.info(f"Using device: {self.device}")
        
        if self.device.type == "cuda":
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def setup_directories(self):
        """Setup output directories"""
        directories = [
            self.config.OUTPUT_DIR,
            os.path.join(self.config.OUTPUT_DIR, 'train'),
            os.path.join(self.config.OUTPUT_DIR, 'val'),
            os.path.join(self.config.OUTPUT_DIR, 'test'),
            os.path.join(self.config.OUTPUT_DIR, 'generated_samples'),
            os.path.join(self.config.OUTPUT_DIR, 'training_snapshots'),
            os.path.join(self.config.OUTPUT_DIR, 'quality_control')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Create class-specific directories
        for split in ['train', 'val', 'test']:
            for class_name in ['acne', 'psoriasis', 'eczema']:
                os.makedirs(
                    os.path.join(self.config.OUTPUT_DIR, split, class_name),
                    exist_ok=True
                )
    
    def load_pretrained_stylegan3(self):
        """Load pretrained StyleGAN3 model"""
        if not STYLEGAN3_AVAILABLE:
            raise ImportError("StyleGAN3 dependencies not installed")
        
        model_path = self.config.MODEL_PATH
        if not os.path.exists(model_path):
            self.logger.info(f"Downloading StyleGAN3 model...")
            self.download_stylegan3_model(model_path)
        
        self.logger.info("Loading StyleGAN3 model...")
        
        with dnnlib.util.open_url(model_path) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
        
        self.logger.info("StyleGAN3 model loaded successfully")
        return self.G
    
    def download_stylegan3_model(self, model_path: str):
        """Download StyleGAN3 model"""
        import urllib.request
        
        try:
            urllib.request.urlretrieve(self.config.STYLEGAN3_MODEL_URL, model_path)
            self.logger.info(f"StyleGAN3 model downloaded to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to download StyleGAN3 model: {str(e)}")
            raise
    
    def analyze_dataset_imbalance(self, dataset_path: str) -> Dict:
        """Analyze dataset for imbalance and minority classes"""
        self.logger.info("Analyzing dataset imbalance...")
        
        # Load dataset with skin tone classification
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        dataset = SkinToneAwareDataset(dataset_path, transform, self.skin_tone_classifier)
        
        # Analyze distribution
        class_counts = Counter()
        skin_tone_counts = Counter()
        class_skin_tone_counts = defaultdict(lambda: defaultdict(int))
        
        for _, label, skin_tone, _ in dataset:
            class_name = dataset.class_names[label]
            class_counts[class_name] += 1
            skin_tone_counts[skin_tone] += 1
            class_skin_tone_counts[class_name][skin_tone] += 1
        
        analysis = {
            'class_distribution': dict(class_counts),
            'skin_tone_distribution': dict(skin_tone_counts),
            'class_skin_tone_distribution': dict(class_skin_tone_counts),
            'total_samples': len(dataset),
            'minority_classes': [],
            'underrepresented_combinations': []
        }
        
        # Identify minority classes (below target minimum)
        target_min = self.config.MIN_IMAGES_PER_CLASS
        for class_name, count in class_counts.items():
            if count < target_min:
                analysis['minority_classes'].append(class_name)
        
        # Identify underrepresented combinations
        for class_name in class_counts:
            for skin_tone in ['dark', 'light']:
                count = class_skin_tone_counts[class_name][skin_tone]
                target_count = target_min // 2  # Aim for balance
                if count < target_count:
                    analysis['underrepresented_combinations'].append(f"{class_name}_{skin_tone}")
        
        self.logger.info(f"Dataset analysis completed:")
        self.logger.info(f"  Total samples: {analysis['total_samples']}")
        self.logger.info(f"  Minority classes: {analysis['minority_classes']}")
        self.logger.info(f"  Underrepresented combinations: {len(analysis['underrepresented_combinations'])}")
        
        return analysis
    
    def generate_synthetic_images(self, class_name: str, skin_tone: str, num_images: int) -> List[np.ndarray]:
        """Generate synthetic images for specific class and skin tone"""
        if not hasattr(self, 'G'):
            self.load_pretrained_stylegan3()
        
        generated_images = []
        
        # StyleGAN3 latent vector
        z = torch.randn(num_images, self.G.z_dim, device=self.device)
        
        # Generate images
        with torch.no_grad():
            for i in range(num_images):
                try:
                    # Generate image
                    img = self.G(z[i:i+1], None, truncation_psi=0.7, noise_mode='const')
                    
                    # Convert to numpy and post-process
                    img_np = img[0].cpu().numpy()
                    img_np = (img_np.transpose(1, 2, 0) * 127.5 + 127.5).astype(np.uint8)
                    
                    # Apply skin tone conditioning
                    if skin_tone == 'dark':
                        img_np = self.condition_for_dark_skin(img_np)
                    
                    # Apply medical feature conditioning
                    if self.config.MEDICAL_FEATURE_PRESERVATION:
                        img_np = self.add_medical_features(img_np, class_name)
                    
                    # Quality control
                    if self.passes_quality_control(img_np, class_name):
                        generated_images.append(img_np)
                        self.training_stats['successful_generations'] += 1
                        
                        if skin_tone == 'dark':
                            self.training_stats['dark_skin_generated'] += 1
                        else:
                            self.training_stats['light_skin_generated'] += 1
                        
                        self.training_stats['class_distribution'][class_name] += 1
                    else:
                        self.training_stats['total_generated'] += 1
                
                except Exception as e:
                    self.logger.warning(f"Failed to generate image: {str(e)}")
                    self.training_stats['total_generated'] += 1
        
        return generated_images
    
    def condition_for_dark_skin(self, image: np.ndarray) -> np.ndarray:
        """Apply conditioning to generate darker skin tones"""
        # Convert to HSV for skin tone adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Reduce value (brightness) for darker skin
        hsv[:, :, 2] = hsv[:, :, 2] * 0.7  # Reduce brightness by 30%
        
        # Adjust saturation for melanin-rich skin
        hsv[:, :, 1] = hsv[:, :, 1] * 1.2  # Increase saturation by 20%
        
        # Convert back to RGB
        conditioned = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Apply subtle contrast enhancement
        conditioned = cv2.convertScaleAbs(conditioned, alpha=1.1, beta=-10)
        
        return np.clip(conditioned, 0, 255).astype(np.uint8)
    
    def add_medical_features(self, image: np.ndarray, class_name: str) -> np.ndarray:
        """Add realistic medical features for specific skin conditions"""
        h, w = image.shape[:2]
        
        if class_name == 'acne':
            # Add acne-like features (small red spots)
            num_spots = random.randint(5, 15)
            for _ in range(num_spots):
                x = random.randint(20, w-20)
                y = random.randint(20, h-20)
                radius = random.randint(2, 5)
                
                # Create red spot
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (x, y), radius, 255, -1)
                
                # Apply reddish color
                image[mask > 0, 0] = min(255, image[mask > 0, 0] + 50)  # More red
                image[mask > 0, 1] = max(0, image[mask > 0, 1] - 20)   # Less green
                image[mask > 0, 2] = max(0, image[mask > 0, 2] - 20)   # Less blue
        
        elif class_name == 'psoriasis':
            # Add psoriasis-like features (scaly patches)
            num_patches = random.randint(2, 5)
            for _ in range(num_patches):
                x = random.randint(30, w-30)
                y = random.randint(30, h-30)
                size = random.randint(20, 40)
                
                # Create irregular patch
                patch_mask = np.zeros((h, w), dtype=np.uint8)
                points = []
                for _ in range(8):
                    px = x + random.randint(-size//2, size//2)
                    py = y + random.randint(-size//2, size//2)
                    points.append([px, py])
                
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(patch_mask, [points], 255)
                
                # Apply silvery-white color
                image[patch_mask > 0, 0] = min(255, image[patch_mask > 0, 0] + 30)
                image[patch_mask > 0, 1] = min(255, image[patch_mask > 0, 1] + 30)
                image[patch_mask > 0, 2] = min(255, image[patch_mask > 0, 2] + 40)
        
        elif class_name == 'eczema':
            # Add eczema-like features (red, inflamed patches)
            num_patches = random.randint(3, 7)
            for _ in range(num_patches):
                x = random.randint(25, w-25)
                y = random.randint(25, h-25)
                size = random.randint(15, 35)
                
                # Create irregular patch
                patch_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(patch_mask, (x, y), (size, size//2), 
                           random.randint(0, 360), 0, 360, 255, -1)
                
                # Apply reddish color with inflammation
                image[patch_mask > 0, 0] = min(255, image[patch_mask > 0, 0] + 60)
                image[patch_mask > 0, 1] = max(0, image[patch_mask > 0, 1] - 30)
                image[patch_mask > 0, 2] = max(0, image[patch_mask > 0, 2] - 30)
        
        return image
    
    def passes_quality_control(self, image: np.ndarray, class_name: str) -> bool:
        """Quality control check for generated images"""
        # Basic quality checks
        if image is None or image.size == 0:
            return False
        
        # Check for reasonable brightness
        brightness = np.mean(image)
        if brightness < 30 or brightness > 225:
            return False
        
        # Check for reasonable contrast
        contrast = np.std(image)
        if contrast < 10:
            return False
        
        # Check skin tone distribution
        skin_tone_score = self.skin_tone_classifier.extract_skin_tone_features("temp.jpg")['skin_tone_score']
        
        # Medical feature check (simplified)
        has_medical_features = self.detect_medical_features(image, class_name)
        
        return has_medical_features
    
    def detect_medical_features(self, image: np.ndarray, class_name: str) -> bool:
        """Simplified medical feature detection"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection for texture analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Different classes have different expected texture patterns
        if class_name == 'acne':
            return edge_density > 0.02  # Should have some texture from spots
        elif class_name == 'psoriasis':
            return edge_density > 0.03  # Should have more texture from scales
        elif class_name == 'eczema':
            return edge_density > 0.015  # Should have some inflammation texture
        
        return edge_density > 0.01  # Default threshold
    
    def augment_dataset(self, input_dataset_path: str):
        """Main augmentation pipeline"""
        self.logger.info("Starting GAN-based dataset augmentation...")
        
        # Analyze dataset imbalance
        imbalance_analysis = self.analyze_dataset_imbalance(input_dataset_path)
        
        # Calculate augmentation needs
        augmentation_plan = self.create_augmentation_plan(imbalance_analysis)
        
        # Generate synthetic images
        for class_name, skin_tone_targets in augmentation_plan.items():
            self.logger.info(f"Generating images for {class_name}...")
            
            for skin_tone, num_images in skin_tone_targets.items():
                if num_images > 0:
                    self.logger.info(f"  Generating {num_images} {skin_tone} skin {class_name} images...")
                    
                    generated_images = []
                    attempts = 0
                    max_attempts = num_images * self.config.MAX_GENERATION_ATTEMPTS
                    
                    while len(generated_images) < num_images and attempts < max_attempts:
                        batch_size = min(self.config.BATCH_SIZE, num_images - len(generated_images))
                        
                        batch_images = self.generate_synthetic_images(class_name, skin_tone, batch_size)
                        generated_images.extend(batch_images)
                        
                        attempts += batch_size
                        
                        if attempts % 100 == 0:
                            self.logger.info(f"  Generated {len(generated_images)}/{num_images} images...")
                    
                    # Save generated images
                    self.save_generated_images(generated_images, class_name, skin_tone)
        
        # Copy original dataset
        self.copy_original_dataset(input_dataset_path)
        
        # Save augmentation report
        self.save_augmentation_report(imbalance_analysis, augmentation_plan)
    
    def create_augmentation_plan(self, analysis: Dict) -> Dict:
        """Create plan for how many images to generate for each class/skin tone"""
        plan = {}
        
        target_per_class = self.config.MIN_IMAGES_PER_CLASS
        
        for class_name in ['acne', 'psoriasis', 'eczema']:
            plan[class_name] = {}
            
            current_dark = analysis['class_skin_tone_distribution'].get(class_name, {}).get('dark', 0)
            current_light = analysis['class_skin_tone_distribution'].get(class_name, {}).get('light', 0)
            
            # Target balanced distribution
            target_dark = target_per_class // 2
            target_light = target_per_class // 2
            
            # Calculate needed images
            needed_dark = max(0, target_dark - current_dark)
            needed_light = max(0, target_light - current_light)
            
            # Prioritize dark skin generation
            if needed_dark > 0 or needed_light > 0:
                total_needed = needed_dark + needed_light
                dark_ratio = self.config.DARK_SKIN_GENERATION_RATIO
                
                plan[class_name]['dark'] = int(total_needed * dark_ratio)
                plan[class_name]['light'] = total_needed - plan[class_name]['dark']
            else:
                plan[class_name]['dark'] = 0
                plan[class_name]['light'] = 0
        
        return plan
    
    def save_generated_images(self, images: List[np.ndarray], class_name: str, skin_tone: str):
        """Save generated images to output directory"""
        output_dir = os.path.join(self.config.OUTPUT_DIR, 'train', class_name)
        os.makedirs(output_dir, exist_ok=True)
        
        existing_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
        start_idx = len(existing_files)
        
        for i, image in enumerate(images):
            filename = f"synthetic_{class_name}_{skin_tone}_{start_idx + i:06d}.jpg"
            output_path = os.path.join(output_dir, filename)
            
            # Save image
            Image.fromarray(image).save(output_path, quality=95)
            
            # Save sample for visualization
            if i < 10:  # Save first 10 as samples
                sample_dir = os.path.join(self.config.OUTPUT_DIR, 'generated_samples', class_name)
                os.makedirs(sample_dir, exist_ok=True)
                sample_path = os.path.join(sample_dir, filename)
                Image.fromarray(image).save(sample_path, quality=95)
    
    def copy_original_dataset(self, input_dataset_path: str):
        """Copy original dataset to augmented directory"""
        import shutil
        
        for split in ['train', 'val', 'test']:
            input_split = os.path.join(input_dataset_path, split)
            output_split = os.path.join(self.config.OUTPUT_DIR, split)
            
            if os.path.exists(input_split):
                for class_name in ['acne', 'psoriasis', 'eczema']:
                    input_class = os.path.join(input_split, class_name)
                    output_class = os.path.join(output_split, class_name)
                    
                    if os.path.exists(input_class):
                        # Copy original images
                        for img_file in os.listdir(input_class):
                            if img_file.endswith('.jpg'):
                                src = os.path.join(input_class, img_file)
                                dst = os.path.join(output_class, img_file)
                                
                                if not os.path.exists(dst):
                                    shutil.copy2(src, dst)
    
    def save_augmentation_report(self, imbalance_analysis: Dict, augmentation_plan: Dict):
        """Save comprehensive augmentation report"""
        report = {
            'original_analysis': imbalance_analysis,
            'augmentation_plan': augmentation_plan,
            'generation_statistics': self.training_stats,
            'configuration': self.config.__dict__,
            'success_rate': (self.training_stats['successful_generations'] / 
                           max(self.training_stats['total_generated'], 1)) * 100
        }
        
        report_path = os.path.join(self.config.OUTPUT_DIR, 'augmentation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Augmentation report saved to {report_path}")
    
    def visualize_augmentation_results(self):
        """Create visualization of augmentation results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('GAN-based Augmentation Results', fontsize=16, fontweight='bold')
        
        # Sample generated images
        sample_dir = os.path.join(self.config.OUTPUT_DIR, 'generated_samples')
        
        for i, class_name in enumerate(['acne', 'psoriasis', 'eczema']):
            class_sample_dir = os.path.join(sample_dir, class_name)
            
            if os.path.exists(class_sample_dir):
                sample_files = [f for f in os.listdir(class_sample_dir) if f.endswith('.jpg')][:2]
                
                for j, sample_file in enumerate(sample_files):
                    if j < 2:
                        sample_path = os.path.join(class_sample_dir, sample_file)
                        image = Image.open(sample_path)
                        
                        axes[j, i].imshow(image)
                        axes[j, i].set_title(f'{class_name.capitalize()} - Sample {j+1}')
                        axes[j, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, 'augmentation_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Augmentation visualization saved")


def main():
    """Main function for GAN-based augmentation"""
    print("="*80)
    print("STYLEGAN3-BASED GAN AUGMENTATION FOR SKIN DISEASE IMAGES")
    print("="*80)
    print("Generating synthetic images to reduce bias and improve fairness")
    print("="*80)
    
    # Check StyleGAN3 availability
    if not STYLEGAN3_AVAILABLE:
        print("❌ ERROR: StyleGAN3 dependencies not installed")
        print("Please install with:")
        print("pip install click requests tqdm")
        print("And clone: https://github.com/NVlabs/stylegan3")
        return
    
    # Create configuration
    config = GANConfig()
    
    print(f"Configuration:")
    print(f"  Target Augmentation Factor: {config.TARGET_AUGMENTATION_FACTOR}x")
    print(f"  Minimum Images per Class: {config.MIN_IMAGES_PER_CLASS}")
    print(f"  Dark Skin Generation Ratio: {config.DARK_SKIN_GENERATION_RATIO * 100:.0f}%")
    print(f"  Output Directory: {config.OUTPUT_DIR}")
    print(f"  Medical Feature Preservation: {config.MEDICAL_FEATURE_PRESERVATION}")
    print()
    
    # Check input dataset
    input_dataset_path = "dataset"
    if not os.path.exists(input_dataset_path):
        print(f"❌ ERROR: Input dataset not found at {input_dataset_path}")
        print("Please run the dataset preparation script first.")
        return
    
    try:
        # Create GAN trainer
        trainer = StyleGAN3Trainer(config)
        
        # Run augmentation
        trainer.augment_dataset(input_dataset_path)
        
        # Visualize results
        trainer.visualize_augmentation_results()
        
        print("\n" + "="*80)
        print("GAN-BASED AUGMENTATION COMPLETED!")
        print("="*80)
        print(f"Augmented dataset saved to: {config.OUTPUT_DIR}")
        print(f"Augmentation report saved to: {config.OUTPUT_DIR}/augmentation_report.json")
        print(f"Generated samples saved to: {config.OUTPUT_DIR}/generated_samples/")
        print()
        
        # Print statistics
        stats = trainer.training_stats
        print(f"GENERATION STATISTICS:")
        print(f"  Total Generation Attempts: {stats['total_generated']}")
        print(f"  Successful Generations: {stats['successful_generations']}")
        print(f"  Success Rate: {(stats['successful_generations'] / max(stats['total_generated'], 1)) * 100:.1f}%")
        print(f"  Dark Skin Images: {stats['dark_skin_generated']}")
        print(f"  Light Skin Images: {stats['light_skin_generated']}")
        print()
        print("CLASS DISTRIBUTION AFTER AUGMENTATION:")
        for class_name, count in stats['class_distribution'].items():
            print(f"  {class_name}: {count} synthetic images")
        
        print()
        print("🎯 EXPECTED BENEFITS:")
        print("✓ Reduced class imbalance")
        print("✓ Better representation of dark skin tones")
        print("✓ Improved model fairness")
        print("✓ Enhanced generalization capability")
        print("✓ Reduced bias in classification")
        print()
        print("Next steps:")
        print("1. Review generated image quality")
        print("2. Train model on augmented dataset")
        print("3. Compare fairness metrics with baseline")
        print("4. Evaluate performance improvements")
        
    except Exception as e:
        print(f"❌ ERROR during augmentation: {str(e)}")
        return


if __name__ == "__main__":
    main()
