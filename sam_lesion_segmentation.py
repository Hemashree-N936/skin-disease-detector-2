#!/usr/bin/env python3
"""
SAM-based Lesion Segmentation Module for Skin Disease Classification

This module implements lesion segmentation using the Segment Anything Model (SAM ViT-H)
to focus on lesion areas before classification, optimized for dermatology use-cases
and dark skin tones.

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
from tqdm import tqdm
import time

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# SAM imports (requires segment-anything package)
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: segment-anything package not installed. Please install with:")
    print("pip install segment-anything")
    print("Also download SAM model: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")

from skin_disease_dataset_preparation import Config


class SAMConfig:
    """Configuration for SAM-based lesion segmentation"""
    
    # SAM model settings
    SAM_MODEL_TYPE = "vit_h"
    SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
    SAM_MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    
    # Device settings
    DEVICE = "auto"  # "auto", "cuda", "cpu"
    
    # Segmentation parameters
    CONFIDENCE_THRESHOLD = 0.5
    MIN_MASK_AREA = 100  # Minimum area in pixels
    MAX_MASK_AREA = 50000  # Maximum area in pixels
    
    # Dark skin optimization
    DARK_SKIN_ENHANCEMENT = True
    CONTRAST_ENHANCEMENT_FACTOR = 1.2
    BRIGHTNESS_ADJUSTMENT = 10
    
    # Dermatology-specific settings
    FOCUS_ON_CENTER = True  # Prioritize central regions where lesions typically appear
    MIN_LESION_SIZE = 50  # Minimum lesion size in pixels
    MAX_LESION_SIZE = 10000  # Maximum lesion size in pixels
    
    # Output settings
    OUTPUT_DIR = "dataset_segmented"
    SAVE_MASKS = True
    SAVE_VISUALIZATIONS = True
    
    # Batch processing
    BATCH_SIZE = 8
    NUM_WORKERS = 4


class SkinToneEnhancer:
    """Enhance images for better SAM performance on dark skin tones"""
    
    def __init__(self, config: SAMConfig):
        self.config = config
    
    def enhance_for_dark_skin(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image for better segmentation on dark skin tones
        
        Args:
            image: Input RGB image
            
        Returns:
            Enhanced image
        """
        if not self.config.DARK_SKIN_ENHANCEMENT:
            return image
        
        # Convert to LAB color space for better skin tone processing
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge back and convert to RGB
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Additional contrast enhancement
        enhanced = cv2.convertScaleAbs(enhanced, alpha=self.config.CONTRAST_ENHANCEMENT_FACTOR, 
                                     beta=self.config.BRIGHTNESS_ADJUSTMENT)
        
        return enhanced
    
    def detect_skin_tone(self, image: np.ndarray) -> str:
        """
        Detect if image contains dark skin tones
        
        Args:
            image: Input RGB image
            
        Returns:
            'dark' or 'light' skin tone classification
        """
        # Sample center region (likely to contain skin)
        h, w = image.shape[:2]
        center_region = image[h//4:3*h//4, w//4:3*w//4]
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
        
        # Calculate average brightness (V channel)
        avg_brightness = np.mean(hsv[:, :, 2])
        
        # Simple threshold for dark vs light skin
        if avg_brightness < 100:  # Dark threshold
            return 'dark'
        else:
            return 'light'


class LesionSegmenter:
    """Main lesion segmentation class using SAM"""
    
    def __init__(self, config: SAMConfig = SAMConfig()):
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.setup_sam()
        self.skin_tone_enhancer = SkinToneEnhancer(config)
        
        # Setup output directories
        self.setup_directories()
        
        # Statistics
        self.segmentation_stats = {
            'total_processed': 0,
            'successful_segmentations': 0,
            'failed_segmentations': 0,
            'dark_skin_images': 0,
            'light_skin_images': 0
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sam_segmentation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_device(self):
        """Setup device for SAM"""
        if self.config.DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.DEVICE)
        
        self.logger.info(f"Using device: {self.device}")
        
        if self.device.type == "cuda":
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def setup_sam(self):
        """Setup SAM model"""
        if not SAM_AVAILABLE:
            raise ImportError("segment-anything package not installed")
        
        # Download SAM model if not present
        sam_checkpoint_path = self.config.SAM_CHECKPOINT
        if not os.path.exists(sam_checkpoint_path):
            self.logger.info(f"Downloading SAM model from {self.config.SAM_MODEL_URL}...")
            self.download_sam_model(sam_checkpoint_path)
        
        # Load SAM model
        self.logger.info("Loading SAM model...")
        sam = sam_model_registry[self.config.SAM_MODEL_TYPE](checkpoint=sam_checkpoint_path)
        sam.to(device=self.device)
        sam.eval()
        
        self.sam = sam
        self.predictor = SamPredictor(sam)
        
        self.logger.info("SAM model loaded successfully")
    
    def download_sam_model(self, checkpoint_path: str):
        """Download SAM model checkpoint"""
        import urllib.request
        
        try:
            urllib.request.urlretrieve(self.config.SAM_MODEL_URL, checkpoint_path)
            self.logger.info(f"SAM model downloaded to {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to download SAM model: {str(e)}")
            raise
    
    def setup_directories(self):
        """Setup output directories"""
        directories = [
            self.config.OUTPUT_DIR,
            os.path.join(self.config.OUTPUT_DIR, 'train'),
            os.path.join(self.config.OUTPUT_DIR, 'val'),
            os.path.join(self.config.OUTPUT_DIR, 'test'),
            os.path.join(self.config.OUTPUT_DIR, 'masks'),
            os.path.join(self.config.OUTPUT_DIR, 'visualizations')
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
    
    def generate_lesion_prompts(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Generate automatic prompts for lesion segmentation
        
        Args:
            image: Input RGB image
            
        Returns:
            List of prompt points
        """
        h, w = image.shape[:2]
        prompts = []
        
        # Strategy 1: Grid-based prompts for comprehensive coverage
        if self.config.FOCUS_ON_CENTER:
            # Focus on center region where lesions typically appear
            center_h, center_w = h // 2, w // 2
            grid_size = min(h, w) // 4
            
            # Create grid of points in center region
            for i in range(-1, 2):
                for j in range(-1, 2):
                    y = center_h + i * grid_size
                    x = center_w + j * grid_size
                    
                    if 0 <= y < h and 0 <= x < w:
                        prompts.append(np.array([[x, y]]))
        else:
            # Full image grid
            grid_size = min(h, w) // 8
            for i in range(0, h, grid_size):
                for j in range(0, w, grid_size):
                    prompts.append(np.array([[j, i]]))
        
        # Strategy 2: Edge-based prompts (lesions often have distinct edges)
        # Use simple edge detection to find potential lesion boundaries
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find edge points
        edge_points = np.argwhere(edges > 0)
        if len(edge_points) > 0:
            # Sample edge points
            num_samples = min(20, len(edge_points))
            sampled_indices = np.random.choice(len(edge_points), num_samples, replace=False)
            
            for idx in sampled_indices:
                y, x = edge_points[idx]
                prompts.append(np.array([[x, y]]))
        
        return prompts
    
    def segment_lesion(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """
        Segment lesion from skin image using SAM
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (segmented_image, mask, metadata)
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_image = image.copy()
            
            # Detect skin tone
            skin_tone = self.skin_tone_enhancer.detect_skin_tone(image)
            
            # Enhance for dark skin if needed
            if skin_tone == 'dark':
                image = self.skin_tone_enhancer.enhance_for_dark_skin(image)
                self.segmentation_stats['dark_skin_images'] += 1
            else:
                self.segmentation_stats['light_skin_images'] += 1
            
            # Set image for SAM predictor
            self.predictor.set_image(image)
            
            # Generate prompts
            prompts = self.generate_lesion_prompts(image)
            
            best_mask = None
            best_score = 0
            
            # Try different prompts and select best mask
            for prompt_points in prompts:
                # Predict mask
                masks, scores, logits = self.predictor.predict(
                    point_coords=prompt_points,
                    point_labels=np.array([1]),  # Positive point
                    multimask_output=True,
                    return_logits=True
                )
                
                # Select best mask for this prompt
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    # Check mask area
                    mask_area = np.sum(mask)
                    
                    if (self.config.MIN_MASK_AREA <= mask_area <= self.config.MAX_MASK_AREA and
                        score > self.config.CONFIDENCE_THRESHOLD and
                        score > best_score):
                        
                        best_mask = mask
                        best_score = score
            
            if best_mask is None:
                # If no suitable mask found, return original image
                self.logger.warning(f"No suitable lesion mask found for {image_path}")
                return original_image, None, {'status': 'no_mask_found', 'skin_tone': skin_tone}
            
            # Apply mask to original image
            mask_3d = np.stack([best_mask] * 3, axis=-1)
            segmented_image = original_image * mask_3d
            
            # Create bounding box around mask
            y_indices, x_indices = np.where(best_mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                y_min, y_max = y_indices.min(), y_indices.max()
                x_min, x_max = x_indices.min(), x_indices.max()
                
                # Add padding
                padding = 20
                y_min = max(0, y_min - padding)
                y_max = min(image.shape[0], y_max + padding)
                x_min = max(0, x_min - padding)
                x_max = min(image.shape[1], x_max + padding)
                
                # Crop to lesion area
                lesion_crop = segmented_image[y_min:y_max, x_min:x_max]
                
                # Resize to standard size if needed
                if lesion_crop.size > 0:
                    # Resize to maintain aspect ratio but fit within standard size
                    target_size = (224, 224)
                    lesion_crop_resized = cv2.resize(lesion_crop, target_size)
                    
                    metadata = {
                        'status': 'success',
                        'skin_tone': skin_tone,
                        'mask_area': np.sum(best_mask),
                        'confidence_score': best_score,
                        'bbox': [x_min, y_min, x_max, y_max],
                        'original_size': original_image.shape[:2]
                    }
                    
                    return lesion_crop_resized, best_mask, metadata
            
            return original_image, best_mask, {'status': 'partial_success', 'skin_tone': skin_tone}
            
        except Exception as e:
            self.logger.error(f"Error segmenting {image_path}: {str(e)}")
            return None, None, {'status': 'error', 'error': str(e)}
    
    def visualize_segmentation(self, original_image: np.ndarray, 
                             segmented_image: np.ndarray, 
                             mask: Optional[np.ndarray],
                             metadata: Dict,
                             save_path: str):
        """
        Visualize segmentation results
        
        Args:
            original_image: Original RGB image
            segmented_image: Segmented lesion image
            mask: Segmentation mask
            metadata: Segmentation metadata
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title(f'Original Image\nSkin Tone: {metadata.get("skin_tone", "unknown")}')
        axes[0].axis('off')
        
        # Segmented image
        axes[1].imshow(segmented_image)
        axes[1].set_title('Segmented Lesion')
        axes[1].axis('off')
        
        # Mask overlay
        axes[2].imshow(original_image)
        if mask is not None:
            # Create overlay
            overlay = original_image.copy()
            mask_colored = np.zeros_like(original_image)
            mask_colored[mask > 0] = [255, 0, 0]  # Red overlay
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            axes[2].imshow(overlay)
            
            # Add bounding box if available
            if 'bbox' in metadata:
                x_min, y_min, x_max, y_max = metadata['bbox']
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                       linewidth=2, edgecolor='g', facecolor='none')
                axes[2].add_patch(rect)
        
        axes[2].set_title(f'Mask Overlay\nConfidence: {metadata.get("confidence_score", 0):.3f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_dataset(self, input_dir: str, split_name: str = 'train'):
        """
        Process entire dataset with segmentation
        
        Args:
            input_dir: Input dataset directory
            split_name: Dataset split (train/val/test)
        """
        self.logger.info(f"Processing {split_name} dataset from {input_dir}")
        
        # Process each class
        for class_name in ['acne', 'psoriasis', 'eczema']:
            class_input_dir = os.path.join(input_dir, class_name)
            class_output_dir = os.path.join(self.config.OUTPUT_DIR, split_name, class_name)
            
            if not os.path.exists(class_input_dir):
                self.logger.warning(f"Class directory not found: {class_input_dir}")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(class_input_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            self.logger.info(f"Processing {len(image_files)} {class_name} images...")
            
            for image_file in tqdm(image_files, desc=f"Segmenting {class_name}"):
                input_path = os.path.join(class_input_dir, image_file)
                
                # Segment lesion
                segmented_image, mask, metadata = self.segment_lesion(input_path)
                
                if segmented_image is not None:
                    # Save segmented image
                    output_path = os.path.join(class_output_dir, image_file)
                    segmented_pil = Image.fromarray(segmented_image)
                    segmented_pil.save(output_path, quality=95)
                    
                    # Save mask if enabled
                    if self.config.SAVE_MASKS and mask is not None:
                        mask_output_dir = os.path.join(self.config.OUTPUT_DIR, 'masks', split_name, class_name)
                        os.makedirs(mask_output_dir, exist_ok=True)
                        mask_path = os.path.join(mask_output_dir, image_file)
                        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                        mask_pil.save(mask_path)
                    
                    # Save visualization if enabled
                    if self.config.SAVE_VISUALIZATIONS:
                        original_image = cv2.imread(input_path)
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                        
                        viz_output_dir = os.path.join(self.config.OUTPUT_DIR, 'visualizations', split_name, class_name)
                        os.makedirs(viz_output_dir, exist_ok=True)
                        viz_path = os.path.join(viz_output_dir, image_file.replace('.jpg', '_viz.png'))
                        
                        self.visualize_segmentation(original_image, segmented_image, mask, metadata, viz_path)
                    
                    self.segmentation_stats['successful_segmentations'] += 1
                else:
                    self.segmentation_stats['failed_segmentations'] += 1
                
                self.segmentation_stats['total_processed'] += 1
    
    def process_all_datasets(self):
        """Process all dataset splits"""
        input_base_dir = "dataset"
        
        for split in ['train', 'val', 'test']:
            input_dir = os.path.join(input_base_dir, split)
            if os.path.exists(input_dir):
                self.process_dataset(input_dir, split)
            else:
                self.logger.warning(f"Dataset split not found: {input_dir}")
    
    def save_segmentation_report(self):
        """Save segmentation statistics report"""
        report = {
            'segmentation_statistics': self.segmentation_stats,
            'configuration': self.config.__dict__,
            'success_rate': (self.segmentation_stats['successful_segmentations'] / 
                           max(self.segmentation_stats['total_processed'], 1)) * 100
        }
        
        report_path = os.path.join(self.config.OUTPUT_DIR, 'segmentation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Segmentation report saved to {report_path}")
        
        # Print summary
        self.logger.info("\n" + "="*50)
        self.logger.info("SEGMENTATION SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Total Images Processed: {self.segmentation_stats['total_processed']}")
        self.logger.info(f"Successful Segmentations: {self.segmentation_stats['successful_segmentations']}")
        self.logger.info(f"Failed Segmentations: {self.segmentation_stats['failed_segmentations']}")
        self.logger.info(f"Success Rate: {report['success_rate']:.2f}%")
        self.logger.info(f"Dark Skin Images: {self.segmentation_stats['dark_skin_images']}")
        self.logger.info(f"Light Skin Images: {self.segmentation_stats['light_skin_images']}")


def main():
    """Main function for SAM-based lesion segmentation"""
    print("="*80)
    print("SAM-BASED LESION SEGMENTATION FOR SKIN DISEASE CLASSIFICATION")
    print("="*80)
    print("Optimizing lesion detection for dermatology use-cases")
    print("="*80)
    
    # Check SAM availability
    if not SAM_AVAILABLE:
        print("❌ ERROR: segment-anything package not installed")
        print("Please install with: pip install segment-anything")
        print("And download SAM model from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return
    
    # Create configuration
    config = SAMConfig()
    
    print(f"Configuration:")
    print(f"  SAM Model: {config.SAM_MODEL_TYPE}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Output Directory: {config.OUTPUT_DIR}")
    print(f"  Dark Skin Enhancement: {config.DARK_SKIN_ENHANCEMENT}")
    print(f"  Confidence Threshold: {config.CONFIDENCE_THRESHOLD}")
    print()
    
    # Check if input dataset exists
    input_dataset_dir = "dataset"
    if not os.path.exists(input_dataset_dir):
        print(f"❌ ERROR: Input dataset not found at {input_dataset_dir}")
        print("Please run the dataset preparation script first to create the dataset.")
        return
    
    try:
        # Create segmenter
        segmenter = LesionSegmenter(config)
        
        # Process all datasets
        segmenter.process_all_datasets()
        
        # Save report
        segmenter.save_segmentation_report()
        
        print("\n" + "="*80)
        print("SEGMENTATION COMPLETED!")
        print("="*80)
        print(f"Segmented dataset saved to: {config.OUTPUT_DIR}")
        print(f"Segmentation report saved to: {config.OUTPUT_DIR}/segmentation_report.json")
        print(f"Visualizations saved to: {config.OUTPUT_DIR}/visualizations/")
        print(f"Masks saved to: {config.OUTPUT_DIR}/masks/")
        print()
        print("Next steps:")
        print("1. Review segmentation quality in the visualizations folder")
        print("2. Use the segmented dataset for training the classification model")
        print("3. Update the dataset preparation script to use segmented images")
        
    except Exception as e:
        print(f"❌ ERROR during segmentation: {str(e)}")
        return


if __name__ == "__main__":
    main()
