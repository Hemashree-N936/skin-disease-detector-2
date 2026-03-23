#!/usr/bin/env python3
"""
Fairness and Bias Evaluator for Skin Disease Classification

This module provides comprehensive fairness evaluation for skin disease models,
with focus on performance differences between dark and light skin tones.

Author: AI Assistant
Date: 2025-03-22
"""

import os
import logging
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import cv2
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import existing modules
from skin_disease_dataset_preparation import SkinDiseaseDataset, Config


class SkinToneClassifier:
    """
    Simple skin tone classifier based on image brightness and color analysis.
    Used to simulate skin tone metadata when actual metadata is not available.
    """
    
    def __init__(self):
        # Thresholds for skin tone classification based on pixel intensity
        # These are heuristic values and should be calibrated with actual skin tone data
        self.DARK_SKIN_THRESHOLD = 0.35  # Lower brightness threshold for dark skin
        self.LIGHT_SKIN_THRESHOLD = 0.55  # Higher threshold for light skin
        
    def extract_skin_tone_features(self, image_path: str) -> Dict[str, float]:
        """
        Extract skin tone features from image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with skin tone features
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {'brightness': 0.5, 'contrast': 0.5, 'skin_tone_score': 0.5}
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Extract central region (assuming skin region is in center)
            h, w = image.shape[:2]
            center_region = image[h//4:3*h//4, w//4:3*w//4]
            
            # Calculate brightness (average intensity)
            brightness = np.mean(center_region) / 255.0
            
            # Calculate contrast (standard deviation)
            contrast = np.std(center_region) / 255.0
            
            # Calculate skin tone score based on HSV values
            # Focus on the V (value/brightness) channel in HSV
            v_channel = hsv[h//4:3*h//4, w//4:3*w//4, 2]
            skin_tone_score = np.mean(v_channel) / 255.0
            
            return {
                'brightness': brightness,
                'contrast': contrast,
                'skin_tone_score': skin_tone_score
            }
            
        except Exception as e:
            logging.warning(f"Error processing {image_path}: {str(e)}")
            return {'brightness': 0.5, 'contrast': 0.5, 'skin_tone_score': 0.5}
    
    def classify_skin_tone(self, image_path: str) -> str:
        """
        Classify skin tone as 'dark' or 'light' based on image analysis
        
        Args:
            image_path: Path to the image file
            
        Returns:
            'dark' or 'light' skin tone classification
        """
        features = self.extract_skin_tone_features(image_path)
        skin_tone_score = features['skin_tone_score']
        
        if skin_tone_score < self.DARK_SKIN_THRESHOLD:
            return 'dark'
        elif skin_tone_score > self.LIGHT_SKIN_THRESHOLD:
            return 'light'
        else:
            # Medium skin tones - classify based on probability
            # For research purposes, we'll assign medium tones to dark
            # to focus on the underrepresented group
            return 'dark' if np.random.random() < 0.6 else 'light'


class FairnessMetrics:
    """Calculate and store fairness metrics"""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.skin_tones = []
        self.image_paths = []
    
    def update(self, predictions: np.ndarray, targets: np.ndarray, 
               skin_tones: List[str], image_paths: List[str]):
        """Update metrics with batch predictions"""
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        self.skin_tones.extend(skin_tones)
        self.image_paths.extend(image_paths)
    
    def calculate_fairness_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive fairness metrics"""
        if len(self.targets) == 0:
            return {}
        
        # Convert to numpy arrays
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        skin_tones = np.array(self.skin_tones)
        
        # Overall metrics
        overall_accuracy = accuracy_score(y_true, y_pred)
        overall_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Skin tone specific metrics
        skin_tone_groups = ['dark', 'light']
        skin_tone_metrics = {}
        
        for tone in skin_tone_groups:
            tone_mask = skin_tones == tone
            if np.sum(tone_mask) > 0:
                tone_y_true = y_true[tone_mask]
                tone_y_pred = y_pred[tone_mask]
                
                tone_accuracy = accuracy_score(tone_y_true, tone_y_pred)
                tone_precision = precision_score(tone_y_true, tone_y_pred, average='weighted', zero_division=0)
                tone_recall = recall_score(tone_y_true, tone_y_pred, average='weighted', zero_division=0)
                tone_f1 = f1_score(tone_y_true, tone_y_pred, average='weighted', zero_division=0)
                
                skin_tone_metrics[tone] = {
                    'accuracy': tone_accuracy,
                    'precision': tone_precision,
                    'recall': tone_recall,
                    'f1_score': tone_f1,
                    'sample_count': np.sum(tone_mask)
                }
                
                # Per-class metrics for this skin tone
                per_class_metrics = {}
                for i, class_name in enumerate(self.class_names):
                    class_mask = tone_y_true == i
                    if np.sum(class_mask) > 0:
                        class_acc = accuracy_score(tone_y_true[class_mask], tone_y_pred[class_mask])
                        per_class_metrics[class_name] = class_acc
                    else:
                        per_class_metrics[class_name] = 0.0
                
                skin_tone_metrics[tone]['per_class_accuracy'] = per_class_metrics
        
        # Calculate performance gaps
        performance_gaps = {}
        if 'dark' in skin_tone_metrics and 'light' in skin_tone_metrics:
            dark_acc = skin_tone_metrics['dark']['accuracy']
            light_acc = skin_tone_metrics['light']['accuracy']
            
            performance_gaps['accuracy_gap'] = light_acc - dark_acc
            performance_gaps['accuracy_gap_percentage'] = ((light_acc - dark_acc) / light_acc) * 100
            performance_gaps['dark_disadvantage_ratio'] = dark_acc / light_acc if light_acc > 0 else 0
            
            # Per-class gaps
            per_class_gaps = {}
            for class_name in self.class_names:
                dark_class_acc = skin_tone_metrics['dark']['per_class_accuracy'].get(class_name, 0)
                light_class_acc = skin_tone_metrics['light']['per_class_accuracy'].get(class_name, 0)
                
                if light_class_acc > 0:
                    per_class_gaps[class_name] = {
                        'gap': light_class_acc - dark_class_acc,
                        'gap_percentage': ((light_class_acc - dark_class_acc) / light_class_acc) * 100,
                        'dark_disadvantage_ratio': dark_class_acc / light_class_acc
                    }
                else:
                    per_class_gaps[class_name] = {
                        'gap': 0,
                        'gap_percentage': 0,
                        'dark_disadvantage_ratio': 0
                    }
            
            performance_gaps['per_class_gaps'] = per_class_gaps
        
        # Bias indicators
        bias_indicators = {}
        if performance_gaps:
            # Significant bias threshold (5% gap)
            significant_bias_threshold = 0.05
            
            bias_indicators['has_significant_accuracy_bias'] = abs(performance_gaps['accuracy_gap']) > significant_bias_threshold
            bias_indicators['dark_skin_disadvantaged'] = performance_gaps['accuracy_gap'] > significant_bias_threshold
            
            # Check per-class bias
            biased_classes = []
            for class_name, gaps in performance_gaps['per_class_gaps'].items():
                if abs(gaps['gap']) > significant_bias_threshold:
                    biased_classes.append(class_name)
            
            bias_indicators['biased_classes'] = biased_classes
            bias_indicators['number_of_biased_classes'] = len(biased_classes)
        
        # Sample distribution
        skin_tone_distribution = dict(Counter(skin_tones))
        
        fairness_report = {
            'overall_metrics': {
                'accuracy': overall_accuracy,
                'f1_score': overall_f1,
                'total_samples': len(y_true)
            },
            'skin_tone_metrics': skin_tone_metrics,
            'performance_gaps': performance_gaps,
            'bias_indicators': bias_indicators,
            'skin_tone_distribution': skin_tone_distribution,
            'sample_size_analysis': {
                'dark_samples': skin_tone_distribution.get('dark', 0),
                'light_samples': skin_tone_distribution.get('light', 0),
                'dark_to_light_ratio': skin_tone_distribution.get('dark', 0) / max(skin_tone_distribution.get('light', 1), 1)
            }
        }
        
        return fairness_report


class FairnessEvaluator:
    """Main fairness evaluation class"""
    
    def __init__(self, model: Any, class_names: List[str], device: torch.device):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.skin_tone_classifier = SkinToneClassifier()
        self.fairness_metrics = FairnessMetrics(class_names)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def evaluate_dataset_fairness(self, dataset: SkinDiseaseDataset, batch_size: int = 32) -> Dict[str, Any]:
        """
        Evaluate fairness on entire dataset
        
        Args:
            dataset: PyTorch dataset
            batch_size: Batch size for evaluation
            
        Returns:
            Comprehensive fairness report
        """
        self.logger.info("Starting fairness evaluation...")
        
        # Reset metrics
        self.fairness_metrics.reset()
        
        # Create data loader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Fairness Evaluation")):
                images = images.to(self.device)
                labels = labels.cpu().numpy()
                
                # Get predictions
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                
                # Get image paths from dataset
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(dataset))
                batch_paths = [dataset.samples[i][0] for i in range(start_idx, end_idx)]
                
                # Classify skin tones
                skin_tones = []
                for path in batch_paths:
                    skin_tone = self.skin_tone_classifier.classify_skin_tone(path)
                    skin_tones.append(skin_tone)
                
                # Update metrics
                self.fairness_metrics.update(predictions, labels, skin_tones, batch_paths)
        
        # Calculate fairness metrics
        fairness_report = self.fairness_metrics.calculate_fairness_metrics()
        
        self.logger.info("Fairness evaluation completed.")
        return fairness_report
    
    def generate_fairness_report(self, fairness_report: Dict[str, Any], save_path: str = "fairness_report.json"):
        """Generate and save comprehensive fairness report"""
        
        # Save detailed report
        with open(save_path, 'w') as f:
            json.dump(fairness_report, f, indent=2)
        
        self.logger.info(f"Fairness report saved to {save_path}")
        
        # Print summary
        self.print_fairness_summary(fairness_report)
        
        # Generate visualizations
        self.generate_fairness_visualizations(fairness_report)
    
    def print_fairness_summary(self, fairness_report: Dict[str, Any]):
        """Print detailed fairness summary"""
        print("\n" + "="*80)
        print("FAIRNESS AND BIAS EVALUATION REPORT")
        print("="*80)
        
        # Overall metrics
        overall = fairness_report['overall_metrics']
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Samples: {overall['total_samples']}")
        print(f"  Overall Accuracy: {overall['accuracy']:.4f}")
        print(f"  Overall F1-Score: {overall['f1_score']:.4f}")
        
        # Sample distribution
        distribution = fairness_report['skin_tone_distribution']
        print(f"\nSAMPLE DISTRIBUTION:")
        print(f"  Dark Skin Samples: {distribution.get('dark', 0)}")
        print(f"  Light Skin Samples: {distribution.get('light', 0)}")
        print(f"  Dark-to-Light Ratio: {fairness_report['sample_size_analysis']['dark_to_light_ratio']:.2f}")
        
        # Skin tone specific performance
        skin_metrics = fairness_report['skin_tone_metrics']
        print(f"\nSKIN TONE SPECIFIC PERFORMANCE:")
        
        for tone in ['dark', 'light']:
            if tone in skin_metrics:
                metrics = skin_metrics[tone]
                print(f"\n  {tone.upper()} SKIN:")
                print(f"    Samples: {metrics['sample_count']}")
                print(f"    Accuracy: {metrics['accuracy']:.4f}")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall: {metrics['recall']:.4f}")
                print(f"    F1-Score: {metrics['f1_score']:.4f}")
                
                print(f"    Per-Class Accuracy:")
                for class_name, acc in metrics['per_class_accuracy'].items():
                    print(f"      {class_name}: {acc:.4f}")
        
        # Performance gaps
        gaps = fairness_report['performance_gaps']
        if gaps:
            print(f"\nPERFORMANCE GAPS (Light vs Dark):")
            print(f"  Accuracy Gap: {gaps['accuracy_gap']:.4f}")
            print(f"  Accuracy Gap Percentage: {gaps['accuracy_gap_percentage']:.2f}%")
            print(f"  Dark Disadvantage Ratio: {gaps['dark_disadvantage_ratio']:.4f}")
            
            print(f"\n  Per-Class Performance Gaps:")
            for class_name, class_gaps in gaps['per_class_gaps'].items():
                print(f"    {class_name}:")
                print(f"      Gap: {class_gaps['gap']:.4f}")
                print(f"      Gap %: {class_gaps['gap_percentage']:.2f}%")
                print(f"      Dark Disadvantage Ratio: {class_gaps['dark_disadvantage_ratio']:.4f}")
        
        # Bias indicators
        bias = fairness_report['bias_indicators']
        print(f"\nBIAS INDICATORS:")
        print(f"  Has Significant Accuracy Bias: {bias.get('has_significant_accuracy_bias', False)}")
        print(f"  Dark Skin Disadvantaged: {bias.get('dark_skin_disadvantaged', False)}")
        print(f"  Number of Biased Classes: {bias.get('number_of_biased_classes', 0)}")
        
        if bias.get('biased_classes'):
            print(f"  Biased Classes: {', '.join(bias['biased_classes'])}")
        
        # Critical findings
        print(f"\n" + "="*50)
        print("CRITICAL FAIRNESS FINDINGS:")
        print("="*50)
        
        if bias.get('dark_skin_disadvantaged', False):
            print("⚠️  WARNING: Model performs worse on dark skin!")
            print(f"   Performance gap: {gaps['accuracy_gap_percentage']:.2f}%")
            print("   This indicates potential bias that needs to be addressed.")
        else:
            print("✅ No significant bias detected against dark skin.")
        
        if bias.get('number_of_biased_classes', 0) > 0:
            print(f"⚠️  {bias['number_of_biased_classes']} classes show bias:")
            for class_name in bias['biased_classes']:
                class_gap = gaps['per_class_gaps'][class_name]['gap_percentage']
                print(f"   - {class_name}: {class_gap:.2f}% gap")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if bias.get('dark_skin_disadvantaged', False):
            print("  1. Collect more diverse training data with dark skin samples")
            print("  2. Apply data augmentation techniques for dark skin tones")
            print("  3. Consider re-weighting loss function for dark skin samples")
            print("  4. Use bias mitigation techniques during training")
        else:
            print("  1. Continue monitoring fairness during model updates")
            print("  2. Validate with larger, more diverse test sets")
        
        print("="*80)
    
    def generate_fairness_visualizations(self, fairness_report: Dict[str, Any]):
        """Generate fairness visualization plots"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fairness and Bias Analysis for Skin Disease Classification', fontsize=16, fontweight='bold')
        
        # 1. Sample Distribution
        distribution = fairness_report['skin_tone_distribution']
        axes[0, 0].pie(distribution.values(), labels=distribution.keys(), autopct='%1.1f%%', 
                      colors=['#8B4513', '#FDBCB4'])
        axes[0, 0].set_title('Sample Distribution by Skin Tone')
        
        # 2. Overall Accuracy Comparison
        skin_metrics = fairness_report['skin_tone_metrics']
        tones = []
        accuracies = []
        
        for tone in ['dark', 'light']:
            if tone in skin_metrics:
                tones.append(tone.capitalize())
                accuracies.append(skin_metrics[tone]['accuracy'])
        
        bars = axes[0, 1].bar(tones, accuracies, color=['#8B4513', '#FDBCB4'])
        axes[0, 1].set_title('Accuracy by Skin Tone')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{acc:.3f}', ha='center', va='bottom')
        
        # 3. Per-Class Accuracy by Skin Tone
        if 'dark' in skin_metrics and 'light' in skin_metrics:
            classes = self.class_names
            dark_acc = [skin_metrics['dark']['per_class_accuracy'].get(cls, 0) for cls in classes]
            light_acc = [skin_metrics['light']['per_class_accuracy'].get(cls, 0) for cls in classes]
            
            x = np.arange(len(classes))
            width = 0.35
            
            axes[0, 2].bar(x - width/2, dark_acc, width, label='Dark Skin', color='#8B4513')
            axes[0, 2].bar(x + width/2, light_acc, width, label='Light Skin', color='#FDBCB4')
            
            axes[0, 2].set_title('Per-Class Accuracy by Skin Tone')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].set_xticks(x)
            axes[0, 2].set_xticklabels(classes, rotation=45)
            axes[0, 2].legend()
            axes[0, 2].set_ylim(0, 1)
        
        # 4. Performance Gap Visualization
        gaps = fairness_report['performance_gaps']
        if gaps and 'per_class_gaps' in gaps:
            classes = list(gaps['per_class_gaps'].keys())
            gap_values = [gaps['per_class_gaps'][cls]['gap_percentage'] for cls in classes]
            
            colors = ['red' if gap > 5 else 'orange' if gap > 0 else 'green' for gap in gap_values]
            
            bars = axes[1, 0].bar(classes, gap_values, color=colors)
            axes[1, 0].set_title('Performance Gap (Light - Dark) by Class')
            axes[1, 0].set_ylabel('Gap Percentage (%)')
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 0].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Significant Bias Threshold')
            axes[1, 0].legend()
            
            # Add value labels
            for bar, gap in zip(bars, gap_values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5 if gap > 0 else bar.get_height() - 0.5, 
                               f'{gap:.1f}%', ha='center', va='bottom' if gap > 0 else 'top')
        
        # 5. Confusion Matrices (if we have the data)
        # This would require storing all predictions and targets per skin tone
        axes[1, 1].text(0.5, 0.5, 'Confusion Matrices\n(Requires detailed prediction data)', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Confusion Matrices by Skin Tone')
        
        # 6. Summary Metrics
        bias = fairness_report['bias_indicators']
        summary_text = f"""
        Bias Summary:
        
        Significant Bias: {'Yes' if bias.get('has_significant_accuracy_bias', False) else 'No'}
        Dark Skin Disadvantaged: {'Yes' if bias.get('dark_skin_disadvantaged', False) else 'No'}
        Biased Classes: {bias.get('number_of_biased_classes', 0)}
        
        Overall Fairness Score: {self.calculate_fairness_score(fairness_report):.3f}
        """
        
        axes[1, 2].text(0.1, 0.5, summary_text, va='center', fontsize=10, 
                       transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Fairness Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('fairness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Fairness visualizations saved to fairness_analysis.png")
    
    def calculate_fairness_score(self, fairness_report: Dict[str, Any]) -> float:
        """Calculate overall fairness score (0-1, higher is better)"""
        gaps = fairness_report['performance_gaps']
        if not gaps:
            return 1.0
        
        # Base score starts at 1.0
        score = 1.0
        
        # Penalize accuracy gaps
        acc_gap = abs(gaps.get('accuracy_gap', 0))
        score -= acc_gap * 2  # Double penalty for accuracy gaps
        
        # Penalize per-class gaps
        if 'per_class_gaps' in gaps:
            for class_gap in gaps['per_class_gaps'].values():
                score -= abs(class_gap['gap']) * 0.5
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return score


def main():
    """Example usage of fairness evaluator"""
    print("Fairness Evaluator Example")
    print("="*50)
    
    # This would typically be used with a trained model
    # For demonstration, we'll show the structure
    
    print("To use the fairness evaluator:")
    print("1. Load your trained model")
    print("2. Create a FairnessEvaluator instance")
    print("3. Call evaluate_dataset_fairness() on your test set")
    print("4. Generate comprehensive fairness report")
    
    print("\nExample code:")
    print("""
    from fairness_evaluator import FairnessEvaluator
    from efficientnet_skin_classifier import EfficientNetB4Classifier
    
    # Load model
    model = EfficientNetB4Classifier()
    model.load_state_dict(torch.load('models/best_model.pth')['model_state_dict'])
    
    # Create evaluator
    evaluator = FairnessEvaluator(model, ['acne', 'psoriasis', 'eczema'], device)
    
    # Evaluate fairness
    fairness_report = evaluator.evaluate_dataset_fairness(test_dataset)
    
    # Generate report
    evaluator.generate_fairness_report(fairness_report)
    """)


if __name__ == "__main__":
    main()
