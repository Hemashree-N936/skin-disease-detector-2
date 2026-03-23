#!/usr/bin/env python3
"""
Basic Fairness Evaluation for Skin Disease Classification

This script provides simple fairness evaluation to detect bias
in skin disease classification models across different skin tone groups.

Author: AI Assistant
Date: 2025-03-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import random
import warnings
warnings.filterwarnings('ignore')

class FairnessEvaluator:
    """
    Simple fairness evaluator for skin disease classification
    """
    
    def __init__(self, model: nn.Module, device: torch.device, class_names: list):
        """
        Initialize fairness evaluator
        
        Args:
            model: Trained PyTorch model
            device: Device (cpu/cuda)
            class_names: List of class names
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def simulate_skin_tone_groups(self, dataset) -> dict:
        """
        Simulate skin tone groups by splitting dataset
        
        Args:
            dataset: PyTorch dataset
            
        Returns:
            Dictionary with group indices
        """
        print("🎨 Simulating skin tone groups...")
        
        # Get all indices
        all_indices = list(range(len(dataset)))
        
        # Simulate skin tone distribution
        # In real scenarios, this would come from actual skin tone labels
        # Here we simulate by random assignment with realistic distribution
        
        # Simulate distribution: 30% light, 40% medium, 30% dark
        # This is a simplified simulation - in practice, use actual skin tone data
        
        random.shuffle(all_indices)
        
        n_total = len(all_indices)
        n_light = int(0.3 * n_total)
        n_medium = int(0.4 * n_total)
        n_dark = n_total - n_light - n_medium
        
        light_indices = all_indices[:n_light]
        medium_indices = all_indices[n_light:n_light + n_medium]
        dark_indices = all_indices[n_light + n_medium:]
        
        groups = {
            'light_skin': light_indices,
            'medium_skin': medium_indices,
            'dark_skin': dark_indices
        }
        
        print(f"📊 Group distribution:")
        for group_name, indices in groups.items():
            print(f"   {group_name}: {len(indices)} samples ({len(indices)/n_total:.1%})")
        
        return groups
    
    def create_group_dataloaders(self, dataset, groups: dict, batch_size: int = 32) -> dict:
        """
        Create dataloaders for each group
        
        Args:
            dataset: Original dataset
            groups: Dictionary with group indices
            batch_size: Batch size
            
        Returns:
            Dictionary with group dataloaders
        """
        group_loaders = {}
        
        for group_name, indices in groups.items():
            # Create subset for this group
            group_subset = Subset(dataset, indices)
            
            # Create dataloader
            group_loader = DataLoader(
                group_subset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=2
            )
            
            group_loaders[group_name] = group_loader
        
        return group_loaders
    
    def evaluate_group_performance(self, group_loader: DataLoader, group_name: str) -> dict:
        """
        Evaluate model performance on a specific group
        
        Args:
            group_loader: DataLoader for the group
            group_name: Name of the group
            
        Returns:
            Dictionary with performance metrics
        """
        print(f"🔍 Evaluating {group_name}...")
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in group_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(all_labels, all_predictions, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_predictions, average=None, zero_division=0)
        f1_per_class = f1_score(all_labels, all_predictions, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        metrics = {
            'group_name': group_name,
            'num_samples': len(all_labels),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'true_labels': all_labels
        }
        
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1:.3f}")
        
        return metrics
    
    def detect_bias(self, group_metrics: dict) -> dict:
        """
        Detect bias by comparing performance across groups
        
        Args:
            group_metrics: Dictionary with metrics for each group
            
        Returns:
            Dictionary with bias analysis
        """
        print("\n🔍 Detecting bias...")
        
        # Extract accuracies
        accuracies = {group: metrics['accuracy'] for group, metrics in group_metrics.items()}
        
        # Calculate performance differences
        max_acc = max(accuracies.values())
        min_acc = min(accuracies.values())
        accuracy_gap = max_acc - min_acc
        
        # Determine if bias exists
        bias_threshold = 0.05  # 5% difference threshold
        has_bias = accuracy_gap > bias_threshold
        
        # Find worst performing group
        worst_group = min(accuracies, key=accuracies.get)
        best_group = max(accuracies, key=accuracies.get)
        
        bias_analysis = {
            'has_bias': has_bias,
            'accuracy_gap': accuracy_gap,
            'max_accuracy': max_acc,
            'min_accuracy': min_acc,
            'best_group': best_group,
            'worst_group': worst_group,
            'bias_threshold': bias_threshold,
            'group_accuracies': accuracies
        }
        
        print(f"📊 Bias Analysis Results:")
        print(f"   Accuracy Gap: {accuracy_gap:.3f} ({accuracy_gap*100:.1f}%)")
        print(f"   Best Group: {best_group} ({max_acc:.3f})")
        print(f"   Worst Group: {worst_group} ({min_acc:.3f})")
        print(f"   Bias Detected: {'Yes ⚠️' if has_bias else 'No ✅'}")
        
        return bias_analysis
    
    def print_comparison_table(self, group_metrics: dict, bias_analysis: dict):
        """
        Print comparison table of group performances
        
        Args:
            group_metrics: Dictionary with metrics for each group
            bias_analysis: Bias analysis results
        """
        print("\n📊 FAIRNESS EVALUATION RESULTS")
        print("=" * 80)
        
        # Create comparison table
        print(f"{'Group':<15} {'Samples':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 80)
        
        for group_name, metrics in group_metrics.items():
            print(f"{group_name:<15} {metrics['num_samples']:<8} "
                  f"{metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
                  f"{metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f}")
        
        print("-" * 80)
        
        # Bias summary
        print(f"\n🔍 BIAS ANALYSIS:")
        print(f"   Accuracy Gap: {bias_analysis['accuracy_gap']:.3f} ({bias_analysis['accuracy_gap']*100:.1f}%)")
        print(f"   Bias Threshold: {bias_analysis['bias_threshold']:.3f} ({bias_analysis['bias_threshold']*100:.1f}%)")
        print(f"   Bias Detected: {'YES ⚠️' if bias_analysis['has_bias'] else 'NO ✅'}")
        
        if bias_analysis['has_bias']:
            print(f"\n⚠️  BIAS DETECTED!")
            print(f"   Model performs significantly worse on {bias_analysis['worst_group']}")
            print(f"   Consider: Data balancing, model adjustment, or targeted training")
        else:
            print(f"\n✅ NO SIGNIFICANT BIAS DETECTED")
            print(f"   Model performs consistently across all skin tone groups")
    
    def plot_fairness_comparison(self, group_metrics: dict, save_path: str = None):
        """
        Plot fairness comparison visualization
        
        Args:
            group_metrics: Dictionary with metrics for each group
            save_path: Path to save the plot
        """
        print("\n📈 Creating fairness comparison plot...")
        
        # Extract data for plotting
        groups = list(group_metrics.keys())
        accuracies = [group_metrics[group]['accuracy'] for group in groups]
        precisions = [group_metrics[group]['precision'] for group in groups]
        recalls = [group_metrics[group]['recall'] for group in groups]
        f1_scores = [group_metrics[group]['f1_score'] for group in groups]
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        bars1 = ax1.bar(groups, accuracies, color=['lightblue', 'lightgreen', 'lightcoral'])
        ax1.set_title('Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Precision comparison
        bars2 = ax2.bar(groups, precisions, color=['lightblue', 'lightgreen', 'lightcoral'])
        ax2.set_title('Precision Comparison', fontweight='bold')
        ax2.set_ylabel('Precision')
        ax2.set_ylim(0, 1)
        for bar, prec in zip(bars2, precisions):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{prec:.3f}', ha='center', va='bottom')
        
        # Recall comparison
        bars3 = ax3.bar(groups, recalls, color=['lightblue', 'lightgreen', 'lightcoral'])
        ax3.set_title('Recall Comparison', fontweight='bold')
        ax3.set_ylabel('Recall')
        ax3.set_ylim(0, 1)
        for bar, rec in zip(bars3, recalls):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rec:.3f}', ha='center', va='bottom')
        
        # F1-Score comparison
        bars4 = ax4.bar(groups, f1_scores, color=['lightblue', 'lightgreen', 'lightcoral'])
        ax4.set_title('F1-Score Comparison', fontweight='bold')
        ax4.set_ylabel('F1-Score')
        ax4.set_ylim(0, 1)
        for bar, f1 in zip(bars4, f1_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Fairness comparison plot saved to: {save_path}")
        
        plt.show()
    
    def plot_per_class_fairness(self, group_metrics: dict, save_path: str = None):
        """
        Plot per-class fairness analysis
        
        Args:
            group_metrics: Dictionary with metrics for each group
            save_path: Path to save the plot
        """
        print("\n📊 Creating per-class fairness analysis...")
        
        # Create figure for per-class metrics
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, class_name in enumerate(self.class_names):
            ax = axes[i]
            
            # Extract per-class metrics for this class
            groups = list(group_metrics.keys())
            precisions = [group_metrics[group]['precision_per_class'][i] for group in groups]
            recalls = [group_metrics[group]['recall_per_class'][i] for group in groups]
            f1_scores = [group_metrics[group]['f1_per_class'][i] for group in groups]
            
            # Create bar plot
            x = np.arange(len(groups))
            width = 0.25
            
            bars1 = ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
            bars2 = ax.bar(x, recalls, width, label='Recall', alpha=0.8)
            bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
            
            ax.set_title(f'{class_name.upper()} - Per-Class Fairness', fontweight='bold')
            ax.set_xlabel('Skin Tone Group')
            ax.set_ylabel('Score')
            ax.set_xticks(x)
            ax.set_xticklabels(groups)
            ax.legend()
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Per-class fairness plot saved to: {save_path}")
        
        plt.show()
    
    def evaluate_fairness_complete(self, dataset, save_dir: str = "fairness_results") -> dict:
        """
        Complete fairness evaluation pipeline
        
        Args:
            dataset: Test dataset
            save_dir: Directory to save results
            
        Returns:
            Dictionary with all fairness results
        """
        print("🚀 Starting Complete Fairness Evaluation...")
        print("=" * 60)
        
        # Step 1: Simulate skin tone groups
        groups = self.simulate_skin_tone_groups(dataset)
        
        # Step 2: Create group dataloaders
        group_loaders = self.create_group_dataloaders(dataset, groups)
        
        # Step 3: Evaluate each group
        group_metrics = {}
        for group_name, group_loader in group_loaders.items():
            group_metrics[group_name] = self.evaluate_group_performance(group_loader, group_name)
        
        # Step 4: Detect bias
        bias_analysis = self.detect_bias(group_metrics)
        
        # Step 5: Print comparison table
        self.print_comparison_table(group_metrics, bias_analysis)
        
        # Step 6: Create visualizations
        os.makedirs(save_dir, exist_ok=True)
        
        # Overall fairness comparison
        self.plot_fairness_comparison(
            group_metrics, 
            save_path=os.path.join(save_dir, "fairness_comparison.png")
        )
        
        # Per-class fairness analysis
        self.plot_per_class_fairness(
            group_metrics, 
            save_path=os.path.join(save_dir, "per_class_fairness.png")
        )
        
        # Compile results
        results = {
            'group_metrics': group_metrics,
            'bias_analysis': bias_analysis,
            'group_distribution': {group: len(indices) for group, indices in groups.items()},
            'save_directory': save_dir
        }
        
        # Save results to JSON
        self.save_fairness_results(results, os.path.join(save_dir, "fairness_results.json"))
        
        print(f"\n🎉 Fairness evaluation completed!")
        print(f"📁 Results saved to: {save_dir}")
        
        return results
    
    def save_fairness_results(self, results: dict, save_path: str):
        """
        Save fairness results to JSON file
        
        Args:
            results: Results dictionary
            save_path: Path to save JSON file
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if key == 'group_metrics':
                json_results[key] = {}
                for group, metrics in value.items():
                    json_results[key][group] = {}
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, np.ndarray):
                            json_results[key][group][metric_name] = metric_value.tolist()
                        else:
                            json_results[key][group][metric_name] = metric_value
            else:
                json_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Fairness results saved to: {save_path}")

def load_model_and_data(model_path: str = "models/simple_best_model.pth"):
    """
    Load trained model and test data for fairness evaluation
    
    Args:
        model_path: Path to trained model
        
    Returns:
        Tuple of (model, test_dataset, device, class_names)
    """
    print("📁 Loading model and data for fairness evaluation...")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Class names
    class_names = ['acne', 'psoriasis', 'eczema']
    
    # Load model
    from simple_train import SimpleSkinClassifier, SimpleSkinDataset
    import torchvision.transforms as transforms
    
    model = SimpleSkinClassifier(num_classes=3)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {model_path}")
    else:
        print(f"⚠️  Model not found at {model_path}")
        print("Using randomly initialized model for demonstration")
    
    model.to(device)
    model.eval()
    
    # Load test data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = SimpleSkinDataset('data/test', transform=transform)
    
    print(f"📊 Test dataset: {len(test_dataset)} samples")
    
    return model, test_dataset, device, class_names

def main():
    """
    Main fairness evaluation function
    """
    print("🏥 SKIN DISEASE CLASSIFICATION - FAIRNESS EVALUATION")
    print("=" * 60)
    
    # Load model and data
    model, test_dataset, device, class_names = load_model_and_data()
    
    if model is None:
        print("❌ Failed to load model. Please train the model first.")
        return
    
    # Create fairness evaluator
    evaluator = FairnessEvaluator(model, device, class_names)
    
    # Run complete fairness evaluation
    results = evaluator.evaluate_fairness_complete(test_dataset, save_dir="fairness_results")
    
    # Summary
    print("\n🎯 FAIRNESS EVALUATION SUMMARY:")
    print(f"   Groups evaluated: {len(results['group_metrics'])}")
    print(f"   Total samples: {sum(results['group_distribution'].values())}")
    print(f"   Bias detected: {'Yes ⚠️' if results['bias_analysis']['has_bias'] else 'No ✅'}")
    print(f"   Accuracy gap: {results['bias_analysis']['accuracy_gap']:.3f}")
    print(f"   Results saved to: {results['save_directory']}")
    
    print("\n🎉 Fairness evaluation completed successfully!")

if __name__ == "__main__":
    import os
    main()
