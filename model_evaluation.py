#!/usr/bin/env python3
"""
Complete Model Evaluation for Skin Disease Classification

This script provides comprehensive evaluation metrics including
accuracy, precision, recall, F1-score, and confusion matrix visualization.

Author: AI Assistant
Date: 2025-03-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import model and dataset
from simple_train import SimpleSkinClassifier, SimpleSkinDataset
import torchvision.transforms as transforms
from PIL import Image

class ModelEvaluator:
    """
    Comprehensive model evaluation class with all metrics
    """
    
    def __init__(self, model: nn.Module, device: torch.device, class_names: List[str]):
        """
        Initialize evaluator
        
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
    
    def predict_dataset(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions for entire dataset
        
        Args:
            dataloader: Test data loader
            
        Returns:
            Tuple of (predictions, true_labels, probabilities)
        """
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        print("🔍 Generating predictions for evaluation...")
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        print("📊 Calculating basic metrics...")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }
        
        return metrics
    
    def calculate_auc_roc(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calculate AUC-ROC scores
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary of AUC scores
        """
        print("📈 Calculating AUC-ROC...")
        
        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        
        # Calculate AUC for each class
        auc_scores = {}
        for i in range(self.num_classes):
            if len(np.unique(y_true_bin[:, i])) > 1:  # Check if both classes present
                auc_scores[f'auc_{self.class_names[i]}'] = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            else:
                auc_scores[f'auc_{self.class_names[i]}'] = 0.0
        
        # Calculate macro average AUC
        macro_auc = np.mean(list(auc_scores.values()))
        auc_scores['auc_macro'] = macro_auc
        
        return auc_scores
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: str = None, normalize: bool = True) -> plt.Figure:
        """
        Plot and save confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
            normalize: Whether to normalize the confusion matrix
            
        Returns:
            Matplotlib figure
        """
        print("📊 Creating confusion matrix...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Proportion' if normalize else 'Count'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add text summary
        accuracy = accuracy_score(y_true, y_pred)
        plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.3f}', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Confusion matrix saved to: {save_path}")
        
        return plt.gcf()
    
    def plot_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 save_path: str = None) -> plt.Figure:
        """
        Plot classification report as heatmap
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        print("📋 Creating classification report visualization...")
        
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names, 
                                     output_dict=True, zero_division=0)
        
        # Convert to DataFrame for plotting
        report_df = pd.DataFrame(report).iloc[:-1, :].T  # Exclude 'accuracy' row
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot heatmap
        sns.heatmap(report_df.iloc[:, :-1].astype(float), annot=True, fmt='.3f', 
                   cmap='RdYlGn', center=0.5,
                   cbar_kws={'label': 'Score'})
        
        plt.title('Classification Report', fontsize=16, fontweight='bold')
        plt.xlabel('Metrics', fontsize=14)
        plt.ylabel('Classes', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Classification report saved to: {save_path}")
        
        return plt.gcf()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray, 
                       save_path: str = None) -> plt.Figure:
        """
        Plot ROC curves for multi-class classification
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        print("📈 Creating ROC curves...")
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
        
        for i in range(self.num_classes):
            if len(np.unique(y_true_bin[:, i])) > 1:  # Check if both classes present
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                auc_score = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                
                plt.plot(fpr, tpr, color=colors[i % len(colors)], 
                        label=f'{self.class_names[i]} (AUC = {auc_score:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curves', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 ROC curves saved to: {save_path}")
        
        return plt.gcf()
    
    def print_detailed_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_prob: np.ndarray) -> None:
        """
        Print detailed evaluation report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
        """
        print("\n" + "="*60)
        print("🏥 SKIN DISEASE CLASSIFICATION - DETAILED EVALUATION REPORT")
        print("="*60)
        
        # Basic metrics
        metrics = self.calculate_basic_metrics(y_true, y_pred)
        
        print("\n📊 OVERALL METRICS:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision_weighted']:.4f}")
        print(f"   Recall:    {metrics['recall_weighted']:.4f}")
        print(f"   F1-Score:  {metrics['f1_weighted']:.4f}")
        
        # Per-class metrics
        print("\n📋 PER-CLASS METRICS:")
        print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 45)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<12} {metrics['precision_per_class'][i]:<10.4f} "
                  f"{metrics['recall_per_class'][i]:<10.4f} {metrics['f1_per_class'][i]:<10.4f}")
        
        # AUC scores
        auc_scores = self.calculate_auc_roc(y_true, y_prob)
        print("\n📈 AUC-ROC SCORES:")
        for key, value in auc_scores.items():
            if key != 'auc_macro':
                print(f"   {key.replace('auc_', '')}: {value:.4f}")
        print(f"   Macro AUC: {auc_scores['auc_macro']:.4f}")
        
        # Classification report
        print("\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_true, y_pred, target_names=self.class_names, 
                                   digits=4, zero_division=0))
        
        # Confusion matrix summary
        cm = confusion_matrix(y_true, y_pred)
        print("\n📊 CONFUSION MATRIX SUMMARY:")
        print(f"   Total samples: {len(y_true)}")
        print(f"   Correct predictions: {np.trace(cm)}")
        print(f"   Incorrect predictions: {len(y_true) - np.trace(cm)}")
        
        print("\n" + "="*60)
    
    def evaluate_complete(self, test_loader: DataLoader, save_dir: str = "evaluation_results") -> Dict[str, Any]:
        """
        Complete evaluation with all metrics and visualizations
        
        Args:
            test_loader: Test data loader
            save_dir: Directory to save results
            
        Returns:
            Dictionary with all evaluation results
        """
        print("🚀 Starting complete model evaluation...")
        
        # Get predictions
        y_pred, y_true, y_prob = self.predict_dataset(test_loader)
        
        # Calculate all metrics
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred)
        auc_scores = self.calculate_auc_roc(y_true, y_prob)
        
        # Create visualizations
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot confusion matrix
        cm_fig = self.plot_confusion_matrix(
            y_true, y_pred, 
            save_path=os.path.join(save_dir, "confusion_matrix.png"),
            normalize=True
        )
        
        # Plot classification report
        report_fig = self.plot_classification_report(
            y_true, y_pred,
            save_path=os.path.join(save_dir, "classification_report.png")
        )
        
        # Plot ROC curves
        roc_fig = self.plot_roc_curves(
            y_true, y_prob,
            save_path=os.path.join(save_dir, "roc_curves.png")
        )
        
        # Print detailed report
        self.print_detailed_report(y_true, y_pred, y_prob)
        
        # Compile results
        results = {
            'basic_metrics': basic_metrics,
            'auc_scores': auc_scores,
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, 
                                                        target_names=self.class_names, 
                                                        output_dict=True, zero_division=0),
            'predictions': y_pred,
            'true_labels': y_true,
            'probabilities': y_prob,
            'num_samples': len(y_true),
            'save_directory': save_dir
        }
        
        # Save results to file
        self.save_results(results, os.path.join(save_dir, "evaluation_results.json"))
        
        print(f"\n🎉 Evaluation completed! Results saved to: {save_dir}")
        
        # Close plots to free memory
        plt.close('all')
        
        return results
    
    def save_results(self, results: Dict[str, Any], save_path: str) -> None:
        """
        Save evaluation results to JSON file
        
        Args:
            results: Results dictionary
            save_path: Path to save JSON file
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = value
            else:
                json_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to: {save_path}")

def load_model_and_data(model_path: str, data_dir: str = "data"):
    """
    Load trained model and test data
    
    Args:
        model_path: Path to trained model
        data_dir: Data directory
        
    Returns:
        Tuple of (model, test_loader, device, class_names)
    """
    print("📁 Loading model and data...")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Class names
    class_names = ['acne', 'psoriasis', 'eczema']
    
    # Load model
    model = SimpleSkinClassifier(num_classes=3)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {model_path}")
        print(f"   Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    else:
        print(f"❌ Model not found at {model_path}")
        return None, None, None, None
    
    model.to(device)
    model.eval()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = SimpleSkinDataset(os.path.join(data_dir, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"📊 Test dataset: {len(test_dataset)} samples")
    
    return model, test_loader, device, class_names

def main():
    """
    Main evaluation function
    """
    print("🏥 SKIN DISEASE CLASSIFICATION - MODEL EVALUATION")
    print("=" * 60)
    
    # Load model and data
    model_path = "models/simple_best_model.pth"
    model, test_loader, device, class_names = load_model_and_data(model_path)
    
    if model is None:
        print("❌ Failed to load model. Please train the model first.")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device, class_names)
    
    # Run complete evaluation
    results = evaluator.evaluate_complete(test_loader, save_dir="evaluation_results")
    
    # Summary
    print("\n🎯 EVALUATION SUMMARY:")
    print(f"   Model: Simple CNN")
    print(f"   Dataset: Test set ({results['num_samples']} samples)")
    print(f"   Accuracy: {results['basic_metrics']['accuracy']:.4f}")
    print(f"   F1-Score: {results['basic_metrics']['f1_weighted']:.4f}")
    print(f"   Macro AUC: {results['auc_scores']['auc_macro']:.4f}")
    print(f"   Results saved to: {results['save_directory']}")
    
    print("\n🎉 Evaluation completed successfully!")

if __name__ == "__main__":
    main()
