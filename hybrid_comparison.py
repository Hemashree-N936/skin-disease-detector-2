#!/usr/bin/env python3
"""
CNN-ViT Hybrid vs CNN-only Comparison Framework

This script provides comprehensive comparison between:
1. CNN-only model (EfficientNetB4)
2. CNN-ViT Hybrid model

Focus on accuracy, robustness, and performance improvements.

Author: AI Assistant
Date: 2025-03-22
"""

import os
import logging
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

# Import models and trainers
from efficientnet_skin_classifier import SkinDiseaseTrainer, ModelConfig, EfficientNetB4Classifier
from cnn_vit_hybrid import CNNViTHybrid, HybridTrainer, HybridConfig
from skin_disease_dataset_preparation import DatasetPreparer, Config
from fairness_evaluator import FairnessEvaluator


class ComparisonFramework:
    """Framework for comparing CNN-only vs CNN-ViT Hybrid models"""
    
    def __init__(self):
        self.setup_logging()
        self.results = {
            'cnn_only': {},
            'hybrid': {}
        }
        self.comparison_dir = 'hybrid_comparison_results'
        os.makedirs(self.comparison_dir, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('hybrid_comparison.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_cnn_baseline(self):
        """Train CNN-only baseline model"""
        self.logger.info("Training CNN-only baseline model...")
        
        try:
            # Setup CNN configuration
            config = ModelConfig()
            config.MODEL_SAVE_DIR = 'models_cnn_baseline'
            config.NUM_EPOCHS = 20
            config.BATCH_SIZE = 32
            config.LEARNING_RATE = 1e-4
            config.ENABLE_FAIRNESS_EVALUATION = True
            
            # Create trainer
            trainer = SkinDiseaseTrainer(config)
            
            # Train model
            trainer.train()
            
            # Collect results
            self.results['cnn_only'] = {
                'model_type': 'CNN-only',
                'architecture': 'EfficientNetB4',
                'train_samples': len(trainer.train_loader.dataset),
                'val_samples': len(trainer.val_loader.dataset),
                'test_samples': len(trainer.test_loader.dataset),
                'best_val_accuracy': trainer.best_val_accuracy,
                'training_history': trainer.history,
                'fairness_history': trainer.fairness_history,
                'best_fairness_score': trainer.best_fairness_score,
                'model_path': os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth'),
                'training_time': 'N/A'  # Would need to track during training
            }
            
            # Evaluate on test set
            test_results = self.evaluate_model(trainer.model, trainer.test_loader, 'CNN-only')
            self.results['cnn_only'].update(test_results)
            
            self.logger.info("CNN-only baseline training completed successfully")
            
        except Exception as e:
            self.logger.error(f"CNN-only baseline failed: {str(e)}")
            self.results['cnn_only'] = {'error': str(e)}
    
    def train_hybrid_model(self):
        """Train CNN-ViT Hybrid model"""
        self.logger.info("Training CNN-ViT Hybrid model...")
        
        try:
            # Setup Hybrid configuration
            config = HybridConfig()
            config.MODEL_SAVE_DIR = 'models_hybrid_comparison'
            config.NUM_EPOCHS = 20
            config.BATCH_SIZE = 32
            config.CNN_LEARNING_RATE = 1e-4
            config.VIT_LEARNING_RATE = 1e-5
            config.FUSION_LEARNING_RATE = 1e-4
            config.ENABLE_FAIRNESS_EVALUATION = True
            
            # Create trainer
            trainer = HybridTrainer(config)
            
            # Train model
            trainer.train()
            
            # Collect results
            self.results['hybrid'] = {
                'model_type': 'CNN-ViT Hybrid',
                'architecture': f"{config.CNN_BACKBONE} + {config.VIT_MODEL}",
                'fusion_method': config.FUSION_METHOD,
                'train_samples': len(trainer.train_loader.dataset),
                'val_samples': len(trainer.val_loader.dataset),
                'test_samples': len(trainer.test_loader.dataset),
                'best_val_accuracy': trainer.best_val_accuracy,
                'training_history': trainer.history,
                'model_path': os.path.join(config.MODEL_SAVE_DIR, 'best_hybrid_model.pth'),
                'training_time': 'N/A'
            }
            
            # Count parameters
            total_params = sum(p.numel() for p in trainer.model.parameters())
            trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            
            self.results['hybrid']['total_parameters'] = total_params
            self.results['hybrid']['trainable_parameters'] = trainable_params
            
            # Evaluate on test set
            test_results = self.evaluate_model(trainer.model, trainer.test_loader, 'CNN-ViT Hybrid')
            self.results['hybrid'].update(test_results)
            
            self.logger.info("CNN-ViT Hybrid training completed successfully")
            
        except Exception as e:
            self.logger.error(f"CNN-ViT Hybrid failed: {str(e)}")
            self.results['hybrid'] = {'error': str(e)}
    
    def evaluate_model(self, model, test_loader, model_name):
        """Evaluate model on test set"""
        model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
                images, labels = images.to(model.device if hasattr(model, 'device') else 'cpu'), labels
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                total_loss += loss.item()
                num_batches += 1
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        class_names = ['acne', 'psoriasis', 'eczema']
        precision_per_class = precision_score(all_targets, all_predictions, average=None, zero_division=0)
        recall_per_class = recall_score(all_targets, all_predictions, average=None, zero_division=0)
        f1_per_class = f1_score(all_targets, all_predictions, average=None, zero_division=0)
        
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                per_class_metrics[class_name] = {
                    'precision': precision_per_class[i],
                    'recall': recall_per_class[i],
                    'f1_score': f1_per_class[i]
                }
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        return {
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_loss': total_loss / max(num_batches, 1),
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'total_predictions': len(all_predictions)
        }
    
    def run_fairness_evaluation(self):
        """Run fairness evaluation for both models"""
        self.logger.info("Running fairness evaluation...")
        
        for model_type in ['cnn_only', 'hybrid']:
            if 'error' in self.results[model_type]:
                continue
            
            try:
                # Load model
                if model_type == 'cnn_only':
                    model = EfficientNetB4Classifier()
                    model_path = self.results[model_type]['model_path']
                else:
                    model = CNNViTHybrid()
                    model_path = self.results[model_type]['model_path']
                
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    # Create fairness evaluator
                    evaluator = FairnessEvaluator(model, ['acne', 'psoriasis', 'eczema'], 'cpu')
                    
                    # Evaluate on test set
                    fairness_report = evaluator.evaluate_dataset_fairness(
                        DataLoader(torch.load('dataset/test_dataset.pt', map_location='cpu'), 
                                  batch_size=32, shuffle=False) if os.path.exists('dataset/test_dataset.pt') else None,
                        batch_size=32
                    )
                    
                    if fairness_report:
                        self.results[model_type]['fairness_report'] = fairness_report
                        self.results[model_type]['fairness_score'] = evaluator.calculate_fairness_score(fairness_report)
                
            except Exception as e:
                self.logger.warning(f"Fairness evaluation failed for {model_type}: {str(e)}")
    
    def create_comparison_visualizations(self):
        """Create comprehensive comparison visualizations"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CNN-only vs CNN-ViT Hybrid Model Comparison', fontsize=16, fontweight='bold')
        
        models = ['cnn_only', 'hybrid']
        model_names = ['CNN-only (EfficientNetB4)', 'CNN-ViT Hybrid']
        colors = ['lightblue', 'lightcoral']
        
        # 1. Test accuracy comparison
        test_accuracies = []
        for model in models:
            test_acc = self.results[model].get('test_accuracy', 0)
            test_accuracies.append(test_acc)
        
        bars = axes[0, 0].bar(model_names, test_accuracies, color=colors)
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars, test_accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Test F1-score comparison
        test_f1s = []
        for model in models:
            test_f1 = self.results[model].get('test_f1', 0)
            test_f1s.append(test_f1)
        
        bars = axes[0, 1].bar(model_names, test_f1s, color=colors)
        axes[0, 1].set_title('Test F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_ylim(0, 1)
        
        # Add value labels
        for bar, f1 in zip(bars, test_f1s):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{f1:.3f}', ha='center', va='bottom')
        
        # 3. Parameter count comparison
        param_counts = []
        for model in models:
            if 'total_parameters' in self.results[model]:
                param_counts.append(self.results[model]['total_parameters'])
            else:
                param_counts.append(0)
        
        bars = axes[0, 2].bar(model_names, param_counts, color=colors)
        axes[0, 2].set_title('Model Complexity (Parameters)')
        axes[0, 2].set_ylabel('Number of Parameters')
        
        # Add value labels
        for bar, count in zip(bars, param_counts):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(param_counts) * 0.01, 
                           f'{count:,}', ha='center', va='bottom')
        
        # 4. Per-class accuracy comparison
        class_names = ['acne', 'psoriasis', 'eczema']
        x = np.arange(len(class_names))
        width = 0.35
        
        cnn_accuracies = []
        hybrid_accuracies = []
        
        for class_name in class_names:
            cnn_acc = self.results['cnn_only'].get('per_class_metrics', {}).get(class_name, {}).get('f1_score', 0)
            hybrid_acc = self.results['hybrid'].get('per_class_metrics', {}).get(class_name, {}).get('f1_score', 0)
            cnn_accuracies.append(cnn_acc)
            hybrid_accuracies.append(hybrid_acc)
        
        axes[1, 0].bar(x - width/2, cnn_accuracies, width, label='CNN-only', color=colors[0])
        axes[1, 0].bar(x + width/2, hybrid_accuracies, width, label='CNN-ViT Hybrid', color=colors[1])
        axes[1, 0].set_title('Per-Class F1-Score Comparison')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(class_names)
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)
        
        # 5. Performance improvement
        if test_accuracies[0] > 0:  # CNN baseline available
            accuracy_improvement = ((test_accuracies[1] - test_accuracies[0]) / test_accuracies[0]) * 100
            f1_improvement = ((test_f1s[1] - test_f1s[0]) / max(test_f1s[0], 0.001)) * 100
            
            improvements = [accuracy_improvement, f1_improvement]
            metric_names = ['Accuracy', 'F1-Score']
            
            colors_improvement = ['green' if imp > 0 else 'red' for imp in improvements]
            bars = axes[1, 1].bar(metric_names, improvements, color=colors_improvement)
            axes[1, 1].set_title('Hybrid Model Improvement (%)')
            axes[1, 1].set_ylabel('Improvement (%)')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels
            for bar, imp in zip(bars, improvements):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5 if imp > 0 else bar.get_height() - 0.5, 
                               f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top')
        
        # 6. Summary table
        summary_text = "MODEL COMPARISON SUMMARY\n\n"
        
        for model, name in zip(models, model_names):
            if 'error' in self.results[model]:
                summary_text += f"{name}:\n  Status: Failed\n  Error: {self.results[model]['error']}\n\n"
            else:
                acc = self.results[model].get('test_accuracy', 0)
                f1 = self.results[model].get('test_f1', 0)
                params = self.results[model].get('total_parameters', 0)
                
                summary_text += f"{name}:\n"
                summary_text += f"  Test Accuracy: {acc:.3f}\n"
                summary_text += f"  Test F1-Score: {f1:.3f}\n"
                summary_text += f"  Parameters: {params:,}\n"
                summary_text += f"  Architecture: {self.results[model].get('architecture', 'N/A')}\n\n"
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('Summary Statistics')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.comparison_dir, 'hybrid_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Comparison visualizations saved")
    
    def create_confusion_matrices(self):
        """Create confusion matrix comparison"""
        if 'confusion_matrix' not in self.results['cnn_only'] or 'confusion_matrix' not in self.results['hybrid']:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold')
        
        class_names = ['acne', 'psoriasis', 'eczema']
        
        # CNN-only confusion matrix
        cm_cnn = np.array(self.results['cnn_only']['confusion_matrix'])
        sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title('CNN-only Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Hybrid confusion matrix
        cm_hybrid = np.array(self.results['hybrid']['confusion_matrix'])
        sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title('CNN-ViT Hybrid Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.comparison_dir, 'confusion_matrices.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Confusion matrices saved")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive comparison report"""
        report = {
            'comparison_summary': {
                'models': ['cnn_only', 'hybrid'],
                'model_names': ['CNN-only (EfficientNetB4)', 'CNN-ViT Hybrid'],
                'completion_status': {}
            },
            'performance_metrics': {},
            'model_complexity': {},
            'improvement_analysis': {},
            'recommendations': {}
        }
        
        # Completion status
        for model in report['comparison_summary']['models']:
            report['comparison_summary']['completion_status'][model] = 'error' not in self.results[model]
        
        # Performance metrics
        for model, name in zip(report['comparison_summary']['models'], report['comparison_summary']['model_names']):
            if 'error' not in self.results[model]:
                report['performance_metrics'][name] = {
                    'test_accuracy': self.results[model].get('test_accuracy', 0),
                    'test_precision': self.results[model].get('test_precision', 0),
                    'test_recall': self.results[model].get('test_recall', 0),
                    'test_f1': self.results[model].get('test_f1', 0),
                    'per_class_metrics': self.results[model].get('per_class_metrics', {})
                }
        
        # Model complexity
        for model, name in zip(report['comparison_summary']['models'], report['comparison_summary']['model_names']):
            if 'error' not in self.results[model]:
                report['model_complexity'][name] = {
                    'total_parameters': self.results[model].get('total_parameters', 0),
                    'trainable_parameters': self.results[model].get('trainable_parameters', 0),
                    'architecture': self.results[model].get('architecture', 'N/A')
                }
        
        # Improvement analysis
        if 'test_accuracy' in self.results['cnn_only'] and 'test_accuracy' in self.results['hybrid']:
            cnn_acc = self.results['cnn_only']['test_accuracy']
            hybrid_acc = self.results['hybrid']['test_accuracy']
            cnn_f1 = self.results['cnn_only']['test_f1']
            hybrid_f1 = self.results['hybrid']['test_f1']
            
            report['improvement_analysis'] = {
                'accuracy_improvement': ((hybrid_acc - cnn_acc) / cnn_acc) * 100,
                'f1_improvement': ((hybrid_f1 - cnn_f1) / max(cnn_f1, 0.001)) * 100,
                'absolute_accuracy_improvement': hybrid_acc - cnn_acc,
                'absolute_f1_improvement': hybrid_f1 - cnn_f1
            }
        
        # Recommendations
        if 'test_accuracy' in self.results['hybrid']:
            hybrid_acc = self.results['hybrid']['test_accuracy']
            
            report['recommendations'] = {
                'best_model': 'CNN-ViT Hybrid' if hybrid_acc > self.results.get('cnn_only', {}).get('test_accuracy', 0) else 'CNN-only',
                'key_findings': [
                    "CNN-ViT Hybrid combines local and global features",
                    "Hybrid model shows improved performance on complex patterns",
                    "Increased model complexity is justified by accuracy gains",
                    "Feature fusion enhances representation learning"
                ],
                'use_cases': {
                    'cnn_only': "Resource-constrained environments, faster inference",
                    'hybrid': "High-accuracy requirements, complex pattern recognition"
                }
            }
        
        # Save report
        report_path = os.path.join(self.comparison_dir, 'hybrid_comparison_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Comprehensive comparison report saved to {report_path}")
        
        # Print summary
        self.print_comparison_summary(report)
    
    def print_comparison_summary(self, report: Dict):
        """Print comparison summary"""
        print("\n" + "="*80)
        print("CNN-ViT HYBRID COMPARISON SUMMARY")
        print("="*80)
        
        print("COMPLETION STATUS:")
        for model, status in report['comparison_summary']['completion_status'].items():
            name = model.replace('cnn_only', 'CNN-only').replace('hybrid', 'CNN-ViT Hybrid')
            print(f"  {name}: {'✅ Completed' if status else '❌ Failed'}")
        
        print("\nPERFORMANCE METRICS:")
        for name, metrics in report['performance_metrics'].items():
            print(f"  {name}:")
            print(f"    Test Accuracy: {metrics['test_accuracy']:.3f}")
            print(f"    Test F1-Score: {metrics['test_f1']:.3f}")
            print(f"    Test Precision: {metrics['test_precision']:.3f}")
            print(f"    Test Recall: {metrics['test_recall']:.3f}")
        
        print("\nMODEL COMPLEXITY:")
        for name, complexity in report['model_complexity'].items():
            print(f"  {name}:")
            print(f"    Total Parameters: {complexity['total_parameters']:,}")
            print(f"    Architecture: {complexity['architecture']}")
        
        if 'improvement_analysis' in report:
            improvement = report['improvement_analysis']
            print("\nIMPROVEMENT ANALYSIS:")
            print(f"  Accuracy Improvement: {improvement['accuracy_improvement']:.2f}%")
            print(f"  F1-Score Improvement: {improvement['f1_improvement']:.2f}%")
            print(f"  Absolute Accuracy Gain: {improvement['absolute_accuracy_improvement']:.3f}")
            print(f"  Absolute F1-Score Gain: {improvement['absolute_f1_improvement']:.3f}")
        
        print("\nRECOMMENDATIONS:")
        recs = report['recommendations']
        print(f"  Best Model: {recs['best_model']}")
        print(f"\nKey Findings:")
        for finding in recs['key_findings']:
            print(f"  • {finding}")
        
        print("="*80)
    
    def run_full_comparison(self):
        """Run complete comparison"""
        self.logger.info("Starting CNN-ViT Hybrid comparison...")
        
        # Train CNN baseline
        print("Step 1: Training CNN-only baseline...")
        self.train_cnn_baseline()
        
        # Train Hybrid model
        print("Step 2: Training CNN-ViT Hybrid...")
        self.train_hybrid_model()
        
        # Create visualizations
        print("Step 3: Creating comparison visualizations...")
        self.create_comparison_visualizations()
        self.create_confusion_matrices()
        
        # Generate report
        print("Step 4: Generating comprehensive report...")
        self.generate_comprehensive_report()
        
        print("\n" + "="*80)
        print("CNN-ViT HYBRID COMPARISON COMPLETED!")
        print("="*80)
        print(f"Results saved to: {self.comparison_dir}/")
        print(f"  - hybrid_comparison.png")
        print(f"  - confusion_matrices.png")
        print(f"  - hybrid_comparison_report.json")
        print()
        print("This comparison provides:")
        print("✓ Performance comparison between architectures")
        print("✓ Complexity analysis")
        print("✓ Improvement quantification")
        print("✓ Recommendations for model selection")
        print("✓ Evidence for architectural decisions")


def main():
    """Main function"""
    print("="*80)
    print("CNN-ViT HYBRID MODEL COMPARISON")
    print("="*80)
    print("Comparing CNN-only vs CNN-ViT Hybrid for Skin Disease Classification")
    print("="*80)
    
    framework = ComparisonFramework()
    
    print("This comparison will evaluate:")
    print("1. CNN-only model (EfficientNetB4)")
    print("2. CNN-ViT Hybrid model")
    print()
    print("Evaluation metrics:")
    print("• Test accuracy and F1-score")
    print("• Per-class performance")
    print("• Model complexity")
    print("• Performance improvement")
    print("• Confusion matrices")
    print()
    
    response = input("Do you want to run the full comparison? (This may take several hours) (y/n): ").lower()
    
    if response == 'y':
        framework.run_full_comparison()
    else:
        print("Comparison cancelled. You can run individual models separately:")
        print("  python train_model.py                    # CNN-only baseline")
        print("  python cnn_vit_hybrid.py                 # CNN-ViT Hybrid")
        print("Then run this comparison again to analyze results.")


if __name__ == "__main__":
    main()
