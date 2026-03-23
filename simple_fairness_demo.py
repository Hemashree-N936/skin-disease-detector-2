#!/usr/bin/env python3
"""
Simple Fairness Evaluation Demo

This script demonstrates basic fairness evaluation for skin disease classification
with simulated skin tone groups and bias detection.

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import warnings
warnings.filterwarnings('ignore')

class SimpleFairnessEvaluator:
    """
    Very simple fairness evaluator for student projects
    """
    
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model.eval()
    
    def create_skin_tone_groups(self, dataset):
        """
        Create simulated skin tone groups (light, medium, dark)
        """
        print("🎨 Creating skin tone groups...")
        
        # Get all indices
        all_indices = list(range(len(dataset)))
        random.shuffle(all_indices)
        
        # Split into 3 groups (simulating different skin tones)
        n_total = len(all_indices)
        n_light = n_total // 3
        n_medium = n_total // 3
        n_dark = n_total - n_light - n_medium
        
        groups = {
            'light_skin': all_indices[:n_light],
            'medium_skin': all_indices[n_light:n_light + n_medium],
            'dark_skin': all_indices[n_light + n_medium:]
        }
        
        print(f"📊 Group sizes:")
        for group, indices in groups.items():
            print(f"   {group}: {len(indices)} samples")
        
        return groups
    
    def evaluate_group(self, dataset, group_indices, group_name):
        """
        Evaluate model on a specific group
        """
        print(f"🔍 Evaluating {group_name}...")
        
        # Create subset and dataloader
        group_subset = Subset(dataset, group_indices)
        group_loader = DataLoader(group_subset, batch_size=16, shuffle=False)
        
        # Get predictions
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in group_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   F1-Score: {f1:.3f}")
        
        return {
            'group': group_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_samples': len(all_labels)
        }
    
    def detect_bias(self, group_results):
        """
        Simple bias detection
        """
        print("\n🔍 Detecting bias...")
        
        # Get accuracies
        accuracies = {result['group']: result['accuracy'] for result in group_results}
        
        # Calculate performance gap
        max_acc = max(accuracies.values())
        min_acc = min(accuracies.values())
        accuracy_gap = max_acc - min_acc
        
        # Bias threshold (5% difference)
        bias_threshold = 0.05
        has_bias = accuracy_gap > bias_threshold
        
        print(f"📊 Accuracy Gap: {accuracy_gap:.3f} ({accuracy_gap*100:.1f}%)")
        print(f"📊 Bias Threshold: {bias_threshold:.3f} ({bias_threshold*100:.1f}%)")
        print(f"🔍 Bias Detected: {'YES ⚠️' if has_bias else 'NO ✅'}")
        
        if has_bias:
            worst_group = min(accuracies, key=accuracies.get)
            best_group = max(accuracies, key=accuracies.get)
            print(f"⚠️  Model performs worse on {worst_group}")
            print(f"✅ Model performs best on {best_group}")
        
        return {
            'has_bias': has_bias,
            'accuracy_gap': accuracy_gap,
            'max_accuracy': max_acc,
            'min_accuracy': min_acc
        }
    
    def print_comparison_table(self, group_results, bias_results):
        """
        Print simple comparison table
        """
        print("\n📊 FAIRNESS EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"{'Group':<15} {'Samples':<8} {'Accuracy':<10} {'F1-Score':<10}")
        print("-" * 60)
        
        for result in group_results:
            print(f"{result['group']:<15} {result['num_samples']:<8} "
                  f"{result['accuracy']:<10.3f} {result['f1']:<10.3f}")
        
        print("-" * 60)
        print(f"\n🔍 BIAS ANALYSIS:")
        print(f"   Accuracy Gap: {bias_results['accuracy_gap']:.3f}")
        print(f"   Bias Detected: {'YES ⚠️' if bias_results['has_bias'] else 'NO ✅'}")
        
        if bias_results['has_bias']:
            print(f"\n⚠️  RECOMMENDATION:")
            print(f"   Consider collecting more diverse data")
            print(f"   Or retraining with balanced samples")
        else:
            print(f"\n✅ GOOD NEWS:")
            print(f"   Model performs consistently across groups")
    
    def plot_simple_comparison(self, group_results, save_path=None):
        """
        Create simple comparison plot
        """
        print("\n📈 Creating fairness comparison plot...")
        
        groups = [result['group'] for result in group_results]
        accuracies = [result['accuracy'] for result in group_results]
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        bars = plt.bar(groups, accuracies, color=colors)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Fairness Evaluation - Accuracy by Skin Tone Group', fontweight='bold', fontsize=14)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Skin Tone Group', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Add bias threshold line
        bias_threshold = 0.05
        max_acc = max(accuracies)
        min_acc = min(accuracies)
        if (max_acc - min_acc) > bias_threshold:
            plt.axhline(y=min_acc + bias_threshold, color='red', linestyle='--', 
                       alpha=0.7, label=f'Bias Threshold ({bias_threshold:.2f})')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to: {save_path}")
        
        plt.show()
    
    def evaluate_fairness(self, dataset, save_dir="fairness_results"):
        """
        Complete fairness evaluation
        """
        print("🚀 Starting Fairness Evaluation...")
        print("=" * 50)
        
        # Step 1: Create groups
        groups = self.create_skin_tone_groups(dataset)
        
        # Step 2: Evaluate each group
        group_results = []
        for group_name, group_indices in groups.items():
            result = self.evaluate_group(dataset, group_indices, group_name)
            group_results.append(result)
        
        # Step 3: Detect bias
        bias_results = self.detect_bias(group_results)
        
        # Step 4: Print results
        self.print_comparison_table(group_results, bias_results)
        
        # Step 5: Create visualization
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        self.plot_simple_comparison(
            group_results, 
            save_path=os.path.join(save_dir, "fairness_comparison.png")
        )
        
        print(f"\n🎉 Fairness evaluation completed!")
        print(f"📁 Results saved to: {save_dir}")
        
        return {
            'group_results': group_results,
            'bias_results': bias_results,
            'save_directory': save_dir
        }

def demo_fairness_evaluation():
    """
    Demonstrate fairness evaluation with sample data
    """
    print("🏥 SKIN DISEASE CLASSIFICATION - FAIRNESS DEMO")
    print("=" * 50)
    
    # Create a simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(32, num_classes)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # Create model and dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel(num_classes=3).to(device)
    
    # Create synthetic dataset
    class SimpleDataset:
        def __init__(self, num_samples=300):
            self.num_samples = num_samples
            self.data = [self.create_sample() for _ in range(num_samples)]
        
        def create_sample(self):
            # Create random image and label
            image = torch.randn(3, 224, 224)
            label = random.randint(0, 2)
            return image, label
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = SimpleDataset(num_samples=300)
    
    # Create evaluator
    class_names = ['acne', 'psoriasis', 'eczema']
    evaluator = SimpleFairnessEvaluator(model, device, class_names)
    
    # Run fairness evaluation
    results = evaluator.evaluate_fairness(dataset, save_dir="demo_fairness_results")
    
    return results

def show_integration_example():
    """
    Show how to integrate with existing model
    """
    print("\n📚 INTEGRATION EXAMPLE")
    print("=" * 30)
    
    print("\n1. Load your trained model:")
    print("   model = load_your_model('models/your_model.pth')")
    
    print("\n2. Load your test dataset:")
    print("   test_dataset = load_test_dataset()")
    
    print("\n3. Create fairness evaluator:")
    print("   evaluator = SimpleFairnessEvaluator(model, device, class_names)")
    
    print("\n4. Run fairness evaluation:")
    print("   results = evaluator.evaluate_fairness(test_dataset)")
    
    print("\n5. Check results:")
    print("   if results['bias_results']['has_bias']:")
    print("       print('⚠️ Bias detected!')")
    print("   else:")
    print("       print('✅ No significant bias')")

if __name__ == "__main__":
    # Run demonstration
    demo_fairness_evaluation()
    
    # Show integration example
    show_integration_example()
    
    print("\n🎉 Simple fairness evaluation demo completed!")
    print("📚 This is perfect for student projects!")
    print("🔧 Easy to integrate with existing models")
