#!/usr/bin/env python3
"""
Evaluate Model After Training - Complete Integration

This script demonstrates how to evaluate your model immediately after training
with comprehensive metrics and visualizations.

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
    classification_report, confusion_matrix, roc_auc_score
)
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_train import SimpleSkinClassifier, SimpleSkinDataset
import torchvision.transforms as transforms
from model_evaluation import ModelEvaluator

def train_and_evaluate():
    """
    Complete pipeline: Train model and evaluate immediately
    """
    print("🚀 TRAINING AND EVALUATION PIPELINE")
    print("=" * 50)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = SimpleSkinDataset('data/train', transform=transform)
    val_dataset = SimpleSkinDataset('data/val', transform=transform)
    test_dataset = SimpleSkinDataset('data/test', transform=transform)
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Create model
    model = SimpleSkinClassifier(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print("\n🏋️  TRAINING MODEL...")
    num_epochs = 5  # Reduced for demo
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        print(f"Epoch {epoch+1}/{num_epochs}: Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'models/demo_model.pth')
    print("✅ Model saved to models/demo_model.pth")
    
    # Evaluation
    print("\n📊 EVALUATING MODEL...")
    
    # Create evaluator
    class_names = ['acne', 'psoriasis', 'eczema']
    evaluator = ModelEvaluator(model, device, class_names)
    
    # Run complete evaluation
    results = evaluator.evaluate_complete(test_loader, save_dir="demo_evaluation")
    
    return results

def quick_evaluation_example():
    """
    Quick evaluation example with existing model
    """
    print("⚡ QUICK EVALUATION EXAMPLE")
    print("=" * 30)
    
    # Check if model exists
    model_path = "models/simple_best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first or run train_and_evaluate()")
        return
    
    # Load model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['acne', 'psoriasis', 'eczema']
    
    # Load model
    model = SimpleSkinClassifier(num_classes=3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load test data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = SimpleSkinDataset('data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Evaluate
    evaluator = ModelEvaluator(model, device, class_names)
    results = evaluator.evaluate_complete(test_loader, save_dir="quick_evaluation")
    
    # Print summary
    print("\n🎯 QUICK RESULTS SUMMARY:")
    print(f"   Accuracy: {results['basic_metrics']['accuracy']:.4f}")
    print(f"   Precision: {results['basic_metrics']['precision_weighted']:.4f}")
    print(f"   Recall: {results['basic_metrics']['recall_weighted']:.4f}")
    print(f"   F1-Score: {results['basic_metrics']['f1_weighted']:.4f}")
    
    return results

def demonstrate_metrics_calculation():
    """
    Demonstrate how metrics are calculated manually
    """
    print("\n🔬 METRICS CALCULATION DEMONSTRATION")
    print("=" * 40)
    
    # Create sample data
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])  # True labels
    y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2, 0])  # Predicted labels
    class_names = ['acne', 'psoriasis', 'eczema']
    
    print("Sample data:")
    print(f"True labels:      {y_true}")
    print(f"Predicted labels: {y_pred}")
    
    # Calculate metrics manually
    print("\n📊 MANUAL METRICS CALCULATION:")
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"  Formula: (Correct predictions) / (Total predictions)")
    print(f"  Calculation: {np.sum(y_true == y_pred)} / {len(y_true)} = {accuracy:.4f}")
    
    # Precision (weighted)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\nPrecision (weighted): {precision:.4f}")
    print(f"  Formula: Weighted average of per-class precision")
    print(f"  Precision per class: {precision_score(y_true, y_pred, average=None, zero_division=0)}")
    
    # Recall (weighted)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\nRecall (weighted): {recall:.4f}")
    print(f"  Formula: Weighted average of per-class recall")
    print(f"  Recall per class: {recall_score(y_true, y_pred, average=None, zero_division=0)}")
    
    # F1-Score (weighted)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\nF1-Score (weighted): {f1:.4f}")
    print(f"  Formula: 2 * (Precision * Recall) / (Precision + Recall)")
    print(f"  F1 per class: {f1_score(y_true, y_pred, average=None, zero_division=0)}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    print("  Rows: True labels")
    print("  Columns: Predicted labels")
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Sample Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('sample_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("💾 Sample confusion matrix saved as 'sample_confusion_matrix.png'")

def show_evaluation_usage_examples():
    """
    Show different ways to use the evaluation system
    """
    print("\n📚 EVALUATION USAGE EXAMPLES")
    print("=" * 30)
    
    print("\n1. Complete Evaluation (Recommended):")
    print("   evaluator = ModelEvaluator(model, device, class_names)")
    print("   results = evaluator.evaluate_complete(test_loader)")
    print("   # Generates: All metrics + 3 visualizations + detailed report")
    
    print("\n2. Basic Metrics Only:")
    print("   y_pred, y_true, y_prob = evaluator.predict_dataset(test_loader)")
    print("   metrics = evaluator.calculate_basic_metrics(y_true, y_pred)")
    print("   # Returns: accuracy, precision, recall, f1-score")
    
    print("\n3. Confusion Matrix Only:")
    print("   fig = evaluator.plot_confusion_matrix(y_true, y_pred, save_path='cm.png')")
    print("   # Saves normalized confusion matrix")
    
    print("\n4. Classification Report:")
    print("   print(classification_report(y_true, y_pred, target_names=class_names))")
    print("   # Detailed per-class metrics")
    
    print("\n5. ROC Curves:")
    print("   fig = evaluator.plot_roc_curves(y_true, y_prob, save_path='roc.png')")
    print("   # Multi-class ROC curves with AUC")

if __name__ == "__main__":
    print("🏥 SKIN DISEASE CLASSIFICATION - EVALUATION DEMONSTRATIONS")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists('data/test'):
        print("❌ Test data not found. Creating sample data...")
        os.system('python create_sample_images.py')
    
    # Option 1: Train and evaluate
    print("\n🔄 Option 1: Train new model and evaluate")
    train_and_evaluate()
    
    # Option 2: Quick evaluation with existing model
    print("\n⚡ Option 2: Quick evaluation with existing model")
    quick_evaluation_example()
    
    # Option 3: Demonstrate metrics calculation
    print("\n🔬 Option 3: Metrics calculation demonstration")
    demonstrate_metrics_calculation()
    
    # Option 4: Show usage examples
    print("\n📚 Option 4: Usage examples")
    show_evaluation_usage_examples()
    
    print("\n🎉 All evaluation demonstrations completed!")
    print("📁 Check the following directories for results:")
    print("   - demo_evaluation/")
    print("   - quick_evaluation/")
    print("   - sample_confusion_matrix.png")
