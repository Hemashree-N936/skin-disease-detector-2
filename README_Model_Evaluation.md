# Model Evaluation for Skin Disease Classification

## Overview
Comprehensive evaluation system for skin disease classification models with all essential metrics: accuracy, precision, recall, F1-score, confusion matrix, ROC curves, and detailed visualizations.

## 📊 Evaluation Metrics Included

### **1. Basic Classification Metrics**
- **Accuracy**: Overall prediction accuracy
- **Precision**: Weighted average precision per class
- **Recall**: Weighted average recall per class  
- **F1-Score**: Weighted average F1-score per class
- **Per-class metrics**: Individual precision, recall, F1 for each disease

### **2. Advanced Metrics**
- **AUC-ROC**: Area under ROC curve for each class
- **Macro AUC**: Average AUC across all classes
- **Confusion Matrix**: Normalized and raw counts
- **Classification Report**: Detailed sklearn classification report

### **3. Visualizations**
- **Confusion Matrix**: Heatmap with normalized values
- **Classification Report**: Visual heatmap of metrics
- **ROC Curves**: Multi-class ROC curves with AUC scores

## 📁 Files Created

### **1. `model_evaluation.py` - Core Evaluation System**
```python
class ModelEvaluator:
    def __init__(self, model, device, class_names):
        # Initialize evaluator
    
    def evaluate_complete(self, test_loader, save_dir):
        # Complete evaluation with all metrics
        # Returns: metrics dict + saves visualizations
```

### **2. `evaluate_after_training.py` - Integration Examples**
```python
def train_and_evaluate():
    # Train model then evaluate immediately
    
def quick_evaluation_example():
    # Quick evaluation with existing model
```

## 🚀 Quick Start

### **Option 1: Complete Evaluation**
```python
# Load model and data
model, test_loader, device, class_names = load_model_and_data("models/simple_best_model.pth")

# Create evaluator
evaluator = ModelEvaluator(model, device, class_names)

# Run complete evaluation
results = evaluator.evaluate_complete(test_loader, save_dir="evaluation_results")
```

### **Option 2: Quick Evaluation**
```bash
python evaluate_after_training.py
```

### **Option 3: Manual Evaluation**
```python
# Get predictions
y_pred, y_true, y_prob = evaluator.predict_dataset(test_loader)

# Calculate specific metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

## 📊 Output Examples

### **1. Detailed Report**
```
🏥 SKIN DISEASE CLASSIFICATION - DETAILED EVALUATION REPORT
============================================================

📊 OVERALL METRICS:
   Accuracy:  0.8333
   Precision: 0.8333
   Recall:    0.8333
   F1-Score:  0.8333

📋 PER-CLASS METRICS:
Class        Precision  Recall     F1-Score  
---------------------------------------------
acne         0.8000     0.8000     0.8000    
psoriasis    0.8000     0.8000     0.8000    
eczema       0.9000     0.9000     0.9000    

📈 AUC-ROC SCORES:
   acne: 0.9500
   psoriasis: 0.9500
   eczema: 0.9500
   Macro AUC: 0.9500

📋 DETAILED CLASSIFICATION REPORT:
              precision    recall  f1-score   support

        acne       0.80      0.80      0.80         5
    psoriasis       0.80      0.80      0.80         5
       eczema       0.90      0.90      0.90        10

    accuracy                           0.83        20
   macro avg       0.83      0.83      0.83        20
weighted avg       0.83      0.83      0.83        20
```

### **2. Confusion Matrix**
![Normalized confusion matrix showing per-class performance](evaluation_results/confusion_matrix.png)

### **3. ROC Curves**
![Multi-class ROC curves with AUC scores](evaluation_results/roc_curves.png)

## 🔧 Integration with Training

### **After Training**
```python
# In your training script
def train_model():
    # ... training code ...
    
    # After training completes
    print("📊 Evaluating trained model...")
    
    # Load test data
    test_loader = create_test_loader()
    
    # Evaluate
    evaluator = ModelEvaluator(model, device, class_names)
    results = evaluator.evaluate_complete(test_loader, save_dir="final_evaluation")
    
    print(f"Final Accuracy: {results['basic_metrics']['accuracy']:.4f}")
    return results
```

### **In Training Loop**
```python
# Evaluate after each epoch
for epoch in range(num_epochs):
    # ... training code ...
    
    # Validation evaluation
    if epoch % 5 == 0:  # Every 5 epochs
        val_results = evaluator.evaluate_complete(val_loader, save_dir=f"val_epoch_{epoch}")
        print(f"Epoch {epoch} Val Accuracy: {val_results['basic_metrics']['accuracy']:.4f}")
```

## 📈 Metrics Explained

### **1. Accuracy**
```python
accuracy = correct_predictions / total_predictions
# Range: 0-1, Higher is better
# Interpretation: Overall correctness of predictions
```

### **2. Precision**
```python
precision = true_positives / (true_positives + false_positives)
# Range: 0-1, Higher is better
# Interpretation: Of all positive predictions, how many were correct
```

### **3. Recall**
```python
recall = true_positives / (true_positives + false_negatives)
# Range: 0-1, Higher is better
# Interpretation: Of all actual positives, how many were correctly predicted
```

### **4. F1-Score**
```python
f1_score = 2 * (precision * recall) / (precision + recall)
# Range: 0-1, Higher is better
# Interpretation: Harmonic mean of precision and recall
```

### **5. Confusion Matrix**
```python
# Rows: True labels
# Columns: Predicted labels
# Diagonal: Correct predictions
# Off-diagonal: Misclassifications
```

## 🎯 Usage Examples

### **1. Basic Evaluation**
```python
# Load model
model = load_trained_model("models/simple_best_model.pth")

# Create evaluator
evaluator = ModelEvaluator(model, device, ['acne', 'psoriasis', 'eczema'])

# Get predictions
y_pred, y_true, y_prob = evaluator.predict_dataset(test_loader)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

### **2. Complete Evaluation**
```python
# Run complete evaluation
results = evaluator.evaluate_complete(test_loader, save_dir="results")

# Access results
accuracy = results['basic_metrics']['accuracy']
precision = results['basic_metrics']['precision_weighted']
recall = results['basic_metrics']['recall_weighted']
f1 = results['basic_metrics']['f1_weighted']

print(f"Results saved to: {results['save_directory']}")
```

### **3. Visualizations Only**
```python
# Get predictions first
y_pred, y_true, y_prob = evaluator.predict_dataset(test_loader)

# Plot confusion matrix
evaluator.plot_confusion_matrix(y_true, y_pred, save_path="cm.png")

# Plot ROC curves
evaluator.plot_roc_curves(y_true, y_prob, save_path="roc.png")

# Plot classification report
evaluator.plot_classification_report(y_true, y_pred, save_path="report.png")
```

## 📁 Output Structure

### **Evaluation Results Directory**
```
evaluation_results/
├── confusion_matrix.png          # Normalized confusion matrix
├── classification_report.png     # Metrics heatmap
├── roc_curves.png               # ROC curves with AUC
├── evaluation_results.json     # All metrics as JSON
└── plots/                       # Additional plots
```

### **JSON Results Format**
```json
{
  "basic_metrics": {
    "accuracy": 0.8333,
    "precision_weighted": 0.8333,
    "recall_weighted": 0.8333,
    "f1_weighted": 0.8333
  },
  "auc_scores": {
    "auc_acne": 0.9500,
    "auc_psoriasis": 0.9500,
    "auc_eczema": 0.9500,
    "auc_macro": 0.9500
  },
  "confusion_matrix": [[4, 1, 0], [1, 4, 0], [0, 1, 9]],
  "num_samples": 20
}
```

## 🧪 Testing and Validation

### **Test Evaluation System**
```bash
# Test with sample data
python evaluate_after_training.py

# This will:
# 1. Train a small model
# 2. Evaluate it completely
# 3. Generate all visualizations
# 4. Show detailed reports
```

### **Validate Metrics**
```python
# Manual verification
y_true = [0, 1, 2, 0, 1]
y_pred = [0, 1, 2, 0, 2]

# Calculate manually
correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
accuracy = correct / len(y_true)

# Compare with sklearn
sklearn_accuracy = accuracy_score(y_true, y_pred)
assert abs(accuracy - sklearn_accuracy) < 1e-6
```

## 🚨 Common Issues

### **1. Data Loading Issues**
```python
# Ensure test data exists
if not os.path.exists('data/test'):
    print("Creating sample test data...")
    create_sample_dataset()
```

### **2. Model Loading Issues**
```python
# Check model path
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    print("Please train the model first")
```

### **3. Memory Issues**
```python
# Use smaller batch size
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Or process in chunks
for batch in test_loader:
    # Process one batch at a time
```

## 🎯 Best Practices

### **1. Always Use Test Set**
```python
# Never evaluate on training data
# Use separate test set for final evaluation
test_loader = create_test_loader()  # Separate from train/val
```

### **2. Save All Results**
```python
# Save both visualizations and raw metrics
results = evaluator.evaluate_complete(test_loader, save_dir="final_results")
```

### **3. Compare Models**
```python
# Evaluate multiple models
for model_name, model in models.items():
    results = evaluator.evaluate_complete(model, test_loader, save_dir=f"results_{model_name}")
    print(f"{model_name} Accuracy: {results['basic_metrics']['accuracy']:.4f}")
```

### **4. Track Progress**
```python
# Log metrics during training
for epoch in range(num_epochs):
    # ... training ...
    val_results = evaluator.evaluate_complete(val_loader, save_dir=f"val_epoch_{epoch}")
    # Log metrics for plotting later
```

## 🎉 Success Indicators

### **✅ Good Performance**
- **Accuracy > 0.8**: Good overall performance
- **F1-Score > 0.8**: Good balance of precision and recall
- **AUC > 0.9**: Excellent discrimination
- **Balanced confusion matrix**: Diagonal dominance

### **✅ Proper Evaluation**
- **Test set only**: No data leakage
- **All metrics calculated**: Complete evaluation
- **Visualizations generated**: Clear interpretation
- **Results saved**: Reproducible results

This evaluation system provides comprehensive, medically-relevant metrics for your skin disease classification model with clear visualizations and detailed reporting.
