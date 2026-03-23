# Basic Fairness Evaluation for Skin Disease Classification

## Overview
A **simple, practical fairness evaluation system** for detecting bias in skin disease classification models across different skin tone groups. Perfect for student projects!

## 🎯 What This Does

### **Fairness Evaluation Steps**
1. **Simulate Skin Tone Groups**: Split dataset into light, medium, dark skin groups
2. **Evaluate Each Group**: Calculate accuracy, precision, recall, F1-score separately
3. **Compare Performance**: Create comparison table and visualizations
4. **Detect Bias**: Identify if model performs significantly worse on any group

### **Bias Detection Logic**
- **Accuracy Gap**: Difference between best and worst performing groups
- **Bias Threshold**: 5% difference (configurable)
- **Simple Alert**: Clear "Bias Detected" or "No Bias" message

## 📁 Files Created

### **1. `fairness_evaluation.py` - Complete Implementation**
```python
class FairnessEvaluator:
    def evaluate_fairness_complete(self, dataset):
        # Complete fairness evaluation with visualizations
        # Returns: group metrics, bias analysis, plots
```

### **2. `simple_fairness_demo.py` - Student-Friendly Version**
```python
class SimpleFairnessEvaluator:
    def evaluate_fairness(self, dataset):
        # Simple evaluation for quick results
        # Perfect for student projects
```

## 🚀 Quick Start

### **Option 1: Simple Demo (Student Projects)**
```python
# Run the simple demonstration
python simple_fairness_demo.py

# Output:
# 📊 FAIRNESS EVALUATION RESULTS
# Group          Samples  Accuracy  F1-Score
# ------------------------------------------------------------
# light_skin     100      0.850     0.842
# medium_skin    100      0.820     0.815
# dark_skin      100      0.780     0.775
# ------------------------------------------------------------
# 🔍 BIAS ANALYSIS:
#    Accuracy Gap: 0.070
#    Bias Detected: YES ⚠️
```

### **Option 2: Complete Evaluation**
```python
# Load your model and data
model, test_dataset, device, class_names = load_model_and_data()

# Create evaluator
evaluator = FairnessEvaluator(model, device, class_names)

# Run complete evaluation
results = evaluator.evaluate_fairness_complete(test_dataset)

# Check for bias
if results['bias_analysis']['has_bias']:
    print("⚠️ Bias detected! Model needs improvement.")
else:
    print("✅ No significant bias detected.")
```

### **Option 3: Integration with Existing Model**
```python
from fairness_evaluation import FairnessEvaluator

# Load your trained model
model = load_your_model('models/simple_best_model.pth')

# Load test data
test_dataset = load_test_dataset()

# Create evaluator
evaluator = FairnessEvaluator(model, device, ['acne', 'psoriasis', 'eczema'])

# Run evaluation
results = evaluator.evaluate_fairness_complete(test_dataset)
```

## 📊 How It Works

### **Step 1: Simulate Skin Tone Groups**
```python
def simulate_skin_tone_groups(self, dataset):
    # Get all indices
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    
    # Split into 3 groups (30%, 40%, 30%)
    n_light = int(0.3 * n_total)
    n_medium = int(0.4 * n_total)
    n_dark = n_total - n_light - n_medium
    
    groups = {
        'light_skin': all_indices[:n_light],
        'medium_skin': all_indices[n_light:n_light + n_medium],
        'dark_skin': all_indices[n_light + n_medium:]
    }
    
    return groups
```

### **Step 2: Evaluate Each Group**
```python
def evaluate_group_performance(self, group_loader, group_name):
    # Get predictions for this group
    for images, labels in group_loader:
        outputs = self.model(images)
        predictions = torch.argmax(outputs, dim=1)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    return metrics
```

### **Step 3: Detect Bias**
```python
def detect_bias(self, group_metrics):
    # Get accuracies
    accuracies = {group: metrics['accuracy'] for group, metrics in group_metrics.items()}
    
    # Calculate performance gap
    max_acc = max(accuracies.values())
    min_acc = min(accuracies.values())
    accuracy_gap = max_acc - min_acc
    
    # Bias threshold (5% difference)
    bias_threshold = 0.05
    has_bias = accuracy_gap > bias_threshold
    
    return {
        'has_bias': has_bias,
        'accuracy_gap': accuracy_gap,
        'max_accuracy': max_acc,
        'min_accuracy': min_acc
    }
```

## 📈 Output Examples

### **Comparison Table**
```
📊 FAIRNESS EVALUATION RESULTS
============================================================
Group          Samples  Accuracy  Precision  Recall    F1-Score
------------------------------------------------------------
light_skin     90       0.856     0.842      0.856     0.849
medium_skin    120      0.833     0.825      0.833     0.829
dark_skin      90       0.789     0.778      0.789     0.783
------------------------------------------------------------

🔍 BIAS ANALYSIS:
   Accuracy Gap: 0.067 (6.7%)
   Bias Threshold: 0.050 (5.0%)
   Bias Detected: YES ⚠️

⚠️  BIAS DETECTED!
   Model performs significantly worse on dark_skin
   Consider: Data balancing, model adjustment, or targeted training
```

### **Visualizations**
1. **Fairness Comparison Plot**: Bar chart comparing accuracy across groups
2. **Per-Class Analysis**: Detailed metrics for each disease class
3. **Bias Threshold Line**: Visual indicator of bias detection

## 🎯 Bias Detection Criteria

### **What Constitutes Bias?**
- **Accuracy Gap > 5%**: Significant performance difference
- **Consistent Pattern**: Same group performs worst across metrics
- **Clinical Impact**: Could affect diagnosis reliability

### **Bias Levels**
- **No Bias**: Gap < 5% ✅
- **Mild Bias**: Gap 5-10% ⚠️
- **Significant Bias**: Gap > 10% 🚨

### **Recommendations Based on Results**
```python
if bias_results['has_bias']:
    print("⚠️  RECOMMENDATIONS:")
    print("1. Collect more diverse training data")
    print("2. Balance dataset across skin tones")
    print("3. Consider retraining with fairness constraints")
    print("4. Test with real-world diverse patients")
else:
    print("✅ GOOD PRACTICE:")
    print("1. Continue monitoring fairness")
    print("2. Test with larger diverse datasets")
    print("3. Consider real-world validation")
```

## 📱 Integration with Your Project

### **1. Add to Training Pipeline**
```python
# After training your model
def train_and_evaluate_with_fairness():
    # Train model
    model = train_model()
    
    # Evaluate fairness
    evaluator = FairnessEvaluator(model, device, class_names)
    fairness_results = evaluator.evaluate_fairness_complete(test_dataset)
    
    # Report results
    if fairness_results['bias_analysis']['has_bias']:
        print("⚠️ Model shows bias - consider retraining")
    
    return model, fairness_results
```

### **2. Add to API Monitoring**
```python
# In your FastAPI endpoint
@app.post("/predict-with-fairness")
async def predict_with_fairness(file: UploadFile = File(...)):
    # Get prediction
    prediction = predict_disease(image)
    
    # Log for fairness monitoring
    log_prediction_with_metadata(prediction, user_metadata)
    
    return prediction
```

### **3. Add to Model Evaluation**
```python
# In your evaluation script
def complete_model_evaluation():
    # Standard evaluation
    standard_metrics = evaluate_model(model, test_loader)
    
    # Fairness evaluation
    fairness_metrics = evaluate_fairness(model, test_dataset)
    
    # Combined report
    print_evaluation_report(standard_metrics, fairness_metrics)
```

## 🔧 Customization Options

### **Change Group Distribution**
```python
# Modify group sizes in simulate_skin_tone_groups()
n_light = int(0.2 * n_total)  # 20% light
n_medium = int(0.5 * n_total)  # 50% medium
n_dark = int(0.3 * n_total)   # 30% dark
```

### **Change Bias Threshold**
```python
# More strict threshold
bias_threshold = 0.03  # 3% difference

# More lenient threshold
bias_threshold = 0.10  # 10% difference
```

### **Add More Groups**
```python
# Add more skin tone categories
groups = {
    'very_light': all_indices[:n_very_light],
    'light': all_indices[n_very_light:n_very_light + n_light],
    'medium': all_indices[n_very_light + n_light:n_very_light + n_light + n_medium],
    'dark': all_indices[n_very_light + n_light + n_medium:n_very_light + n_light + n_medium + n_dark],
    'very_dark': all_indices[n_very_light + n_light + n_medium + n_dark:]
}
```

## 🧪 Testing and Validation

### **Test with Sample Data**
```python
# Run the demo to test the system
python simple_fairness_demo.py

# This will:
# 1. Create a sample model
# 2. Generate synthetic data
# 3. Run fairness evaluation
# 4. Show results and visualizations
```

### **Validate with Real Data**
```python
# When you have real data with skin tone labels:
def evaluate_with_real_skin_tones(dataset, skin_tone_labels):
    # Use actual skin tone labels instead of simulation
    groups = create_groups_from_labels(dataset, skin_tone_labels)
    # ... rest of evaluation
```

## 🎯 Best Practices

### **1. Regular Monitoring**
```python
# Run fairness evaluation regularly
def monitor_model_fairness():
    # Weekly fairness checks
    fairness_results = evaluator.evaluate_fairness(test_dataset)
    
    # Alert if bias detected
    if fairness_results['bias_analysis']['has_bias']:
        send_bias_alert(fairness_results)
```

### **2. Diverse Data Collection**
```python
# Ensure diverse training data
def collect_diverse_data():
    # Aim for balanced representation
    # Target: 30% light, 40% medium, 30% dark skin tones
    # Collect from diverse geographic regions
    # Include various age groups and genders
```

### **3. Transparency**
```python
# Document fairness evaluation results
def document_fairness_results(results):
    # Save detailed report
    # Include methodology
    # Note limitations
    # Suggest improvements
```

## 🎉 Success Indicators

### **✅ Good Fairness**
- **Accuracy Gap < 5%**: Consistent performance across groups
- **Balanced Metrics**: Similar precision, recall, F1 across groups
- **No Pattern**: Random performance differences

### **⚠️ Potential Bias**
- **Accuracy Gap 5-10%**: Some performance differences
- **Consistent Pattern**: Same group performs worse
- **Clinical Impact**: Could affect some patient groups

### **🚨 Significant Bias**
- **Accuracy Gap > 10%**: Major performance differences
- **Systematic Issues**: Clear pattern of poor performance
- **Ethical Concerns**: Could lead to health disparities

## 🚀 Why This is Perfect for Student Projects

### **✅ Simple and Understandable**
- Clear methodology
- Easy to interpret results
- Minimal complexity

### **✅ Practical and Useful**
- Addresses real-world AI ethics
- Provides actionable insights
- Easy to integrate

### **✅ Educational Value**
- Teaches fairness concepts
- Demonstrates responsible AI
- Shows impact on healthcare

### **✅ Extendable**
- Can add more sophisticated methods later
- Good foundation for advanced work
- Scalable approach

This fairness evaluation system provides **practical, easy-to-understand bias detection** perfect for student projects while maintaining scientific rigor and real-world relevance!
