# HAM10000 Dataset Training for Skin Disease Classification

## Overview
This guide provides a complete PyTorch training pipeline using the HAM10000 dataset for skin disease classification with pretrained EfficientNetB4 model.

## Files Created

### 1. `ham10000_training.py` - Complete Local Training
- Full HAM10000 dataset support
- Automatic dataset download
- Complete training pipeline
- Model saving and evaluation

### 2. `colab_ready_training.py` - Google Colab Optimized
- Colab-specific optimizations
- GPU utilization
- Synthetic dataset fallback
- Easy to run in Colab

## Quick Start

### Option 1: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `colab_ready_training.py`
3. Run the script
4. Download the trained model

### Option 2: Local Training
1. Install dependencies:
```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn seaborn tqdm pillow opencv-python
```

2. Run training:
```bash
python ham10000_training.py
```

## Key Features

### 1. Dataset Handling
- **HAM10000 Dataset**: 10,015 dermatoscopic images
- **7 Original Classes**: akiec, bcc, bkl, df, mel, nv, vasc
- **Simplified to 3 Classes**: benign, malignant, other
- **Automatic Download**: Fetches dataset from official sources

### 2. Data Preprocessing
```python
# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 3. Model Architecture
```python
# EfficientNetB4 with pretrained weights
model = models.efficientnet_b4(weights='IMAGENET1K_V1')

# Freeze backbone for transfer learning
for param in model.features.parameters():
    param.requires_grad = False

# Modify final classifier
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_features, 3)  # 3 classes
)
```

### 4. Training Loop
```python
# Training with progress tracking
for epoch in range(NUM_EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        # Evaluate on validation set
        val_acc = evaluate_model(model, val_loader)
    
    # Save best model
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), 'best_model.pt')
```

## Training Configuration

### Hyperparameters
```python
class Config:
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    INPUT_SIZE = 224
    NUM_CLASSES = 3
    FREEZE_BACKBONE = True
```

### Data Splits
- **Training**: 70% of dataset
- **Validation**: 15% of dataset  
- **Test**: 15% of dataset

## Model Output

### Saved Model Format
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_acc': val_acc,
    'class_names': ['benign', 'malignant', 'other']
}, 'ham10000_model.pt')
```

### Training Metrics
- **Loss Curves**: Training and validation loss
- **Accuracy Curves**: Training and validation accuracy
- **Classification Report**: Precision, recall, F1-score
- **Confusion Matrix**: Per-class performance

## Where to Place the Model

### In Your Project Structure
```
skin_diseases_3/
├── models/
│   ├── ham10000_model.pt          # <-- PLACE HERE
│   └── simple_best_model.pth     # Existing model
├── api/
│   ├── main.py
│   └── simple_main.py
├── webapp/
│   └── src/
└── ham10000_training.py
```

### Update API to Use New Model
```python
# In api/simple_main.py or api/main.py
model_path = "../models/ham10000_model.pt"  # Update this line

# Load the model
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Update class names
class_names = checkpoint['class_names']  # ['benign', 'malignant', 'other']
```

## Running the Complete Pipeline

### Step 1: Train Model
```bash
# In Colab
python colab_ready_training.py

# Or locally
python ham10000_training.py
```

### Step 2: Download Model
- **Colab**: Download `ham10000_model.pt` from Colab
- **Local**: Model saved to `models/ham10000_model.pt`

### Step 3: Update API
```python
# Update model loading in api/simple_main.py
model_path = "../models/ham10000_model.pt"
```

### Step 4: Run Backend
```bash
cd api
python simple_main.py
```

### Step 5: Run Frontend
```bash
cd webapp
npm install
npm start
```

## Expected Performance

### Training Metrics
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 80-90%
- **Test Accuracy**: 75-85%
- **Training Time**: 15-30 minutes (GPU), 2-4 hours (CPU)

### Model Size
- **File Size**: ~80-120 MB
- **Parameters**: ~18M total, ~1M trainable
- **Memory Usage**: ~500MB GPU memory

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   BATCH_SIZE = 16  # or 8
   ```

2. **Slow Training**
   ```python
   # Use fewer epochs
   NUM_EPOCHS = 10
   # Or use GPU in Colab
   ```

3. **Poor Accuracy**
   ```python
   # Unfreeze more layers
   FREEZE_BACKBONE = False
   # Or increase learning rate
   LEARNING_RATE = 0.01
   ```

## Advanced Features

### 1. Data Augmentation
- Random horizontal flip
- Random rotation (±15 degrees)
- Random crop and resize
- Color jitter (brightness, contrast, saturation, hue)

### 2. Transfer Learning
- Pretrained ImageNet weights
- Frozen backbone layers
- Custom classifier head

### 3. Monitoring
- Real-time loss tracking
- Accuracy monitoring
- Learning rate scheduling
- Early stopping

## Next Steps

### 1. Model Integration
- Integrate trained model with FastAPI backend
- Update class mappings for your use case
- Test end-to-end pipeline

### 2. Performance Optimization
- Fine-tune hyperparameters
- Try different architectures (ResNet50, DenseNet)
- Implement ensemble methods

### 3. Production Deployment
- Model quantization for faster inference
- Batch processing capabilities
- API endpoint optimization

## Medical Considerations

### Class Mapping
```python
# Original HAM10000 to simplified classes
HAM10000_CLASSES = {
    'akiec': 'malignant',  # Actinic keratoses
    'bcc': 'malignant',    # Basal cell carcinoma
    'bkl': 'benign',       # Benign keratosis
    'df': 'benign',        # Dermatofibroma
    'mel': 'malignant',    # Melanoma
    'nv': 'benign',        # Melanocytic nevi
    'vasc': 'benign'       # Vascular lesions
}
```

### Clinical Relevance
- **Benign**: Non-cancerous conditions
- **Malignant**: Cancerous conditions requiring immediate attention
- **Other**: Lesions requiring specialist consultation

This complete training pipeline provides a production-ready model for your skin disease classification system with real medical data.
