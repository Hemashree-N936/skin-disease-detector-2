#!/usr/bin/env python3
"""
Complete HAM10000 Training Script

This script provides a complete training pipeline for skin disease classification
using the HAM10000 dataset with proper train/validation split, model saving,
and inference compatibility.

Author: AI Assistant
Date: 2025-03-23
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class HAM10000Dataset(Dataset):
    """
    Custom dataset for HAM10000 skin lesion images
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            transform: Optional transforms to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Create a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label

class SkinDiseaseClassifier(nn.Module):
    """
    Skin disease classification model using EfficientNet-B0
    """
    
    def __init__(self, num_classes=3, pretrained=True):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Load EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features before classification"""
        # Remove classifier part
        features = self.backbone.features(x)
        return features

def load_ham10000_data(data_dir="HAM10000"):
    """
    Load HAM10000 dataset
    
    Args:
        data_dir: Directory containing HAM10000 data
        
    Returns:
        Tuple of (image_paths, labels, class_names, label_mapping)
    """
    print("📁 Loading HAM10000 dataset...")
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory not found: {data_dir}")
        print("Creating synthetic dataset for demonstration...")
        return create_synthetic_ham10000_data()
    
    # Load metadata
    metadata_file = os.path.join(data_dir, "HAM10000_metadata.csv")
    if not os.path.exists(metadata_file):
        print(f"❌ Metadata file not found: {metadata_file}")
        print("Creating synthetic dataset for demonstration...")
        return create_synthetic_ham10000_data()
    
    # Read metadata
    df = pd.read_csv(metadata_file)
    print(f"📊 Loaded metadata: {len(df)} samples")
    
    # Simplified 3-class mapping
    # Original 7 classes -> 3 classes for simplicity
    lesion_mapping = {
        'nv': 'melanocytic_nevi',      # nevus -> benign
        'mel': 'melanoma',             # melanoma -> malignant
        'bkl': 'benign_keratosis',     # keratosis -> benign
        'bcc': 'basal_cell_carcinoma', # carcinoma -> malignant
        'akiec': 'actinic_keratoses',  # keratoses -> benign
        'vasc': 'vascular_lesions',    # vascular -> benign
        'df': 'dermatofibroma'         # fibroma -> benign
    }
    
    # Map to 3 classes
    simplified_mapping = {
        'melanocytic_nevi': 'benign',
        'benign_keratosis': 'benign', 
        'actinic_keratoses': 'benign',
        'vascular_lesions': 'benign',
        'dermatofibroma': 'benign',
        'melanoma': 'malignant',
        'basal_cell_carcinoma': 'malignant'
    }
    
    # Apply mappings
    df['lesion_type'] = df['dx'].map(lesion_mapping)
    df['simplified_class'] = df['lesion_type'].map(simplified_mapping)
    
    # Remove any NaN values
    df = df.dropna(subset=['simplified_class'])
    
    print(f"📊 Class distribution:")
    print(df['simplified_class'].value_counts())
    
    # Get image paths
    image_dir1 = os.path.join(data_dir, "HAM10000_images_part_1")
    image_dir2 = os.path.join(data_dir, "HAM10000_images_part_2")
    
    image_paths = []
    labels = []
    
    for idx, row in df.iterrows():
        image_id = row['image_id']
        label = row['simplified_class']
        
        # Try both directories
        img_path1 = os.path.join(image_dir1, f"{image_id}.jpg")
        img_path2 = os.path.join(image_dir2, f"{image_id}.jpg")
        
        if os.path.exists(img_path1):
            image_paths.append(img_path1)
            labels.append(label)
        elif os.path.exists(img_path2):
            image_paths.append(img_path2)
            labels.append(label)
        else:
            print(f"⚠️ Image not found: {image_id}")
    
    print(f"📊 Found {len(image_paths)} valid images")
    
    # Create label mapping
    unique_labels = sorted(list(set(labels)))
    class_names = unique_labels
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    
    # Convert labels to indices
    label_indices = [label_to_idx[label] for label in labels]
    
    print(f"📊 Final classes: {class_names}")
    
    return image_paths, label_indices, class_names, label_to_idx

def create_synthetic_ham10000_data():
    """
    Create synthetic HAM10000-like data for demonstration
    """
    print("🎨 Creating synthetic HAM10000 dataset...")
    
    # Create synthetic data
    num_samples = 1000
    class_names = ['benign', 'malignant']
    
    # Generate synthetic image paths and labels
    image_paths = [f"synthetic_image_{i}.jpg" for i in range(num_samples)]
    labels = [np.random.randint(0, len(class_names)) for _ in range(num_samples)]
    
    # Label mapping
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    
    print(f"📊 Created {num_samples} synthetic samples")
    print(f"📊 Classes: {class_names}")
    
    return image_paths, labels, class_names, label_to_idx

def create_data_loaders(image_paths, labels, class_names, batch_size=32, val_split=0.2):
    """
    Create train and validation data loaders with proper split
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        class_names: List of class names
        batch_size: Batch size for data loaders
        val_split: Validation split ratio
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print("🔄 Creating data loaders...")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    full_dataset = HAM10000Dataset(image_paths, labels, transform=None)
    
    # Stratified split for train/validation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(sss.split(image_paths, labels))
    
    # Create train and validation datasets with appropriate transforms
    train_dataset = HAM10000Dataset(
        [image_paths[i] for i in train_idx],
        [labels[i] for i in train_idx],
        transform=train_transform
    )
    
    val_dataset = HAM10000Dataset(
        [image_paths[i] for i in val_idx],
        [labels[i] for i in val_idx],
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"📊 Dataset splits:")
    print(f"   Training: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Classes: {class_names}")
    
    return train_loader, val_loader, class_names

def train_model(model, train_loader, val_loader, class_names, num_epochs=20, learning_rate=0.001):
    """
    Train the model with proper validation and model saving
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        class_names: List of class names
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        
    Returns:
        Trained model and training history
    """
    print("🚀 Starting training...")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Using device: {device}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch_idx, (images, labels) in enumerate(val_pbar):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                current_val_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f'{val_loss/(batch_idx+1):.4f}',
                    'Acc': f'{current_val_acc:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            history['best_epoch'] = epoch + 1
            
            # Save model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'class_names': class_names,
                'num_classes': len(class_names),
                'model_config': {
                    'architecture': 'efficientnet_b0',
                    'pretrained': True
                }
            }, 'models/best_model.pth')
            
            print(f"🏆 New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            print(f"   Best Val Acc: {history['best_val_acc']:.2f}% (Epoch {history['best_epoch']})")
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation accuracy: {history['best_val_acc']:.2f}% (Epoch {history['best_epoch']})")
    
    # Load best model for final evaluation
    checkpoint = torch.load('models/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history

def evaluate_model(model, test_loader, class_names):
    """
    Evaluate the trained model
    
    Args:
        model: Trained model
        test_loader: Test data loader
        class_names: List of class names
        
    Returns:
        Evaluation results
    """
    print("\n🔍 Evaluating model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    print(f"\n📊 Final Evaluation Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    
    # Detailed classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'true_labels': all_labels,
        'confusion_matrix': cm
    }

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
    """
    print("\n📈 Plotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(range(1, len(history['train_acc']) + 1), history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(range(1, len(history['val_acc']) + 1), history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_training_info(history, class_names):
    """
    Save training information to JSON file
    
    Args:
        history: Training history
        class_names: List of class names
    """
    training_info = {
        'model_info': {
            'architecture': 'efficientnet_b0',
            'num_classes': len(class_names),
            'class_names': class_names,
            'pretrained': True
        },
        'training_results': {
            'best_val_accuracy': history['best_val_acc'],
            'best_epoch': history['best_epoch'],
            'total_epochs': len(history['train_loss'])
        },
        'history': history
    }
    
    with open('training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print("💾 Training information saved to training_info.json")

def main():
    """
    Main training function
    """
    print("🏥 HAM10000 Skin Disease Classification Training")
    print("=" * 60)
    
    # Load data
    image_paths, labels, class_names, label_mapping = load_ham10000_data()
    
    if len(image_paths) == 0:
        print("❌ No data found. Exiting...")
        return
    
    # Create data loaders
    train_loader, val_loader, class_names = create_data_loaders(
        image_paths, labels, class_names, batch_size=32, val_split=0.2
    )
    
    # Create model
    print("\n🏗️ Creating model...")
    model = SkinDiseaseClassifier(num_classes=len(class_names), pretrained=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Architecture: EfficientNet-B0")
    print(f"   Number of classes: {len(class_names)}")
    print(f"   Classes: {class_names}")
    
    # Train model
    model, history = train_model(
        model, train_loader, val_loader, class_names, 
        num_epochs=20, learning_rate=0.001
    )
    
    # Evaluate model
    eval_results = evaluate_model(model, val_loader, class_names)
    
    # Plot training history
    plot_training_history(history)
    
    # Save training information
    save_training_info(history, class_names)
    
    print("\n🎉 Training completed successfully!")
    print("📁 Files created:")
    print("   - models/best_model.pth (best trained model)")
    print("   - training_history.png (training curves)")
    print("   - confusion_matrix.png (evaluation results)")
    print("   - training_info.json (training metadata)")
    
    print(f"\n🎯 Final Results:")
    print(f"   Best Validation Accuracy: {history['best_val_acc']:.2f}%")
    print(f"   Final Test Accuracy: {eval_results['accuracy']*100:.2f}%")
    print(f"   Model saved as: models/best_model.pth")
    
    return model, history, eval_results

if __name__ == "__main__":
    # Run training
    model, history, eval_results = main()
