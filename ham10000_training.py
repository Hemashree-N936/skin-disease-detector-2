#!/usr/bin/env python3
"""
Complete HAM10000 Dataset Training Pipeline for Skin Disease Classification

This script provides a complete training pipeline using the HAM10000 dataset
with pretrained EfficientNetB4 model for multi-class classification.

Requirements:
- torch
- torchvision
- pandas
- numpy
- scikit-learn
- matplotlib
- tqdm
- pillow
- opencv-python

Author: AI Assistant
Date: 2025-03-23
"""

import os
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import zipfile
import requests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
class Config:
    # Dataset paths
    DATASET_URL = "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/IGQNL0/8Q3P0F"
    DATA_DIR = "ham10000_data"
    IMAGES_DIR = os.path.join(DATA_DIR, "HAM10000_images")
    METADATA_FILE = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Model parameters
    NUM_CLASSES = 7  # HAM10000 has 7 classes
    INPUT_SIZE = 224
    FREEZE_BACKBONE = True
    
    # Paths
    MODEL_SAVE_PATH = "models/ham10000_model.pt"
    PLOTS_DIR = "training_plots"
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# HAM10000 class mapping
HAM10000_CLASSES = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}

# Simplified mapping for our 3-class system
SIMPLIFIED_CLASSES = {
    'akiec': 'benign',
    'bcc': 'malignant', 
    'bkl': 'benign',
    'df': 'benign',
    'mel': 'malignant',
    'nv': 'benign',
    'vasc': 'benign'
}

class HAM10000Dataset(Dataset):
    """Custom Dataset for HAM10000 skin disease classification"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return image, label

def download_ham10000():
    """Download and extract HAM10000 dataset"""
    print("📥 Downloading HAM10000 dataset...")
    
    # Create data directory
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    # Download metadata (smaller file)
    metadata_url = "https://raw.githubusercontent.com/ieee8023/ham10000/master/HAM10000_metadata.csv"
    
    if not os.path.exists(Config.METADATA_FILE):
        print("Downloading metadata...")
        response = requests.get(metadata_url)
        with open(Config.METADATA_FILE, 'w') as f:
            f.write(response.text)
        print(f"✅ Metadata saved to {Config.METADATA_FILE}")
    else:
        print("✅ Metadata already exists")
    
    # Note: For full dataset, you'll need to download images separately
    # This is a placeholder - in practice, download from the official source
    print("\n⚠️  NOTE: You need to download HAM10000 images separately!")
    print("📥 Download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IGQNL0")
    print(f"📁 Extract images to: {Config.IMAGES_DIR}")
    
    return Config.METADATA_FILE

def load_and_prepare_data():
    """Load HAM10000 dataset and prepare for training"""
    print("📊 Loading and preparing dataset...")
    
    # Load metadata
    df = pd.read_csv(Config.METADATA_FILE)
    print(f"📋 Loaded metadata: {len(df)} samples")
    print(f"🏷️  Classes found: {df['dx'].unique()}")
    
    # Create image paths
    image_paths = []
    labels = []
    
    # Check if images directory exists
    if not os.path.exists(Config.IMAGES_DIR):
        print(f"❌ Images directory not found: {Config.IMAGES_DIR}")
        print("Please download and extract HAM10000 images to this directory")
        return None, None, None, None
    
    # Process each image
    for idx, row in df.iterrows():
        image_id = row['image_id']
        image_path = os.path.join(Config.IMAGES_DIR, f"{image_id}.jpg")
        
        if os.path.exists(image_path):
            image_paths.append(image_path)
            # Use simplified 3-class system
            simplified_label = SIMPLIFIED_CLASSES[row['dx']]
            labels.append(simplified_label)
    
    print(f"✅ Found {len(image_paths)} valid images")
    
    # Convert labels to indices
    unique_labels = list(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    labels_indices = [label_to_idx[label] for label in labels]
    
    print(f"🏷️  Simplified classes: {unique_labels}")
    print(f"📊 Class distribution:")
    for label in unique_labels:
        count = labels.count(label)
        percentage = (count / len(labels)) * 100
        print(f"   {label}: {count} ({percentage:.1f}%)")
    
    return image_paths, labels_indices, unique_labels, label_to_idx

def get_data_transforms():
    """Create data augmentation and normalization transforms"""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(Config.INPUT_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/Test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.CenterCrop(Config.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(num_classes):
    """Create EfficientNetB4 model with modified classifier"""
    print("🏗️  Creating EfficientNetB4 model...")
    
    # Load pretrained EfficientNetB4
    model = models.efficientnet_b4(weights='IMAGENET1K_V1')
    
    # Freeze backbone if specified
    if Config.FREEZE_BACKBONE:
        print("🔒 Freezing backbone layers...")
        for param in model.features.parameters():
            param.requires_grad = False
    
    # Get the number of features in the final layer
    num_features = model.classifier[1].in_features
    
    # Replace classifier for our number of classes
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, num_classes)
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    """Training loop with loss and accuracy tracking"""
    print("🚀 Starting training...")
    
    device = Config.DEVICE
    model.to(device)
    
    # Track metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_path = None
    
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
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
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
        
        # Calculate training metrics
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss_avg)
        train_accuracies.append(train_acc)
        
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
        
        # Calculate validation metrics
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss_avg)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss_avg)
        
        # Print epoch results
        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"   Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'val_loss': val_loss_avg,
                'train_loss': train_loss_avg,
                'class_names': unique_labels
            }, Config.MODEL_SAVE_PATH)
            best_model_path = Config.MODEL_SAVE_PATH
            print(f"🏆 New best model saved! Val Acc: {val_acc:.2f}%")
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Best model saved to: {best_model_path}")
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training and validation curves"""
    print("📈 Plotting training curves...")
    
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PLOTS_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Training curves saved to {Config.PLOTS_DIR}/training_curves.png")

def evaluate_model(model, test_loader, class_names):
    """Evaluate model on test set"""
    print("🔍 Evaluating model on test set...")
    
    device = Config.DEVICE
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"🎯 Test Accuracy: {accuracy:.2f}%")
    
    # Classification report
    print("\n📋 Classification Report:")
    report = classification_report(
        all_labels, all_predictions, 
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.PLOTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy

def main():
    """Main training pipeline"""
    print("🏥 HAM10000 Skin Disease Classification Training")
    print("=" * 60)
    
    # Download dataset
    metadata_file = download_ham10000()
    
    # Load and prepare data
    image_paths, labels, unique_labels, label_to_idx = load_and_prepare_data()
    
    if image_paths is None:
        print("❌ Cannot proceed without images. Please download HAM10000 dataset.")
        return
    
    global unique_labels  # Make it available for saving
    unique_labels = unique_labels
    
    # Create data transforms
    train_transform, val_transform = get_data_transforms()
    
    # Split data
    print("\n🔄 Splitting data...")
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=Config.VALIDATION_SPLIT + Config.TEST_SPLIT, 
        random_state=42, stratify=labels
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, 
        test_size=Config.TEST_SPLIT / (Config.VALIDATION_SPLIT + Config.TEST_SPLIT),
        random_state=42, stratify=temp_labels
    )
    
    print(f"📊 Data split:")
    print(f"   Training: {len(train_paths)} samples")
    print(f"   Validation: {len(val_paths)} samples")
    print(f"   Test: {len(test_paths)} samples")
    
    # Create datasets
    train_dataset = HAM10000Dataset(train_paths, train_labels, train_transform)
    val_dataset = HAM10000Dataset(val_paths, val_labels, val_transform)
    test_dataset = HAM10000Dataset(test_paths, test_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                          shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                        shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                         shuffle=False, num_workers=2, pin_memory=True)
    
    # Create model
    model = create_model(len(unique_labels))
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Train model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, Config.NUM_EPOCHS
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Evaluate on test set
    test_accuracy = evaluate_model(model, test_loader, unique_labels)
    
    print("\n🎉 Training pipeline completed successfully!")
    print(f"📁 Model saved to: {Config.MODEL_SAVE_PATH}")
    print(f"📊 Plots saved to: {Config.PLOTS_DIR}")
    print(f"🎯 Final test accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
