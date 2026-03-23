#!/usr/bin/env python3
"""
Google Colab Ready HAM10000 Training Script

This script is optimized for Google Colab with automatic dataset download,
GPU utilization, and complete training pipeline.

Run in Colab: https://colab.research.google.com/
"""

# Install required packages
!pip install torch torchvision pandas numpy matplotlib scikit-learn seaborn tqdm pillow opencv-python

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
import warnings
warnings.filterwarnings('ignore')

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")
if torch.cuda.is_available():
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Configuration
class Config:
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 15  # Reduced for Colab
    INPUT_SIZE = 224
    NUM_CLASSES = 3  # Simplified to 3 classes for our use case
    
    # Colab specific paths
    DATA_DIR = "/content/ham10000_data"
    MODEL_SAVE_PATH = "/content/ham10000_model.pt"
    PLOTS_DIR = "/content/plots"

# Download and extract HAM10000 dataset
def download_ham10000_colab():
    """Download HAM10000 dataset in Colab"""
    print("📥 Downloading HAM10000 dataset...")
    
    # Create directories
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    
    # Download using kagglehub (if available) or alternative source
    try:
        import kagglehub
        print("📥 Using Kaggle Hub...")
        path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Find images and metadata
        import glob
        images = glob.glob(f"{path}/*/*.jpg")
        metadata_file = glob.glob(f"{path}/*.csv")[0]
        
        return images, metadata_file
        
    except:
        # Alternative: Download from GitHub (smaller sample)
        print("📥 Downloading sample dataset from GitHub...")
        import requests
        
        # Download metadata
        metadata_url = "https://raw.githubusercontent.com/ieee8023/ham10000/master/HAM10000_metadata.csv"
        metadata_response = requests.get(metadata_url)
        metadata_file = "/content/HAM10000_metadata.csv"
        with open(metadata_file, 'w') as f:
            f.write(metadata_response.text)
        
        # Note: For full dataset, you'll need to upload images manually
        print("⚠️  Please upload HAM10000 images manually to Colab!")
        print("📥 Download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IGQNL0")
        
        return [], metadata_file

# Simplified dataset class for Colab
class SimpleHAM10000Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # For demonstration, create synthetic images if real ones not available
        if len(self.image_paths) == 0:
            # Create synthetic skin disease image
            image = np.random.randint(180, 255, (224, 224, 3), dtype=np.uint8)
            
            # Add disease-specific patterns
            label_idx = self.labels[idx]
            if label_idx == 0:  # Benign
                for _ in range(5):
                    x, y = np.random.randint(20, 204, 2)
                    image[y:y+3, x:x+3] = [200, 180, 180]
            elif label_idx == 1:  # Malignant
                for _ in range(10):
                    x, y = np.random.randint(20, 204, 2)
                    image[y:y+4, x:x+4] = [200, 50, 50]
            else:  # Other
                for _ in range(7):
                    x, y = np.random.randint(20, 204, 2)
                    image[y:y+3, x:x+3] = [255, 200, 150]
            
            image = Image.fromarray(image)
        else:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

def create_synthetic_dataset(num_samples=1000):
    """Create synthetic dataset for demonstration"""
    print("🎨 Creating synthetic dataset for demonstration...")
    
    image_paths = []  # Empty for synthetic generation
    labels = []
    
    # Create balanced dataset
    samples_per_class = num_samples // 3
    for class_idx in range(3):
        labels.extend([class_idx] * samples_per_class)
    
    return image_paths, labels

def main():
    """Main training pipeline for Colab"""
    print("🏥 HAM10000 Training on Google Colab")
    print("=" * 50)
    
    # Download dataset
    image_paths, metadata_file = download_ham10000_colab()
    
    # If no real images, create synthetic dataset for demonstration
    if len(image_paths) == 0:
        print("🎨 Using synthetic dataset for demonstration...")
        image_paths, labels = create_synthetic_dataset(num_samples=900)
        class_names = ['benign', 'malignant', 'other']
    else:
        # Load real metadata
        df = pd.read_csv(metadata_file)
        print(f"📋 Loaded metadata: {len(df)} samples")
        
        # Process real data
        labels = []
        for idx, row in df.iterrows():
            # Simplify to 3 classes
            dx = row['dx']
            if dx in ['akiec', 'bcc', 'mel']:
                labels.append(1)  # malignant
            elif dx in ['bkl', 'df', 'nv', 'vasc']:
                labels.append(0)  # benign
            else:
                labels.append(2)  # other
        
        class_names = ['benign', 'malignant', 'other']
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(Config.INPUT_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.CenterCrop(Config.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Split data
    train_labels, temp_labels = train_test_split(labels, test_size=0.3, random_state=42, stratify=labels)
    val_labels, test_labels = train_test_split(temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)
    
    # Create datasets
    train_dataset = SimpleHAM10000Dataset(image_paths, train_labels, train_transform)
    val_dataset = SimpleHAM10000Dataset(image_paths, val_labels, val_transform)
    test_dataset = SimpleHAM10000Dataset(image_paths, test_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"📊 Dataset splits:")
    print(f"   Training: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")
    
    # Create model
    print("🏗️  Creating EfficientNetB4 model...")
    model = models.efficientnet_b4(weights='IMAGENET1K_V1')
    
    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Modify classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, Config.NUM_CLASSES)
    )
    
    model = model.to(device)
    
    print(f"📊 Model created with {Config.NUM_CLASSES} classes")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    print("🚀 Starting training...")
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n📅 Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
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
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)
        
        scheduler.step(val_losses[-1])
        
        print(f"📊 Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"📊 Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, Config.MODEL_SAVE_PATH)
            print(f"🏆 New best model saved! Val Acc: {val_acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{Config.PLOTS_DIR}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Final evaluation
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
    
    test_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\n🎯 Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"📁 Model saved to: {Config.MODEL_SAVE_PATH}")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    return model, Config.MODEL_SAVE_PATH

if __name__ == "__main__":
    trained_model, model_path = main()
    print(f"\n🎉 Training completed! Model saved at: {model_path}")
    print("📥 Download the model file from Colab to use in your project")
