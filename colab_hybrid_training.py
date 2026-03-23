#!/usr/bin/env python3
"""
Google Colab Ready Hybrid CNN-ViT Training Script

This script provides complete training pipeline for the hybrid CNN-ViT model
optimized for Google Colab with GPU support.

Author: AI Assistant
Date: 2025-03-23
"""

# Install required packages
!pip install torch torchvision pandas numpy matplotlib scikit-learn seaborn tqdm pillow opencv-python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our hybrid model
import sys
import os

# Since we're in Colab, we'll define the model here
# (In practice, you'd upload hybrid_cnn_vit.py and import it)

class PatchEmbedding(nn.Module):
    """Convert CNN features to patch embeddings for Vision Transformer"""
    
    def __init__(self, feature_dim: int, embed_dim: int, patch_size: int = 1):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        self.projection = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, embed_dim))
        
    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        num_patches = x.shape[1]
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, :num_patches+1, :]
        
        return x, num_patches

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        batch_size, num_patches, embed_dim = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, num_patches, embed_dim)
        output = self.proj(attn_output)
        
        return output

class TransformerBlock(nn.Module):
    """Transformer block with Multi-Head Attention and MLP"""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        mlp_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        attn_output = self.attention(self.norm1(x))
        x = x + self.dropout(attn_output)
        
        mlp_output = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_output)
        
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer for processing CNN features"""
    
    def __init__(self, feature_dim: int, embed_dim: int = 256, num_heads: int = 8, 
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(feature_dim, embed_dim)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.embed_dim = embed_dim
        
    def forward(self, x: torch.Tensor):
        x, num_patches = self.patch_embedding(x)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        cls_token = x[:, 0, :]
        return cls_token

class HybridCNNViT(nn.Module):
    """Hybrid CNN-Vision Transformer model for image classification"""
    
    def __init__(self, num_classes: int = 3, cnn_backbone: str = 'efficientnet_b0', 
                 vit_embed_dim: int = 256, vit_num_heads: int = 8, vit_num_layers: int = 4,
                 dropout: float = 0.1, pretrained: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        self.cnn_backbone = cnn_backbone
        
        # CNN Backbone
        if cnn_backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            model.classifier = nn.Identity()
            self.cnn_backbone_model = model.features
        elif cnn_backbone == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            model.fc = nn.Identity()
            model.avgpool = nn.Identity()
            self.cnn_backbone_model = model
        else:
            raise ValueError(f"Unsupported backbone: {cnn_backbone}")
        
        # Get CNN feature dimension
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.cnn_backbone_model(dummy_input)
        self.cnn_feature_dim = features.shape[1] if len(features.shape) == 2 else features.shape[1]
        
        # Vision Transformer
        self.vit = VisionTransformer(
            feature_dim=self.cnn_feature_dim,
            embed_dim=vit_embed_dim,
            num_heads=vit_num_heads,
            num_layers=vit_num_layers,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(vit_embed_dim, vit_embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(vit_embed_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor):
        # Step 1: CNN Feature Extraction
        cnn_features = self.cnn_backbone_model(x)
        
        # Step 2: Vision Transformer Processing
        vit_features = self.vit(cnn_features)
        
        # Step 3: Classification
        logits = self.classifier(vit_features)
        
        return logits

# Dataset class (same as before)
class SimpleSkinDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if len(self.image_paths) == 0:
            # Create synthetic image
            image = np.random.randint(180, 255, (224, 224, 3), dtype=np.uint8)
            label_idx = self.labels[idx]
            
            if label_idx == 0:  # acne
                for _ in range(5):
                    x, y = np.random.randint(20, 204, 2)
                    image[y:y+3, x:x+3] = [200, 50, 50]
            elif label_idx == 1:  # psoriasis
                for _ in range(3):
                    x, y = np.random.randint(20, 180, 2)
                    image[y:y+5, x:x+5] = [255, 200, 150]
            else:  # eczema
                for _ in range(7):
                    x, y = np.random.randint(20, 204, 2)
                    image[y:y+3, x:x+3] = [200, 180, 180]
            
            image = Image.fromarray(image)
        else:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

def create_synthetic_dataset(num_samples=1000):
    """Create synthetic dataset for demonstration"""
    print("Creating synthetic dataset for demonstration...")
    
    image_paths = []  # Empty for synthetic generation
    labels = []
    
    samples_per_class = num_samples // 3
    for class_idx in range(3):
        labels.extend([class_idx] * samples_per_class)
    
    return image_paths, labels

def train_hybrid_model():
    """Train the hybrid CNN-ViT model"""
    print("🚀 Training Hybrid CNN-ViT Model")
    print("=" * 50)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create synthetic dataset
    image_paths, labels = create_synthetic_dataset(num_samples=900)
    class_names = ['acne', 'psoriasis', 'eczema']
    
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
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_labels, temp_labels = train_test_split(labels, test_size=0.3, random_state=42, stratify=labels)
    val_labels, test_labels = train_test_split(temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)
    
    # Create datasets
    train_dataset = SimpleSkinDataset(image_paths, train_labels, train_transform)
    val_dataset = SimpleSkinDataset(image_paths, val_labels, val_transform)
    test_dataset = SimpleSkinDataset(image_paths, test_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"Dataset splits:")
    print(f"   Training: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")
    
    # Create hybrid model
    print("\n🏗️ Creating Hybrid CNN-ViT Model...")
    model = HybridCNNViT(
        num_classes=3,
        cnn_backbone='efficientnet_b0',
        vit_embed_dim=256,
        vit_num_heads=8,
        vit_num_layers=4,
        dropout=0.1,
        pretrained=True
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   CNN backbone: {model.cnn_backbone}")
    print(f"   ViT embed dim: {model.vit.embed_dim}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Training loop
    print("\n🚀 Starting Training...")
    num_epochs = 10  # Reduced for Colab
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(train_pbar):
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
            
            # Update progress bar
            current_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)
        
        # Validation
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
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Print epoch results
        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"   Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'class_names': class_names,
                'model_config': {
                    'cnn_backbone': model.cnn_backbone,
                    'vit_embed_dim': model.vit.embed_dim,
                    'vit_num_heads': 8,
                    'vit_num_layers': 4
                }
            }, '/content/hybrid_cnn_vit_model.pt')
            print(f"🏆 New best model saved! Val Acc: {val_acc:.2f}%")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
    plt.title('Hybrid CNN-ViT Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Hybrid CNN-ViT Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/content/hybrid_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Final evaluation
    print("\n🔍 Final Evaluation on Test Set...")
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
    print(f"🎯 Final Test Accuracy: {test_accuracy:.2f}%")
    
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Hybrid CNN-ViT Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('/content/hybrid_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Model saved to: /content/hybrid_cnn_vit_model.pt")
    print(f"📊 Training curves saved to: /content/hybrid_training_curves.png")
    print(f"📊 Confusion matrix saved to: /content/hybrid_confusion_matrix.png")
    print(f"🎯 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"🎯 Final test accuracy: {test_accuracy:.2f}%")
    
    return model

def demonstrate_model_architecture():
    """Demonstrate the model architecture and forward pass"""
    print("\n🏗️ Hybrid CNN-ViT Architecture Demonstration")
    print("=" * 50)
    
    # Create model
    model = HybridCNNViT(
        num_classes=3,
        cnn_backbone='efficientnet_b0',
        vit_embed_dim=256,
        vit_num_heads=8,
        vit_num_layers=4,
        dropout=0.1
    )
    
    # Create sample input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: 224x224x3")
    
    model.eval()
    with torch.no_grad():
        # Step-by-step forward pass
        print("\n🔬 Forward Pass Steps:")
        
        # Step 1: CNN features
        cnn_features = model.cnn_backbone_model(input_tensor)
        print(f"1. CNN features: {cnn_features.shape}")
        print(f"   EfficientNet-B0 extracts spatial features")
        
        # Step 2: Patch embedding
        patch_embeddings, num_patches = model.vit.patch_embedding(cnn_features)
        print(f"2. Patch embeddings: {patch_embeddings.shape}")
        print(f"   {num_patches} patches + 1 class token")
        
        # Step 3: Transformer processing
        vit_input = patch_embeddings
        for i, block in enumerate(model.vit.transformer_blocks):
            vit_input = block(vit_input)
            print(f"3.{i+1}. Transformer block {i+1}: {vit_input.shape}")
        
        # Step 4: Class token extraction
        cls_token = vit_input[:, 0, :]
        print(f"4. Class token: {cls_token.shape}")
        
        # Step 5: Classification
        logits = model.classifier(cls_token)
        print(f"5. Logits: {logits.shape}")
        
        # Step 6: Probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)
        print(f"6. Probabilities: {probs}")
        print(f"   Predicted classes: {torch.argmax(probs, dim=1)}")
    
    print("\n✅ Architecture demonstration completed!")

if __name__ == "__main__":
    print("🏥 Hybrid CNN-ViT Training for Google Colab")
    print("=" * 60)
    
    # Demonstrate architecture
    demonstrate_model_architecture()
    
    # Train model
    trained_model = train_hybrid_model()
    
    print("\n🎉 Hybrid CNN-ViT training completed!")
    print("📥 Download the trained model from Colab:")
    print("   - hybrid_cnn_vit_model.pt")
    print("   - hybrid_training_curves.png")
    print("   - hybrid_confusion_matrix.png")
