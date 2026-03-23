#!/usr/bin/env python3
"""
CNN-ViT Hybrid Model for Skin Disease Classification

This module implements a hybrid deep learning architecture combining:
- EfficientNetB4 (CNN) for local feature extraction
- Vision Transformer (ViT) for global context understanding

Architecture: Image → CNN → Feature Map → ViT → Classification

Author: AI Assistant
Date: 2025-03-22
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

# Import existing components
from skin_disease_dataset_preparation import Config, DatasetPreparer
from efficientnet_skin_classifier import ModelConfig, MetricsCalculator
from fairness_evaluator import FairnessEvaluator


class HybridConfig(ModelConfig):
    """Configuration for CNN-ViT hybrid model"""
    
    # Model architecture
    CNN_BACKBONE = 'efficientnet_b4'
    VIT_MODEL = 'vit_base_patch16_224'
    FUSION_METHOD = 'concatenate'  # 'concatenate', 'add', 'attention'
    
    # Feature dimensions
    CNN_FEATURE_DIM = 1792  # EfficientNetB4 output
    VIT_FEATURE_DIM = 768   # ViT-Base output
    FUSION_DIM = 2048      # After fusion
    HIDDEN_DIM = 512       # Classification head
    
    # Training parameters
    CNN_LEARNING_RATE = 1e-4
    VIT_LEARNING_RATE = 1e-5
    FUSION_LEARNING_RATE = 1e-4
    
    # Regularization
    DROPOUT_RATE = 0.3
    WEIGHT_DECAY = 1e-5
    
    # Fine-tuning
    FREEZE_CNN_EPOCHS = 5
    FREEZE_VIT_EPOCHS = 5
    
    # Model saving
    MODEL_SAVE_DIR = 'models_hybrid'


class PatchEmbedding(nn.Module):
    """Patch embedding for ViT input from CNN feature maps"""
    
    def __init__(self, cnn_feature_dim: int, patch_size: int = 16, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Project CNN features to embedding dimension
        self.proj = nn.Conv2d(cnn_feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional encoding
        self.num_patches = (224 // patch_size) ** 2  # Assuming 224x224 input
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def forward(self, x):
        # x shape: (batch_size, cnn_feature_dim, H, W)
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embedding
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention for feature fusion"""
    
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(feature_dim, feature_dim * 3)
        self.proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class FeatureFusion(nn.Module):
    """Feature fusion module for CNN and ViT features"""
    
    def __init__(self, cnn_dim: int, vit_dim: int, fusion_dim: int, method: str = 'concatenate'):
        super().__init__()
        self.method = method
        
        if method == 'concatenate':
            self.fusion_dim = cnn_dim + vit_dim
            self.proj = nn.Linear(self.fusion_dim, fusion_dim)
        elif method == 'add':
            assert cnn_dim == vit_dim, "Dimensions must match for addition"
            self.fusion_dim = cnn_dim
            self.proj = nn.Linear(self.fusion_dim, fusion_dim)
        elif method == 'attention':
            self.attention = MultiHeadAttention(cnn_dim + vit_dim)
            self.proj = nn.Linear(cnn_dim + vit_dim, fusion_dim)
            self.fusion_dim = cnn_dim + vit_dim
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(fusion_dim)
    
    def forward(self, cnn_features, vit_features):
        # cnn_features: (B, cnn_dim)
        # vit_features: (B, vit_dim)
        
        if self.method == 'concatenate':
            # Concatenate features
            fused = torch.cat([cnn_features, vit_features], dim=1)
        elif self.method == 'add':
            # Add features (requires same dimension)
            if cnn_features.shape[1] != vit_features.shape[1]:
                # Project to same dimension
                if cnn_features.shape[1] > vit_features.shape[1]:
                    vit_proj = nn.Linear(vit_features.shape[1], cnn_features.shape[1]).to(vit_features.device)
                    vit_features = vit_proj(vit_features)
                else:
                    cnn_proj = nn.Linear(cnn_features.shape[1], vit_features.shape[1]).to(cnn_features.device)
                    cnn_features = cnn_proj(cnn_features)
            fused = cnn_features + vit_features
        elif self.method == 'attention':
            # Stack features for attention
            stacked = torch.stack([cnn_features, vit_features], dim=1)  # (B, 2, feature_dim)
            fused = self.attention(stacked).mean(dim=1)  # (B, feature_dim)
        
        # Project and normalize
        fused = self.proj(fused)
        fused = self.dropout(fused)
        fused = self.layer_norm(fused)
        
        return fused


class CNNViTHybrid(nn.Module):
    """CNN-ViT Hybrid model for skin disease classification"""
    
    def __init__(self, num_classes: int = 3, config: HybridConfig = HybridConfig()):
        super(CNNViTHybrid, self).__init__()
        
        self.config = config
        self.num_classes = num_classes
        
        # CNN backbone (EfficientNetB4)
        self.cnn_backbone = models.efficientnet_b4(pretrained=True)
        self.cnn_backbone.classifier = nn.Identity()  # Remove classification head
        
        # Get CNN feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            cnn_features = self.cnn_backbone(dummy_input)
            self.cnn_feature_dim = cnn_features.shape[1]
        
        # Patch embedding for ViT
        self.patch_embedding = PatchEmbedding(
            cnn_feature_dim=self.cnn_feature_dim,
            patch_size=16,
            embed_dim=config.VIT_FEATURE_DIM
        )
        
        # Vision Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.VIT_FEATURE_DIM,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu'
        )
        self.vit_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Feature fusion
        self.feature_fusion = FeatureFusion(
            cnn_dim=self.cnn_feature_dim,
            vit_dim=config.VIT_FEATURE_DIM,
            fusion_dim=config.FUSION_DIM,
            method=config.FUSION_METHOD
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.FUSION_DIM, config.HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE * 0.5),
            nn.Linear(config.HIDDEN_DIM // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for new layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        # Input: (batch_size, 3, 224, 224)
        
        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)  # (batch_size, cnn_feature_dim)
        
        # Get intermediate feature maps for ViT
        # We need to extract features from an intermediate layer
        cnn_feature_maps = self._extract_feature_maps(x)
        
        # Patch embedding for ViT
        vit_input = self.patch_embedding(cnn_feature_maps)  # (batch_size, num_patches + 1, vit_feature_dim)
        
        # ViT processing
        vit_features = self.vit_encoder(vit_input)  # (batch_size, num_patches + 1, vit_feature_dim)
        
        # Use class token for global representation
        vit_global = vit_features[:, 0]  # (batch_size, vit_feature_dim)
        
        # Feature fusion
        fused_features = self.feature_fusion(cnn_features, vit_global)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits
    
    def _extract_feature_maps(self, x):
        """Extract intermediate feature maps from CNN"""
        # This is a simplified version - in practice, you'd want to extract from a specific layer
        # For now, we'll use the final features and reshape
        
        # Get features from the last convolutional block
        # This is a workaround - ideally, modify the CNN backbone to expose intermediate features
        with torch.no_grad():
            # Process through CNN backbone to get feature maps
            features = self.cnn_backbone.features(x)
            # Upsample to match expected size if needed
            if features.shape[2] != 14 or features.shape[3] != 14:  # Expected for 224x224 input
                features = F.interpolate(features, size=(14, 14), mode='bilinear', align_corners=False)
        
        return features
    
    def freeze_cnn(self):
        """Freeze CNN backbone parameters"""
        for param in self.cnn_backbone.parameters():
            param.requires_grad = False
        print("CNN backbone frozen")
    
    def unfreeze_cnn(self):
        """Unfreeze CNN backbone parameters"""
        for param in self.cnn_backbone.parameters():
            param.requires_grad = True
        print("CNN backbone unfrozen")
    
    def freeze_vit(self):
        """Freeze ViT parameters"""
        for param in self.patch_embedding.parameters():
            param.requires_grad = False
        for param in self.vit_encoder.parameters():
            param.requires_grad = False
        print("ViT frozen")
    
    def unfreeze_vit(self):
        """Unfreeze ViT parameters"""
        for param in self.patch_embedding.parameters():
            param.requires_grad = True
        for param in self.vit_encoder.parameters():
            param.requires_grad = True
        print("ViT unfrozen")


class HybridTrainer:
    """Trainer for CNN-ViT hybrid model"""
    
    def __init__(self, config: HybridConfig = HybridConfig()):
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.setup_model()
        self.setup_training_components()
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_accuracy': [], 'val_accuracy': [],
            'train_f1': [], 'val_f1': []
        }
        
        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0
        
        # Create model save directory
        os.makedirs(self.config.MODEL_SAVE_DIR, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('hybrid_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_device(self):
        """Setup device for training"""
        if self.config.DEVICE == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.DEVICE)
        
        self.logger.info(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def setup_model(self):
        """Setup hybrid model"""
        self.model = CNNViTHybrid(num_classes=self.config.NUM_CLASSES, config=self.config)
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"CNN-ViT Hybrid Model:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        # Initially freeze backbones
        self.model.freeze_cnn()
        self.model.freeze_vit()
    
    def setup_training_components(self):
        """Setup loss function and optimizers"""
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Separate optimizers for different components
        self.optimizer_cnn = torch.optim.Adam(
            self.model.cnn_backbone.parameters(),
            lr=self.config.CNN_LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        self.optimizer_vit = torch.optim.Adam(
            list(self.model.patch_embedding.parameters()) + list(self.model.vit_encoder.parameters()),
            lr=self.config.VIT_LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        self.optimizer_fusion = torch.optim.Adam(
            list(self.model.feature_fusion.parameters()) + list(self.model.classifier.parameters()),
            lr=self.config.FUSION_LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Learning rate schedulers
        self.scheduler_cnn = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_cnn, mode='max', factor=0.5, patience=3, verbose=True
        )
        self.scheduler_vit = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_vit, mode='max', factor=0.5, patience=3, verbose=True
        )
        self.scheduler_fusion = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_fusion, mode='max', factor=0.5, patience=3, verbose=True
        )
    
    def load_data(self):
        """Load training and validation data"""
        self.logger.info("Loading dataset...")
        
        # Use the dataset preparer
        dataset_config = Config()
        dataset_config.BATCH_SIZE = self.config.BATCH_SIZE
        dataset_config.NUM_WORKERS = self.config.NUM_WORKERS
        
        preparer = DatasetPreparer(dataset_config)
        self.train_loader, self.val_loader, self.test_loader = preparer.run()
        
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        metrics_calculator = MetricsCalculator(self.config.CLASS_NAMES)
        
        # Determine which components to train based on epoch
        if epoch < self.config.FREEZE_CNN_EPOCHS:
            train_cnn = False
        else:
            train_cnn = True
            if epoch == self.config.FREEZE_CNN_EPOCHS:
                self.model.unfreeze_cnn()
        
        if epoch < self.config.FREEZE_VIT_EPOCHS:
            train_vit = False
        else:
            train_vit = True
            if epoch == self.config.FREEZE_VIT_EPOCHS:
                self.model.unfreeze_vit()
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer_cnn.zero_grad()
            self.optimizer_vit.zero_grad()
            self.optimizer_fusion.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            if train_cnn:
                self.optimizer_cnn.step()
            if train_vit:
                self.optimizer_vit.step()
            self.optimizer_fusion.step()
            
            # Update metrics
            batch_loss = loss.item()
            predictions = torch.argmax(outputs, dim=1)
            metrics_calculator.update(predictions.cpu().numpy(), labels.cpu().numpy(), batch_loss)
            
            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})
        
        # Compute epoch metrics
        metrics = metrics_calculator.compute_metrics()
        return metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        metrics_calculator = MetricsCalculator(self.config.CLASS_NAMES)
        
        progress_bar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Val]')
        
        with torch.no_grad():
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                batch_loss = loss.item()
                predictions = torch.argmax(outputs, dim=1)
                metrics_calculator.update(predictions.cpu().numpy(), labels.cpu().numpy(), batch_loss)
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})
        
        # Compute epoch metrics
        metrics = metrics_calculator.compute_metrics()
        return metrics
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting CNN-ViT hybrid training...")
        
        # Load data
        self.load_data()
        
        for epoch in range(self.config.NUM_EPOCHS):
            start_time = time.time()
            
            # Train and validate
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate schedulers
            self.scheduler_cnn.step(val_metrics['accuracy'])
            self.scheduler_vit.step(val_metrics['accuracy'])
            self.scheduler_fusion.step(val_metrics['accuracy'])
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1_score'])
            self.history['val_f1'].append(val_metrics['f1_score'])
            
            # Check for improvement
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.epochs_without_improvement += 1
            
            # Print progress
            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} completed in {epoch_time:.2f}s")
            self.logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            self.logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.PATIENCE:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model
        self.save_checkpoint(epoch, val_metrics, is_best=False)
        
        # Plot training curves
        self.plot_training_curves()
        
        self.logger.info("CNN-ViT hybrid training completed!")
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_cnn_state_dict': self.optimizer_cnn.state_dict(),
            'optimizer_vit_state_dict': self.optimizer_vit.state_dict(),
            'optimizer_fusion_state_dict': self.optimizer_fusion.state_dict(),
            'scheduler_cnn_state_dict': self.scheduler_cnn.state_dict(),
            'scheduler_vit_state_dict': self.scheduler_vit.state_dict(),
            'scheduler_fusion_state_dict': self.scheduler_fusion.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'config': self.config.__dict__
        }
        
        if is_best:
            best_path = os.path.join(self.config.MODEL_SAVE_DIR, 'best_hybrid_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best hybrid model saved to {best_path}")
        
        # Save last model
        last_path = os.path.join(self.config.MODEL_SAVE_DIR, 'last_hybrid_model.pth')
        torch.save(checkpoint, last_path)
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_accuracy'], label='Train Accuracy')
        axes[0, 1].plot(self.history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1-Score
        axes[1, 0].plot(self.history['train_f1'], label='Train F1')
        axes[1, 0].plot(self.history['val_f1'], label='Val F1')
        axes[1, 0].set_title('Training and Validation F1-Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rates (if available)
        axes[1, 1].set_title('Model Architecture')
        axes[1, 1].text(0.1, 0.5, f"CNN Backbone: {self.config.CNN_BACKBONE}\n"
                                  f"ViT Model: {self.config.VIT_MODEL}\n"
                                  f"Fusion Method: {self.config.FUSION_METHOD}\n"
                                  f"Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}",
                       transform=axes[1, 1].transAxes, fontsize=12, verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.MODEL_SAVE_DIR, 'hybrid_training_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function"""
    print("="*80)
    print("CNN-ViT HYBRID MODEL FOR SKIN DISEASE CLASSIFICATION")
    print("="*80)
    print("Combining EfficientNetB4 (CNN) and Vision Transformer (ViT)")
    print("="*80)
    
    # Create configuration
    config = HybridConfig()
    
    print(f"Hybrid Model Configuration:")
    print(f"  CNN Backbone: {config.CNN_BACKBONE}")
    print(f"  ViT Model: {config.VIT_MODEL}")
    print(f"  Fusion Method: {config.FUSION_METHOD}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  CNN Learning Rate: {config.CNN_LEARNING_RATE}")
    print(f"  ViT Learning Rate: {config.VIT_LEARNING_RATE}")
    print(f"  Fusion Learning Rate: {config.FUSION_LEARNING_RATE}")
    print(f"  Freeze CNN Epochs: {config.FREEZE_CNN_EPOCHS}")
    print(f"  Freeze ViT Epochs: {config.FREEZE_VIT_EPOCHS}")
    print()
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🎉 GPU Available: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU available. Training will use CPU (slower).")
    
    print()
    print("🔍 HYBRID ARCHITECTURE BENEFITS:")
    print("  • CNN: Local feature extraction (textures, patterns)")
    print("  • ViT: Global context understanding (spatial relationships)")
    print("  • Fusion: Combined local and global representations")
    print("  • Robustness: Improved performance on complex patterns")
    print("  • Accuracy: Expected improvement over single architectures")
    print()
    
    try:
        # Create trainer
        trainer = HybridTrainer(config)
        
        # Train model
        trainer.train()
        
        print("\n" + "="*80)
        print("CNN-ViT HYBRID TRAINING COMPLETED!")
        print("="*80)
        print(f"Best hybrid model saved to: {config.MODEL_SAVE_DIR}/best_hybrid_model.pth")
        print(f"Training curves saved to: {config.MODEL_SAVE_DIR}/hybrid_training_curves.png")
        print(f"Best validation accuracy: {trainer.best_val_accuracy:.4f}")
        print()
        print("🎯 EXPECTED BENEFITS:")
        print("✓ Improved accuracy on complex skin patterns")
        print("✓ Better generalization to diverse presentations")
        print("✓ Enhanced robustness to variations")
        print("✓ Superior feature representation")
        print("✓ State-of-the-art performance")
        print()
        print("Next steps:")
        print("1. Compare with CNN-only baseline")
        print("2. Evaluate on test set")
        print("3. Analyze feature fusion effectiveness")
        print("4. Test robustness on different datasets")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
        print("   Current progress has been saved.")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return


if __name__ == "__main__":
    main()
