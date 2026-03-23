#!/usr/bin/env python3
"""
Hybrid CNN-Vision Transformer Model for Image Classification

This implementation combines CNN feature extraction with Vision Transformer
for improved image classification performance.

Author: AI Assistant
Date: 2025-03-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from typing import Tuple, Optional

class PatchEmbedding(nn.Module):
    """
    Convert CNN features to patch embeddings for Vision Transformer
    """
    
    def __init__(self, feature_dim: int, embed_dim: int, patch_size: int = 1):
        """
        Args:
            feature_dim: Dimension of CNN features
            embed_dim: Embedding dimension for ViT
            patch_size: Size of patches (default 1 for feature maps)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Linear projection to embed features
        self.projection = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, embed_dim))  # Max 1000 patches
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: CNN features of shape (batch_size, channels, height, width)
        
        Returns:
            Tuple of (patch_embeddings, num_patches)
        """
        batch_size = x.shape[0]
        
        # Project features to embedding dimension
        x = self.projection(x)  # (batch_size, embed_dim, h, w)
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        
        num_patches = x.shape[1]
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, num_patches+1, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :num_patches+1, :]
        
        return x, num_patches

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism for Vision Transformer
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_patches, embed_dim)
        
        Returns:
            Output tensor of same shape
        """
        batch_size, num_patches, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, num_patches, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (batch_size, num_heads, num_patches, num_patches)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)  # (batch_size, num_heads, num_patches, head_dim)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, num_patches, embed_dim)
        output = self.proj(attn_output)
        
        return output

class TransformerBlock(nn.Module):
    """
    Transformer block with Multi-Head Attention and MLP
    """
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
        """
        super().__init__()
        
        # Multi-Head Attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # MLP
        mlp_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_patches, embed_dim)
        
        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual connection
        attn_output = self.attention(self.norm1(x))
        x = x + self.dropout(attn_output)
        
        # MLP with residual connection
        mlp_output = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_output)
        
        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer for processing CNN features
    """
    
    def __init__(self, feature_dim: int, embed_dim: int = 256, num_heads: int = 8, 
                 num_layers: int = 4, dropout: float = 0.1):
        """
        Args:
            feature_dim: Dimension of CNN features
            embed_dim: Embedding dimension for ViT
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(feature_dim, embed_dim)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.embed_dim = embed_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CNN features of shape (batch_size, channels, height, width)
        
        Returns:
            Class token output of shape (batch_size, embed_dim)
        """
        # Convert to patch embeddings
        x, num_patches = self.patch_embedding(x)  # (batch_size, num_patches+1, embed_dim)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Extract class token (first token)
        cls_token = x[:, 0, :]  # (batch_size, embed_dim)
        
        return cls_token

class HybridCNNViT(nn.Module):
    """
    Hybrid CNN-Vision Transformer model for image classification
    
    Architecture:
    1. CNN backbone (EfficientNet/ResNet) for feature extraction
    2. Vision Transformer for processing features
    3. Combined classification head
    """
    
    def __init__(self, num_classes: int = 3, cnn_backbone: str = 'efficientnet_b0', 
                 vit_embed_dim: int = 256, vit_num_heads: int = 8, vit_num_layers: int = 4,
                 dropout: float = 0.1, pretrained: bool = True):
        """
        Args:
            num_classes: Number of output classes
            cnn_backbone: CNN backbone architecture ('efficientnet_b0', 'resnet18', etc.)
            vit_embed_dim: Embedding dimension for Vision Transformer
            vit_num_heads: Number of attention heads
            vit_num_layers: Number of transformer layers
            dropout: Dropout rate
            pretrained: Whether to use pretrained CNN backbone
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.cnn_backbone = cnn_backbone
        
        # CNN Backbone
        self.cnn_backbone_model = self._create_cnn_backbone(cnn_backbone, pretrained)
        
        # Get CNN feature dimension
        self.cnn_feature_dim = self._get_cnn_feature_dim()
        
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
        
    def _create_cnn_backbone(self, backbone_name: str, pretrained: bool) -> nn.Module:
        """Create CNN backbone model"""
        if backbone_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            # Remove classifier, keep features
            model.classifier = nn.Identity()
            return model.features
        elif backbone_name == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            # Remove classifier and avgpool, keep features
            model.fc = nn.Identity()
            model.avgpool = nn.Identity()
            return model
        elif backbone_name == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            model.fc = nn.Identity()
            model.avgpool = nn.Identity()
            return model
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    def _get_cnn_feature_dim(self) -> int:
        """Get the output dimension of CNN backbone"""
        # Create dummy input to determine feature dimension
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.cnn_backbone_model(dummy_input)
        
        if len(features.shape) == 4:  # (batch, channels, h, w)
            return features.shape[1]
        elif len(features.shape) == 2:  # (batch, features)
            return features.shape[1]
        else:
            raise ValueError("Unexpected CNN output shape")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid model
        
        Args:
            x: Input images of shape (batch_size, 3, height, width)
        
        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        # Step 1: CNN Feature Extraction
        cnn_features = self.cnn_backbone_model(x)  # (batch_size, channels, h, w)
        
        # Step 2: Vision Transformer Processing
        vit_features = self.vit(cnn_features)  # (batch_size, embed_dim)
        
        # Step 3: Classification
        logits = self.classifier(vit_features)  # (batch_size, num_classes)
        
        return logits
    
    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """
        Get intermediate feature maps for visualization
        
        Args:
            x: Input images
        
        Returns:
            Dictionary containing intermediate features
        """
        with torch.no_grad():
            # CNN features
            cnn_features = self.cnn_backbone_model(x)
            
            # ViT features
            vit_features = self.vit(cnn_features)
            
            return {
                'cnn_features': cnn_features,
                'vit_features': vit_features,
                'logits': self.classifier(vit_features)
            }

def create_hybrid_model(num_classes: int = 3, model_size: str = 'small') -> HybridCNNViT:
    """
    Create hybrid CNN-ViT model with predefined configurations
    
    Args:
        num_classes: Number of output classes
        model_size: Model size ('small', 'medium', 'large')
    
    Returns:
        HybridCNNViT model
    """
    if model_size == 'small':
        return HybridCNNViT(
            num_classes=num_classes,
            cnn_backbone='efficientnet_b0',
            vit_embed_dim=256,
            vit_num_heads=8,
            vit_num_layers=4,
            dropout=0.1
        )
    elif model_size == 'medium':
        return HybridCNNViT(
            num_classes=num_classes,
            cnn_backbone='resnet18',
            vit_embed_dim=384,
            vit_num_heads=12,
            vit_num_layers=6,
            dropout=0.1
        )
    elif model_size == 'large':
        return HybridCNNViT(
            num_classes=num_classes,
            cnn_backbone='resnet50',
            vit_embed_dim=512,
            vit_num_heads=16,
            vit_num_layers=8,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")

def test_hybrid_model():
    """Test the hybrid CNN-ViT model"""
    print("🧪 Testing Hybrid CNN-ViT Model...")
    
    # Create model
    model = create_hybrid_model(num_classes=3, model_size='small')
    
    # Create dummy input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        features = model.get_feature_maps(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"CNN features shape: {features['cnn_features'].shape}")
    print(f"ViT features shape: {features['vit_features'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test with different model sizes
    print("\n📊 Testing different model sizes:")
    for size in ['small', 'medium', 'large']:
        try:
            model = create_hybrid_model(num_classes=3, model_size=size)
            params = sum(p.numel() for p in model.parameters())
            print(f"  {size.capitalize()}: {params:,} parameters")
        except Exception as e:
            print(f"  {size.capitalize()}: Error - {e}")
    
    print("✅ Hybrid CNN-ViT model test completed!")

def demonstrate_forward_pass():
    """Demonstrate the forward pass step by step"""
    print("\n🔬 Demonstrating Forward Pass Step by Step...")
    
    # Create model
    model = create_hybrid_model(num_classes=3, model_size='small')
    model.eval()
    
    # Create sample input
    x = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        # Step 1: CNN Feature Extraction
        print("\n📊 Step 1: CNN Feature Extraction")
        cnn_features = model.cnn_backbone_model(x)
        print(f"CNN features shape: {cnn_features.shape}")
        print(f"   CNN extracts spatial features from input images")
        
        # Step 2: Patch Embedding
        print("\n🔧 Step 2: Patch Embedding")
        patch_embeddings, num_patches = model.vit.patch_embedding(cnn_features)
        print(f"Patch embeddings shape: {patch_embeddings.shape}")
        print(f"   Number of patches: {num_patches}")
        print(f"   Features converted to transformer-compatible format")
        
        # Step 3: Transformer Processing
        print("\n🤖 Step 3: Vision Transformer Processing")
        vit_input = patch_embeddings
        for i, block in enumerate(model.vit.transformer_blocks):
            vit_input = block(vit_input)
            print(f"   Transformer block {i+1}: {vit_input.shape}")
        
        # Extract class token
        cls_token = vit_input[:, 0, :]
        print(f"Class token shape: {cls_token.shape}")
        print(f"   Class token contains global image representation")
        
        # Step 4: Classification
        print("\n🎯 Step 4: Classification")
        logits = model.classifier(cls_token)
        print(f"Logits shape: {logits.shape}")
        print(f"   Final classification predictions")
        
        # Apply softmax for probabilities
        probs = F.softmax(logits, dim=1)
        print(f"Probabilities: {probs}")
        print(f"   Predicted classes: {torch.argmax(probs, dim=1)}")
    
    print("\n✅ Forward pass demonstration completed!")

if __name__ == "__main__":
    print("🏥 Hybrid CNN-Vision Transformer for Image Classification")
    print("=" * 60)
    
    # Test the model
    test_hybrid_model()
    
    # Demonstrate forward pass
    demonstrate_forward_pass()
    
    print("\n🎉 Hybrid CNN-ViT model is ready for training!")
    print("📚 Usage:")
    print("   model = create_hybrid_model(num_classes=3, model_size='small')")
    print("   output = model(input_tensor)")
    print("   features = model.get_feature_maps(input_tensor)")
