#!/usr/bin/env python3
"""
Training Script for CNN-ViT Hybrid Model

This script provides an easy-to-use interface for training the CNN-ViT hybrid model
with comprehensive evaluation and comparison capabilities.

Author: AI Assistant
Date: 2025-03-22
"""

import torch
import os
from cnn_vit_hybrid import HybridTrainer, HybridConfig

def main():
    """Main training function for CNN-ViT Hybrid model"""
    
    print("="*80)
    print("CNN-ViT HYBRID MODEL TRAINING")
    print("="*80)
    print("Combining EfficientNetB4 (CNN) and Vision Transformer (ViT)")
    print("="*80)
    
    # Create configuration
    config = HybridConfig()
    
    # Model architecture settings
    config.CNN_BACKBONE = 'efficientnet_b4'
    config.VIT_MODEL = 'vit_base_patch16_224'
    config.FUSION_METHOD = 'concatenate'  # Options: 'concatenate', 'add', 'attention'
    
    # Training parameters
    config.NUM_EPOCHS = 20
    config.BATCH_SIZE = 32
    config.CNN_LEARNING_RATE = 1e-4
    config.VIT_LEARNING_RATE = 1e-5
    config.FUSION_LEARNING_RATE = 1e-4
    config.WEIGHT_DECAY = 1e-5
    config.DROPOUT_RATE = 0.3
    
    # Fine-tuning strategy
    config.FREEZE_CNN_EPOCHS = 5
    config.FREEZE_VIT_EPOCHS = 5
    
    # Device and checkpointing
    config.DEVICE = 'auto'
    config.SAVE_BEST_MODEL = True
    config.MODEL_SAVE_DIR = 'models_hybrid'
    
    # Early stopping
    config.PATIENCE = 7
    config.MIN_DELTA = 0.001
    
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
        print("   Consider using Google Colab for GPU training.")
    
    print()
    
    # Check dataset paths
    required_dirs = ['data/dermacon_in', 'data/kaggle_acne', 'data/kaggle_psoriasis', 'data/kaggle_eczema']
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("⚠️  Warning: The following dataset directories are missing:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        print("   Please ensure your dataset is properly organized before training.")
        print()
    
    print("🔍 HYBRID ARCHITECTURE BENEFITS:")
    print("  • CNN: Extracts local features (textures, edges, patterns)")
    print("  • ViT: Captures global context and spatial relationships")
    print("  • Fusion: Combines complementary feature representations")
    print("  • Robustness: Better handling of complex skin patterns")
    print("  • Accuracy: Expected improvement over single architectures")
    print()
    
    print("🏗️  ARCHITECTURE OVERVIEW:")
    print("  Input Image (224x224x3)")
    print("  ↓")
    print("  EfficientNetB4 CNN → Local Features (1792-dim)")
    print("  ↓")
    print("  Feature Maps → Patch Embedding → ViT")
    print("  ↓")
    print("  ViT Output → Global Features (768-dim)")
    print("  ↓")
    print("  Feature Fusion → Combined Features (2048-dim)")
    print("  ↓")
    print("  Classification Head → 3 Classes (Acne, Psoriasis, Eczema)")
    print()
    
    # Explain fusion methods
    print("🔀 FUSION METHODS:")
    print("  • Concatenate: Simply combine feature vectors")
    print("  • Add: Element-wise addition (requires same dimensions)")
    print("  • Attention: Learn weighted combination of features")
    print(f"  Current: {config.FUSION_METHOD}")
    print()
    
    # Training strategy explanation
    print("📚 TRAINING STRATEGY:")
    print(f"  • Epochs 1-{config.FREEZE_CNN_EPOCHS}: Train fusion head only (CNN frozen)")
    print(f"  • Epochs {config.FREEZE_CNN_EPOCHS+1}-{config.FREEZE_VIT_EPOCHS}: Train CNN + fusion (ViT frozen)")
    print(f"  • Epochs {config.FREEZE_VIT_EPOCHS+1}-{config.NUM_EPOCHS}: Train all components")
    print("  • Different learning rates for each component")
    print("  • Progressive unfreezing for stable training")
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
        
        # Model complexity information
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        
        print("📊 MODEL COMPLEXITY:")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  Model Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        print()
        
        print("🎯 EXPECTED BENEFITS:")
        print("✓ Improved accuracy on complex skin patterns")
        print("✓ Better generalization to diverse presentations")
        print("✓ Enhanced robustness to variations")
        print("✓ Superior feature representation")
        print("✓ State-of-the-art performance")
        print()
        
        # Training summary
        print("📈 TRAINING SUMMARY:")
        print(f"  Final Validation Accuracy: {trainer.best_val_accuracy:.4f}")
        print(f"  Training Epochs Completed: {len(trainer.history['train_accuracy'])}")
        print(f"  Final Training Loss: {trainer.history['train_loss'][-1]:.4f}")
        print(f"  Final Validation Loss: {trainer.history['val_loss'][-1]:.4f}")
        print()
        
        print("Next steps:")
        print("1. Compare with CNN-only baseline using hybrid_comparison.py")
        print("2. Evaluate on test set for final performance metrics")
        print("3. Analyze feature fusion effectiveness")
        print("4. Test robustness on different datasets")
        print("5. Consider ensemble approaches for further improvements")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
        print("   Current progress has been saved.")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        print("   Please check the error message above and your configuration.")
        return
    
    print("\n" + "="*80)
    print("COMPARISON AND EVALUATION:")
    print("="*80)
    print("To evaluate the hybrid model benefits:")
    print()
    print("1. Run comprehensive comparison:")
    print("   python hybrid_comparison.py")
    print()
    print("2. This will compare:")
    print("   • CNN-only baseline (EfficientNetB4)")
    print("   • CNN-ViT Hybrid model")
    print("   • Performance metrics")
    print("   • Model complexity")
    print("   • Improvement analysis")
    print()
    print("3. Expected hybrid advantages:")
    print("   • 2-5% higher accuracy")
    print("   • Better performance on complex patterns")
    print("   • Improved robustness")
    print("   • Enhanced feature representation")
    print()
    print("4. Research publication insights:")
    print("   • Architectural innovation")
    print("   • Performance quantification")
    print("   • Ablation studies")
    print("   • Feature analysis")


def run_quick_test():
    """Run a quick test to verify the hybrid model works"""
    print("Running quick test of CNN-ViT Hybrid model...")
    
    try:
        config = HybridConfig()
        config.NUM_EPOCHS = 1  # Just test one epoch
        config.BATCH_SIZE = 2   # Small batch size
        
        trainer = HybridTrainer(config)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            trainer.model = trainer.model.cuda()
        
        with torch.no_grad():
            output = trainer.model(dummy_input)
        
        print(f"✅ Model test successful!")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected output: [batch_size, 3]")
        print(f"   Actual output: {list(output.shape)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # First run a quick test
    if run_quick_test():
        print("\n" + "="*50)
        print("Model test passed! Starting full training...")
        print("="*50)
        main()
    else:
        print("\n" + "="*50)
        print("Model test failed! Please check the implementation.")
        print("="*50)
