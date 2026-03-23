#!/usr/bin/env python3
"""
Training Script for EfficientNetB4 Skin Disease Classifier

Example script demonstrating how to train the skin disease classification model
with proper configuration and monitoring.

Author: AI Assistant
Date: 2025-03-22
"""

import torch
import os
from efficientnet_skin_classifier import SkinDiseaseTrainer, ModelConfig

def main():
    """Main training function"""
    
    print("="*80)
    print("EFFICIENTNETB4 SKIN DISEASE CLASSIFIER - TRAINING")
    print("="*80)
    print("Detecting Acne, Psoriasis, and Eczema on Dark Skin Tones")
    print("="*80)
    
    # Create configuration
    config = ModelConfig()
    
    # Modify configuration for your specific needs
    config.NUM_EPOCHS = 20
    config.BATCH_SIZE = 32
    config.LEARNING_RATE = 1e-4
    config.WEIGHT_DECAY = 1e-5
    config.DROPOUT_RATE = 0.3
    config.FREEZE_BACKBONE = True  # Start with frozen backbone
    
    # Device configuration (auto-detect GPU/CPU)
    config.DEVICE = 'auto'
    
    # Checkpoint settings
    config.SAVE_BEST_MODEL = True
    config.SAVE_LAST_MODEL = True
    config.MODEL_SAVE_DIR = 'models'
    
    # Early stopping
    config.PATIENCE = 7
    config.MIN_DELTA = 0.001
    
    # Class names
    config.CLASS_NAMES = ['acne', 'psoriasis', 'eczema']
    
    print(f"Training Configuration:")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Dropout Rate: {config.DROPOUT_RATE}")
    print(f"  Freeze Backbone: {config.FREEZE_BACKBONE}")
    print(f"  Classes: {config.CLASS_NAMES}")
    print(f"  Device: {config.DEVICE}")
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
    
    # Create trainer and start training
    try:
        trainer = SkinDiseaseTrainer(config)
        trainer.train()
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Best model saved to: {config.MODEL_SAVE_DIR}/best_model.pth")
        print(f"Training curves saved to: {config.MODEL_SAVE_DIR}/training_curves.png")
        print(f"Training history saved to: {config.MODEL_SAVE_DIR}/training_history.json")
        print(f"Test results saved to: {config.MODEL_SAVE_DIR}/test_results.json")
        print()
        print("You can now use the trained model for inference.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
        print("   Current progress has been saved.")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("   Please check the error message above and your dataset configuration.")
        return
    
    print("\nNext steps:")
    print("1. Review the training curves and metrics")
    print("2. Use the best model for inference on new images")
    print("3. Consider fine-tuning with unfrozen backbone if needed")
    print("4. Evaluate model performance on different skin tone subsets")

if __name__ == "__main__":
    main()
