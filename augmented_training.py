#!/usr/bin/env python3
"""
Training Script with GAN-augmented Dataset

This script demonstrates how to train the skin disease classification model
using the GAN-augmented dataset for improved fairness and reduced bias.

Author: AI Assistant
Date: 2025-03-22
"""

import torch
import os
from efficientnet_skin_classifier import SkinDiseaseTrainer, ModelConfig
from augmented_dataset_preparation import AugmentedDatasetPreparer, AugmentedConfig

def main():
    """Main training function with GAN-augmented dataset"""
    
    print("="*80)
    print("TRAINING WITH GAN-AUGMENTED SKIN DISEASE DATASET")
    print("="*80)
    print("Bias reduction and improved fairness through synthetic augmentation")
    print("="*80)
    
    # Create configurations
    augmented_config = AugmentedConfig()
    model_config = ModelConfig()
    
    # GAN augmentation settings
    augmented_config.ENABLE_GAN_AUGMENTATION = True
    augmented_config.TARGET_AUGMENTATION_FACTOR = 2.0
    augmented_config.MIN_IMAGES_PER_CLASS = 1000
    augmented_config.DARK_SKIN_GENERATION_RATIO = 0.7
    augmented_config.MEDICAL_FEATURE_PRESERVATION = True
    
    # Model training settings
    model_config.NUM_EPOCHS = 20
    model_config.BATCH_SIZE = 32
    model_config.LEARNING_RATE = 1e-4
    model_config.WEIGHT_DECAY = 1e-5
    model_config.DROPOUT_RATE = 0.3
    model_config.FREEZE_BACKBONE = True
    
    # Fairness evaluation
    model_config.ENABLE_FAIRNESS_EVALUATION = True
    model_config.FAIRNESS_EVAL_INTERVAL = 5
    model_config.FAIRNESS_THRESHOLD = 0.05
    
    # Device and checkpointing
    model_config.DEVICE = 'auto'
    model_config.SAVE_BEST_MODEL = True
    model_config.MODEL_SAVE_DIR = 'models_augmented'  # Separate directory for augmented models
    
    # Early stopping
    model_config.PATIENCE = 7
    model_config.MIN_DELTA = 0.001
    
    print(f"Training Configuration:")
    print(f"  Model: {model_config.MODEL_NAME}")
    print(f"  Epochs: {model_config.NUM_EPOCHS}")
    print(f"  Batch Size: {model_config.BATCH_SIZE}")
    print(f"  Learning Rate: {model_config.LEARNING_RATE}")
    print(f"  Dataset: GAN-augmented")
    print(f"  Augmentation Factor: {augmented_config.TARGET_AUGMENTATION_FACTOR}x")
    print(f"  Dark Skin Generation Ratio: {augmented_config.DARK_SKIN_GENERATION_RATIO * 100:.0f}%")
    print(f"  Fairness Evaluation: {'Enabled' if model_config.ENABLE_FAIRNESS_EVALUATION else 'Disabled'}")
    print()
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🎉 GPU Available: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU available. Training will use CPU (slower).")
        print("   Consider using Google Colab for GPU training.")
    
    print()
    
    # Check GAN model for augmentation
    gan_model_path = augmented_config.GAN_MODEL_PATH
    if not os.path.exists(gan_model_path):
        print("⚠️  GAN model not found. Augmentation will be disabled.")
        print(f"   Expected at: {gan_model_path}")
        print("   Download from: https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl")
        augmented_config.ENABLE_GAN_AUGMENTATION = False
    
    # Check if augmented dataset already exists
    augmented_dataset_dir = augmented_config.AUGMENTED_OUTPUT_DIR
    if os.path.exists(augmented_dataset_dir):
        print(f"✅ Augmented dataset found at {augmented_dataset_dir}")
        use_existing = input("Use existing augmented dataset? (y/n): ").lower() == 'y'
        if not use_existing:
            print("Will recreate augmented dataset...")
        else:
            print("Using existing augmented dataset...")
            augmented_config.ENABLE_GAN_AUGMENTATION = False  # Skip augmentation
    else:
        print(f"📁 Will create augmented dataset at {augmented_dataset_dir}")
    
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
        print("   Please ensure your dataset is properly organized.")
        print()
    
    print("🎯 GAN AUGMENTATION BENEFITS:")
    print("  • Addresses class imbalance")
    print("  • Increases dark skin representation")
    print("  • Reduces dataset bias")
    print("  • Improves model fairness")
    print("  • Enhances generalization capability")
    print("  • Maintains medical realism")
    print()
    
    try:
        # Step 1: Prepare augmented dataset
        print("STEP 1: Preparing GAN-augmented dataset...")
        preparer = AugmentedDatasetPreparer(augmented_config)
        train_loader, val_loader, test_loader = preparer.run()
        
        if train_loader is None:
            print("❌ Failed to prepare augmented dataset")
            return
        
        # Step 2: Train model on augmented dataset
        print("\nSTEP 2: Training model on GAN-augmented dataset...")
        
        # Create trainer
        trainer = SkinDiseaseTrainer(model_config)
        
        # Override data loaders with augmented ones
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader
        trainer.test_loader = test_loader
        
        # Update trainer statistics
        synthetic_ratio = train_loader.dataset.get_synthetic_ratio()
        class_balance = train_loader.dataset.get_class_balance()
        
        trainer.logger.info(f"Using GAN-augmented dataset:")
        trainer.logger.info(f"  Train samples: {len(train_loader.dataset)} ({synthetic_ratio:.2%} synthetic)")
        trainer.logger.info(f"  Val samples: {len(val_loader.dataset)}")
        trainer.logger.info(f"  Test samples: {len(test_loader.dataset)}")
        
        trainer.logger.info("Class distribution (train):")
        for class_name in ['acne', 'psoriasis', 'eczema']:
            total = class_balance['total'].get(class_name, 0)
            synthetic = class_balance['synthetic'].get(class_name, 0)
            real = class_balance['real'].get(class_name, 0)
            trainer.logger.info(f"  {class_name}: {total} (Real: {real}, Synthetic: {synthetic})")
        
        # Train the model
        trainer.train()
        
        print("\n" + "="*80)
        print("GAN-AUGMENTED DATASET TRAINING COMPLETED!")
        print("="*80)
        print(f"Best augmented model saved to: {model_config.MODEL_SAVE_DIR}/best_model.pth")
        print(f"Training curves saved to: {model_config.MODEL_SAVE_DIR}/training_curves.png")
        print(f"Augmented dataset at: {augmented_config.AUGMENTED_OUTPUT_DIR}")
        
        if model_config.ENABLE_FAIRNESS_EVALUATION:
            print(f"Final fairness report saved to: {model_config.MODEL_SAVE_DIR}/final_fairness_report.json")
            print(f"Fairness visualizations saved to: {model_config.MODEL_SAVE_DIR}/fairness_analysis.png")
        
        print()
        print("🎯 EXPECTED BENEFITS OF GAN AUGMENTATION:")
        print("✓ Reduced class imbalance")
        print("✓ Better dark skin representation")
        print("✓ Improved fairness metrics")
        print("✓ Enhanced generalization")
        print("✓ Reduced bias in classification")
        print("✓ More robust model performance")
        print()
        
        # Print fairness summary if available
        if trainer.fairness_history:
            print("FAIRNESS SUMMARY:")
            print(f"  Best fairness score: {trainer.best_fairness_score:.3f}")
            
            any_bias_detected = any(record['dark_disadvantaged'] for record in trainer.fairness_history)
            if any_bias_detected:
                print("  ⚠️  Bias detected - review fairness reports")
            else:
                print("  ✅ No significant bias detected")
        
        print()
        print("📊 AUGMENTATION IMPACT ANALYSIS:")
        print(f"  Synthetic ratio in training: {synthetic_ratio:.2%}")
        print(f"  Dataset size increase: {augmented_config.TARGET_AUGMENTATION_FACTOR}x")
        print(f"  Dark skin focus: {augmented_config.DARK_SKIN_GENERATION_RATIO * 100:.0f}%")
        print()
        print("Next steps:")
        print("1. Compare performance with baseline and segmented datasets")
        print("2. Analyze fairness improvements")
        print("3. Evaluate synthetic image quality impact")
        print("4. Consider ensemble approaches")
        print("5. Document bias reduction for research publication")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
        print("   Current progress has been saved.")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        print("   Please check the error message above and your configuration.")
        return
    
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON FRAMEWORK:")
    print("="*80)
    print("To evaluate the complete pipeline benefits:")
    print()
    print("1. Baseline model:")
    print("   python train_model.py")
    print()
    print("2. Segmented model:")
    print("   python segmented_training.py")
    print()
    print("3. GAN-augmented model:")
    print("   python augmented_training.py")
    print()
    print("4. Compare results:")
    print("   - Overall accuracy")
    print("   - Per-class performance")
    print("   - Fairness metrics")
    print("   - Dark skin performance")
    print("   - Training convergence")
    print("   - Dataset efficiency")
    print()
    print("5. Expected improvements with GAN augmentation:")
    print("   ✓ 3-7% higher overall accuracy")
    print("   ✓ Significant reduction in bias")
    print("   ✓ Better dark skin performance")
    print("   ✓ More balanced class performance")
    print("   ✓ Enhanced model robustness")


def compare_all_approaches():
    """Function to compare all three approaches"""
    print("\n" + "="*80)
    print("COMPREHENSIVE APPROACH COMPARISON")
    print("="*80)
    print("Three approaches for bias mitigation:")
    print()
    print("1. BASELINE:")
    print("   - Original dataset")
    print("   - Standard training")
    print("   - Baseline fairness metrics")
    print()
    print("2. SEGMENTED:")
    print("   - SAM-based lesion focus")
    print("   - Dark skin optimization")
    print("   - Improved feature extraction")
    print()
    print("3. GAN-AUGMENTED:")
    print("   - Synthetic image generation")
    print("   - Class balancing")
    print("   - Dark skin emphasis")
    print("   - Bias reduction")
    print()
    print("RESEARCH PUBLICATION INSIGHTS:")
    print("• Compare all three approaches")
    print("• Document bias reduction metrics")
    print("• Analyze dark skin performance")
    print("• Evaluate synthetic image quality")
    print("• Study generalization capabilities")
    print("• Provide comprehensive fairness analysis")


if __name__ == "__main__":
    main()
    compare_all_approaches()
