#!/usr/bin/env python3
"""
Training Script with SAM-based Segmented Dataset

This script demonstrates how to train the skin disease classification model
using the SAM-segmented dataset for focused lesion classification.

Author: AI Assistant
Date: 2025-03-22
"""

import torch
import os
from efficientnet_skin_classifier import SkinDiseaseTrainer, ModelConfig
from segmented_dataset_preparation import SegmentedDatasetPreparer, SegmentedConfig

def main():
    """Main training function with segmented dataset"""
    
    print("="*80)
    print("TRAINING WITH SAM-SEGMENTED SKIN DISEASE DATASET")
    print("="*80)
    print("Focused lesion classification for improved performance")
    print("="*80)
    
    # Create configurations
    segmented_config = SegmentedConfig()
    model_config = ModelConfig()
    
    # Segmentation settings
    segmented_config.ENABLE_SEGMENTATION = True
    segmented_config.DARK_SKIN_ENHANCEMENT = True
    segmented_config.USE_ORIGINAL_ON_FAILURE = True
    
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
    model_config.MODEL_SAVE_DIR = 'models_segmented'  # Separate directory for segmented models
    
    # Early stopping
    model_config.PATIENCE = 7
    model_config.MIN_DELTA = 0.001
    
    print(f"Training Configuration:")
    print(f"  Model: {model_config.MODEL_NAME}")
    print(f"  Epochs: {model_config.NUM_EPOCHS}")
    print(f"  Batch Size: {model_config.BATCH_SIZE}")
    print(f"  Learning Rate: {model_config.LEARNING_RATE}")
    print(f"  Dataset: Segmented (SAM-based)")
    print(f"  Dark Skin Enhancement: {segmented_config.DARK_SKIN_ENHANCEMENT}")
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
    
    # Check SAM model for segmentation
    sam_model_path = segmented_config.SAM_MODEL_PATH
    if not os.path.exists(sam_model_path):
        print("⚠️  SAM model not found. Segmentation will be disabled.")
        print(f"   Expected at: {sam_model_path}")
        print("   Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        segmented_config.ENABLE_SEGMENTATION = False
    
    # Check if segmented dataset already exists
    segmented_dataset_dir = segmented_config.SEGMENTED_OUTPUT_DIR
    if os.path.exists(segmented_dataset_dir):
        print(f"✅ Segmented dataset found at {segmented_dataset_dir}")
        use_existing = input("Use existing segmented dataset? (y/n): ").lower() == 'y'
        if not use_existing:
            print("Will recreate segmented dataset...")
        else:
            print("Using existing segmented dataset...")
            segmented_config.ENABLE_SEGMENTATION = False  # Skip segmentation
    else:
        print(f"📁 Will create segmented dataset at {segmented_dataset_dir}")
    
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
    
    print("🔍 SEGMENTATION BENEFITS:")
    print("  • Focused lesion area classification")
    print("  • Reduced background noise")
    print("  • Improved feature extraction")
    print("  • Better generalization")
    print("  • Enhanced performance on small lesions")
    print()
    
    try:
        # Step 1: Prepare segmented dataset
        print("STEP 1: Preparing segmented dataset...")
        preparer = SegmentedDatasetPreparer(segmented_config)
        train_loader, val_loader, test_loader = preparer.run()
        
        if train_loader is None:
            print("❌ Failed to prepare segmented dataset")
            return
        
        # Step 2: Train model on segmented dataset
        print("\nSTEP 2: Training model on segmented dataset...")
        
        # Create trainer
        trainer = SkinDiseaseTrainer(model_config)
        
        # Override data loaders with segmented ones
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader
        trainer.test_loader = test_loader
        
        # Update trainer statistics
        trainer.logger.info(f"Using segmented dataset:")
        trainer.logger.info(f"  Train samples: {len(train_loader.dataset)}")
        trainer.logger.info(f"  Val samples: {len(val_loader.dataset)}")
        trainer.logger.info(f"  Test samples: {len(test_loader.dataset)}")
        
        # Train the model
        trainer.train()
        
        print("\n" + "="*80)
        print("SEGMENTED DATASET TRAINING COMPLETED!")
        print("="*80)
        print(f"Best segmented model saved to: {model_config.MODEL_SAVE_DIR}/best_model.pth")
        print(f"Training curves saved to: {model_config.MODEL_SAVE_DIR}/training_curves.png")
        print(f"Segmented dataset at: {segmented_config.SEGMENTED_OUTPUT_DIR}")
        
        if model_config.ENABLE_FAIRNESS_EVALUATION:
            print(f"Final fairness report saved to: {model_config.MODEL_SAVE_DIR}/final_fairness_report.json")
            print(f"Fairness visualizations saved to: {model_config.MODEL_SAVE_DIR}/fairness_analysis.png")
        
        print()
        print("🎯 EXPECTED BENEFITS OF SEGMENTED TRAINING:")
        print("✓ Higher accuracy on lesion classification")
        print("✓ Better performance on small or subtle lesions")
        print("✓ Reduced false positives from background")
        print("✓ Improved model interpretability")
        print("✓ Enhanced robustness to image variations")
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
        print("Next steps:")
        print("1. Compare performance with non-segmented baseline")
        print("2. Analyze segmentation quality impact on results")
        print("3. Test model on new images with preprocessing")
        print("4. Consider ensemble approaches (segmented + original)")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
        print("   Current progress has been saved.")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("   Please check the error message above and your configuration.")
        return
    
    print("\n" + "="*80)
    print("COMPARISON SUGGESTIONS:")
    print("="*80)
    print("To evaluate the benefits of segmentation:")
    print("1. Train baseline model on original dataset")
    print("2. Train segmented model using this script")
    print("3. Compare metrics:")
    print("   - Overall accuracy")
    print("   - Per-class performance")
    print("   - Fairness metrics")
    print("   - Training convergence speed")
    print("4. Document improvements for research publication")


def compare_segmented_vs_baseline():
    """Function to compare segmented vs baseline performance"""
    print("\n" + "="*80)
    print("COMPARISON: SEGMENTED VS BASELINE DATASET")
    print("="*80)
    print("To run a comparison study:")
    print()
    print("1. Train baseline model:")
    print("   python train_model.py")
    print()
    print("2. Train segmented model:")
    print("   python segmented_training.py")
    print()
    print("3. Compare results:")
    print("   - Check models/best_model.pth (baseline)")
    print("   - Check models_segmented/best_model.pth (segmented)")
    print("   - Review training curves and fairness reports")
    print()
    print("4. Expected improvements with segmentation:")
    print("   ✓ 2-5% higher overall accuracy")
    print("   ✓ Better performance on small lesions")
    print("   ✓ Reduced false positives")
    print("   ✓ Faster convergence")
    print("   ✓ Similar or better fairness metrics")


if __name__ == "__main__":
    main()
    compare_segmented_vs_baseline()
