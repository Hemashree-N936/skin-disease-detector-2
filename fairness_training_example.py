#!/usr/bin/env python3
"""
Fairness-Aware Training Example

This script demonstrates how to train the skin disease classification model
with comprehensive fairness evaluation and bias monitoring.

Author: AI Assistant
Date: 2025-03-22
"""

import torch
import os
from efficientnet_skin_classifier import SkinDiseaseTrainer, ModelConfig

def main():
    """Main function for fairness-aware training"""
    
    print("="*80)
    print("FAIRNESS-AWARE SKIN DISEASE CLASSIFICATION TRAINING")
    print("="*80)
    print("Training with comprehensive bias evaluation for dark vs light skin tones")
    print("="*80)
    
    # Create configuration with fairness evaluation enabled
    config = ModelConfig()
    
    # Training parameters
    config.NUM_EPOCHS = 20
    config.BATCH_SIZE = 32
    config.LEARNING_RATE = 1e-4
    config.WEIGHT_DECAY = 1e-5
    config.DROPOUT_RATE = 0.3
    config.FREEZE_BACKBONE = True
    
    # Fairness evaluation settings
    config.ENABLE_FAIRNESS_EVALUATION = True
    config.FAIRNESS_EVAL_INTERVAL = 5  # Evaluate every 5 epochs
    config.FAIRNESS_THRESHOLD = 0.05   # 5% bias threshold
    
    # Device and checkpointing
    config.DEVICE = 'auto'
    config.SAVE_BEST_MODEL = True
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
    print(f"  Fairness Evaluation: {'Enabled' if config.ENABLE_FAIRNESS_EVALUATION else 'Disabled'}")
    print(f"  Fairness Eval Interval: Every {config.FAIRNESS_EVAL_INTERVAL} epochs")
    print(f"  Classes: {config.CLASS_NAMES}")
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
    
    print("🔍 FAIRNESS EVALUATION FEATURES:")
    print("  • Automatic skin tone classification (dark vs light)")
    print("  • Performance gap calculation between skin tones")
    print("  • Per-class bias analysis")
    print("  • Real-time bias monitoring during training")
    print("  • Fairness-aware early stopping")
    print("  • Comprehensive fairness reporting")
    print("  • Visualization of bias metrics")
    print()
    
    # Create trainer and start training
    try:
        trainer = SkinDiseaseTrainer(config)
        trainer.train()
        
        print("\n" + "="*80)
        print("FAIRNESS-AWARE TRAINING COMPLETED!")
        print("="*80)
        print(f"Best model saved to: {config.MODEL_SAVE_DIR}/best_model.pth")
        print(f"Training curves saved to: {config.MODEL_SAVE_DIR}/training_curves.png")
        print(f"Final fairness report saved to: {config.MODEL_SAVE_DIR}/final_fairness_report.json")
        print(f"Fairness visualizations saved to: {config.MODEL_SAVE_DIR}/fairness_analysis.png")
        print()
        
        # Print fairness summary
        if trainer.fairness_history:
            print("FAIRNESS TRAINING SUMMARY:")
            print(f"  Best fairness score achieved: {trainer.best_fairness_score:.3f}")
            print(f"  Number of fairness evaluations: {len(trainer.fairness_history)}")
            
            # Check if bias was detected
            any_bias_detected = any(record['dark_disadvantaged'] for record in trainer.fairness_history)
            if any_bias_detected:
                print("  ⚠️  Bias against dark skin was detected during training")
                print("     Review fairness reports for detailed analysis")
            else:
                print("  ✅ No significant bias detected against dark skin")
        
        print()
        print("Next steps:")
        print("1. Review the final fairness report for detailed bias analysis")
        print("2. Examine fairness visualizations to understand performance gaps")
        print("3. If bias is detected, consider:")
        print("   - Collecting more diverse training data")
        print("   - Applying bias mitigation techniques")
        print("   - Re-weighting loss functions")
        print("4. Use the trained model with fairness monitoring in production")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
        print("   Current progress and fairness evaluations have been saved.")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("   Please check the error message above and your dataset configuration.")
        return
    
    print("\n" + "="*80)
    print("FAIRNESS EVALUATION OUTPUTS:")
    print("="*80)
    print("Files generated for research publication:")
    print("📊 fairness_report_epoch_N.json - Fairness metrics at each evaluation")
    print("📊 final_fairness_report.json - Comprehensive final fairness analysis")
    print("📈 fairness_analysis.png - Visual fairness analysis plots")
    print("📈 training_curves.png - Model training progress")
    print("📋 training.log - Detailed training and fairness logs")
    print("📋 test_results.json - Final test performance with fairness metrics")
    print()
    print("Key metrics for publication:")
    print("• Overall accuracy and per-class performance")
    print("• Dark vs light skin accuracy gap")
    print("• Per-class bias analysis")
    print("• Fairness score over training epochs")
    print("• Sample distribution by skin tone")

if __name__ == "__main__":
    main()
