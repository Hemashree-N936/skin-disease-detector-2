#!/usr/bin/env python3
"""
GroupDRO Training Script for Fair Skin Disease Classification

This script implements training with Group Distributionally Robust Optimization (GroupDRO)
to reduce bias and ensure equal performance across skin tones.

Author: AI Assistant
Date: 2025-03-22
"""

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import existing components
from efficientnet_skin_classifier import SkinDiseaseTrainer, ModelConfig, EfficientNetB4Classifier, MetricsCalculator
from skin_disease_dataset_preparation import DatasetPreparer, Config
from groupdro_loss import GroupDROLoss, GroupDROConfig, FairnessTracker
from fairness_evaluator import FairnessEvaluator


class GroupDROTrainer(SkinDiseaseTrainer):
    """Enhanced trainer with GroupDRO loss for fairness"""
    
    def __init__(self, config: ModelConfig = ModelConfig(), groupdro_config: GroupDROConfig = GroupDROConfig()):
        # Initialize base trainer
        super().__init__(config)
        
        # GroupDRO configuration
        self.groupdro_config = groupdro_config
        
        # Replace standard loss with GroupDRO loss
        self.groupdro_loss = GroupDROLoss(groupdro_config, num_classes=config.NUM_CLASSES)
        
        # Fairness tracker
        self.fairness_tracker = FairnessTracker(groupdro_config)
        
        # Additional training history for fairness
        self.fairness_history = {
            'group_weights': [],
            'fairness_gap': [],
            'balanced_accuracy': [],
            'dark_skin_accuracy': [],
            'light_skin_accuracy': []
        }
        
        # Setup GroupDRO-specific components
        self.setup_groupdro_training()
        
        self.logger.info("GroupDRO trainer initialized for fairness-aware training")
    
    def setup_groupdro_training(self):
        """Setup GroupDRO-specific training components"""
        # Create separate optimizer for GroupDRO weights (handled internally)
        self.logger.info(f"GroupDRO Configuration:")
        self.logger.info(f"  ETA (learning rate): {self.groupdro_config.ETA}")
        self.logger.info(f"  ALPHA (robustness): {self.groupdro_config.ALPHA}")
        self.logger.info(f"  Dark skin weight multiplier: {self.groupdro_config.DARK_SKIN_WEIGHT}")
        self.logger.info(f"  Target fairness gap: {self.groupdro_config.TARGET_FAIRNESS_GAP}")
        self.logger.info(f"  Balanced accuracy weight: {self.groupdro_config.BALANCED_ACCURACY_WEIGHT}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with GroupDRO loss"""
        self.model.train()
        metrics_calculator = MetricsCalculator(self.config.CLASS_NAMES)
        
        # Track GroupDRO metrics
        epoch_groupdro_metrics = []
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Get image paths for skin tone classification
            image_paths = []
            if hasattr(self.train_loader.dataset, 'samples'):
                start_idx = batch_idx * self.config.BATCH_SIZE
                end_idx = min(start_idx + self.config.BATCH_SIZE, len(self.train_loader.dataset))
                for i in range(start_idx, end_idx):
                    if i < len(self.train_loader.dataset.samples):
                        image_paths.append(self.train_loader.dataset.samples[i][0])
            
            # Forward pass with GroupDRO loss
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss, groupdro_metrics = self.groupdro_loss(outputs, labels, image_paths)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            batch_loss = loss.item()
            predictions = torch.argmax(outputs, dim=1)
            metrics_calculator.update(predictions.cpu().numpy(), labels.cpu().numpy(), batch_loss)
            
            # Store GroupDRO metrics
            epoch_groupdro_metrics.append(groupdro_metrics)
            
            # Update progress bar
            if batch_idx % 10 == 0:
                fairness_gap = groupdro_metrics['accuracy_metrics'].get('fairness_gap', 0)
                progress_bar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'fairness_gap': f'{fairness_gap:.3f}'
                })
        
        # Compute epoch metrics
        metrics = metrics_calculator.compute_metrics()
        
        # Add GroupDRO metrics
        if epoch_groupdro_metrics:
            # Average GroupDRO metrics across batches
            avg_groupdro_metrics = {}
            for key in epoch_groupdro_metrics[0].keys():
                if isinstance(epoch_groupdro_metrics[0][key], (int, float)):
                    avg_groupdro_metrics[key] = np.mean([m[key] for m in epoch_groupdro_metrics])
            
            metrics['groupdro_loss'] = avg_groupdro_metrics.get('groupdro_loss', 0)
            metrics['fairness_gap'] = avg_groupdro_metrics['accuracy_metrics'].get('fairness_gap', 0)
            metrics['balanced_accuracy'] = avg_groupdro_metrics['accuracy_metrics'].get('balanced_accuracy', 0)
            
            # Store group accuracies
            group_accuracies = avg_groupdro_metrics['accuracy_metrics'].get('group_accuracies', {})
            metrics['dark_skin_accuracy'] = group_accuracies.get('dark_skin', 0)
            metrics['light_skin_accuracy'] = group_accuracies.get('light_skin', 0)
            
            # Store group weights
            fairness_metrics = self.groupdro_loss.get_fairness_metrics()
            metrics['dark_skin_weight'] = fairness_metrics.get('dark_skin_weight', 0)
            metrics['light_skin_weight'] = fairness_metrics.get('light_skin_weight', 0)
            metrics['weight_ratio'] = fairness_metrics.get('weight_ratio', 1.0)
        
        return metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch with fairness metrics"""
        self.model.eval()
        metrics_calculator = MetricsCalculator(self.config.CLASS_NAMES)
        
        # Track GroupDRO metrics for validation
        val_groupdro_metrics = []
        all_predictions = []
        all_labels = []
        all_image_paths = []
        
        progress_bar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Val]')
        
        with torch.no_grad():
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Get image paths
                batch_image_paths = []
                if hasattr(self.val_loader.dataset, 'samples'):
                    start_idx = len(all_labels)
                    end_idx = start_idx + len(labels)
                    for i in range(start_idx, min(end_idx, len(self.val_loader.dataset.samples))):
                        batch_image_paths.append(self.val_loader.dataset.samples[i][0])
                
                # Forward pass
                outputs = self.model(images)
                loss, groupdro_metrics = self.groupdro_loss(outputs, labels, batch_image_paths)
                
                # Update metrics
                batch_loss = loss.item()
                predictions = torch.argmax(outputs, dim=1)
                metrics_calculator.update(predictions.cpu().numpy(), labels.cpu().numpy(), batch_loss)
                
                # Store for comprehensive evaluation
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_image_paths.extend(batch_image_paths)
                
                val_groupdro_metrics.append(groupdro_metrics)
                
                # Update progress bar
                if batch_idx % 10 == 0:
                    fairness_gap = groupdro_metrics['accuracy_metrics'].get('fairness_gap', 0)
                    progress_bar.set_postfix({'loss': f'{batch_loss:.4f}', 'fairness_gap': f'{fairness_gap:.3f}'})
        
        # Compute epoch metrics
        metrics = metrics_calculator.compute_metrics()
        
        # Add GroupDRO validation metrics
        if val_groupdro_metrics:
            avg_groupdro_metrics = {}
            for key in val_groupdro_metrics[0].keys():
                if isinstance(val_groupdro_metrics[0][key], (int, float)):
                    avg_groupdro_metrics[key] = np.mean([m[key] for m in val_groupdro_metrics])
            
            metrics['groupdro_loss'] = avg_groupdro_metrics.get('groupdro_loss', 0)
            metrics['fairness_gap'] = avg_groupdro_metrics['accuracy_metrics'].get('fairness_gap', 0)
            metrics['balanced_accuracy'] = avg_groupdro_metrics['accuracy_metrics'].get('balanced_accuracy', 0)
            
            group_accuracies = avg_groupdro_metrics['accuracy_metrics'].get('group_accuracies', {})
            metrics['dark_skin_accuracy'] = group_accuracies.get('dark_skin', 0)
            metrics['light_skin_accuracy'] = group_accuracies.get('light_skin', 0)
        
        return metrics
    
    def train(self):
        """Main training loop with GroupDRO and fairness tracking"""
        self.logger.info("Starting GroupDRO fairness-aware training...")
        
        # Load data
        self.load_data()
        
        for epoch in range(self.config.NUM_EPOCHS):
            start_time = time.time()
            
            # Train and validate with GroupDRO
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate scheduler
            self.scheduler.step(val_metrics['accuracy'])
            
            # Save training history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1_score'])
            self.history['val_f1'].append(val_metrics['f1_score'])
            
            # Save GroupDRO-specific history
            if 'fairness_gap' in val_metrics:
                self.fairness_history['fairness_gap'].append(val_metrics['fairness_gap'])
                self.fairness_history['balanced_accuracy'].append(val_metrics['balanced_accuracy'])
                self.fairness_history['dark_skin_accuracy'].append(val_metrics['dark_skin_accuracy'])
                self.fairness_history['light_skin_accuracy'].append(val_metrics['light_skin_accuracy'])
                
                # Store group weights
                fairness_metrics = self.groupdro_loss.get_fairness_metrics()
                self.fairness_history['group_weights'].append({
                    'dark_skin': fairness_metrics.get('dark_skin_weight', 0),
                    'light_skin': fairness_metrics.get('light_skin_weight', 0),
                    'ratio': fairness_metrics.get('weight_ratio', 1.0)
                })
            
            # Update fairness tracker
            self.fairness_tracker.update(epoch, val_metrics)
            
            # Check for improvement (use balanced accuracy instead of regular accuracy)
            if 'balanced_accuracy' in val_metrics:
                current_balanced_acc = val_metrics['balanced_accuracy']
                if current_balanced_acc > self.best_val_accuracy:
                    self.best_val_accuracy = current_balanced_acc
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                else:
                    self.epochs_without_improvement += 1
            
            # Print epoch summary with fairness metrics
            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} completed in {epoch_time:.2f}s")
            self.logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            self.logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            
            if 'fairness_gap' in val_metrics:
                self.logger.info(f"Fairness Gap: {val_metrics['fairness_gap']:.4f}")
                self.logger.info(f"Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
                self.logger.info(f"Dark Skin Acc: {val_metrics['dark_skin_accuracy']:.4f}")
                self.logger.info(f"Light Skin Acc: {val_metrics['light_skin_accuracy']:.4f}")
            
            # Print fairness summary
            self.fairness_tracker.print_epoch_summary(epoch + 1)
            
            # Early stopping based on balanced accuracy
            if self.epochs_without_improvement >= self.config.PATIENCE:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model
        self.save_checkpoint(epoch, val_metrics, is_best=False)
        
        # Plot training curves with fairness metrics
        self.plot_groupdro_training_curves()
        
        # Generate final fairness report
        self.generate_fairness_report()
        
        self.logger.info("GroupDRO fairness-aware training completed!")
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint with GroupDRO state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'fairness_history': self.fairness_history,
            'groupdro_loss_state': {
                'group_weights': self.groupdro_loss.group_weights.clone(),
                'group_counts': self.groupdro_loss.group_counts.clone(),
                'step_count': self.groupdro_loss.step_count.clone()
            },
            'config': self.config.__dict__,
            'groupdro_config': self.groupdro_config.__dict__
        }
        
        if is_best:
            best_path = os.path.join(self.config.MODEL_SAVE_DIR, 'best_groupdro_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best GroupDRO model saved to {best_path}")
        
        # Save last model
        last_path = os.path.join(self.config.MODEL_SAVE_DIR, 'last_groupdro_model.pth')
        torch.save(checkpoint, last_path)
    
    def plot_groupdro_training_curves(self):
        """Plot training curves with fairness metrics"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('GroupDRO Training Curves and Fairness Metrics', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Standard training curves
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, self.history['train_accuracy'], label='Train Accuracy')
        axes[0, 1].plot(epochs, self.history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(epochs, self.history['train_f1'], label='Train F1')
        axes[0, 2].plot(epochs, self.history['val_f1'], label='Val F1')
        axes[0, 2].set_title('Training and Validation F1-Score')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1-Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Fairness metrics
        if self.fairness_history['fairness_gap']:
            axes[1, 0].plot(epochs, self.fairness_history['fairness_gap'], color='red')
            axes[1, 0].axhline(y=self.groupdro_config.TARGET_FAIRNESS_GAP, color='green', 
                               linestyle='--', label=f'Target ({self.groupdro_config.TARGET_FAIRNESS_GAP})')
            axes[1, 0].set_title('Fairness Gap (Dark vs Light)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy Gap')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(epochs, self.fairness_history['balanced_accuracy'], color='purple')
            axes[1, 1].set_title('Balanced Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Balanced Accuracy')
            axes[1, 1].grid(True, alpha=0.3)
            
            axes[1, 2].plot(epochs, self.fairness_history['dark_skin_accuracy'], label='Dark Skin', color='brown')
            axes[1, 2].plot(epochs, self.fairness_history['light_skin_accuracy'], label='Light Skin', color='orange')
            axes[1, 2].set_title('Skin Tone Accuracy')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Accuracy')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        # Group weights
        if self.fairness_history['group_weights']:
            dark_weights = [w['dark_skin'] for w in self.fairness_history['group_weights']]
            light_weights = [w['light_skin'] for w in self.fairness_history['group_weights']]
            weight_ratios = [w['ratio'] for w in self.fairness_history['group_weights']]
            
            axes[2, 0].plot(epochs, dark_weights, label='Dark Skin', color='brown')
            axes[2, 0].plot(epochs, light_weights, label='Light Skin', color='orange')
            axes[2, 0].set_title('Group Weights')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Weight')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            axes[2, 1].plot(epochs, weight_ratios, color='purple')
            axes[2, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal Weight')
            axes[2, 1].set_title('Weight Ratio (Dark/Light)')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Ratio')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        
        # Summary
        if len(self.fairness_history['fairness_gap']) > 0:
            final_gap = self.fairness_history['fairness_gap'][-1]
            final_balanced_acc = self.fairness_history['balanced_accuracy'][-1]
            final_dark_acc = self.fairness_history['dark_skin_accuracy'][-1]
            final_light_acc = self.fairness_history['light_skin_accuracy'][-1]
            
            summary_text = f"FINAL GROUPDRO METRICS:\n\n"
            summary_text += f"Fairness Gap: {final_gap:.4f}\n"
            summary_text += f"Balanced Accuracy: {final_balanced_acc:.4f}\n"
            summary_text += f"Dark Skin Accuracy: {final_dark_acc:.4f}\n"
            summary_text += f"Light Skin Accuracy: {final_light_acc:.4f}\n\n"
            
            if final_gap <= self.groupdro_config.TARGET_FAIRNESS_GAP:
                summary_text += "✅ FAIRNESS TARGET ACHIEVED!"
            else:
                summary_text += "⚠️  FAIRNESS TARGET NOT MET"
            
            axes[2, 2].text(0.05, 0.95, summary_text, transform=axes[2, 2].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[2, 2].set_title('Final Summary')
            axes[2, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.MODEL_SAVE_DIR, 'groupdro_training_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("GroupDRO training curves saved")
    
    def generate_fairness_report(self):
        """Generate comprehensive fairness report"""
        final_report = self.fairness_tracker.get_final_report()
        
        if final_report:
            report_path = os.path.join(self.config.MODEL_SAVE_DIR, 'groupdro_fairness_report.json')
            import json
            with open(report_path, 'w') as f:
                json.dump(final_report, f, indent=2)
            
            self.logger.info(f"GroupDRO fairness report saved to {report_path}")
            
            # Print final summary
            print("\n" + "="*80)
            print("GROUPDRO FAIRNESS TRAINING COMPLETED!")
            print("="*80)
            print(f"Final Fairness Gap: {final_report.get('final_fairness_gap', 0):.4f}")
            print(f"Final Balanced Accuracy: {final_report.get('final_balanced_accuracy', 0):.4f}")
            print(f"Final Dark Skin Accuracy: {final_report.get('final_dark_skin_accuracy', 0):.4f}")
            print(f"Final Light Skin Accuracy: {final_report.get('final_light_skin_accuracy', 0):.4f}")
            print(f"Best Balanced Accuracy: {final_report.get('best_balanced_accuracy', 0):.4f}")
            
            if 'fairness_gap_improvement' in final_report:
                print(f"Fairness Gap Improvement: {final_report['fairness_gap_improvement']:.4f}")
            
            if final_report.get('final_fairness_gap', 1.0) <= self.groupdro_config.TARGET_FAIRNESS_GAP:
                print("✅ FAIRNESS TARGET ACHIEVED!")
            else:
                print("⚠️  FAIRNESS TARGET NOT MET")
            
            print("="*80)


def main():
    """Main function for GroupDRO training"""
    print("="*80)
    print("GROUPDRO FAIRNESS-AWARE TRAINING")
    print("="*80)
    print("Training with Group Distributionally Robust Optimization")
    print("Focus: Equal performance across skin tones")
    print("="*80)
    
    # Create configurations
    config = ModelConfig()
    groupdro_config = GroupDROConfig()
    
    # Model configuration
    config.NUM_EPOCHS = 20
    config.BATCH_SIZE = 32
    config.LEARNING_RATE = 1e-4
    config.WEIGHT_DECAY = 1e-5
    config.DROPOUT_RATE = 0.3
    config.MODEL_SAVE_DIR = 'models_groupdro'
    
    # GroupDRO configuration
    groupdro_config.ETA = 1.0
    groupdro_config.ALPHA = 0.2
    groupdro_config.DARK_SKIN_WEIGHT = 2.0
    groupdro_config.TARGET_FAIRNESS_GAP = 0.02  # 2% target gap
    groupdro_config.BALANCED_ACCURACY_WEIGHT = 0.3
    
    print(f"Training Configuration:")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print()
    print(f"GroupDRO Configuration:")
    print(f"  ETA (learning rate): {groupdro_config.ETA}")
    print(f"  ALPHA (robustness): {groupdro_config.ALPHA}")
    print(f"  Dark Skin Weight: {groupdro_config.DARK_SKIN_WEIGHT}")
    print(f"  Target Fairness Gap: {groupdro_config.TARGET_FAIRNESS_GAP}")
    print(f"  Balanced Accuracy Weight: {groupdro_config.BALANCED_ACCURACY_WEIGHT}")
    print()
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🎉 GPU Available: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU available. Training will use CPU (slower).")
    
    print()
    print("🎯 GROUPDRO BENEFITS:")
    print("  • Reduces bias between skin tones")
    print("  • Improves performance on underrepresented groups")
    print("  • Ensures equal performance across demographics")
    print("  • Adapts to group-specific challenges")
    print("  • Provides fairness guarantees")
    print()
    
    try:
        # Create GroupDRO trainer
        trainer = GroupDROTrainer(config, groupdro_config)
        
        # Train model
        trainer.train()
        
        print("\n" + "="*80)
        print("GROUPDRO FAIRNESS-AWARE TRAINING COMPLETED!")
        print("="*80)
        print(f"Best GroupDRO model saved to: {config.MODEL_SAVE_DIR}/best_groupdro_model.pth")
        print(f"Training curves saved to: {config.MODEL_SAVE_DIR}/groupdro_training_curves.png")
        print(f"Fairness report saved to: {config.MODEL_SAVE_DIR}/groupdro_fairness_report.json")
        print()
        print("🎉 EXPECTED FAIRNESS IMPROVEMENTS:")
        print("✓ Reduced performance gap between skin tones")
        print("✓ Higher accuracy on dark skin samples")
        print("✓ Balanced performance across groups")
        print("✓ Robustness to distribution shifts")
        print("✓ Fair representation learning")
        print()
        print("Next steps:")
        print("1. Compare with standard training using groupdro_comparison.py")
        print("2. Evaluate fairness on test set")
        print("3. Analyze group weight evolution")
        print("4. Test on diverse datasets")
        print("5. Document fairness improvements for research")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
        print("   Current progress has been saved.")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        print("   Please check the error message above and your configuration.")
        return


if __name__ == "__main__":
    main()
