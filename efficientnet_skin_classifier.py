#!/usr/bin/env python3
"""
EfficientNetB4-based Skin Disease Classifier

Multi-class classification model for detecting Acne, Psoriasis, and Eczema
with focus on dark skin tones (Fitzpatrick IV-VI).

Author: AI Assistant
Date: 2025-03-22
"""

import os
import logging
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import dataset preparation
from skin_disease_dataset_preparation import DatasetPreparer, Config
from fairness_evaluator import FairnessEvaluator


class ModelConfig:
    """Configuration for EfficientNetB4 model training"""
    
    # Model parameters
    MODEL_NAME = 'efficientnet_b4'
    NUM_CLASSES = 3
    DROPOUT_RATE = 0.3
    FREEZE_BACKBONE = True
    
    # Training parameters
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Loss and optimizer
    CRITERION = 'cross_entropy'
    OPTIMIZER = 'adam'
    
    # Device and checkpointing
    DEVICE = 'auto'  # 'auto', 'cpu', 'cuda'
    SAVE_BEST_MODEL = True
    SAVE_LAST_MODEL = True
    MODEL_SAVE_DIR = 'models'
    CHECKPOINT_INTERVAL = 5
    
    # Early stopping
    PATIENCE = 7
    MIN_DELTA = 0.001
    
    # Fairness evaluation
    ENABLE_FAIRNESS_EVALUATION = True
    FAIRNESS_EVAL_INTERVAL = 5  # Evaluate fairness every N epochs
    FAIRNESS_THRESHOLD = 0.05  # 5% performance gap threshold
    
    # Class names
    CLASS_NAMES = ['acne', 'psoriasis', 'eczema']


class EfficientNetB4Classifier(nn.Module):
    """EfficientNetB4-based classifier for skin disease detection"""
    
    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.3, pretrained: bool = True):
        super(EfficientNetB4Classifier, self).__init__()
        
        # Load pretrained EfficientNetB4
        self.backbone = models.efficientnet_b4(pretrained=pretrained)
        
        # Get the number of features from the last layer
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier with custom layers
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning"""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("Backbone frozen. Only classifier parameters will be trained.")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning"""
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        print("Backbone unfrozen. All parameters will be trained.")


class MetricsCalculator:
    """Calculate and track training metrics"""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
        """Update metrics with batch predictions"""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        pred_classes = np.argmax(predictions, axis=1)
        
        self.predictions.extend(pred_classes)
        self.targets.extend(targets)
        self.losses.append(loss)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute comprehensive metrics"""
        if len(self.targets) == 0:
            return {}
        
        # Convert to numpy arrays
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Per-class accuracy
        per_class_accuracy = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
                per_class_accuracy[class_name] = class_acc
            else:
                per_class_accuracy[class_name] = 0.0
        
        # Average loss
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'loss': avg_loss,
            'per_class_accuracy': per_class_accuracy,
            'precision_per_class': dict(zip(self.class_names, precision_per_class)),
            'recall_per_class': dict(zip(self.class_names, recall_per_class)),
            'f1_per_class': dict(zip(self.class_names, f1_per_class))
        }
        
        return metrics


class SkinDiseaseTrainer:
    """Main training class for skin disease classification"""
    
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.setup_model()
        self.setup_training_components()
        
        # Training history
        self.history = defaultdict(list)
        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0
        
        # Fairness tracking
        self.fairness_history = []
        self.best_fairness_score = 0.0
        self.fairness_evaluator = None
        
        # Create model save directory
        os.makedirs(self.config.MODEL_SAVE_DIR, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_device(self):
        """Setup device (CPU/GPU)"""
        if self.config.DEVICE == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.DEVICE)
        
        self.logger.info(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def setup_model(self):
        """Setup EfficientNetB4 model"""
        self.model = EfficientNetB4Classifier(
            num_classes=self.config.NUM_CLASSES,
            dropout_rate=self.config.DROPOUT_RATE,
            pretrained=True
        )
        
        if self.config.FREEZE_BACKBONE:
            self.model.freeze_backbone()
        
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model: {self.config.MODEL_NAME}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def setup_training_components(self):
        """Setup loss function, optimizer, and scheduler"""
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
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
    
    def evaluate_fairness(self, epoch: int) -> Dict[str, Any]:
        """Evaluate model fairness on validation set"""
        if not self.config.ENABLE_FAIRNESS_EVALUATION:
            return {}
        
        self.logger.info(f"Evaluating fairness at epoch {epoch+1}...")
        
        # Initialize fairness evaluator if not already done
        if self.fairness_evaluator is None:
            self.fairness_evaluator = FairnessEvaluator(
                self.model, 
                self.config.CLASS_NAMES, 
                self.device
            )
        
        # Evaluate fairness on validation set
        fairness_report = self.fairness_evaluator.evaluate_dataset_fairness(
            self.val_loader.dataset, 
            batch_size=self.config.BATCH_SIZE
        )
        
        # Calculate fairness score
        fairness_score = self.fairness_evaluator.calculate_fairness_score(fairness_report)
        
        # Store fairness metrics
        fairness_record = {
            'epoch': epoch,
            'fairness_score': fairness_score,
            'accuracy_gap': fairness_report['performance_gaps'].get('accuracy_gap', 0),
            'dark_disadvantaged': fairness_report['bias_indicators'].get('dark_skin_disadvantaged', False),
            'biased_classes': fairness_report['bias_indicators'].get('biased_classes', [])
        }
        
        self.fairness_history.append(fairness_record)
        
        # Update best fairness score
        if fairness_score > self.best_fairness_score:
            self.best_fairness_score = fairness_score
        
        # Log fairness results
        self.logger.info(f"Fairness Score: {fairness_score:.3f}")
        self.logger.info(f"Accuracy Gap: {fairness_record['accuracy_gap']:.4f}")
        
        if fairness_record['dark_disadvantaged']:
            self.logger.warning("⚠️  Model shows bias against dark skin!")
        
        if fairness_record['biased_classes']:
            self.logger.warning(f"⚠️  Bias detected in classes: {', '.join(fairness_record['biased_classes'])}")
        
        return fairness_report
    
    def fairness_aware_early_stopping(self, fairness_report: Dict[str, Any]) -> bool:
        """Check if early stopping should be triggered due to fairness issues"""
        if not fairness_report:
            return False
        
        bias_indicators = fairness_report['bias_indicators']
        
        # Check for severe bias
        severe_bias_threshold = 0.10  # 10% performance gap
        
        if bias_indicators.get('dark_skin_disadvantaged', False):
            accuracy_gap = fairness_report['performance_gaps'].get('accuracy_gap_percentage', 0)
            
            if accuracy_gap > severe_bias_threshold * 100:  # Convert to percentage
                self.logger.warning(f"🚨 SEVERE BIAS DETECTED: {accuracy_gap:.1f}% accuracy gap!")
                self.logger.warning("Consider stopping training and addressing bias issues.")
                return True
        
        return False

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        metrics_calculator = MetricsCalculator(self.config.CLASS_NAMES)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            batch_loss = loss.item()
            metrics_calculator.update(outputs, labels, batch_loss)
            
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
                metrics_calculator.update(outputs, labels, batch_loss)
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})
        
        # Compute epoch metrics
        metrics = metrics_calculator.compute_metrics()
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.MODEL_SAVE_DIR, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved to {best_path}")
        
        # Save last model
        if self.config.SAVE_LAST_MODEL:
            last_path = os.path.join(self.config.MODEL_SAVE_DIR, 'last_model.pth')
            torch.save(checkpoint, last_path)
        
        # Save periodic checkpoint
        if (epoch + 1) % self.config.CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(self.config.MODEL_SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
    
    def print_epoch_summary(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Print detailed epoch summary"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"EPOCH {epoch+1}/{self.config.NUM_EPOCHS} SUMMARY")
        self.logger.info(f"{'='*80}")
        
        # Training metrics
        self.logger.info(f"Training - Loss: {train_metrics['loss']:.4f}, "
                        f"Acc: {train_metrics['accuracy']:.4f}, "
                        f"F1: {train_metrics['f1_score']:.4f}")
        
        # Validation metrics
        self.logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                        f"Acc: {val_metrics['accuracy']:.4f}, "
                        f"F1: {val_metrics['f1_score']:.4f}")
        
        # Per-class accuracy (important for fairness)
        self.logger.info(f"\nPer-Class Accuracy (Validation):")
        for class_name, acc in val_metrics['per_class_accuracy'].items():
            self.logger.info(f"  {class_name}: {acc:.4f}")
        
        # Learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Best model tracking
        if val_metrics['accuracy'] > self.best_val_accuracy:
            self.best_val_accuracy = val_metrics['accuracy']
            self.epochs_without_improvement = 0
            self.logger.info(f"🎉 NEW BEST VALIDATION ACCURACY: {self.best_val_accuracy:.4f}")
        else:
            self.epochs_without_improvement += 1
            self.logger.info(f"Epochs without improvement: {self.epochs_without_improvement}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Configuration: {self.config.__dict__}")
        
        # Load data
        self.load_data()
        
        # Training loop
        for epoch in range(self.config.NUM_EPOCHS):
            start_time = time.time()
            
            # Train and validate
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate scheduler
            self.scheduler.step(val_metrics['accuracy'])
            
            # Save history
            for key, value in train_metrics.items():
                if key != 'per_class_accuracy' and key != 'precision_per_class' and key != 'recall_per_class' and key != 'f1_per_class':
                    self.history[f'train_{key}'].append(value)
            
            for key, value in val_metrics.items():
                if key != 'per_class_accuracy' and key != 'precision_per_class' and key != 'recall_per_class' and key != 'f1_per_class':
                    self.history[f'val_{key}'].append(value)
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_val_accuracy
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Print summary
            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch time: {epoch_time:.2f}s")
            self.print_epoch_summary(epoch, train_metrics, val_metrics)
            
            # Evaluate fairness at intervals
            fairness_report = {}
            if (epoch + 1) % self.config.FAIRNESS_EVAL_INTERVAL == 0:
                fairness_report = self.evaluate_fairness(epoch)
                
                # Check for fairness-aware early stopping
                if self.fairness_aware_early_stopping(fairness_report):
                    self.logger.info("Early stopping triggered due to severe bias detection")
                    break
                
                # Save fairness report
                if fairness_report:
                    self.fairness_evaluator.generate_fairness_report(
                        fairness_report, 
                        f'fairness_report_epoch_{epoch+1}.json'
                    )
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.PATIENCE:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save training history
            self.save_training_history()
            
            # Final evaluation on test set
            self.evaluate_on_test()
        
        # Final fairness evaluation on test set
        if self.config.ENABLE_FAIRNESS_EVALUATION:
            self.logger.info("\n" + "="*80)
            self.logger.info("FINAL FAIRNESS EVALUATION ON TEST SET")
            self.logger.info("="*80)
            
            final_fairness_report = self.evaluate_fairness(self.config.NUM_EPOCHS)
            if final_fairness_report:
                self.fairness_evaluator.generate_fairness_report(
                    final_fairness_report, 
                    'final_fairness_report.json'
                )
        
        self.logger.info("Training completed!")
    
    def evaluate_on_test(self):
        """Final evaluation on test set"""
        self.logger.info("\n" + "="*80)
        self.logger.info("FINAL TEST EVALUATION")
        self.logger.info("="*80)
        
        # Load best model
        best_model_path = os.path.join(self.config.MODEL_SAVE_DIR, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Loaded best model for evaluation")
        
        # Evaluate on test set
        self.model.eval()
        metrics_calculator = MetricsCalculator(self.config.CLASS_NAMES)
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Test Evaluation'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                metrics_calculator.update(outputs, labels, loss.item())
        
        # Compute and print test metrics
        test_metrics = metrics_calculator.compute_metrics()
        
        self.logger.info(f"\nTest Results:")
        self.logger.info(f"Loss: {test_metrics['loss']:.4f}")
        self.logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {test_metrics['precision']:.4f}")
        self.logger.info(f"Recall: {test_metrics['recall']:.4f}")
        self.logger.info(f"F1-Score: {test_metrics['f1_score']:.4f}")
        
        self.logger.info(f"\nPer-Class Test Performance:")
        for class_name in self.config.CLASS_NAMES:
            acc = test_metrics['per_class_accuracy'][class_name]
            prec = test_metrics['precision_per_class'][class_name]
            rec = test_metrics['recall_per_class'][class_name]
            f1 = test_metrics['f1_per_class'][class_name]
            self.logger.info(f"  {class_name}:")
            self.logger.info(f"    Accuracy: {acc:.4f}")
            self.logger.info(f"    Precision: {prec:.4f}")
            self.logger.info(f"    Recall: {rec:.4f}")
            self.logger.info(f"    F1-Score: {f1:.4f}")
        
        # Generate detailed classification report
        y_true = np.array(metrics_calculator.targets)
        y_pred = np.array(metrics_calculator.predictions)
        
        report = classification_report(y_true, y_pred, target_names=self.config.CLASS_NAMES, digits=4)
        self.logger.info(f"\nDetailed Classification Report:\n{report}")
        
        # Save test results
        test_results = {
            'test_metrics': test_metrics,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        with open(os.path.join(self.config.MODEL_SAVE_DIR, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
    
    def save_training_history(self):
        """Save training history to file"""
        history_path = os.path.join(self.config.MODEL_SAVE_DIR, 'training_history.json')
        
        # Convert defaultdict to regular dict for JSON serialization
        history_dict = dict(self.history)
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_path}")
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
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
        axes[1, 0].plot(self.history['train_f1_score'], label='Train F1')
        axes[1, 0].plot(self.history['val_f1_score'], label='Val F1')
        axes[1, 0].set_title('Training and Validation F1-Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Precision and Recall
        axes[1, 1].plot(self.history['train_precision'], label='Train Precision')
        axes[1, 1].plot(self.history['val_precision'], label='Val Precision')
        axes[1, 1].plot(self.history['train_recall'], label='Train Recall')
        axes[1, 1].plot(self.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Precision and Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.MODEL_SAVE_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Training curves plotted and saved")


def main():
    """Main function"""
    config = ModelConfig()
    
    # You can modify configuration here
    config.NUM_EPOCHS = 20
    config.BATCH_SIZE = 32
    config.LEARNING_RATE = 1e-4
    
    trainer = SkinDiseaseTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
