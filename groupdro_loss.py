#!/usr/bin/env python3
"""
Group Distributionally Robust Optimization (GroupDRO) Loss for Fairness

This module implements GroupDRO loss function to reduce bias in skin disease classification
by assigning higher weights to dark skin samples and penalizing errors more for underrepresented groups.

Author: AI Assistant
Date: 2025-03-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import defaultdict, Counter

# Import existing components
from fairness_evaluator import SkinToneClassifier


class GroupDROConfig:
    """Configuration for GroupDRO loss"""
    
    # GroupDRO parameters
    ETA = 1.0  # Learning rate for group weights
    ALPHA = 0.2  # Robustness parameter (higher = more emphasis on worst groups)
    TAU = 10.0  # Temperature for weight smoothing
    
    # Group definitions
    GROUPS = ['dark_skin', 'light_skin']
    SUBGROUPS = ['dark_skin_acne', 'dark_skin_psoriasis', 'dark_skin_eczema',
                 'light_skin_acne', 'light_skin_psoriasis', 'light_skin_eczema']
    
    # Weighting strategy
    DARK_SKIN_WEIGHT = 2.0  # Base weight multiplier for dark skin
    MINORITY_CLASS_WEIGHT = 1.5  # Additional weight for minority classes
    
    # Fairness targets
    TARGET_FAIRNESS_GAP = 0.02  # Target maximum accuracy gap (2%)
    BALANCED_ACCURACY_WEIGHT = 0.3  # Weight for balanced accuracy in loss
    
    # Logging and tracking
    TRACK_GROUP_METRICS = True
    LOG_INTERVAL = 100


class GroupDROLoss(nn.Module):
    """
    Group Distributionally Robust Optimization Loss
    
    This loss function implements GroupDRO to reduce bias by:
    1. Assigning higher weights to dark skin samples
    2. Penalizing errors more for underrepresented groups
    3. Optimizing for worst-group performance
    """
    
    def __init__(self, config: GroupDROConfig = GroupDROConfig(), num_classes: int = 3):
        super(GroupDROLoss, self).__init__()
        
        self.config = config
        self.num_classes = num_classes
        
        # Initialize group weights
        self.register_buffer('group_weights', torch.ones(len(config.GROUPS)))
        self.register_buffer('subgroup_weights', torch.ones(len(config.SUBGROUPS)))
        
        # Track group statistics
        self.register_buffer('group_losses', torch.zeros(len(config.GROUPS)))
        self.register_buffer('group_counts', torch.zeros(len(config.GROUPS)))
        self.register_buffer('subgroup_losses', torch.zeros(len(config.SUBGROUPS)))
        self.register_buffer('subgroup_counts', torch.zeros(len(config.SUBGROUPS)))
        
        # Step count for learning rate scheduling
        self.register_buffer('step_count', torch.zeros(1))
        
        # Skin tone classifier for group assignment
        self.skin_tone_classifier = SkinToneClassifier()
        
        # Group to index mapping
        self.group_to_idx = {group: idx for idx, group in enumerate(config.GROUPS)}
        self.subgroup_to_idx = {subgroup: idx for idx, subgroup in enumerate(config.SUBGROUPS)}
        
        # Class names
        self.class_names = ['acne', 'psoriasis', 'eczema']
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # History tracking
        self.history = {
            'group_weights': [],
            'group_losses': [],
            'group_accuracies': [],
            'fairness_gap': [],
            'balanced_accuracy': []
        }
    
    def assign_groups(self, images: torch.Tensor, labels: torch.Tensor, 
                     image_paths: Optional[List[str]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign samples to groups based on skin tone and class
        
        Args:
            images: Batch of images
            labels: Batch of labels
            image_paths: Optional image paths for skin tone classification
            
        Returns:
            Tuple of (group_indices, subgroup_indices)
        """
        batch_size = images.shape[0]
        group_indices = torch.zeros(batch_size, dtype=torch.long)
        subgroup_indices = torch.zeros(batch_size, dtype=torch.long)
        
        for i in range(batch_size):
            label = labels[i].item()
            class_name = self.class_names[label]
            
            # Determine skin tone
            if image_paths and i < len(image_paths):
                skin_tone = self.skin_tone_classifier.classify_skin_tone(image_paths[i])
            else:
                # Fallback: estimate from image statistics (less accurate)
                skin_tone = self._estimate_skin_tone_from_image(images[i])
            
            # Assign group indices
            if skin_tone == 'dark':
                group_idx = self.group_to_idx['dark_skin']
            else:
                group_idx = self.group_to_idx['light_skin']
            
            # Assign subgroup indices
            subgroup_name = f"{skin_tone}_skin_{class_name}"
            subgroup_idx = self.subgroup_to_idx[subgroup_name]
            
            group_indices[i] = group_idx
            subgroup_indices[i] = subgroup_idx
        
        return group_indices, subgroup_indices
    
    def _estimate_skin_tone_from_image(self, image: torch.Tensor) -> str:
        """
        Estimate skin tone from image statistics (fallback method)
        
        Args:
            image: Single image tensor
            
        Returns:
            'dark' or 'light' skin tone classification
        """
        # Convert to numpy and calculate brightness
        img_np = image.cpu().numpy()
        
        # Calculate average brightness (simple heuristic)
        if img_np.ndim == 3:  # CHW format
            brightness = np.mean(img_np)
        else:  # HWC format
            brightness = np.mean(img_np)
        
        # Simple threshold for skin tone classification
        if brightness < 0.4:  # Dark threshold
            return 'dark'
        else:
            return 'light'
    
    def compute_group_losses(self, logits: torch.Tensor, labels: torch.Tensor, 
                           group_indices: torch.Tensor, subgroup_indices: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute losses for each group/subgroup
        
        Args:
            logits: Model predictions
            labels: Ground truth labels
            group_indices: Group assignments
            subgroup_indices: Subgroup assignments
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Compute standard cross-entropy loss
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # Initialize group losses
        group_losses = torch.zeros(len(self.config.GROUPS), device=logits.device)
        subgroup_losses = torch.zeros(len(self.config.SUBGROUPS), device=logits.device)
        
        # Accumulate losses per group
        for i in range(len(labels)):
            group_idx = group_indices[i]
            subgroup_idx = subgroup_indices[i]
            
            group_losses[group_idx] += ce_loss[i]
            subgroup_losses[subgroup_idx] += ce_loss[i]
        
        # Update counts
        for group_idx in range(len(self.config.GROUPS)):
            count = (group_indices == group_idx).sum().float()
            self.group_counts[group_idx] += count
            
            if count > 0:
                self.group_losses[group_idx] = group_losses[group_idx] / count
        
        for subgroup_idx in range(len(self.config.SUBGROUPS)):
            count = (subgroup_indices == subgroup_idx).sum().float()
            self.subgroup_counts[subgroup_idx] += count
            
            if count > 0:
                self.subgroup_losses[subgroup_idx] = subgroup_losses[subgroup_idx] / count
        
        # Apply group weights
        weighted_losses = torch.zeros_like(group_losses)
        for group_idx in range(len(self.config.GROUPS)):
            weighted_losses[group_idx] = group_losses[group_idx] * self.group_weights[group_idx]
        
        total_loss = weighted_losses.sum()
        
        # Create loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'group_losses': self.group_losses.clone(),
            'subgroup_losses': self.subgroup_losses.clone(),
            'group_weights': self.group_weights.clone(),
            'subgroup_weights': self.subgroup_weights.clone(),
            'group_counts': self.group_counts.clone(),
            'subgroup_counts': self.subgroup_counts.clone()
        }
        
        return total_loss, loss_dict
    
    def update_group_weights(self, loss_dict: Dict):
        """
        Update group weights using GroupDRO algorithm
        
        Args:
            loss_dict: Dictionary containing loss information
        """
        # Update step count
        self.step_count += 1
        
        # Get current group losses
        group_losses = loss_dict['group_losses']
        
        # Compute worst-group loss
        worst_loss = torch.max(group_losses)
        
        # Update group weights using exponential moving average
        for group_idx in range(len(self.config.GROUPS)):
            if self.group_counts[group_idx] > 0:
                # GroupDRO weight update
                weight_update = torch.exp(self.config.ETA * group_losses[group_idx] / worst_loss)
                self.group_weights[group_idx] = (1 - self.config.ETA) * self.group_weights[group_idx] + self.config.ETA * weight_update
        
        # Apply temperature smoothing
        if self.config.TAU > 0:
            self.group_weights = F.softmax(self.group_weights * self.config.TAU, dim=0)
        
        # Normalize weights
        self.group_weights = self.group_weights / self.group_weights.sum()
        
        # Apply additional weighting for dark skin
        dark_skin_idx = self.group_to_idx['dark_skin']
        self.group_weights[dark_skin_idx] *= self.config.DARK_SKIN_WEIGHT
        
        # Renormalize
        self.group_weights = self.group_weights / self.group_weights.sum()
    
    def compute_balanced_accuracy(self, logits: torch.Tensor, labels: torch.Tensor, 
                                group_indices: torch.Tensor) -> Dict[str, float]:
        """
        Compute balanced accuracy metrics
        
        Args:
            logits: Model predictions
            labels: Ground truth labels
            group_indices: Group assignments
            
        Returns:
            Dictionary of balanced accuracy metrics
        """
        predictions = torch.argmax(logits, dim=1)
        
        # Overall accuracy
        overall_acc = (predictions == labels).float().mean().item()
        
        # Group-wise accuracy
        group_accuracies = {}
        for group_name, group_idx in self.group_to_idx.items():
            mask = group_indices == group_idx
            if mask.sum() > 0:
                group_acc = (predictions[mask] == labels[mask]).float().mean().item()
                group_accuracies[group_name] = group_acc
            else:
                group_accuracies[group_name] = 0.0
        
        # Balanced accuracy (average of group accuracies)
        balanced_acc = np.mean(list(group_accuracies.values()))
        
        # Fairness gap
        if 'dark_skin' in group_accuracies and 'light_skin' in group_accuracies:
            fairness_gap = abs(group_accuracies['dark_skin'] - group_accuracies['light_skin'])
        else:
            fairness_gap = 0.0
        
        return {
            'overall_accuracy': overall_acc,
            'balanced_accuracy': balanced_acc,
            'group_accuracies': group_accuracies,
            'fairness_gap': fairness_gap
        }
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, 
                image_paths: Optional[List[str]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass of GroupDRO loss
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            labels: Ground truth labels (batch_size,)
            image_paths: Optional image paths for skin tone classification
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Assign groups
        group_indices, subgroup_indices = self.assign_groups(logits, labels, image_paths)
        
        # Compute group losses
        total_loss, loss_dict = self.compute_group_losses(logits, labels, group_indices, subgroup_indices)
        
        # Update group weights
        self.update_group_weights(loss_dict)
        
        # Compute balanced accuracy
        accuracy_metrics = self.compute_balanced_accuracy(logits, labels, group_indices)
        
        # Combine with standard cross-entropy for stability
        standard_loss = F.cross_entropy(logits, labels)
        combined_loss = (1 - self.config.BALANCED_ACCURACY_WEIGHT) * standard_loss + \
                       self.config.BALANCED_ACCURACY_WEIGHT * total_loss
        
        # Create comprehensive metrics dictionary
        metrics_dict = {
            'loss': combined_loss.item(),
            'standard_loss': standard_loss.item(),
            'groupdro_loss': total_loss.item(),
            'accuracy_metrics': accuracy_metrics,
            'group_weights': self.group_weights.clone(),
            'group_losses': self.group_losses.clone(),
            'group_counts': self.group_counts.clone(),
            'step': self.step_count.item()
        }
        
        return combined_loss, metrics_dict
    
    def get_fairness_metrics(self) -> Dict:
        """
        Get comprehensive fairness metrics
        
        Returns:
            Dictionary of fairness metrics
        """
        # Calculate current fairness metrics
        if self.group_counts.sum() > 0:
            group_accuracies = {}
            for group_name, group_idx in self.group_to_idx.items():
                if self.group_counts[group_idx] > 0:
                    # Approximate accuracy from losses (inverse relationship)
                    # This is a rough estimate; actual accuracy should be computed during evaluation
                    loss = self.group_losses[group_idx].item()
                    accuracy = max(0.0, 1.0 - loss)  # Rough approximation
                    group_accuracies[group_name] = accuracy
                else:
                    group_accuracies[group_name] = 0.0
            
            # Fairness gap
            if 'dark_skin' in group_accuracies and 'light_skin' in group_accuracies:
                fairness_gap = abs(group_accuracies['dark_skin'] - group_accuracies['light_skin'])
            else:
                fairness_gap = 0.0
            
            # Balanced accuracy
            balanced_acc = np.mean(list(group_accuracies.values()))
            
            return {
                'group_accuracies': group_accuracies,
                'fairness_gap': fairness_gap,
                'balanced_accuracy': balanced_acc,
                'group_weights': self.group_weights.tolist(),
                'dark_skin_weight': self.group_weights[self.group_to_idx['dark_skin']].item(),
                'light_skin_weight': self.group_weights[self.group_to_idx['light_skin']].item(),
                'weight_ratio': self.group_weights[self.group_to_idx['dark_skin']].item() / 
                              max(self.group_weights[self.group_to_idx['light_skin']].item(), 1e-6)
            }
        else:
            return {
                'group_accuracies': {},
                'fairness_gap': 0.0,
                'balanced_accuracy': 0.0,
                'group_weights': self.group_weights.tolist(),
                'dark_skin_weight': 0.0,
                'light_skin_weight': 0.0,
                'weight_ratio': 1.0
            }
    
    def reset_statistics(self):
        """Reset accumulated statistics"""
        self.group_losses.zero_()
        self.group_counts.zero_()
        self.subgroup_losses.zero_()
        self.subgroup_counts.zero_()
        self.step_count.zero_()
        
        # Reset weights to uniform
        self.group_weights.fill_(1.0 / len(self.config.GROUPS))
        self.subgroup_weights.fill_(1.0 / len(self.config.SUBGROUPS))
        
        # Clear history
        self.history = {
            'group_weights': [],
            'group_losses': [],
            'group_accuracies': [],
            'fairness_gap': [],
            'balanced_accuracy': []
        }


class FairnessTracker:
    """Tracker for fairness metrics during training"""
    
    def __init__(self, config: GroupDROConfig = GroupDROConfig()):
        self.config = config
        self.metrics_history = {
            'epoch': [],
            'overall_accuracy': [],
            'balanced_accuracy': [],
            'dark_skin_accuracy': [],
            'light_skin_accuracy': [],
            'fairness_gap': [],
            'dark_skin_weight': [],
            'light_skin_weight': [],
            'weight_ratio': []
        }
        
        self.logger = logging.getLogger(__name__)
    
    def update(self, epoch: int, metrics: Dict):
        """Update metrics history"""
        accuracy_metrics = metrics.get('accuracy_metrics', {})
        
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['overall_accuracy'].append(accuracy_metrics.get('overall_accuracy', 0.0))
        self.metrics_history['balanced_accuracy'].append(accuracy_metrics.get('balanced_accuracy', 0.0))
        self.metrics_history['dark_skin_accuracy'].append(accuracy_metrics.get('group_accuracies', {}).get('dark_skin', 0.0))
        self.metrics_history['light_skin_accuracy'].append(accuracy_metrics.get('group_accuracies', {}).get('light_skin', 0.0))
        self.metrics_history['fairness_gap'].append(accuracy_metrics.get('fairness_gap', 0.0))
        
        fairness_metrics = metrics.get('fairness_metrics', {})
        self.metrics_history['dark_skin_weight'].append(fairness_metrics.get('dark_skin_weight', 0.0))
        self.metrics_history['light_skin_weight'].append(fairness_metrics.get('light_skin_weight', 0.0))
        self.metrics_history['weight_ratio'].append(fairness_metrics.get('weight_ratio', 1.0))
    
    def print_epoch_summary(self, epoch: int):
        """Print fairness summary for the epoch"""
        if len(self.metrics_history['epoch']) == 0:
            return
        
        idx = -1  # Latest epoch
        
        print(f"\n{'='*60}")
        print(f"FAIRNESS METRICS - EPOCH {epoch}")
        print(f"{'='*60}")
        
        overall_acc = self.metrics_history['overall_accuracy'][idx]
        balanced_acc = self.metrics_history['balanced_accuracy'][idx]
        dark_acc = self.metrics_history['dark_skin_accuracy'][idx]
        light_acc = self.metrics_history['light_skin_accuracy'][idx]
        fairness_gap = self.metrics_history['fairness_gap'][idx]
        dark_weight = self.metrics_history['dark_skin_weight'][idx]
        light_weight = self.metrics_history['light_skin_accuracy'][idx]
        weight_ratio = self.metrics_history['weight_ratio'][idx]
        
        print(f"Overall Accuracy:     {overall_acc:.4f}")
        print(f"Balanced Accuracy:    {balanced_acc:.4f}")
        print(f"Dark Skin Accuracy:    {dark_acc:.4f}")
        print(f"Light Skin Accuracy:   {light_acc:.4f}")
        print(f"Fairness Gap:          {fairness_gap:.4f}")
        print(f"Dark Skin Weight:      {dark_weight:.4f}")
        print(f"Light Skin Weight:     {light_weight:.4f}")
        print(f"Weight Ratio (D/L):    {weight_ratio:.2f}")
        
        # Check if fairness target is met
        if fairness_gap <= self.config.TARGET_FAIRNESS_GAP:
            print(f"✅ FAIRNESS TARGET MET: Gap ≤ {self.config.TARGET_FAIRNESS_GAP:.2f}")
        else:
            print(f"⚠️  FAIRNESS TARGET NOT MET: Gap > {self.config.TARGET_FAIRNESS_GAP:.2f}")
        
        print(f"{'='*60}")
    
    def get_final_report(self) -> Dict:
        """Generate final fairness report"""
        if len(self.metrics_history['epoch']) == 0:
            return {}
        
        final_metrics = {}
        
        # Final values
        final_metrics['final_epoch'] = self.metrics_history['epoch'][-1]
        final_metrics['final_overall_accuracy'] = self.metrics_history['overall_accuracy'][-1]
        final_metrics['final_balanced_accuracy'] = self.metrics_history['balanced_accuracy'][-1]
        final_metrics['final_dark_skin_accuracy'] = self.metrics_history['dark_skin_accuracy'][-1]
        final_metrics['final_light_skin_accuracy'] = self.metrics_history['light_skin_accuracy'][-1]
        final_metrics['final_fairness_gap'] = self.metrics_history['fairness_gap'][-1]
        final_metrics['final_weight_ratio'] = self.metrics_history['weight_ratio'][-1]
        
        # Best values
        final_metrics['best_balanced_accuracy'] = max(self.metrics_history['balanced_accuracy'])
        final_metrics['best_fairness_gap'] = min(self.metrics_history['fairness_gap'])
        
        # Improvement metrics
        if len(self.metrics_history['fairness_gap']) > 1:
            initial_gap = self.metrics_history['fairness_gap'][0]
            final_gap = self.metrics_history['fairness_gap'][-1]
            final_metrics['fairness_gap_improvement'] = initial_gap - final_gap
        
        return final_metrics
    
    def plot_fairness_progress(self, save_path: str = 'fairness_progress.png'):
        """Plot fairness metrics over training"""
        import matplotlib.pyplot as plt
        
        if len(self.metrics_history['epoch']) == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('GroupDRO Fairness Metrics Progress', fontsize=16, fontweight='bold')
        
        epochs = self.metrics_history['epoch']
        
        # 1. Accuracy comparison
        axes[0, 0].plot(epochs, self.metrics_history['overall_accuracy'], label='Overall', color='blue')
        axes[0, 0].plot(epochs, self.metrics_history['balanced_accuracy'], label='Balanced', color='green')
        axes[0, 0].set_title('Accuracy Progress')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Skin tone accuracy
        axes[0, 1].plot(epochs, self.metrics_history['dark_skin_accuracy'], label='Dark Skin', color='brown')
        axes[0, 1].plot(epochs, self.metrics_history['light_skin_accuracy'], label='Light Skin', color='orange')
        axes[0, 1].set_title('Skin Tone Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Fairness gap
        axes[0, 2].plot(epochs, self.metrics_history['fairness_gap'], color='red')
        axes[0, 2].axhline(y=self.config.TARGET_FAIRNESS_GAP, color='green', linestyle='--', 
                           label=f'Target ({self.config.TARGET_FAIRNESS_GAP})')
        axes[0, 2].set_title('Fairness Gap (Dark vs Light)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy Gap')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Group weights
        axes[1, 0].plot(epochs, self.metrics_history['dark_skin_weight'], label='Dark Skin', color='brown')
        axes[1, 0].plot(epochs, self.metrics_history['light_skin_weight'], label='Light Skin', color='orange')
        axes[1, 0].set_title('Group Weights')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Weight')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Weight ratio
        axes[1, 1].plot(epochs, self.metrics_history['weight_ratio'], color='purple')
        axes[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal Weight')
        axes[1, 1].set_title('Weight Ratio (Dark/Light)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Summary statistics
        if len(epochs) > 0:
            summary_text = f"FINAL METRICS (Epoch {epochs[-1]}):\n\n"
            summary_text += f"Overall Accuracy: {self.metrics_history['overall_accuracy'][-1]:.4f}\n"
            summary_text += f"Balanced Accuracy: {self.metrics_history['balanced_accuracy'][-1]:.4f}\n"
            summary_text += f"Dark Skin Accuracy: {self.metrics_history['dark_skin_accuracy'][-1]:.4f}\n"
            summary_text += f"Light Skin Accuracy: {self.metrics_history['light_skin_accuracy'][-1]:.4f}\n"
            summary_text += f"Fairness Gap: {self.metrics_history['fairness_gap'][-1]:.4f}\n"
            summary_text += f"Weight Ratio: {self.metrics_history['weight_ratio'][-1]:.2f}\n\n"
            
            if self.metrics_history['fairness_gap'][-1] <= self.config.TARGET_FAIRNESS_GAP:
                summary_text += "✅ FAIRNESS TARGET ACHIEVED!"
            else:
                summary_text += "⚠️  FAIRNESS TARGET NOT MET"
            
            axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1, 2].set_title('Final Summary')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Fairness progress plot saved to {save_path}")


def main():
    """Test GroupDRO loss functionality"""
    print("="*60)
    print("GROUPDRO LOSS FOR FAIRNESS TESTING")
    print("="*60)
    
    # Create test data
    config = GroupDROConfig()
    groupdro_loss = GroupDROLoss(config)
    
    # Test with dummy data
    batch_size = 8
    num_classes = 3
    
    # Create dummy logits and labels
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test forward pass
    loss, metrics = groupdro_loss(logits, labels)
    
    print(f"✅ GroupDRO loss test successful!")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Group weights: {metrics['group_weights']}")
    print(f"   Fairness metrics: {metrics['accuracy_metrics']}")
    
    # Test multiple steps
    print("\nTesting weight updates...")
    for step in range(5):
        loss, metrics = groupdro_loss(logits, labels)
        fairness_metrics = groupdro_loss.get_fairness_metrics()
        print(f"   Step {step+1}: Dark weight = {fairness_metrics['dark_skin_weight']:.4f}, "
              f"Fairness gap = {fairness_metrics['fairness_gap']:.4f}")
    
    print("\n✅ GroupDRO loss implementation complete!")


if __name__ == "__main__":
    main()
