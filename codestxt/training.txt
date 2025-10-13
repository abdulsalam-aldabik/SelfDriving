# -*- coding: utf-8 -*-
"""
Assetto Corsa Autonomous Driving - OPTIMIZED Training Script
RTX 3060 Ti + ResNet50 - Enhanced Performance
"""

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================

from fastai.vision.all import *
from fastai.callback.fp16 import *
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import json

sns.set_style("whitegrid")

# GPU Check
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY!'}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"PyTorch Version: {torch.__version__}")

if not torch.cuda.is_available():
    print("⚠️  WARNING: CUDA not detected! Training will be slow.")
    print("   Check: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()  # Clear cache
    print("\n✓ cuDNN Autotuning: Enabled")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Suppress scientific notation globally
np.set_printoptions(suppress=True, precision=1)
pd.options.display.float_format = '{:.1f}'.format
torch.set_printoptions(precision=1, sci_mode=False)

# ============================================================================
# DATA SETUP
# ============================================================================
print("\n" + "=" * 80)
print("DATA LOADING")
print("=" * 80)

data_path = Path('ac_training_data_cleaned')
csv_path = data_path / 'labels_cleaned.csv'
df = pd.read_csv(csv_path)

print(f"\n✓ Dataset loaded: {len(df)} samples")
print(f"\nData columns: {list(df.columns)}")
print(f"\nStatistics:")
print(df[['steer', 'throttle', 'brake', 'speed']].describe())

# ============================================================================
# DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("DATA DISTRIBUTION ANALYSIS")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram
axes[0].hist(df['steer'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(-0.05, color='r', linestyle='--', label='Threshold')
axes[0].axvline(0.05, color='r', linestyle='--')
axes[0].set_xlabel('Steering Angle')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Steering Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Speed vs Steering
axes[1].scatter(df['speed'], np.abs(df['steer']), alpha=0.3, s=1)
axes[1].set_xlabel('Speed (normalized)')
axes[1].set_ylabel('Abs(Steering)')
axes[1].set_title('Steering vs Speed')
axes[1].grid(True, alpha=0.3)

# Steering categories
df['steer_category'] = pd.cut(df['steer'],
                               bins=[-1.0, -0.05, 0.05, 1.0],
                               labels=['Left', 'Straight', 'Right'])
category_counts = df['steer_category'].value_counts()
axes[2].bar(category_counts.index, category_counts.values, color=['red', 'gray', 'blue'])
axes[2].set_xlabel('Direction')
axes[2].set_ylabel('Count')
axes[2].set_title('Steering Balance')
axes[2].grid(True, alpha=0.3)

for i, (cat, count) in enumerate(category_counts.items()):
    axes[2].text(i, count + 50, f'{count}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show(block=False)

balance_ratio = category_counts.min() / category_counts.max()
print(f"\n📊 Steering Balance Ratio: {balance_ratio:.2f}")
if balance_ratio < 0.7:
    print("⚠️  WARNING: Data imbalance detected!")
else:
    print("✓ Good data balance")

# ============================================================================
# OPTIMIZED DATABLOCK - ENHANCED AUGMENTATION
# ============================================================================
print("\n" + "=" * 80)
print("CREATING DATALOADERS (Optimized for RTX 3060 Ti)")
print("=" * 80)

dblock = DataBlock(
    blocks=(ImageBlock, RegressionBlock(n_out=3)),
    get_x=lambda row: data_path / row['image'].replace('\\', '/'),
    get_y=ColReader(['steer', 'throttle', 'brake']),
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(480),  # Increased from 460
    batch_tfms=[
        *aug_transforms(
            size=224,
            do_flip=False,
            max_rotate=5.0,        # Increased from 3.0
            max_lighting=0.5,      # Increased from 0.4
            max_warp=0.2,          # Increased from 0.15
            min_scale=0.80,        # Decreased from 0.85 (more zoom variation)
            p_affine=0.75,         # Increased from 0.6
            p_lighting=0.85        # Increased from 0.8
        ),
        Brightness(max_lighting=0.4, p=0.8),    # Increased
        Contrast(max_lighting=0.3, p=0.7),      # Increased
        Normalize.from_stats(*imagenet_stats)
    ]
)

# OPTIMIZED: Increased batch size for RTX 3060 Ti (8GB VRAM)
dls = dblock.dataloaders(df, bs=160, num_workers=0, pin_memory=True)

print(f"\n✓ DataLoaders created")
print(f"  Training batches: {len(dls.train)}")
print(f"  Validation batches: {len(dls.valid)}")
print(f"  Batch size: {dls.bs} (optimized for 8GB VRAM)")

dls.show_batch(max_n=6, nrows=2)
plt.show(block=False)

# ============================================================================
# OPTIMIZED LOSS FUNCTION
# ============================================================================
print("\n" + "=" * 80)
print("LOSS FUNCTION (Steering-Focused)")
print("=" * 80)

class SteeringFocusedLoss(nn.Module):
    """Enhanced loss with stronger steering focus."""
    
    def __init__(self, steer_weight=15.0, throttle_weight=0.3, brake_weight=0.3):
        super().__init__()
        self.steer_weight = steer_weight
        self.throttle_weight = throttle_weight
        self.brake_weight = brake_weight
    
    def forward(self, pred, target):
        # Stronger penalty for sharp turns
        turn_magnitude = torch.abs(target[:, 0])
        turn_multiplier = 1.0 + 4.0 * turn_magnitude  # Increased from 3.0
        
        steer_loss = self.steer_weight * turn_multiplier * (pred[:, 0] - target[:, 0])**2
        throttle_loss = self.throttle_weight * (pred[:, 1] - target[:, 1])**2
        brake_loss = self.brake_weight * (pred[:, 2] - target[:, 2])**2
        
        return (steer_loss + throttle_loss + brake_loss).mean()

print("✓ Loss: 15x steering weight + 4x multiplier for sharp turns")

# ============================================================================
# METRICS
# ============================================================================
print("\n" + "=" * 80)
print("METRICS SETUP")
print("=" * 80)

def steering_mae(pred, targ):
    return torch.abs(pred[:, 0] - targ[:, 0]).mean()

def steering_rmse(pred, targ):
    return torch.sqrt(((pred[:, 0] - targ[:, 0])**2).mean())

def steering_r2(pred, targ):
    ss_res = ((targ[:, 0] - pred[:, 0])**2).sum()
    ss_tot = ((targ[:, 0] - targ[:, 0].mean())**2).sum()
    return 1 - (ss_res / ss_tot)

def steering_accuracy_tight(pred, targ, threshold=0.05):
    return (torch.abs(pred[:, 0] - targ[:, 0]) < threshold).float().mean()

def steering_accuracy_loose(pred, targ, threshold=0.15):
    return (torch.abs(pred[:, 0] - targ[:, 0]) < threshold).float().mean()

print("✓ Metrics: MAE, RMSE, R², Accuracy (tight & loose)")

# ============================================================================
# MODEL CREATION - RESNET50
# ============================================================================
print("\n" + "=" * 80)
print("MODEL CREATION (ResNet50)")
print("=" * 80)

learn = vision_learner(
    dls,
    resnet50,  # ✓ Changed from resnet34
    n_out=3,
    loss_func=SteeringFocusedLoss(steer_weight=15.0),
    metrics=[
        mae,
        steering_mae,
        steering_rmse,
        steering_r2,
        steering_accuracy_tight,
        steering_accuracy_loose
    ]
)

# Mixed precision
if torch.cuda.is_available():
    learn = learn.to_fp16()
    print("\n✓ Mixed Precision (FP16): Enabled")
else:
    print("\n⚠️  Mixed Precision: Disabled (CPU mode)")

print("✓ Model: ResNet50 (25M parameters)")
print("  Enhanced: 15x steering focus + 4x turn multiplier")
print("  Metrics: 6 comprehensive metrics")

# ============================================================================
# LEARNING RATE FINDER
# ============================================================================
print("\n" + "=" * 80)
print("LEARNING RATE FINDER")
print("=" * 80)

lr_suggestion = learn.lr_find()
lr_max = lr_suggestion.valley

print(f"\n✓ Suggested LR: {lr_max:.2e}")
plt.show(block=False)

learn.fine_tune(
    epochs=30,                    # epochs for unfrozen training
    base_lr=lr_max,              # maximum learning rate
    freeze_epochs=8,             # epochs for head-only training
    lr_mult=100,                 # discriminative LR ratio (default)
    wd=0.01,                     # weight decay
    pct_start=0.3                # percentage of cycle for LR increase
)


# ============================================================================
# TRAINING HISTORY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING HISTORY")
print("=" * 80)

learn.recorder.plot_loss()
plt.title('Training & Validation Loss')
plt.grid(True, alpha=0.3)
plt.show(block=False)

# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("COMPREHENSIVE MODEL EVALUATION")
print("=" * 80)

preds, targets = learn.get_preds()
preds_np = preds.cpu().numpy()
targets_np = targets.cpu().numpy()

steer_mae_val = mean_absolute_error(targets_np[:, 0], preds_np[:, 0])
steer_rmse_val = np.sqrt(mean_squared_error(targets_np[:, 0], preds_np[:, 0]))
steer_r2_val = r2_score(targets_np[:, 0], preds_np[:, 0])
steer_corr, steer_pvalue = pearsonr(targets_np[:, 0], preds_np[:, 0])

print(f"\n📊 STEERING METRICS:")
print(f"  MAE:              {steer_mae_val:.4f}")
print(f"  RMSE:             {steer_rmse_val:.4f}")
print(f"  R² Score:         {steer_r2_val:.4f}")
print(f"  Pearson Corr:     {steer_corr:.4f} (p={steer_pvalue:.2e})")

print(f"\n🎯 MODEL QUALITY ASSESSMENT:")
if steer_mae_val < 0.05 and steer_r2_val > 0.92:
    print("  ★★★★★ EXCELLENT - Production ready!")
elif steer_mae_val < 0.08 and steer_r2_val > 0.88:
    print("  ★★★★☆ VERY GOOD - Should work well")
elif steer_mae_val < 0.12 and steer_r2_val > 0.80:
    print("  ★★★☆☆ GOOD - Acceptable performance")
elif steer_mae_val < 0.18 and steer_r2_val > 0.70:
    print("  ★★☆☆☆ FAIR - Needs improvement")
else:
    print("  ★☆☆☆☆ POOR - Collect more data and retrain")

throttle_mae_val = mean_absolute_error(targets_np[:, 1], preds_np[:, 1])
brake_mae_val = mean_absolute_error(targets_np[:, 2], preds_np[:, 2])

print(f"\n📊 OTHER OUTPUTS:")
print(f"  Throttle MAE:     {throttle_mae_val:.4f}")
print(f"  Brake MAE:        {brake_mae_val:.4f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED VISUALIZATIONS")
print("=" * 80)

fig = plt.figure(figsize=(16, 10))

# 1. Scatter: Predicted vs Actual
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(targets_np[:, 0], preds_np[:, 0], alpha=0.3, s=2)
ax1.plot([-1, 1], [-1, 1], 'r--', lw=2, label='Perfect')
ax1.set_xlabel('True Steering')
ax1.set_ylabel('Predicted Steering')
ax1.set_title(f'Steering: R²={steer_r2_val:.3f}, MAE={steer_mae_val:.3f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Error distribution
ax2 = plt.subplot(2, 3, 2)
errors = preds_np[:, 0] - targets_np[:, 0]
ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='coral')
ax2.axvline(0, color='r', linestyle='--', lw=2)
ax2.axvline(errors.mean(), color='blue', linestyle='--', lw=2, label=f'Mean={errors.mean():.3f}')
ax2.set_xlabel('Prediction Error')
ax2.set_ylabel('Frequency')
ax2.set_title(f'Error Distribution (Std={errors.std():.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Error vs True Steering
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(targets_np[:, 0], np.abs(errors), alpha=0.3, s=2)
ax3.axhline(0.05, color='r', linestyle='--', label='0.05 target')
ax3.set_xlabel('True Steering')
ax3.set_ylabel('Absolute Error')
ax3.set_title('Error vs Steering')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4-6. Throttle, Brake, Cumulative
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(targets_np[:, 1], preds_np[:, 1], alpha=0.3, s=2, color='green')
ax4.plot([0, 1], [0, 1], 'r--', lw=2)
ax4.set_xlabel('True Throttle')
ax4.set_ylabel('Predicted Throttle')
ax4.set_title(f'Throttle: MAE={throttle_mae_val:.3f}')
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(2, 3, 5)
ax5.scatter(targets_np[:, 2], preds_np[:, 2], alpha=0.3, s=2, color='red')
ax5.plot([0, 1], [0, 1], 'r--', lw=2)
ax5.set_xlabel('True Brake')
ax5.set_ylabel('Predicted Brake')
ax5.set_title(f'Brake: MAE={brake_mae_val:.3f}')
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(2, 3, 6)
sorted_errors = np.sort(np.abs(errors))
cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
ax6.plot(sorted_errors, cumulative, lw=2)
ax6.axvline(0.05, color='g', linestyle='--', label='0.05')
ax6.axvline(0.10, color='orange', linestyle='--', label='0.10')
ax6.set_xlabel('Absolute Error')
ax6.set_ylabel('Cumulative %')
ax6.set_title('Cumulative Error')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=False)

# ============================================================================
# ERROR ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("ERROR ANALYSIS BY TURN SEVERITY")
print("=" * 80)

abs_steer = np.abs(targets_np[:, 0])

categories = [
    ('Straight (<0.1)', abs_steer < 0.1),
    ('Gentle (0.1-0.3)', (abs_steer >= 0.1) & (abs_steer < 0.3)),
    ('Moderate (0.3-0.6)', (abs_steer >= 0.3) & (abs_steer < 0.6)),
    ('Sharp (>0.6)', abs_steer >= 0.6)
]

print(f"\n{'Category':<20} {'Count':>8} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
print("-" * 62)

for name, mask in categories:
    if mask.sum() > 0:
        mae = np.abs(preds_np[mask, 0] - targets_np[mask, 0]).mean()
        rmse = np.sqrt(((preds_np[mask, 0] - targets_np[mask, 0])**2).mean())
        r2 = r2_score(targets_np[mask, 0], preds_np[mask, 0])
        print(f"{name:<20} {mask.sum():>8} {mae:>10.4f} {rmse:>10.4f} {r2:>10.4f}")

# ============================================================================
# DIRECTIONAL BIAS
# ============================================================================
print("\n" + "=" * 80)
print("DIRECTIONAL BIAS ANALYSIS")
print("=" * 80)

left_mask = targets_np[:, 0] < -0.1
right_mask = targets_np[:, 0] > 0.1

if left_mask.sum() > 0 and right_mask.sum() > 0:
    left_bias = (preds_np[left_mask, 0] - targets_np[left_mask, 0]).mean()
    right_bias = (preds_np[right_mask, 0] - targets_np[right_mask, 0]).mean()
    
    print(f"\nLeft turn bias:  {left_bias:+.4f}")
    print(f"Right turn bias: {right_bias:+.4f}")
    
    if abs(left_bias) < 0.03 and abs(right_bias) < 0.03:
        print("✓ Excellent - No significant bias")
    elif abs(left_bias) < 0.05 and abs(right_bias) < 0.05:
        print("✓ Good - Minor bias")
    else:
        print("⚠️  WARNING: Significant bias detected!")

# ============================================================================
# SAMPLE PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS")
print("=" * 80)

sample_indices = np.random.choice(len(df), size=5, replace=False)

print(f"\n{'Image':<40} {'True':>8} {'Pred':>8} {'Error':>8}")
print("-" * 72)

for idx in sample_indices:
    row = df.iloc[idx]
    img_path = data_path / row['image'].replace('\\', '/')
    
    pred_result = learn.predict(PILImage.create(img_path))
    pred_steer = float(pred_result[0][0])
    true_steer = row['steer']
    error = abs(pred_steer - true_steer)
    
    print(f"{row['image'][-35:]:<40} {true_steer:>+8.3f} {pred_steer:>+8.3f} {error:>8.3f}")

# ============================================================================
# SAVE METRICS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING METRICS")
print("=" * 80)

metrics_dict = {
    'steering': {
        'mae': float(steer_mae_val),
        'rmse': float(steer_rmse_val),
        'r2_score': float(steer_r2_val),
        'pearson_correlation': float(steer_corr)
    },
    'throttle': {'mae': float(throttle_mae_val)},
    'brake': {'mae': float(brake_mae_val)},
    'training': {
        'model': 'ResNet50',
        'epochs_phase1': 8,
        'epochs_phase2': 40,
        'batch_size': 192,
        'total_samples': len(df)
    }
}

output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)

metrics_file = output_dir / 'training_metrics.json'
with open(metrics_file, 'w') as f:
    json.dump(metrics_dict, f, indent=2)

print(f"\n✓ Metrics saved to: {metrics_file}")

# ============================================================================
# MODEL EXPORT
# ============================================================================
print("\n" + "=" * 80)
print("MODEL EXPORT")
print("=" * 80)

model_filename = output_dir / 'drive_model_ac_resnet50.pkl'
learn.export(model_filename)

print(f"\n✓ Model saved: {model_filename}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)

print(f"\n📊 FINAL METRICS:")
print(f"  Steering MAE:     {steer_mae_val:.4f} {'✓' if steer_mae_val < 0.08 else '⚠️'}")
print(f"  Steering R²:      {steer_r2_val:.4f} {'✓' if steer_r2_val > 0.85 else '⚠️'}")

print(f"\n📁 SAVED FILES:")
print(f"  • {model_filename}")
print(f"  • {metrics_file}")

print(f"\n🚗 NEXT STEPS:")
print("  1. Review visualizations")
print("  2. Test model on actual track")
print("  3. If MAE > 0.08, collect more diverse data")

print("\n" + "=" * 80)
print("✓ Close all plot windows to exit")
plt.show()
