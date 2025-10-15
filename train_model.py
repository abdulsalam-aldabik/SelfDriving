# -*- coding: utf-8 -*-
"""
Assetto Corsa Autonomous Driving - STEERING ONLY
Pure steering regression with fastai's built-in MSE loss
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
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from scipy.stats import pearsonr
import json
from fastai.metrics import mae, rmse, R2Score

sns.set_style("whitegrid")

# GPU Check
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY!'}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Suppress scientific notation
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format
torch.set_printoptions(precision=4, sci_mode=False)

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
print(f"\nStatistics:")
print(df[['steer', 'throttle', 'brake', 'speed']].describe())

# ============================================================================
# DATA VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("DATA DISTRIBUTION VALIDATION")
print("=" * 80)

# Check steering range
print(f"\nSteering value range:")
print(f"  Min: {df['steer'].min():.4f}")
print(f"  Max: {df['steer'].max():.4f}")
print(f"  Mean: {df['steer'].mean():.4f}")
print(f"  Std: {df['steer'].std():.4f}")

# Check straight vs turn distribution
straight_threshold = 0.05
straight_count = ((df['steer'].abs() <= straight_threshold).sum())
turn_count = len(df) - straight_count
straight_pct = straight_count / len(df) * 100

print(f"\nStraight vs Turn (threshold ±{straight_threshold}):")
print(f"  Straight: {straight_count} ({straight_pct:.2f}%)")
print(f"  Turns: {turn_count} ({100-straight_pct:.2f}%)")

if straight_pct > 25.0:
    print(f"\n⚠️  WARNING: Straight data >{25}%! Model will likely fail.")
    print(f"  Recommendation: Re-run cleaning with STRAIGHT_PERCENTAGE = 0.15")
elif straight_pct > 20.0:
    print(f"\n⚠️  CAUTION: Straight data is high. Monitor for straight bias.")
else:
    print(f"\n✓ Straight percentage is acceptable.")

# ============================================================================
# STEERING CLASSIFICATION CATEGORIES (5 CLASSES)
# ============================================================================
print("\n" + "=" * 80)
print("STEERING CATEGORIZATION (5 Classes)")
print("=" * 80)

def categorize_steering(steer_value):
    """Convert continuous steering to 5 categories"""
    if steer_value < -0.25:
        return 0  # Sharp Left
    elif steer_value < -0.05:
        return 1  # Left
    elif steer_value <= 0.05:
        return 2  # Straight
    elif steer_value <= 0.25:
        return 3  # Right
    else:
        return 4  # Sharp Right

# Add category column
df['steer_category'] = df['steer'].apply(categorize_steering)

# Category names for visualization
category_names = ['Sharp Left', 'Left', 'Straight', 'Right', 'Sharp Right']
category_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

# Display distribution
print("\nSteering Category Distribution:")
for i, name in enumerate(category_names):
    count = (df['steer_category'] == i).sum()
    percentage = count / len(df) * 100
    print(f"  {i} - {name:12s}: {count:6d} ({percentage:5.2f}%)")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram with category boundaries
axes[0].hist(df['steer'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(-0.25, color='red', linestyle='--', linewidth=2, label='Category Boundaries')
axes[0].axvline(-0.05, color='red', linestyle='--', linewidth=2)
axes[0].axvline(0.05, color='red', linestyle='--', linewidth=2)
axes[0].axvline(0.25, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Steering Angle', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Steering Distribution with Category Boundaries', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bar chart
category_counts = df['steer_category'].value_counts().sort_index()
bars = axes[1].bar(range(5), category_counts.values, color=category_colors, edgecolor='black', linewidth=1.5)
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(category_names, rotation=45, ha='right')
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Steering Category Distribution', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# Add count labels on bars
for i, (bar, count) in enumerate(zip(bars, category_counts.values)):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('output/steering_categories.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: output/steering_categories.png")
plt.show(block=False)
plt.close()

# ============================================================================
# DATABLOCK SETUP - STEERING ONLY
# ============================================================================
print("\n" + "=" * 80)
print("CREATING DATALOADERS (STEERING ONLY)")
print("=" * 80)

dblock = DataBlock(
    blocks=(ImageBlock, RegressionBlock(n_out=1)),
    get_x=lambda row: data_path / row['image'].replace('\\', '/'),
    get_y=lambda row: row['steer'],
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(480),
    batch_tfms=[
        *aug_transforms(
            size=224,
            do_flip=False,
            max_rotate=10.0,
            max_lighting=0.9,
            max_warp=0.35,
            min_scale=0.60,
            p_affine=0.85,
            p_lighting=0.90
        ),
        Brightness(max_lighting=0.5, p=0.9),
        Contrast(max_lighting=0.4, p=0.8),
        Normalize.from_stats(*imagenet_stats)
    ]
)

dls = dblock.dataloaders(df, bs=32, num_workers=0, pin_memory=True)

print(f"\n✓ DataLoaders created")
print(f"  Training batches: {len(dls.train)}")
print(f"  Validation batches: {len(dls.valid)}")
print(f"  Batch size: {dls.bs}")

# ============================================================================
# METRICS - STEERING ONLY
# ============================================================================

# 1. MAE - Built-in (Mean Absolute Error)
# Already handles flattening automatically
steering_mae = mae

# 2. RMSE - Built-in (Root Mean Squared Error)
# Already handles flattening automatically
steering_rmse = rmse

# 3. R² - Built-in
# Already handles flattening automatically
steering_r2 = R2Score()
def steering_accuracy_tight(pred, targ):
    """Percentage of predictions within ±0.05 of target"""
    pred, targ = flatten_check(pred, targ)
    return (torch.abs(pred - targ) < 0.05).float().mean()

def steering_accuracy_loose(pred, targ):
    """Percentage of predictions within ±0.15 of target"""
    pred, targ = flatten_check(pred, targ)
    return (torch.abs(pred - targ) < 0.15).float().mean()

# ============================================================================
# MODEL CREATION WITH FASTAI'S MSELossFlat
# ============================================================================
print("\n" + "=" * 80)
print("MODEL CREATION (ResNet50 - STEERING ONLY)")
print("=" * 80)

output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)

# Use fastai's built-in MSELossFlat for regression
learn = vision_learner(
    dls,
    resnet50,
    n_out=1,
    loss_func=MSELossFlat(),  # ✓ fastai built-in MSE loss
    metrics=[
        steering_mae,
        steering_rmse,
        steering_r2,
        steering_accuracy_tight,
        steering_accuracy_loose
    ]
)

if torch.cuda.is_available():
    learn = learn.to_fp16()
    print("\n✓ Mixed Precision (FP16): Enabled")
else:
    print("\n⚠️  Mixed Precision: Disabled (CPU mode)")

print("✓ Model: ResNet50")
print("✓ Output: Steering ONLY (single value)")
print("✓ Loss: MSELossFlat() [fastai built-in]")

# ============================================================================
# DATA INTEGRITY CHECKS
# ============================================================================
print("\n" + "=" * 80)
print("CRITICAL DATA CHECKS")
print("=" * 80)

# Verify straight percentage
straight_mask = df['steer'].abs() <= 0.05
actual_straight_pct = straight_mask.sum() / len(df) * 100

print(f"\n1. Straight percentage check:")
print(f"   Actual: {actual_straight_pct:.2f}%")
if actual_straight_pct > 20:
    print(f"   ⚠️  CRITICAL: Still >20%! Data cleaning failed!")
    print(f"   Action: Stop training, re-run cleaning.py")
else:
    print(f"   ✓ Acceptable")

# Check data integrity
print(f"\n2. Data integrity check:")
print(f"   NaN values: {df['steer'].isna().sum()}")
print(f"   Infinite values: {np.isinf(df['steer']).sum()}")
print(f"   Out of range (|x|>1): {(df['steer'].abs() > 1).sum()}")

if df['steer'].isna().sum() > 0 or np.isinf(df['steer']).sum() > 0:
    print(f"   ⚠️  CRITICAL: Data is corrupted!")
else:
    print(f"   ✓ Data integrity OK")

# Sample a batch and verify DataLoader
print(f"\n3. DataLoader check:")
xb, yb = dls.one_batch()
print(f"   Batch shape: x={xb.shape}, y={yb.shape}")
print(f"   Target range: [{yb.min():.4f}, {yb.max():.4f}]")
print(f"   Target mean: {yb.mean():.4f}")
print(f"   Target std: {yb.std():.4f}")

if yb.mean().abs() > 0.15:
    print(f"   ⚠️  WARNING: Target mean not near 0!")
else:
    print(f"   ✓ DataLoader OK")

# ============================================================================
# TRAINING WITH GRADIENT CLIPPING
# ============================================================================
print("\n" + "=" * 80)
print("LEARNING RATE FINDER")
print("=" * 80)

lr_suggestion = learn.lr_find()
lr_max = lr_suggestion.valley

# Safety check for learning rate
if lr_max > 0.01:
    print(f"\n⚠️  Suggested LR ({lr_max:.2e}) seems high, using conservative 5e-3")
    lr_max = 5e-3
else:
    print(f"\n✓ Using suggested LR: {lr_max:.2e}")

plt.savefig('output/lr_finder.png', dpi=150, bbox_inches='tight')
plt.show(block=False)
plt.close()

print("\n" + "=" * 80)
print("TRAINING WITH GRADIENT CLIPPING")
print("=" * 80)

# Import gradient clipping callback
from fastai.callback.training import GradientClip

learn.fine_tune(
    epochs=15,
    base_lr=lr_max,
    freeze_epochs=3,  # Increased from 8
    wd=0.01,  # Reduced from 0.02
    pct_start=0.3,
    cbs=[GradientClip(1.0)]  # Prevent gradient explosions
)

# Save training history
learn.recorder.plot_loss()
plt.title('Training & Validation Loss (Steering Only)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('output/training_loss.png', dpi=150, bbox_inches='tight')
plt.show(block=False)
plt.close()

# ============================================================================
# GET PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

preds, targets = learn.get_preds()
preds_np = preds.cpu().numpy().squeeze()
targets_np = targets.cpu().numpy().squeeze()

# Convert predictions to categories
pred_categories = np.array([categorize_steering(s) for s in preds_np])
true_categories = np.array([categorize_steering(s) for s in targets_np])

print(f"\n✓ Generated {len(preds_np)} predictions")
print(f"✓ Converted to 5 steering categories")

# ============================================================================
# REGRESSION METRICS
# ============================================================================
print("\n" + "=" * 80)
print("REGRESSION METRICS (Steering)")
print("=" * 80)

steer_mae_val = mean_absolute_error(targets_np, preds_np)
steer_rmse_val = np.sqrt(mean_squared_error(targets_np, preds_np))
steer_r2_val = r2_score(targets_np, preds_np)
steer_corr, steer_pvalue = pearsonr(targets_np, preds_np)

print(f"\nSteering Regression Metrics:")
print(f"  MAE:              {steer_mae_val:.4f}")
print(f"  RMSE:             {steer_rmse_val:.4f}")
print(f"  R² Score:         {steer_r2_val:.4f}")
print(f"  Pearson Corr:     {steer_corr:.4f} (p={steer_pvalue:.2e})")

# Quality assessment
if steer_mae_val < 0.06:
    print("\n✓ EXCELLENT: MAE < 0.06")
elif steer_mae_val < 0.08:
    print("\n✓ GOOD: MAE < 0.08")
elif steer_mae_val < 0.10:
    print("\n⚠️  ACCEPTABLE: MAE < 0.10")
else:
    print("\n⚠️  POOR: MAE > 0.10 - Model needs improvement")

# ============================================================================
# CONFUSION MATRIX (5 CLASSES)
# ============================================================================
print("\n" + "=" * 80)
print("CONFUSION MATRIX (5-Class Steering)")
print("=" * 80)

cm = confusion_matrix(true_categories, pred_categories)

# Calculate accuracy
classification_accuracy = np.trace(cm) / np.sum(cm)
print(f"\nClassification Accuracy: {classification_accuracy:.4f} ({classification_accuracy*100:.2f}%)")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(true_categories, pred_categories, 
                          target_names=category_names, digits=4))

# Visualize confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=category_names, yticklabels=category_names,
            cbar_kws={'label': 'Count'}, ax=axes[0], square=True)
axes[0].set_xlabel('Predicted Category', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Category', fontsize=12, fontweight='bold')
axes[0].set_title(f'Confusion Matrix (Counts)\nAccuracy: {classification_accuracy:.4f}', 
                 fontsize=14, fontweight='bold')

# Normalized (percentages)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=category_names, yticklabels=category_names,
            cbar_kws={'label': 'Percentage'}, ax=axes[1], square=True)
axes[1].set_xlabel('Predicted Category', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Category', fontsize=12, fontweight='bold')
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('output/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: output/confusion_matrix.png")
plt.show(block=False)
plt.close()

# ============================================================================
# ROC CURVES
# ============================================================================
print("\n" + "=" * 80)
print("ROC CURVES (One-vs-Rest)")
print("=" * 80)

from sklearn.preprocessing import label_binarize
true_categories_bin = label_binarize(true_categories, classes=range(5))

# Calculate soft probabilities
category_centers = np.array([-0.5, -0.15, 0.0, 0.15, 0.5])

def calculate_soft_probabilities(pred_values, centers):
    pred_values = np.clip(pred_values, -1, 1)
    distances = np.abs(pred_values[:, np.newaxis] - centers[np.newaxis, :])
    sigma = 0.15
    weights = np.exp(-distances**2 / (2 * sigma**2))
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    return probabilities

pred_probabilities = calculate_soft_probabilities(preds_np, category_centers)

# Calculate ROC curve and AUC
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(true_categories_bin[:, i], pred_probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(true_categories_bin.ravel(), 
                                           pred_probabilities.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(fpr["micro"], tpr["micro"],
        label=f'Micro-average (AUC = {roc_auc["micro"]:.4f})',
        color='deeppink', linestyle=':', linewidth=3)

for i, (name, color) in enumerate(zip(category_names, category_colors)):
    ax.plot(fpr[i], tpr[i], color=color, lw=2.5,
            label=f'{name} (AUC = {roc_auc[i]:.4f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves - Steering Classification (One-vs-Rest)', 
            fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/roc_curves.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: output/roc_curves.png")
plt.show(block=False)
plt.close()

# ============================================================================
# PRECISION-RECALL CURVES
# ============================================================================
print("\n" + "=" * 80)
print("PRECISION-RECALL CURVES")
print("=" * 80)

precision = dict()
recall = dict()
pr_auc = dict()

for i in range(5):
    precision[i], recall[i], _ = precision_recall_curve(
        true_categories_bin[:, i], pred_probabilities[:, i])
    pr_auc[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(
    true_categories_bin.ravel(), pred_probabilities.ravel())
pr_auc["micro"] = auc(recall["micro"], precision["micro"])

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(recall["micro"], precision["micro"],
        label=f'Micro-average (AUC = {pr_auc["micro"]:.4f})',
        color='deeppink', linestyle=':', linewidth=3)

for i, (name, color) in enumerate(zip(category_names, category_colors)):
    ax.plot(recall[i], precision[i], color=color, lw=2.5,
            label=f'{name} (AUC = {pr_auc[i]:.4f})')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curves - Steering Classification', 
            fontsize=14, fontweight='bold')
ax.legend(loc="lower left", fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/precision_recall_curves.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: output/precision_recall_curves.png")
plt.show(block=False)
plt.close()

# ============================================================================
# REGRESSION VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("REGRESSION VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Predicted vs Actual
ax = axes[0, 0]
scatter = ax.scatter(targets_np, preds_np, c=true_categories, 
                    cmap='viridis', alpha=0.4, s=10, edgecolors='none')
ax.plot([-1, 1], [-1, 1], 'r--', lw=2.5, label='Perfect Prediction')
ax.set_xlabel('True Steering', fontsize=11, fontweight='bold')
ax.set_ylabel('Predicted Steering', fontsize=11, fontweight='bold')
ax.set_title(f'Predicted vs True Steering\nR² = {steer_r2_val:.4f}, MAE = {steer_mae_val:.4f}',
            fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('True Category', fontsize=10)

# 2. Error Distribution
ax = axes[0, 1]
errors = preds_np - targets_np
ax.hist(errors, bins=60, edgecolor='black', alpha=0.7, color='coral')
ax.axvline(0, color='red', linestyle='--', lw=2.5, label='Zero Error')
ax.axvline(errors.mean(), color='blue', linestyle='--', lw=2, 
          label=f'Mean = {errors.mean():.4f}')
ax.set_xlabel('Prediction Error', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title(f'Error Distribution\nStd = {errors.std():.4f}', 
            fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 3. Error vs True Steering
ax = axes[1, 0]
scatter = ax.scatter(targets_np, np.abs(errors), c=true_categories,
                    cmap='viridis', alpha=0.4, s=10, edgecolors='none')
ax.axhline(0.05, color='green', linestyle='--', lw=2, label='0.05 threshold')
ax.axhline(0.10, color='orange', linestyle='--', lw=2, label='0.10 threshold')
ax.set_xlabel('True Steering', fontsize=11, fontweight='bold')
ax.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
ax.set_title('Absolute Error vs True Steering', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('True Category', fontsize=10)

# 4. Cumulative Error Distribution
ax = axes[1, 1]
sorted_errors = np.sort(np.abs(errors))
cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
ax.plot(sorted_errors, cumulative, lw=2.5, color='darkblue')
ax.axvline(0.05, color='green', linestyle='--', lw=2, label='±0.05')
ax.axvline(0.10, color='orange', linestyle='--', lw=2, label='±0.10')
ax.axvline(0.15, color='red', linestyle='--', lw=2, label='±0.15')
ax.set_xlabel('Absolute Error', fontsize=11, fontweight='bold')
ax.set_ylabel('Cumulative Percentage (%)', fontsize=11, fontweight='bold')
ax.set_title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/regression_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: output/regression_analysis.png")
plt.show(block=False)
plt.close()

# ============================================================================
# ERROR ANALYSIS BY CATEGORY
# ============================================================================
print("\n" + "=" * 80)
print("ERROR ANALYSIS BY CATEGORY")
print("=" * 80)

print(f"\n{'Category':<15} {'Count':>8} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
print("-" * 57)

for i, name in enumerate(category_names):
    mask = true_categories == i
    if mask.sum() > 0:
        mae = np.abs(preds_np[mask] - targets_np[mask]).mean()
        rmse = np.sqrt(((preds_np[mask] - targets_np[mask])**2).mean())
        r2 = r2_score(targets_np[mask], preds_np[mask])
        print(f"{name:<15} {mask.sum():>8} {mae:>10.4f} {rmse:>10.4f} {r2:>10.4f}")

# ============================================================================
# SAVE METRICS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING COMPREHENSIVE METRICS")
print("=" * 80)

metrics_dict = {
    'regression': {
        'mae': float(steer_mae_val),
        'rmse': float(steer_rmse_val),
        'r2_score': float(steer_r2_val),
        'pearson_correlation': float(steer_corr)
    },
    'classification': {
        'accuracy': float(classification_accuracy),
        'confusion_matrix': cm.tolist(),
        'roc_auc': {name: float(roc_auc[i]) for i, name in enumerate(category_names)},
        'roc_auc_micro': float(roc_auc['micro']),
        'pr_auc': {name: float(pr_auc[i]) for i, name in enumerate(category_names)},
        'pr_auc_micro': float(pr_auc['micro'])
    },
    'training': {
        'model': 'ResNet50',
        'output': 'steering_only',
        'loss_function': 'MSELossFlat',
        'total_samples': len(df),
        'validation_samples': len(preds_np),
        'categories': category_names,
        'straight_threshold': 0.05
    }
}

metrics_file = output_dir / 'comprehensive_metrics.json'
with open(metrics_file, 'w') as f:
    json.dump(metrics_dict, f, indent=2)

print(f"\n✓ Metrics saved to: {metrics_file}")

# ============================================================================
# MODEL EXPORT
# ============================================================================
print("\n" + "=" * 80)
print("MODEL EXPORT")
print("=" * 80)

model_filename = output_dir / 'steering_only_model.pkl'
learn.export(model_filename)

print(f"\n✓ Model saved: {model_filename}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)

print(f"\n📊 REGRESSION METRICS:")
print(f"  MAE:              {steer_mae_val:.4f}")
print(f"  RMSE:             {steer_rmse_val:.4f}")
print(f"  R² Score:         {steer_r2_val:.4f}")

print(f"\n📊 CLASSIFICATION METRICS (5 Categories):")
print(f"  Accuracy:         {classification_accuracy:.4f} ({classification_accuracy*100:.2f}%)")
print(f"  ROC AUC (micro):  {roc_auc['micro']:.4f}")
print(f"  PR AUC (micro):   {pr_auc['micro']:.4f}")

print(f"\n📁 SAVED FILES:")
print(f"  • {model_filename}")
print(f"  • {metrics_file}")
print(f"  • output/confusion_matrix.png")
print(f"  • output/roc_curves.png")
print(f"  • output/precision_recall_curves.png")
print(f"  • output/regression_analysis.png")
print(f"  • output/steering_categories.png")
print(f"  • output/training_loss.png")

print(f"\n🚗 MODEL SPECIFICATIONS:")
print(f"  • Output: Single steering value [-1, 1]")
print(f"  • Architecture: ResNet50 with single output neuron")
print(f"  • Loss: MSELossFlat() [fastai built-in]")
print(f"  • Gradient clipping: 1.0")
print(f"  • Categories: 5 (Sharp Left, Left, Straight, Right, Sharp Right)")

print("\n" + "=" * 80)
print("✓ All complete! Close plot windows when ready.")
plt.show()
