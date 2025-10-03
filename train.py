"""
Assetto Corsa Autonomous Driving - Training Script (Google Colab Version)
Following fastai best practices for transfer learning and regression
"""

# ============================================================================
# COLAB SETUP
# ============================================================================
print("=" * 80)
print("GOOGLE COLAB SETUP")
print("=" * 80)

# Install fastai if not already installed
print("Installing fastai...")
!pip install -q fastai

# Mount Google Drive for data storage
from google.colab import drive
drive.mount('/content/drive')
print("✓ Google Drive mounted at /content/drive")

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================
from fastai.vision.all import *
from fastai.callback.fp16 import *
import pandas as pd
import numpy as np
from pathlib import Path
from google.colab import files
import matplotlib.pyplot as plt
import warnings

# Suppress FutureWarnings from fastai's internal torch.cuda.amp usage
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.cuda.amp.*')

# Enable inline plotting
%matplotlib inline

# Check GPU availability
import torch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  WARNING: GPU not enabled!")
    print("Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU")

# ============================================================================
# DATA SETUP
# ============================================================================
print("\n" + "=" * 80)
print("DATA SETUP")
print("=" * 80)

# Update this path to match your Google Drive folder structure
gdrive_data_path = '/content/drive/MyDrive/data_AssitoCorsa/ac_training_data'

# Copy data from Drive to local Colab storage for faster training
data_path = Path('/content/ac_training_data')
print(f"Copying data from Drive to local storage for faster training...")
!cp -r '{gdrive_data_path}' '/content/'
print(f"✓ Data copied to: {data_path}")

# ============================================================================
# DATA PREPARATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: Loading Data")
print("=" * 80)

csv_file = data_path / 'labels.csv'

# Load CSV with telemetry labels
df = pd.read_csv(csv_file)
print(f"Loaded {len(df)} training samples")

# Normalize path separators for Linux/Colab
df['image'] = df['image'].str.replace('\\', '/', regex=False)
print("✓ Path separators normalized for Linux")

print("\nFirst few rows:")
print(df.head())

# Check data distribution
print("\nData statistics:")
print(df[['steer', 'throttle', 'brake', 'speed']].describe())

# ============================================================================
# DATABLOCK CREATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Creating DataBlock")
print("=" * 80)

# Define DataBlock for regression with 3 outputs
dblock = DataBlock(
    blocks=(ImageBlock, RegressionBlock(n_out=3)),
    get_x=ColReader('image', pref=str(data_path) + '/'),
    get_y=ColReader(['steer', 'throttle', 'brake']),
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(460),
    batch_tfms=[
        *aug_transforms(
            size=224,
            do_flip=False,
            max_rotate=0,
            max_warp=0,
            max_lighting=0.2,
            min_scale=0.75,
            p_lighting=0.5
        ),
        Normalize.from_stats(*imagenet_stats)
    ]
)

# Create DataLoaders
print("Creating DataLoaders...")
dls = dblock.dataloaders(df, bs=32)


print(f"Training batches: {len(dls.train)}")
print(f"Validation batches: {len(dls.valid)}")
print(f"Batch size: {dls.bs}")

# Show sample batch
print("\nDisplaying sample training batch...")
dls.show_batch(max_n=4, nrows=1)
plt.show()

# ============================================================================
# MODEL CREATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Creating Vision Learner")
print("=" * 80)

learn = vision_learner(
    dls,
    resnet34,
    n_out=3,
    loss_func=MSELossFlat(),
    metrics=[mae]
).to_fp16()

print("✓ Model created successfully")
print("Architecture: ResNet34 with pretrained ImageNet weights")
print("Loss function: MSE")
print("Metrics: MAE")
# ============================================================================
# LEARNING RATE FINDER
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Finding Optimal Learning Rate")
print("=" * 80)

lr_suggestion = learn.lr_find()
print(f"Suggested learning rate (valley): {lr_suggestion.valley:.2e}")
plt.show()

# ============================================================================
# TRAINING WITH fine_tune() - PROPER TRANSFER LEARNING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Training with fine_tune() Method")
print("=" * 80)
print("Using fastai's built-in fine_tune() for proper transfer learning")
print("This method automatically:")
print("  1. Freezes backbone and trains head for freeze_epochs")
print("  2. Unfreezes all layers")
print("  3. Applies discriminative learning rates automatically")

# fine_tune() handles everything properly:
# - Trains frozen layers first
# - Unfreezes gradually
# - Uses discriminative learning rates (early layers get lower LR)
learn.fine_tune(
    epochs=12,              # Total epochs after unfreezing
    freeze_epochs=3,        # Epochs to train with frozen backbone
    base_lr=lr_suggestion.valley  # Base learning rate from lr_find
)

# ============================================================================
# TRAINING VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: Training Results")
print("=" * 80)

learn.recorder.plot_loss()
plt.show()

# ============================================================================
# MODEL VISUALIZATION AND ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP X: Model Performance Visualization")
print("=" * 80)

# Show sample results
learn.show_results(max_n=6, figsize=(8,8))

# Get predictions for analysis
preds, targets = learn.get_preds()
preds_np = preds.cpu().numpy()
targets_np = targets.cpu().numpy()

# Scatter plots: Predicted vs Actual
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(targets_np[:, 0], preds_np[:, 0], alpha=0.5)
axes[0].plot([-1, 1], [-1, 1], 'r--')
axes[0].set_xlabel('Actual Steering')
axes[0].set_ylabel('Predicted Steering')
axes[0].set_title('Steering Predictions')
axes[0].grid(True)

axes[1].scatter(targets_np[:, 1], preds_np[:, 1], alpha=0.5)
axes[1].plot([0, 1], [0, 1], 'r--')
axes[1].set_xlabel('Actual Throttle')
axes[1].set_ylabel('Predicted Throttle')
axes[1].set_title('Throttle Predictions')
axes[1].grid(True)

axes[2].scatter(targets_np[:, 2], preds_np[:, 2], alpha=0.5)
axes[2].plot([0, 1], [0, 1], 'r--')
axes[2].set_xlabel('Actual Brake')
axes[2].set_ylabel('Predicted Brake')
axes[2].set_title('Brake Predictions')
axes[2].grid(True)

plt.tight_layout()
plt.show()

# Plot top losses
print("\nAnalyzing samples with highest prediction errors...")
losses = F.mse_loss(preds, targets, reduction='none').mean(dim=1)
top_losses_idx = losses.argsort(descending=True)[:10]

for i, idx in enumerate(top_losses_idx[:5]):
    row = df.iloc[idx]
    print(f"\n#{i+1} - Loss: {losses[idx]:.4f}")
    print(f"  Image: {row['image']}")
    print(f"  True:  steer={row['steer']:+.3f}, throttle={row['throttle']:.3f}, brake={row['brake']:.3f}")
    print(f"  Pred:  steer={preds_np[idx, 0]:+.3f}, throttle={preds_np[idx, 1]:.3f}, brake={preds_np[idx, 2]:.3f}")


# ============================================================================
# MODEL VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: Model Validation")
print("=" * 80)

# Sample predictions
print("\nSample predictions:")
sample_indices = np.random.choice(len(df), size=5, replace=False)

for idx in sample_indices:
    row = df.iloc[idx]
    img_path = data_path / row['image']
    
    pred_result = learn.predict(PILImage.create(img_path))
    pred = pred_result[0]
    
    print(f"\nImage: {row['image']}")
    print(f"  True:  steer={row['steer']:+.3f}, throttle={row['throttle']:.3f}, brake={row['brake']:.3f}")
    print(f"  Pred:  steer={pred[0]:+.3f}, throttle={pred[1]:.3f}, brake={pred[2]:.3f}")
    print(f"  Error: steer={abs(pred[0]-row['steer']):.3f}, throttle={abs(pred[1]-row['throttle']):.3f}, brake={abs(pred[2]-row['brake']):.3f}")

# Overall validation metrics
print("\n" + "=" * 80)
print("Overall Validation Performance")
print("=" * 80)

val_preds, val_targets = learn.get_preds()
val_preds_np = val_preds.cpu().numpy()
val_targets_np = val_targets.cpu().numpy()

steer_mae = np.abs(val_preds_np[:, 0] - val_targets_np[:, 0]).mean()
throttle_mae = np.abs(val_preds_np[:, 1] - val_targets_np[:, 1]).mean()
brake_mae = np.abs(val_preds_np[:, 2] - val_targets_np[:, 2]).mean()

print(f"Steering MAE:  {steer_mae:.4f} (target: <0.10)")
print(f"Throttle MAE:  {throttle_mae:.4f} (target: <0.05)")
print(f"Brake MAE:     {brake_mae:.4f} (target: <0.05)")

# ============================================================================
# MODEL EXPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: Exporting Model")
print("=" * 80)

# Save model locally in Colab
model_filename = 'drive_model_ac.pkl'
learn.export(model_filename)
print(f"✓ Model exported to: {model_filename}")

# Save weights
learn.save('drive_model_weights')
print(f"✓ Model weights saved")

# Save to Google Drive for persistence
gdrive_save_path = '/content/drive/MyDrive/ac_models/'
!mkdir -p '{gdrive_save_path}'
!cp '{model_filename}' '{gdrive_save_path}'
print(f"✓ Model backed up to Google Drive: {gdrive_save_path}")

# ============================================================================
# QUICK INFERENCE TEST
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: Testing Exported Model")
print("=" * 80)

learn_inference = load_learner(model_filename)
print("✓ Model loaded successfully")

test_img_path = data_path / df.iloc[0]['image']
pred = learn_inference.predict(PILImage.create(test_img_path))
print(f"\nTest prediction:")
print(f"  Steering: {pred[0][0]:+.3f}")
print(f"  Throttle: {pred[0][1]:.3f}")
print(f"  Brake:    {pred[0][2]:.3f}")

# ============================================================================
# TRAINING COMPLETE
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print("\nYour model is saved in:")
print(f"  • Colab session: /content/{model_filename}")
print(f"  • Google Drive: {gdrive_save_path}{model_filename}")
print("\nNext steps:")
print("1. Review training curves and validation metrics")
print("2. Download model or access from Google Drive")
print("3. Use for live inference in Assetto Corsa")
print("=" * 80)



learn.show_results(max_n=6, figsize=(8,8))# ============================================================================
# DIRECT DOWNLOAD TO LOCAL DEVICE
# ============================================================================
print("\n" + "=" * 80)
print("DOWNLOADING MODEL TO LOCAL DEVICE")
print("=" * 80)

from google.colab import files

# Download the main model file
files.download(model_filename)
print(f"✓ Downloaded {model_filename} to your Downloads folder")

# Download model weights (optional)
files.download('models/drive_model_weights.pth')
print("✓ Downloaded model weights file")
