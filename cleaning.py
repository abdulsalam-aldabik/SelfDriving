# ============================================================================
# DATA CLEANING AND BALANCING NOTEBOOK SECTION (WITH DYNAMIC SPEED NORMALIZATION)
# ============================================================================

import pandas as pd
import os
import numpy as np
from PIL import Image
import shutil

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input directory (where your raw data is stored)
INPUT_DIR = "ac_training_data"
INPUT_FRAMES_DIR = os.path.join(INPUT_DIR, "frames")
INPUT_CSV = os.path.join(INPUT_DIR, "labels.csv")

# Output directory (cleaned data ready for upload)
OUTPUT_DIR = "ac_training_data_cleaned"
OUTPUT_FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "labels_cleaned.csv")

# Create output directories
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

print("=" * 80)
print("DATA CLEANING PIPELINE WITH DYNAMIC SPEED NORMALIZATION")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\nSTEP 1: Loading data...")
df = pd.read_csv(INPUT_CSV)

print(f"Original dataset: {len(df)} samples")
print("\nOriginal data statistics:")
print(df[['steer', 'throttle', 'brake', 'speed']].describe())

# ============================================================================
# STEP 2: REMOVE DUPLICATES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Removing duplicate entries")
print("=" * 80)

# Remove exact duplicates based on telemetry data
df_no_duplicates = df.drop_duplicates(
    subset=['steer', 'throttle', 'brake', 'speed'],
    keep='first'
)

duplicates_removed = len(df) - len(df_no_duplicates)
print(f"Removed {duplicates_removed} duplicate entries ({duplicates_removed/len(df)*100:.1f}%)")
print(f"Dataset after duplicate removal: {len(df_no_duplicates)} samples")

# ============================================================================
# STEP 3: REMOVE INVALID SAMPLES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Removing invalid samples")
print("=" * 80)

# Check for missing or corrupted image files
valid_samples = []
invalid_count = 0

for idx, row in df_no_duplicates.iterrows():
    image_path = os.path.join(INPUT_DIR, row['image'])
    
    # Check if file exists
    if not os.path.exists(image_path):
        invalid_count += 1
        continue
    
    # Check if file can be opened as valid image
    try:
        with Image.open(image_path) as img:
            img.verify()
        valid_samples.append(idx)
    except Exception as e:
        invalid_count += 1
        print(f"Invalid image: {row['image']}")

df_valid = df_no_duplicates.loc[valid_samples].reset_index(drop=True)

print(f"Removed {invalid_count} invalid/missing images ({invalid_count/len(df_no_duplicates)*100:.1f}%)")
print(f"Dataset after validation: {len(df_valid)} samples")

# ============================================================================
# STEP 4: NORMALIZE SPEED (DYNAMIC RANGE)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Normalizing speed (Dynamic Range)")
print("=" * 80)

# Store original speed range for display purposes
original_speed_min = df_valid['speed'].min()
original_speed_max = df_valid['speed'].max()
original_speed_mean = df_valid['speed'].mean()
original_speed_std = df_valid['speed'].std()

print(f"Original speed range in data:")
print(f"  Minimum: {original_speed_min:.2f} km/h")
print(f"  Maximum: {original_speed_max:.2f} km/h")
print(f"  Mean: {original_speed_mean:.2f} km/h")
print(f"  Std: {original_speed_std:.2f} km/h")

# Calculate dynamic min/max from actual data
SPEED_MIN = original_speed_min
SPEED_MAX = original_speed_max

# Min-Max Normalization (0-1 range) using dynamic range
df_valid['speed'] = (df_valid['speed'] - SPEED_MIN) / (SPEED_MAX - SPEED_MIN)

# Round to 1 decimal place
df_valid['speed'] = df_valid['speed'].round(1)


# Handle edge case where all speeds are the same
if SPEED_MAX == SPEED_MIN:
    print("WARNING: All speed values are identical. Setting normalized speed to 0.5")
    df_valid['speed'] = 0.5

print(f"\nNormalized speed statistics:")
print(f"  Min: {df_valid['speed'].min():.4f}")
print(f"  Max: {df_valid['speed'].max():.4f}")
print(f"  Mean: {df_valid['speed'].mean():.4f}")
print(f"  Std: {df_valid['speed'].std():.4f}")

# Save normalization parameters for later use during inference
normalization_params = {
    'speed_min': float(SPEED_MIN),
    'speed_max': float(SPEED_MAX)
}

import json
normalization_file = os.path.join(OUTPUT_DIR, 'normalization_params.json')
with open(normalization_file, 'w') as f:
    json.dump(normalization_params, f, indent=4)
print(f"\nSaved normalization parameters to: {normalization_file}")
print(f"  Speed range: {SPEED_MIN:.2f} - {SPEED_MAX:.2f} km/h")

# ============================================================================
# STEP 5: BALANCE STEERING DATA
# ============================================================================

def balance_steering_data(df, threshold=0.05, max_samples_per_bin=300):
    """
    Remove excessive straight-driving samples to balance dataset.
    Steering range: -1.0 (full left) to 1.0 (full right)
    
    Args:
        df: DataFrame with steering data
        threshold: Steering angle considered "straight" (absolute value)
        max_samples_per_bin: Maximum samples to keep for straight driving
    
    Returns:
        Balanced DataFrame
    """
    # Identify straight driving samples
    straight_mask = (df['steer'].abs() < threshold)
    
    # Sample straight driving with limit
    straight_samples = df[straight_mask].sample(
        n=min(max_samples_per_bin, straight_mask.sum()),
        random_state=42
    )
    
    # Keep all turning samples
    turn_samples = df[~straight_mask]
    
    # Combine and shuffle
    balanced_df = pd.concat([straight_samples, turn_samples]).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    
    # Print statistics
    print(f"Before balancing: {len(df)} samples")
    print(f"  Straight driving (|steer| < {threshold}): {straight_mask.sum()} ({straight_mask.sum()/len(df)*100:.1f}%)")
    print(f"  Turning (|steer| >= {threshold}): {(~straight_mask).sum()} ({(~straight_mask).sum()/len(df)*100:.1f}%)")
    print(f"\nAfter balancing: {len(balanced_df)} samples")
    print(f"  Straight driving: {len(straight_samples)} ({len(straight_samples)/len(balanced_df)*100:.1f}%)")
    print(f"  Turning: {len(turn_samples)} ({len(turn_samples)/len(balanced_df)*100:.1f}%)")
    
    return balanced_df

print("\n" + "=" * 80)
print("STEP 5: Balancing steering data...")
print("=" * 80)
df_balanced = balance_steering_data(
    df_valid, 
    threshold=0.05,             # Steering angle threshold for "straight"
    max_samples_per_bin=50      # Max straight-driving samples to keep
)

# ============================================================================
# STEP 6: COPY IMAGES AND SAVE CLEANED DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Copying images...")
print("=" * 80)
cleaned_rows = []

for idx, row in df_balanced.iterrows():
    source_path = os.path.join(INPUT_DIR, row['image'])
    dest_path = os.path.join(OUTPUT_DIR, row['image'])
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    try:
        shutil.copy2(source_path, dest_path)
        cleaned_rows.append(row)
    except Exception as e:
        print(f"Error copying {row['image']}: {e}")

df_cleaned = pd.DataFrame(cleaned_rows)

# Save with only normalized speed column
df_cleaned.to_csv(OUTPUT_CSV, index=False)

print(f"Copied {len(df_cleaned)} images")
print(f"Saved cleaned CSV to {OUTPUT_CSV}")

# ============================================================================
# STEP 7: FINAL STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("FINAL STATISTICS")
print("=" * 80)

print(f"\nOriginal dataset: {len(df)} samples")
print(f"Cleaned dataset: {len(df_cleaned)} samples")
print(f"Reduction: {len(df) - len(df_cleaned)} samples ({(len(df) - len(df_cleaned))/len(df)*100:.1f}%)")

print("\nCleaned data statistics:")
print(df_cleaned[['steer', 'throttle', 'brake', 'speed']].describe())

print("\nSteering distribution:")
print(f"  Mean: {df_cleaned['steer'].mean():.3f}")
print(f"  Std: {df_cleaned['steer'].std():.3f}")
print(f"  Min: {df_cleaned['steer'].min():.3f}")
print(f"  Max: {df_cleaned['steer'].max():.3f}")

print("\nSpeed distribution (normalized):")
print(f"  Mean: {df_cleaned['speed'].mean():.4f}")
print(f"  Std: {df_cleaned['speed'].std():.4f}")
print(f"  Min: {df_cleaned['speed'].min():.4f}")
print(f"  Max: {df_cleaned['speed'].max():.4f}")

print(f"\nOriginal speed range (before normalization):")
print(f"  {original_speed_min:.2f} - {original_speed_max:.2f} km/h")

print("\n" + "=" * 80)
print(f"CLEANING COMPLETE! Upload '{OUTPUT_DIR}' folder to Google Drive.")
print("=" * 80)
