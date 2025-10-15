"""
Data Cleaning and Balancing Script
- Validates images
- Balances steering distribution (left/straight/right)
- Normalizes speed values to [0, 1] range
- Saves normalization parameters for inference
- Supports incremental mode (appending to existing cleaned data)
"""

import pandas as pd
import os
import numpy as np
from PIL import Image
import shutil
import json


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_DIR = "ac_training_data"
INPUT_CSV = os.path.join(INPUT_DIR, "labels.csv")

OUTPUT_DIR = "ac_training_data_cleaned"
OUTPUT_FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "labels_cleaned.csv")

# Balancing parameters
STEERING_THRESHOLD = 0.05  # Angle threshold for "straight" driving
STRAIGHT_PERCENTAGE = 0.15  # Straight data as 25% of total dataset (recommended: 20-30%)
TARGET_DATASET_SIZE = None  # Set to None for dynamic calculation, or specify a number  

os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)
# Check if output CSV already exists
output_csv_exists = os.path.isfile(OUTPUT_CSV)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("Loading data...")
df = pd.read_csv(INPUT_CSV)
print(f"Original dataset: {len(df)} samples")


# ============================================================================
# STEP 2: VALIDATE IMAGE FILES
# ============================================================================

print("\nValidating images...")
valid_samples = []

for idx, row in df.iterrows():
    image_path = os.path.join(INPUT_DIR, row['image'])
    
    if os.path.exists(image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()
            valid_samples.append(idx)
        except:
            pass

df_valid = df.loc[valid_samples].reset_index(drop=True)
print(f"Valid samples: {len(df_valid)}")


# ============================================================================
# STEP 3: NORMALIZE SPEED
# ============================================================================

print("\nNormalizing speed...")

normalization_file = os.path.join(OUTPUT_DIR, 'normalization_params.json')

# Load existing normalization params if they exist
if os.path.exists(normalization_file):
    with open(normalization_file, 'r') as f:
        normalization_params = json.load(f)
    original_speed_min = normalization_params['speed_min']
    original_speed_max = normalization_params['speed_max']
    print(f"Using existing normalization: {original_speed_min:.2f} - {original_speed_max:.2f} km/h")
else:
    # Calculate new normalization params from current data
    original_speed_min = df_valid['speed'].min()
    original_speed_max = df_valid['speed'].max()
    print(f"New speed range: {original_speed_min:.2f} - {original_speed_max:.2f} km/h")

# Handle edge case of identical speeds
if original_speed_max == original_speed_min:
    df_valid['speed'] = 0.5
else:
    # Min-Max normalization to [0, 1]
    df_valid['speed'] = (df_valid['speed'] - original_speed_min) / (original_speed_max - original_speed_min)
    df_valid['speed'] = df_valid['speed'].round(1)

# Save normalization parameters for later use in inference
normalization_params = {
    'speed_min': float(original_speed_min),
    'speed_max': float(original_speed_max)
}

with open(normalization_file, 'w') as f:
    json.dump(normalization_params, f, indent=4)

print(f"Normalization params saved to: {normalization_file}")


# ============================================================================
# STEP 4: BALANCE STEERING DATA
# ============================================================================

def balance_steering_data(df, threshold, straight_percentage=0.25, target_size=None):
    """
    Balance dataset with controlled straight data percentage
    
    Args:
        df: DataFrame with steering data
        threshold: Steering angle threshold for categorization
        straight_percentage: Percentage of total data that should be straight (0.0-1.0)
        target_size: Target total dataset size (None = dynamic based on available data)
    
    Returns:
        Balanced DataFrame
    """
    # Categorize steering into three bins
    left_mask = df['steer'] < -threshold
    straight_mask = (df['steer'] >= -threshold) & (df['steer'] <= threshold)
    right_mask = df['steer'] > threshold
    
    left_samples = df[left_mask]
    straight_samples = df[straight_mask]
    right_samples = df[right_mask]
    
    print(f"\nOriginal distribution:")
    print(f"  Left: {len(left_samples)}, Straight: {len(straight_samples)}, Right: {len(right_samples)}")
    
    # Calculate dynamic target size if not specified
    if target_size is None:
        # Use minimum of left/right samples to determine max possible balanced size
        max_steering_samples = min(len(left_samples), len(right_samples))
        
        # Calculate total size based on straight percentage
        # If straight is 25%, then steering (left+right) is 75%
        # So: steering_samples * 2 = 0.75 * total_size
        # Therefore: total_size = (steering_samples * 2) / (1 - straight_percentage)
        steering_percentage = 1.0 - straight_percentage
        target_size = int((max_steering_samples * 2) / steering_percentage)
        
        # Ensure we don't exceed available straight samples
        max_straight_needed = int(target_size * straight_percentage)
        if max_straight_needed > len(straight_samples):
            target_size = int((len(straight_samples) / straight_percentage))
            print(f"WARNING: Limited by straight samples. Adjusting target size to {target_size}")
    
    # Calculate samples per category
    straight_count = int(target_size * straight_percentage)
    steering_count = int((target_size - straight_count) / 2)  # Split remaining between left/right
    
    print(f"\nTarget distribution (based on {straight_percentage*100:.0f}% straight):")
    print(f"  Total target: {target_size} samples")
    print(f"  Straight: {straight_count} ({straight_percentage*100:.0f}%)")
    print(f"  Left: {steering_count} ({(steering_count/target_size)*100:.1f}%)")
    print(f"  Right: {steering_count} ({(steering_count/target_size)*100:.1f}%)")
    
    # Verify we have enough samples
    if steering_count > min(len(left_samples), len(right_samples)):
        steering_count = min(len(left_samples), len(right_samples))
        straight_count = int(steering_count * 2 * straight_percentage / (1 - straight_percentage))
        print(f"\nWARNING: Insufficient steering data. Adjusted to:")
        print(f"  Left/Right: {steering_count} each")
        print(f"  Straight: {straight_count}")
    
    if straight_count > len(straight_samples):
        straight_count = len(straight_samples)
        print(f"WARNING: Insufficient straight data. Using {straight_count} samples")
    
    # Sample from each category
    left_balanced = left_samples.sample(n=steering_count, random_state=42)
    right_balanced = right_samples.sample(n=steering_count, random_state=42)
    straight_balanced = straight_samples.sample(n=straight_count, random_state=42)
    
    # Combine and shuffle
    balanced_df = pd.concat([left_balanced, straight_balanced, right_balanced]).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    
    # Calculate actual percentages
    actual_straight_pct = len(straight_balanced) / len(balanced_df) * 100
    actual_left_pct = len(left_balanced) / len(balanced_df) * 100
    actual_right_pct = len(right_balanced) / len(balanced_df) * 100
    
    print(f"\nFinal balanced distribution:")
    print(f"  Left: {len(left_balanced)} ({actual_left_pct:.1f}%)")
    print(f"  Straight: {len(straight_balanced)} ({actual_straight_pct:.1f}%)")
    print(f"  Right: {len(right_balanced)} ({actual_right_pct:.1f}%)")
    print(f"  Total: {len(balanced_df)} samples")
    
    return balanced_df


print("\nBalancing steering data...")
df_balanced = balance_steering_data(
    df_valid, 
    STEERING_THRESHOLD, 
    straight_percentage=STRAIGHT_PERCENTAGE,
    target_size=TARGET_DATASET_SIZE
)


# ============================================================================
# STEP 5: COPY IMAGES AND SAVE (INCREMENTAL MODE)
# ============================================================================

print("\nCopying images to output directory...")
cleaned_rows = []

# Load existing cleaned data if it exists
if output_csv_exists:
    existing_cleaned = pd.read_csv(OUTPUT_CSV)
    existing_images = set(existing_cleaned['image'].values)
    print(f"Found {len(existing_cleaned)} existing cleaned samples")
else:
    existing_images = set()

for idx, row in df_balanced.iterrows():
    source_path = os.path.join(INPUT_DIR, row['image'])
    dest_path = os.path.join(OUTPUT_DIR, row['image'])
    
    # Skip if image already exists in cleaned dataset
    if row['image'] in existing_images:
        continue
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    try:
        shutil.copy2(source_path, dest_path)
        cleaned_rows.append(row)
    except Exception:
        pass

df_new_cleaned = pd.DataFrame(cleaned_rows)

# Append to existing CSV or create new one
if output_csv_exists:
    df_new_cleaned.to_csv(OUTPUT_CSV, mode='a', index=False, header=False)
    print(f"Appended {len(df_new_cleaned)} new samples to existing cleaned data")
else:
    df_new_cleaned.to_csv(OUTPUT_CSV, index=False)
    print(f"Created new cleaned dataset with {len(df_new_cleaned)} samples")

# Load final combined dataset for reporting
df_final = pd.read_csv(OUTPUT_CSV)

print(f"\nCleaning complete!")
print(f"  Total cleaned samples: {len(df_final)}")
print(f"  New samples added: {len(df_new_cleaned)}")
print(f"  Saved to: {OUTPUT_DIR}")
