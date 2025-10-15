"""
YOLOv11 Traffic Sign Detection - TRAINING SCRIPT
Optimized for 4GB GPU (RTX 3050)
Run this ONCE to train your model
"""

import os
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

if __name__ == '__main__':
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ARCHIVE_DIR = os.path.join(BASE_DIR, "Archive")
    DATASET_DIR = os.path.join(ARCHIVE_DIR, "ts", "ts")
    WORK_DIR = os.path.join(BASE_DIR, "yolo_traffic_signs")
    
    # Training parameters - Optimized for 4GB GPU
    EPOCHS = 100
    BATCH = 8
    IMG_SIZE = 480
    MODEL = 'yolo11s.pt'
    
    print("="*60)
    print("YOLO TRAFFIC SIGN TRAINING")
    print("="*60)
    print(f"Model: {MODEL} (Small - optimized for 4GB GPU)")
    print(f"Batch Size: {BATCH}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print("="*60)
    
    # ============================================================
    # SETUP DIRECTORIES
    # ============================================================
    print("\n[1/6] Setting up directories...")
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(WORK_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(WORK_DIR, 'labels', split), exist_ok=True)
    
    # ============================================================
    # COPY DATASET
    # ============================================================
    print("[2/6] Organizing dataset...")
    
    def copy_files(txt_file, split):
        count = 0
        with open(os.path.join(ARCHIVE_DIR, txt_file), 'r') as f:
            for line in f:
                filename = os.path.basename(line.strip())
                if not filename:
                    continue
                
                # Copy image
                img_src = os.path.join(DATASET_DIR, filename)
                img_dst = os.path.join(WORK_DIR, 'images', split, filename)
                if os.path.exists(img_src):
                    shutil.copy2(img_src, img_dst)
                    count += 1
                
                # Copy label
                label = filename.replace('.jpg', '.txt').replace('.png', '.txt')
                lbl_src = os.path.join(DATASET_DIR, label)
                lbl_dst = os.path.join(WORK_DIR, 'labels', split, label)
                if os.path.exists(lbl_src):
                    shutil.copy2(lbl_src, lbl_dst)
        
        return count
    
    train_count = copy_files('train.txt', 'train')
    test_count = copy_files('test.txt', 'test')
    print(f"   ✓ Copied {train_count} train images")
    print(f"   ✓ Copied {test_count} test images")
    
    # ============================================================
    # CREATE VALIDATION SPLIT
    # ============================================================
    print("[3/6] Creating validation split...")
    train_imgs = os.listdir(os.path.join(WORK_DIR, 'images', 'train'))
    train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.1, random_state=42)
    
    for img in val_imgs:
        shutil.move(
            os.path.join(WORK_DIR, 'images', 'train', img),
            os.path.join(WORK_DIR, 'images', 'val', img)
        )
        label = img.replace('.jpg', '.txt').replace('.png', '.txt')
        lbl_path = os.path.join(WORK_DIR, 'labels', 'train', label)
        if os.path.exists(lbl_path):
            shutil.move(lbl_path, os.path.join(WORK_DIR, 'labels', 'val', label))
    
    print(f"   ✓ Train: {len(train_imgs)} images")
    print(f"   ✓ Val: {len(val_imgs)} images")
    
    # ============================================================
    # CREATE CONFIG FILE
    # ============================================================
    print("[4/6] Creating config file...")
    config = f"""path: {WORK_DIR.replace(os.sep, '/')}
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: prohibitory
  1: danger
  2: mandatory
  3: other
"""
    
    config_path = os.path.join(WORK_DIR, 'data.yaml')
    with open(config_path, 'w') as f:
        f.write(config)
    print(f"   ✓ Config saved: {config_path}")
    
    # ============================================================
    # TRAIN MODEL
    # ============================================================
    print("\n[5/6] Starting training...")
    print("="*60)
    model = YOLO(MODEL)
    
    results = model.train(
        data=config_path,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        device=0,
        workers=4,
        cache=False,
        amp=True,
        project=os.path.join(WORK_DIR, 'runs'),
        name='train',
        exist_ok=True
    )
    
    # ============================================================
    # EVALUATE
    # ============================================================
    print("\n[6/6] Evaluating model...")
    val_metrics = model.val()
    test_metrics = model.val(split='test')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Validation mAP50: {val_metrics.box.map50:.4f}")
    print(f"Test mAP50: {test_metrics.box.map50:.4f}")
    
    best_model = os.path.join(WORK_DIR, 'runs', 'train', 'weights', 'best.pt')
    print(f"\n✓ Best model saved: {best_model}")
    print(f"\nUse this model path in game_detector.py!")
    print("="*60)
