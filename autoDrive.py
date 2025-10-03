"""
Assetto Corsa Autonomous Driving - Real-Time Inference
Uses trained fastai model with virtual Xbox 360 controller
"""

import mss
import numpy as np
import time
import keyboard
import win32api
from PIL import Image
from pathlib import Path
import vgamepad as vg
from fastai.vision.all import *
import torch
import pandas as pd

print("=" * 80)
print("ASSETTO CORSA AUTONOMOUS DRIVER")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_WEIGHTS = 'models/drive_model_weights.pth'  # Use .pth file instead
TARGET_FPS = 10
SHOW_DEBUG = True

# ============================================================================
# LOAD MODEL - SAFE METHOD
# ============================================================================
print("\n[1/4] Loading trained model...")
try:
    # Create a dummy image for initialization
    dummy_img_path = Path('dummy.jpg')
    if not dummy_img_path.exists():
        Image.new('RGB', (224, 224)).save(dummy_img_path)
    
    # Create dummy DataFrame with NUMERIC values (not strings)
    dummy_df = pd.DataFrame({
        'image': ['dummy.jpg'],
        'steer': [0.0],      # Float, not string
        'throttle': [0.0],   # Float, not string
        'brake': [0.0]       # Float, not string
    })
    
    # Ensure columns are numeric
    dummy_df['steer'] = dummy_df['steer'].astype('float32')
    dummy_df['throttle'] = dummy_df['throttle'].astype('float32')
    dummy_df['brake'] = dummy_df['brake'].astype('float32')
    
    # Create DataBlock matching your training configuration
    dblock = DataBlock(
        blocks=(ImageBlock, RegressionBlock(n_out=3)),
        get_x=ColReader('image', pref=''),
        get_y=ColReader(['steer', 'throttle', 'brake']),
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=Resize(224),
        batch_tfms=[Normalize.from_stats(*imagenet_stats)]
    )
    
    # Create DataLoaders
    dls = dblock.dataloaders(dummy_df, bs=1, path='.')
    
    # Create learner with same architecture as training
    learn = vision_learner(
        dls,
        resnet34,
        n_out=3,
        loss_func=MSELossFlat(),
        metrics=[mae]
    )
    
    # Load the saved weights (cross-platform safe)
    # Remove 'models/' and '.pth' from the path
    learn.load('drive_model_weights')
    
    print(f"✓ Model weights loaded from: {MODEL_WEIGHTS}")
    
    # Clean up dummy file
    if dummy_img_path.exists():
        dummy_img_path.unlink()
    
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Move model to GPU if available
if torch.cuda.is_available():
    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    learn.model = learn.model.cuda()
else:
    print("⚠ Using CPU (slower)")

# Set model to evaluation mode
learn.model.eval()

# ============================================================================
# SETUP VIRTUAL XBOX 360 CONTROLLER
# ============================================================================
print("\n[2/4] Initializing virtual Xbox 360 controller...")
try:
    gamepad = vg.VX360Gamepad()
    print("✓ Virtual Xbox 360 controller created")
    print("  Make sure Assetto Corsa is set to use Xbox controller input!")
except Exception as e:
    print(f"✗ Failed to create virtual controller: {e}")
    print("  Install ViGEmBus driver: pip install vgamepad")
    exit(1)

# ============================================================================
# SETUP SCREEN CAPTURE
# ============================================================================
print("\n[3/4] Setting up screen capture...")
sct = mss.mss()
monitor = {
    "top": 0,
    "left": 0,
    "width": win32api.GetSystemMetrics(0),
    "height": win32api.GetSystemMetrics(1)
}
print(f"✓ Screen capture ready: {monitor['width']}x{monitor['height']}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clip_predictions(steer, throttle, brake):
    """Clip predictions to valid control ranges"""
    steer = np.clip(steer, -1.0, 1.0)
    throttle = np.clip(throttle, 0.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    return steer, throttle, brake

def capture_and_preprocess():
    """Capture screen and preprocess for model input"""
    screenshot = sct.grab(monitor)
    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
    img = img.resize((224, 224), Image.LANCZOS)
    return img

def send_controls_to_game(steer, throttle, brake):
    """Send control inputs to virtual Xbox 360 controller"""
    gamepad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
    gamepad.right_trigger_float(value_float=throttle)
    gamepad.left_trigger_float(value_float=brake)
    gamepad.update()

def reset_controls():
    """Reset all controls to neutral/zero"""
    gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
    gamepad.right_trigger_float(value_float=0.0)
    gamepad.left_trigger_float(value_float=0.0)
    gamepad.update()

# ============================================================================
# MAIN INFERENCE LOOP
# ============================================================================
print("\n[4/4] Starting autonomous driving system...")
print("\nControls:")
print("  Press 'a' to START autonomous driving")
print("  Press 's' to STOP autonomous driving")
print("  Press 'q' to QUIT program")
print("\n" + "=" * 80)

driving = False
frame_count = 0
start_time = time.time()
fps_update_interval = 10

try:
    print("\n⏸  Waiting for command... (Press 'a' to start)")
    
    while True:
        if keyboard.is_pressed('a') and not driving:
            print("\n▶  STARTING autonomous driving...")
            driving = True
            frame_count = 0
            start_time = time.time()
            time.sleep(0.2)
            
        elif keyboard.is_pressed('s') and driving:
            print("\n⏸  STOPPING autonomous driving...")
            driving = False
            reset_controls()
            print("   Controls reset to neutral")
            time.sleep(0.2)
            
        elif keyboard.is_pressed('q'):
            print("\n⏹  QUITTING...")
            break
        
        if driving:
            loop_start = time.time()
            
            try:
                img = capture_and_preprocess()
                pred = learn.predict(img)
                steer_raw, throttle_raw, brake_raw = pred[0][0], pred[0][1], pred[0][2]
                steer, throttle, brake = clip_predictions(
                    float(steer_raw),
                    float(throttle_raw),
                    float(brake_raw)
                )
                send_controls_to_game(steer, throttle, brake)
                frame_count += 1
                
                if SHOW_DEBUG:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    if frame_count % fps_update_interval == 0:
                        print(f"Frame {frame_count:4d} | "
                              f"FPS: {current_fps:5.1f} | "
                              f"Steer: {steer:+.3f} | "
                              f"Throttle: {throttle:.3f} | "
                              f"Brake: {brake:.3f}")
                
                inference_time = time.time() - loop_start
                target_frame_time = 1.0 / TARGET_FPS
                
                if inference_time < target_frame_time:
                    time.sleep(target_frame_time - inference_time)
                    
            except Exception as e:
                print(f"⚠ Inference error: {e}")
                continue
        else:
            time.sleep(0.01)
            
except KeyboardInterrupt:
    print("\n\n⏹  Interrupted by user")
    
finally:
    print("\nCleaning up...")
    reset_controls()
    sct.close()
    print("✓ Screen capture closed")
    print("✓ Virtual controller reset")
    print("\n" + "=" * 80)
    print("AUTONOMOUS DRIVING SESSION COMPLETE")
    print("=" * 80)
