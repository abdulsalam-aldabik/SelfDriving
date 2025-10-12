"""
Assetto Corsa Autonomous Driving - Real-Time Inference (Windows)
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
import torch
import warnings

# CRITICAL: Fix PosixPath issue for models trained on Linux/Colab
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Now import fastai
from fastai.vision.all import *

# Restore PosixPath after import (best practice)
pathlib.PosixPath = temp

warnings.filterwarnings('ignore')

print("=" * 80)
print("ASSETTO CORSA AUTONOMOUS DRIVER")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = 'models/drive_model_ac_optimized_cross_platform.pkl'
TARGET_FPS = 30
SHOW_DEBUG = True

# ============================================================================
# LOAD MODEL
# ============================================================================
print("\n[1/4] Loading trained model...")
try:
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    print(f"Loading model from: {MODEL_PATH}")
    
    # Temporarily set PosixPath for loading
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    
    try:
        learn = load_learner(MODEL_PATH, cpu=not torch.cuda.is_available())
        print("✓ Model loaded successfully")
    finally:
        # Restore original PosixPath
        pathlib.PosixPath = temp
    
    # Optimize for inference
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        learn.model.cuda()
    else:
        print("⚠ Using CPU (slower)")
    
    # Set model to evaluation mode
    learn.model.eval()
    
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

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
    print("  Install ViGEmBus driver: https://github.com/ViGEm/ViGEmBus/releases")
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
    """Capture screen and convert to PIL Image"""
    screenshot = sct.grab(monitor)
    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
    return img

def get_prediction_fast(img):
    """
    Fast prediction using direct model inference.
    Bypasses DataLoader overhead for real-time performance.
    """
    # Convert PIL Image to tensor manually
    img_resized = img.resize((224, 224), Image.LANCZOS)
    img_tensor = tensor(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
    
    # Apply ImageNet normalization (must match training)
    mean = tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    
    # Direct model inference (fastest method)
    with torch.no_grad():
        preds = learn.model(img_tensor)
    
    # Extract predictions
    steer = float(preds[0, 0]) * 2.2
    throttle = 0.2  # Your hardcoded value
    brake = 0.0     # Your hardcoded value
    
    return steer, throttle, brake

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
# WARM-UP INFERENCE
# ============================================================================
print("\n[4/4] Warming up model...")
try:
    dummy_img = Image.new('RGB', (224, 224))
    _ = get_prediction_fast(dummy_img)
    print("✓ Model warmed up and ready")
except Exception as e:
    print(f"⚠ Warm-up warning: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# MAIN INFERENCE LOOP
# ============================================================================
print("\nStarting autonomous driving system...")
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
        # Check keyboard inputs
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
                # Capture screen
                img = capture_and_preprocess()
                
                # Get prediction (fast method)
                steer_raw, throttle_raw, brake_raw = get_prediction_fast(img)
                
                # Clip to valid ranges
                steer, throttle, brake = clip_predictions(steer_raw, throttle_raw, brake_raw)
                
                # Send to game
                send_controls_to_game(steer, throttle, brake)
                
                frame_count += 1
                
                # Display debug info
                if SHOW_DEBUG and frame_count % fps_update_interval == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    print(f"Frame {frame_count:4d} | "
                          f"FPS: {current_fps:5.1f} | "
                          f"Steer: {steer:+.3f} | "
                          f"Throttle: {throttle:.3f} | "
                          f"Brake: {brake:.3f}")
                
                # Frame rate limiting
                inference_time = time.time() - loop_start
                target_frame_time = 1.0 / TARGET_FPS
                
                if inference_time < target_frame_time:
                    time.sleep(target_frame_time - inference_time)
                    
            except Exception as e:
                print(f"⚠ Inference error: {e}")
                import traceback
                traceback.print_exc()
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
