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
# from fastai.vision.all import *
import torch
import dill
import warnings

# ADD THIS BEFORE FASTAI IMPORT
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Now import fastai
from fastai.vision.all import *

warnings.filterwarnings('ignore')


print("=" * 80)
print("ASSETTO CORSA AUTONOMOUS DRIVER")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = 'drive_model_ac_optimized_2.pkl'  # Use .pkl exported model
TARGET_FPS = 30
SHOW_DEBUG = True

# ============================================================================
# LOAD MODEL - METHOD 1: Using Exported Model (Recommended)
# ============================================================================
print("\n[1/4] Loading trained model...")
try:
    if Path(MODEL_PATH).exists():
        # Load complete exported model (includes transforms)
        print(f"Loading model from: {MODEL_PATH}")
        learn = load_learner(MODEL_PATH, cpu=not torch.cuda.is_available(), pickle_module=dill)
        print("✓ Model loaded successfully using learn.export method")
    else:
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        learn.model = learn.model.cuda()
    else:
        print("⚠ Using CPU (slower)")
    
    # Set model to evaluation mode
    learn.model.eval()
    
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    
    # Fallback: Try loading weights only (Method 2)
    print("\n⚠ Attempting fallback: Loading weights only...")
    try:
        # Recreate model architecture (MUST match training exactly)
        class WeightedMSELoss(nn.Module):
            def __init__(self, steer_weight=2.0, throttle_weight=1.0, brake_weight=1.0):
                super().__init__()
                self.steer_weight = steer_weight
                self.throttle_weight = throttle_weight
                self.brake_weight = brake_weight
            
            def forward(self, pred, target):
                turn_intensity = 1.0 + self.steer_weight * torch.abs(target[:, 0])
                steer_loss = turn_intensity * (pred[:, 0] - target[:, 0])**2
                throttle_loss = self.throttle_weight * (pred[:, 1] - target[:, 1])**2
                brake_loss = self.brake_weight * (pred[:, 2] - target[:, 2])**2
                return (steer_loss + throttle_loss + brake_loss).mean()
        
        # Create dummy data for DataBlock initialization
        dummy_img = Image.new('RGB', (224, 224))
        dummy_img.save('temp_dummy.jpg')
        dummy_df = pd.DataFrame({
            'image': ['temp_dummy.jpg'],
            'steer': [0.0],
            'throttle': [0.0],
            'brake': [0.0]
        })
        
        # Create DataBlock (MUST match training configuration)
        dblock = DataBlock(
            blocks=(ImageBlock, RegressionBlock(n_out=3)),
            get_x=ColReader('image', pref=''),
            get_y=ColReader(['steer', 'throttle', 'brake']),
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            item_tfms=Resize(224),
            batch_tfms=[Normalize.from_stats(*imagenet_stats)]
        )
        
        dls = dblock.dataloaders(dummy_df, bs=1, path='.')
        
        # Create learner with same architecture
        learn = vision_learner(
            dls,
            resnet34,
            n_out=3,
            loss_func=WeightedMSELoss(steer_weight=2.0),
            metrics=[mae]
        )
        
        # Load weights
        learn.load('models/ac_drive_model_weights')
        learn.model.eval()
        
        # Cleanup
        Path('temp_dummy.jpg').unlink(missing_ok=True)
        
        print("✓ Fallback successful: Weights loaded")
        
    except Exception as e2:
        print(f"✗ Fallback also failed: {e2}")
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
    print("  Then: pip install vgamepad")
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
    # Resize to match training input size
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
# WARM-UP INFERENCE
# ============================================================================
print("\n[4/4] Warming up model...")
try:
    dummy_img = Image.new('RGB', (224, 224))
    _ = learn.predict(dummy_img)
    print("✓ Model warmed up and ready")
except Exception as e:
    print(f"⚠ Warm-up warning: {e}")

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
            time.sleep(0.2)  # Debounce
            
        elif keyboard.is_pressed('s') and driving:
            print("\n⏸  STOPPING autonomous driving...")
            driving = False
            reset_controls()
            print("   Controls reset to neutral")
            time.sleep(0.2)  # Debounce
            
        elif keyboard.is_pressed('q'):
            print("\n⏹  QUITTING...")
            break
        
        if driving:
            loop_start = time.time()
            
            try:
                # Capture screen and run inference
                img = capture_and_preprocess()
                
                # Get prediction (returns tuple: (prediction, class_idx, probabilities))
                with torch.no_grad():  # Disable gradient computation for inference
                    pred_result = learn.predict(img)
                    pred = pred_result[0]  # Extract tensor predictions
                
                # Extract control values
                steer_raw = float(pred[0])
                throttle_raw = float(pred[1])
                brake_raw = float(pred[2])
                
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
                continue
        else:
            time.sleep(0.01)  # Small sleep when not driving
            
except KeyboardInterrupt:
    print("\n\n⏹  Interrupted by user")
    
finally:
    # Cleanup
    print("\nCleaning up...")
    reset_controls()
    sct.close()
    print("✓ Screen capture closed")
    print("✓ Virtual controller reset")
    print("\n" + "=" * 80)
    print("AUTONOMOUS DRIVING SESSION COMPLETE")
    print("=" * 80)
