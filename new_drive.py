"""
Assetto Corsa AI Driver - Xbox 360 Controller with Constant Throttle
Model predicts steering only, throttle is constant
Uses virtual Xbox 360 controller (vgamepad/ViGEmBus)
Direct model output - No steering modifications
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
import pathlib
import json

# ============================================================================
# FIX POSIX PATH ISSUE (for models trained on Linux/Colab)
# ============================================================================
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from fastai.vision.all import *

pathlib.PosixPath = temp  # Restore
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'models/drive_model_ac_steering_focused.pkl'
NORMALIZATION_PARAMS_PATH = "models/normalization_params.json"

# CONSTANT THROTTLE SETTING
CONSTANT_THROTTLE = 0.4  # Adjust for your car/track
                          # 0.4 = Technical tracks
                          # 0.6 = Medium speed
                          # 0.8 = High speed


# Performance
TARGET_FPS = 60
SHOW_DEBUG = True

print("=" * 80)
print("ASSETTO CORSA AI - XBOX 360 CONTROLLER MODE")
print("=" * 80)

# ============================================================================
# LOAD MODEL
# ============================================================================
print("\n[1/4] Loading trained model...")
try:
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print(f"Loading: {MODEL_PATH}")

    # Handle PosixPath for cross-platform compatibility
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    try:
        learn = load_learner(MODEL_PATH, cpu=not torch.cuda.is_available())
        print("✓ Model loaded successfully")
    finally:
        pathlib.PosixPath = temp

    # Optimize for inference
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        learn.model.cuda()
    else:
        print("⚠ Using CPU (slower)")

    learn.model.eval()

except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Load normalization parameters
norm_params = None
if Path(NORMALIZATION_PARAMS_PATH).exists():
    with open(NORMALIZATION_PARAMS_PATH, 'r') as f:
        norm_params = json.load(f)
    print(f"✓ Normalization params loaded")

# ============================================================================
# SETUP VIRTUAL XBOX 360 CONTROLLER
# ============================================================================
print("\n[2/4] Initializing virtual Xbox 360 controller...")
try:
    gamepad = vg.VX360Gamepad()
    print("✓ Virtual Xbox 360 controller created")
    print("  Configure Assetto Corsa to use Xbox controller!")
    print("  Left stick = Steering, Right trigger = Throttle, Left trigger = Brake")
except Exception as e:
    print(f"✗ Failed to create controller: {e}")
    print("  Install ViGEmBus: https://github.com/ViGEm/ViGEmBus/releases")
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
# AI DRIVER CLASS
# ============================================================================

class AIDriverXbox:
    """AI driver using Xbox 360 controller with constant throttle"""

    def __init__(self, model, gamepad, constant_throttle):
        self.model = model
        self.gamepad = gamepad
        self.constant_throttle = constant_throttle


        # Statistics
        self.frame_count = 0
        self.total_time = 0.0

        print(f"✓ AI Driver initialized (throttle: {constant_throttle:.2f})")


    def capture_and_preprocess(self):
        """Capture screen and convert to PIL Image"""
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        return img

    def get_prediction_fast(self, img):
        """
        Fast prediction using direct model inference.   
        Returns steering, throttle, and brake predictions.
        """
        # Resize and convert to tensor
        img_resized = img.resize((224, 224), Image.LANCZOS)
        img_tensor = tensor(np.array(img_resized)).permute(2, 0, 1).float() / 255.0

        # ImageNet normalization
        mean = tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        # Move to GPU if available
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        # Inference
        with torch.no_grad():
            preds = self.model(img_tensor)

        # Extract all three controls
        steer = float(preds[0, 0])
        throttle = float(preds[0, 1])
        brake = float(preds[0, 2])


        return steer, throttle, brake


    def send_controls(self, steer, throttle, brake):
        """Send controls to Xbox 360 controller"""
        # Clamp values
        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        # Send to gamepad
        self.gamepad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
        self.gamepad.right_trigger_float(value_float=throttle)
        self.gamepad.left_trigger_float(value_float=brake)
        self.gamepad.update()

    def reset_controls(self):
        """Reset all controls to neutral"""
        self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.right_trigger_float(value_float=0.0)
        self.gamepad.left_trigger_float(value_float=0.0)
        self.gamepad.update()


    def run(self):
        """Main inference loop"""


        print("\n" + "=" * 80)
        print("CONTROLS")
        print("=" * 80)
        print("  'a' - START autonomous driving")
        print("  's' - STOP autonomous driving")
        print("  'q' - QUIT program")
        print("  '+' - Increase throttle (+0.05)")
        print("  '-' - Decrease throttle (-0.05)")
        print("=" * 80)
        print("\n⏸  Waiting for command... (Press 'a' to start)")

        driving = False
        fps_update_interval = 10

        try:
            while True:
                # Keyboard controls
                if keyboard.is_pressed('a') and not driving:
                    print("\n▶  STARTING autonomous driving...")
                    print(f"   Throttle: {self.constant_throttle:.2f}")
                    driving = True
                    self.frame_count = 0
                    start_time = time.time()
                    time.sleep(0.2)

                elif keyboard.is_pressed('s') and driving:
                    print("\n⏸  STOPPING autonomous driving...")
                    driving = False
                    self.reset_controls()
                    print("   Controls reset to neutral")
                    time.sleep(0.2)

                elif keyboard.is_pressed('q'):
                    print("\n⏹  QUITTING...")
                    break

                elif keyboard.is_pressed('+') or keyboard.is_pressed('='):
                    self.constant_throttle = min(1.0, self.constant_throttle + 0.05)
                    print(f"\nThrottle: {self.constant_throttle:.2f}")
                    time.sleep(0.2)

                elif keyboard.is_pressed('-') or keyboard.is_pressed('_'):
                    self.constant_throttle = max(0.0, self.constant_throttle - 0.05)
                    print(f"\nThrottle: {self.constant_throttle:.2f}")
                    time.sleep(0.2)


                if driving:
                    loop_start = time.time()

                    try:
                        # Capture screen
                        img = self.capture_and_preprocess()


                        # Get predictions
                        steer, throttle_pred, brake = self.get_prediction_fast(img)


                        # Use constant throttle, use predicted brake
                        throttle = self.constant_throttle
                        brake = np.clip(brake, 0.0, 1.0)


                        # Send to controller
                        self.send_controls(steer, throttle, brake)


                        self.frame_count += 1

                        # Display debug info
                        if SHOW_DEBUG and self.frame_count % fps_update_interval == 0:
                            elapsed = time.time() - start_time
                            current_fps = self.frame_count / elapsed if elapsed > 0 else 0

                            print(f"Frame {self.frame_count:4d} | "
                                f"FPS: {current_fps:5.1f} | "
                                f"Steer: {steer:+.3f} | "
                                f"Throttle: {throttle:.2f} | "
                                f"Brake: {brake:.2f}")

                        # Frame rate limiting
                        inference_time = time.time() - loop_start
                        target_frame_time = 1.0 / TARGET_FPS

                        if inference_time < target_frame_time:
                            time.sleep(target_frame_time - inference_time)

                    except Exception as e:
                        print(f"\n⚠ Inference error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                else:
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n\n⏹  Interrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("\n" + "=" * 80)
        print("CLEANUP")
        print("=" * 80)

        self.reset_controls()
        print("✓ Controller reset")

        sct.close()
        print("✓ Screen capture closed")

        if self.frame_count > 0:
            avg_fps = self.frame_count / self.total_time if self.total_time > 0 else 0
            print(f"\nStatistics:")
            print(f"  Frames: {self.frame_count}")
            print(f"  Avg FPS: {avg_fps:.1f}")

        print("\n" + "=" * 80)
        print("SESSION COMPLETE")
        print("=" * 80)

# ============================================================================
# WARM-UP
# ============================================================================
print("\n[4/4] Warming up model...")
try:
    dummy_img = Image.new('RGB', (224, 224))
    img_tensor = tensor(np.array(dummy_img)).permute(2, 0, 1).float() / 255.0
    mean = tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    with torch.no_grad():
        _ = learn.model(img_tensor)

    print("✓ Model warmed up")
except Exception as e:
    print(f"⚠ Warm-up warning: {e}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Initialize AI driver
    driver = AIDriverXbox(
        model=learn.model,
        gamepad=gamepad,
        constant_throttle=CONSTANT_THROTTLE
    )

    # Run inference loop
    driver.run()

if __name__ == "__main__":
    main()
