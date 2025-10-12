"""
Assetto Corsa Data Capture Script
Captures game frames and telemetry data for training autonomous driving model
Controls: Press 'c' to start/stop capture, 'q' to quit
"""

import mss
import os
import time
import datetime
import keyboard
import csv
from PIL import Image
from pyaccsharedmemory import accSharedMemory
import win32api


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "ac_training_data"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
CAPTURE_FPS = 30
CAPTURE_DELAY = 1.0 / CAPTURE_FPS

os.makedirs(FRAMES_DIR, exist_ok=True)


# ============================================================================
# CAPTURE CLASS
# ============================================================================

class Capture:
    """Handles screen capture and telemetry data recording"""
    
    def __init__(self):
        # Initialize screen capture
        self.sct = mss.mss()
        self.monitor = {
            "top": 0,
            "left": 0,
            "width": win32api.GetSystemMetrics(0),
            "height": win32api.GetSystemMetrics(1)
        }
        
        # Initialize Assetto Corsa shared memory
        self.asm = accSharedMemory()
        
        # Setup CSV file with headers
        self.csv_path = os.path.join(OUTPUT_DIR, "labels.csv")
        file_exists = os.path.isfile(self.csv_path)
        if not file_exists:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image', 'steer', 'throttle', 'brake', 'speed'])
    
    def read_telemetry_data(self):
        """Read telemetry data from Assetto Corsa shared memory"""
        try:
            sm = self.asm.read_shared_memory()
            if sm is not None:
                # Clamp steering to valid range [-1.0, 1.0]
                steer = max(-1.0, min(1.0, sm.Physics.steer_angle))
                return {
                    'speed': sm.Physics.speed_kmh,
                    'steer': steer,
                    'throttle': sm.Physics.gas,
                    'brake': sm.Physics.brake,
                }
            return None
        except Exception:
            return None
    
    def capture_frame(self):
        """Capture single frame with telemetry data and save to disk"""
        telemetry = self.read_telemetry_data()
        if not telemetry:
            return False
        
        # Generate unique timestamp-based filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"frame_{timestamp}.png"
        filepath = os.path.join(FRAMES_DIR, filename)
        
        try:
            # Capture screenshot and convert to RGB
            screenshot = self.sct.grab(self.monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            
            # Resize to 224x224 for neural network input
            img = img.resize((224, 224), Image.LANCZOS)
            img.save(filepath)
            
            # Round telemetry values for consistency
            steer = round(float(max(-1.0, min(1.0, telemetry['steer']))), 1)
            if steer == 0.0:
                steer = 0.0
            throttle = round(telemetry['throttle'], 1)
            brake = round(telemetry['brake'], 1)
            speed = round(telemetry['speed'], 1)
            
            # Append to CSV file
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    os.path.join("frames", filename),
                    steer,
                    throttle,
                    brake,
                    speed
                ])
            
            return True
        except Exception:
            return False
    
    def cleanup(self):
        """Release resources"""
        if hasattr(self, 'asm') and self.asm:
            self.asm.close()
        if hasattr(self, 'sct') and self.sct:
            self.sct.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("Press 'c' to start/stop capture, 'q' to quit")
    
    capture = Capture()
    
    # Test telemetry connection
    telemetry = capture.read_telemetry_data()
    if telemetry is None:
        print("ERROR: Cannot connect to AC telemetry. Is the game running?")
        return
    
    print(f"Connected. Speed: {telemetry['speed']:.1f} km/h")
    
    capturing = False
    frame_count = 0
    
    try:
        while True:
            # Toggle capture on 'c' press
            if keyboard.is_pressed('c') and not capturing:
                print("Capture started...")
                capturing = True
                time.sleep(0.2)
            elif keyboard.is_pressed('c') and capturing:
                print(f"Capture paused. Frames captured: {frame_count}")
                capturing = False
                time.sleep(0.2)
            elif keyboard.is_pressed('q'):
                print(f"Stopping. Total frames: {frame_count}")
                break
            
            # Capture frame if active
            if capturing:
                if capture.capture_frame():
                    frame_count += 1
                time.sleep(CAPTURE_DELAY)
            else:
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        print(f"\nInterrupted. Total frames: {frame_count}")
    finally:
        capture.cleanup()


if __name__ == "__main__":
    main()
