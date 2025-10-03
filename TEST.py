import mss
import os
import time
import datetime
import keyboard
from PIL import Image
from pyaccsharedmemory import accSharedMemory
import win32api

# Output settings
OUTPUT_DIR = "ac_screenshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ScreenshotCapture:
    def __init__(self):
        """Initialize screenshot capture and telemetry reader"""
        # Initialize MSS
        self.sct = mss.mss()
        
        # Get screen dimensions
        self.monitor = {
            "top": 0,
            "left": 0, 
            "width": win32api.GetSystemMetrics(0),
            "height": win32api.GetSystemMetrics(1)
        }
        
        # Initialize Assetto Corsa shared memory
        self.asm = accSharedMemory()
        print("MSS and AC telemetry initialized successfully")

    def read_telemetry_data(self):
        """Read telemetry data from Assetto Corsa shared memory"""
        try:
            sm = self.asm.read_shared_memory()
            
            if sm is not None:
                return {
                    'speed': sm.Physics.speed_kmh,
                    'steering': sm.Physics.steer_angle,
                    'throttle': sm.Physics.gas,
                    'brake': sm.Physics.brake,
                }
            return None
        except Exception as e:
            print(f"Error reading telemetry: {e}")
            return None

    def capture_screenshot(self):
        """Capture screenshot with telemetry data"""
        telemetry = self.read_telemetry_data()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Handle telemetry data
        if telemetry:
            speed = telemetry['speed']
            brake = telemetry['brake']
            steering = telemetry['steering']
            throttle = telemetry['throttle']

            steer_right = 0
            steer_left = 0

            if steering == 0.0:
                steer_str  = 0
            elif steering > 0.01:
                steer_right  = 1
            else:
                steer_left  = 1
            
            filename = (f"{OUTPUT_DIR}/ac_{timestamp}_"
                       f"speed_{speed:.1f}_brake_{brake:.2f}_"
                       f"streer_right_{steer_right}_throttle_{throttle:.2f}"
                       f"steer_left_{steer_left}.png")
        else:
            filename = f"{OUTPUT_DIR}/ac_{timestamp}_no_telemetry.png"
        
        # Capture screenshot using MSS
        try:
            screenshot = self.sct.grab(self.monitor)
            # Convert to PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            img.save(filename)
            print(f"Saved {filename}")
            
            # Display telemetry info
            if telemetry:
                print(f"  Speed: {telemetry['speed']:.1f} km/h")
                print(f"  Steering: {telemetry['steering']:.3f}")
                print(f"  Throttle: {telemetry['throttle']:.2f}")
                print(f"  Brake: {telemetry['brake']:.2f}")
            return True
            
        except Exception as e:
            print(f"Failed to capture screenshot: {e}")
            return False

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'asm') and self.asm:
            self.asm.close()
        if hasattr(self, 'sct') and self.sct:
            self.sct.close()

def main():
    print("Screenshot capture ready. Press 'c' to capture, 'q' to quit.")
    print("Make sure Assetto Corsa is running and in a session!")
    
    try:
        capture = ScreenshotCapture()
        
        # Test telemetry connection
        telemetry = capture.read_telemetry_data()
        if telemetry is None:
            print("WARNING: Cannot connect to Assetto Corsa telemetry!")
            print("Make sure Assetto Corsa is running and you're in a session.")
            print("Screenshots will still work but without telemetry data.")
        else:
            print(f"✓ Connected to AC telemetry. Current speed: {telemetry['speed']:.1f} km/h")
        
        capturing = False
        while True:
            try:
                if keyboard.is_pressed('c') and not capturing:
                    print("Starting continuous capture (1 screenshot/second)...")
                    capturing = True
                    time.sleep(0.2)  # Prevent multiple toggles
                    
                elif keyboard.is_pressed('q'):
                    print("Exiting...")
                    break
                
                if capturing:
                    capture.capture_screenshot()
                    time.sleep(0.033)  # 1 screenshot per second
                else:
                    time.sleep(0.01)  # Small delay to reduce CPU usage
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
                
    except Exception as e:
        print(f"Error: {e}")
        return
    finally:
        if 'capture' in locals():
            capture.cleanup()


if __name__ == "__main__":
    main()
