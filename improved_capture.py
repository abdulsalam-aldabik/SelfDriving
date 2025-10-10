import mss
import os
import time
import datetime
import keyboard
import csv
from PIL import Image
from pyaccsharedmemory import accSharedMemory
import win32api

# Output settings
OUTPUT_DIR = "ac_training_data"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

class Capture:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = {
            "top": 0,
            "left": 0,
            "width": win32api.GetSystemMetrics(0),
            "height": win32api.GetSystemMetrics(1)
        }
        self.asm = accSharedMemory()
        # CSV path and header setup
        self.csv_path = os.path.join(OUTPUT_DIR, "labels.csv")
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'steer', 'throttle', 'brake', 'speed'])
        print("Capture initialized successfully and CSV header written.")

    def read_telemetry_data(self):
        try:
            sm = self.asm.read_shared_memory()
            if sm is not None:
                steer = max(-1.0, min(1.0, sm.Physics.steer_angle))
                return {
                    'speed': sm.Physics.speed_kmh,
                    'steer': steer,
                    'throttle': sm.Physics.gas,
                    'brake': sm.Physics.brake,
                }
            return None
        except Exception as e:
            print(f"Error reading telemetry: {e}")
            return None

    def capture_frame(self):
        telemetry = self.read_telemetry_data()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        if telemetry:
            filename = f"frame_{timestamp}.png"
            filepath = os.path.join(FRAMES_DIR, filename)

            try:
                screenshot = self.sct.grab(self.monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                img = img.resize((224, 224), Image.LANCZOS)
                img.save(filepath)
                
                
                steer = round(float(max(-1.0, min(1.0, telemetry['steer']))),1)
                if steer == 0.0:
                    steer = 0.0

                
                    
                throttle = telemetry['throttle']
                brake = telemetry['brake']
                speed = telemetry['speed']

                # Write directly to CSV for every frame
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        os.path.join("frames", filename),
                        steer,
                        round(throttle, 1),
                        round(brake, 1),
                        round(speed, 1)
                    ])

                print(f"Frame: steer={telemetry['steer']:.3f}, throttle={telemetry['throttle']:.2f}, speed={telemetry['speed']:.2f}")
                return True
            except Exception as e:
                print(f"Capture failed: {e}")
                return False
        return False

    def cleanup(self):
        if hasattr(self, 'asm') and self.asm:
            self.asm.close()
        if hasattr(self, 'sct') and self.sct:
            self.sct.close()

def main():
    print("Press 'c' to start/stop capture, 'q' to quit and save")

    capture = Capture()
    telemetry = capture.read_telemetry_data()

    if telemetry is None:
        print("WARNING: Cannot connect to AC telemetry!")
        return
    else:
        print(f"✓ Connected. Speed: {telemetry['speed']:.1f} km/h")

    capturing = False
    try:
        while True:
            if keyboard.is_pressed('c') and not capturing:
                print("Starting capture at ~30 FPS...")
                capturing = True
                time.sleep(0.2)
            elif keyboard.is_pressed('c') and capturing:
                print("Pausing capture...")
                capturing = False
                time.sleep(0.2)
            elif keyboard.is_pressed('q'):
                print("Stopping and saving...")
                break

            if capturing:
                capture.capture_frame()
                time.sleep(0.033)  # ~30 FPS
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nInterrupted...")
    finally:
        capture.cleanup()

if __name__ == "__main__":
    main()
