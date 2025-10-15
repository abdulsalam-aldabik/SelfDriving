"""
YOLOv11 Real-Time Traffic Sign Detection for Assetto Corsa
Detects signs with speed limits in real-time
High-performance screen capture + YOLO detection
"""

import cv2
import numpy as np
import mss
import time
from ultralytics import YOLO
import easyocr
from collections import deque

class GameTrafficSignDetector:
    """Real-time traffic sign detector for racing games"""
    
    def __init__(self, model_path, monitor_number=1):
        """
        Initialize detector
        Args:
            model_path: Path to trained YOLO model (.pt file)
            monitor_number: Which monitor to capture (1=primary)
        """
        print("Initializing Traffic Sign Detector...")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        print(f"✓ YOLO model loaded: {model_path}")
        
        # Initialize OCR for speed limit detection
        self.reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        print("✓ OCR initialized")
        
        # Screen capture
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_number]
        print(f"✓ Capturing monitor {monitor_number}: {self.monitor['width']}x{self.monitor['height']}")
        
        # Speed limit detection
        self.speed_limits = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130]
        self.current_speed_limit = None
        self.speed_limit_confidence = 0
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.fps = 0
        
        # Colors for visualization
        self.colors = {
            'prohibitory': (0, 0, 255),    # Red
            'danger': (0, 165, 255),       # Orange
            'mandatory': (255, 0, 0),      # Blue
            'other': (0, 255, 0)           # Green
        }
        
    def capture_screen(self):
        """Capture screen using MSS (fast)"""
        img = np.array(self.sct.grab(self.monitor))
        # Convert BGRA to BGR
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    def detect_speed_limit(self, img_crop):
        """Extract speed limit number from cropped sign"""
        try:
            # Preprocess for better OCR
            gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Threshold
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR
            results = self.reader.readtext(thresh, detail=1)
            
            # Extract numbers
            for (bbox, text, conf) in results:
                # Get only digits
                numbers = ''.join(filter(str.isdigit, text))
                
                if numbers and len(numbers) >= 2:
                    speed = int(numbers)
                    # Match to closest standard speed limit
                    closest = min(self.speed_limits, key=lambda x: abs(x - speed))
                    if abs(closest - speed) <= 10 and conf > 0.3:
                        return closest, conf
            
            return None, 0
        except:
            return None, 0
    
    def is_circular(self, img_crop):
        """Quick check if sign is circular (speed limits are circular)"""
        h, w = img_crop.shape[:2]
        if h < 20 or w < 20:
            return False
        
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, 
            minRadius=int(min(h, w) * 0.3), 
            maxRadius=int(min(h, w) * 0.5)
        )
        return circles is not None
    
    def process_frame(self, frame):
        """Detect signs and recognize speed limits"""
        # YOLO detection
        results = self.model.predict(frame, conf=0.4, verbose=False, device=0)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box info
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls]
                
                # Crop sign
                img_crop = frame[y1:y2, x1:x2]
                
                # Try to detect speed limit on prohibitory signs
                speed_limit = None
                speed_conf = 0
                
                if class_name == 'prohibitory' and img_crop.size > 0:
                    if self.is_circular(img_crop):
                        speed_limit, speed_conf = self.detect_speed_limit(img_crop)
                        
                        # Update current speed limit with confidence
                        if speed_limit and speed_conf > self.speed_limit_confidence:
                            self.current_speed_limit = speed_limit
                            self.speed_limit_confidence = speed_conf
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class': class_name,
                    'confidence': conf,
                    'speed_limit': speed_limit
                })
        
        return detections
    
    def draw_overlay(self, frame, detections):
        """Draw detections with overlay"""
        overlay = frame.copy()
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            conf = det['confidence']
            speed = det['speed_limit']
            
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Draw semi-transparent box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            if speed:
                label = f"{speed} km/h"
                # Bigger font for speed limits
                font_scale = 1.0
                thickness = 3
            else:
                label = f"{class_name}"
                font_scale = 0.6
                thickness = 2
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(overlay, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(overlay, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Draw HUD - Current Speed Limit
        if self.current_speed_limit:
            hud_text = f"SPEED LIMIT: {self.current_speed_limit} km/h"
            hud_size = 1.5
            hud_thickness = 4
            (w, h), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, hud_size, hud_thickness)
            
            # Draw HUD background
            hud_x = (frame.shape[1] - w) // 2
            hud_y = 50
            cv2.rectangle(overlay, (hud_x - 20, hud_y - h - 20), 
                         (hud_x + w + 20, hud_y + 10), (0, 0, 0), -1)
            cv2.putText(overlay, hud_text, (hud_x, hud_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, hud_size, (0, 0, 255), hud_thickness)
        
        # Draw FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(overlay, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Blend overlay with original frame
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame
    
    def run(self, show_window=True):
        """Main detection loop"""
        print("\n" + "="*60)
        print("TRAFFIC SIGN DETECTOR - RUNNING")
        print("="*60)
        print("Controls:")
        print("  Q - Quit")
        print("  F - Toggle fullscreen")
        print("  H - Toggle HUD")
        print("="*60 + "\n")
        
        show_hud = True
        
        try:
            while True:
                start_time = time.time()
                
                # Capture screen
                frame = self.capture_screen()
                
                # Detect signs
                detections = self.process_frame(frame)
                
                # Draw overlay
                if show_hud:
                    frame = self.draw_overlay(frame, detections)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                self.fps_history.append(1.0 / elapsed if elapsed > 0 else 0)
                self.fps = np.mean(self.fps_history)
                
                # Show frame
                if show_window:
                    # Resize for display (optional)
                    display_frame = cv2.resize(frame, (1280, 720))
                    cv2.imshow('Traffic Sign Detector', display_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('h'):
                        show_hud = not show_hud
                
        except KeyboardInterrupt:
            print("\nStopping detector...")
        finally:
            cv2.destroyAllWindows()
            print("Detector stopped.")


if __name__ == '__main__':
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    # Path to your trained model
    MODEL_PATH = 'yolo_traffic_signs/runs/train/weights/best.pt'
    
    # Which monitor to capture (1 = primary monitor, 2 = secondary, etc.)
    MONITOR = 1
    
    # ============================================================
    # RUN DETECTOR
    # ============================================================
    
    # Initialize detector
    detector = GameTrafficSignDetector(
        model_path=MODEL_PATH,
        monitor_number=MONITOR
    )
    
    # Run detection
    detector.run(show_window=True)
