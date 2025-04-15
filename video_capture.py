import cv2
import numpy as np
import time
import threading
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

class VideoCapture:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Capture")
        self.root.geometry("800x600")
        
        # Setup UI
        self.setup_ui()
        
        # Initialize video capture with optimized settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Full HD resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        # Performance tracking
        self.frame_times = []
        
        # Control variables
        self.running = True
        self.current_frame = None
        self.frame_count = 0
        
        # Start video thread
        self.thread = threading.Thread(target=self.video_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video display
        self.video_frame = ttk.LabelFrame(main_frame, text="Video Feed")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # FPS display
        self.fps_label = ttk.Label(control_frame, text="FPS: 0")
        self.fps_label.pack(side=tk.LEFT, padx=20, pady=5)
        
        # Start/Stop button
        self.toggle_button = ttk.Button(control_frame, text="Stop", command=self.toggle_capture)
        self.toggle_button.pack(side=tk.RIGHT, padx=20, pady=5)
    
    def toggle_capture(self):
        if self.running:
            self.running = False
            self.toggle_button.config(text="Start")
        else:
            self.running = True
            self.toggle_button.config(text="Stop")
            if not self.thread.is_alive():
                self.thread = threading.Thread(target=self.video_loop)
                self.thread.daemon = True
                self.thread.start()
    
    def preprocess_frame(self, frame):
        """Enhance video frame quality"""
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        
        # Replace enhanced luminance channel
        lab[:, :, 0] = enhanced_l
        enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_frame
    
    def video_loop(self):
        """Main video processing loop"""
        while self.running:
            start_time = time.time()
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Store current frame
            self.current_frame = frame.copy()
            
            # Optional: Apply video enhancements
            display_frame = self.preprocess_frame(frame)
            
            # Add frame count and timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, f"Frame: {self.frame_count} | {timestamp}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Convert to format for Tkinter
            cv_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            tk_image = ImageTk.PhotoImage(image=pil_image)
            
            # Update UI
            self.video_label.configure(image=tk_image)
            self.video_label.image = tk_image
            
            # Update FPS counter
            self.frame_times.append(time.time() - start_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
                
            fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            
            self.frame_count += 1
            
            # Control loop speed for consistent frame rate
            elapsed = time.time() - start_time
            if elapsed < 0.033:  # Target ~30fps
                time.sleep(0.033 - elapsed)
    
    def release(self):
        """Release resources"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoCapture(root)
    
    # Handle window close event
    def on_closing():
        app.release()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop() 