import cv2
import time
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")
        self.root.geometry("900x600")
        
        # Initialize variables before UI setup
        self.frame_times = []
        self.detection_times = []
        self.frame_count = 0
        self.detection_interval = 5  # Only run detection every N frames
        self.faces = []
        self.running = True
        
        # Setup UI
        self.setup_ui()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.log_message("Error: Could not open camera.")
            return
            
        self.log_message("Camera initialized successfully.")
        
        # Get camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.log_message(f"Camera properties: {width}x{height} at {fps} FPS")
        
        # Initialize face detector (using Haar cascade for simplicity)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            self.log_message(f"Warning: Could not find cascade file at {cascade_path}")
            self.log_message("Will attempt to use a generic path")
            cascade_path = 'haarcascade_frontalface_default.xml'
        
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        
        # Start processing thread
        self.thread = threading.Thread(target=self.video_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Video display
        self.video_frame = ttk.LabelFrame(main_frame, text="Video Feed")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel - Controls and info
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(control_frame, text="Performance")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.fps_label = ttk.Label(stats_frame, text="FPS: 0")
        self.fps_label.pack(padx=5, pady=2)
        
        self.detection_label = ttk.Label(stats_frame, text="Detection: 0ms")
        self.detection_label.pack(padx=5, pady=2)
        
        self.face_count_label = ttk.Label(stats_frame, text="Faces: 0")
        self.face_count_label.pack(padx=5, pady=2)
        
        # Control buttons
        button_frame = ttk.LabelFrame(control_frame, text="Controls")
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_stop_btn = ttk.Button(button_frame, text="Stop", command=self.toggle_capture)
        self.start_stop_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.detection_interval_lbl = ttk.Label(button_frame, text="Detection Interval:")
        self.detection_interval_lbl.pack(padx=5, pady=2)
        
        interval_frame = ttk.Frame(button_frame)
        interval_frame.pack(padx=5, pady=2, fill=tk.X)
        
        interval_values = [1, 2, 5, 10, 15]
        self.interval_var = tk.IntVar(value=self.detection_interval)
        
        for val in interval_values:
            ttk.Radiobutton(interval_frame, text=str(val), variable=self.interval_var, 
                           value=val, command=self.update_interval).pack(side=tk.LEFT, padx=5)
        
        # Log frame
        log_frame = ttk.LabelFrame(control_frame, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_frame, height=10, width=30)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
    
    def toggle_capture(self):
        if self.running:
            self.running = False
            self.start_stop_btn.config(text="Start")
            self.log_message("Video capture stopped")
        else:
            self.running = True
            self.start_stop_btn.config(text="Stop")
            self.log_message("Video capture started")
            if not self.thread.is_alive():
                self.thread = threading.Thread(target=self.video_loop)
                self.thread.daemon = True
                self.thread.start()
    
    def update_interval(self):
        self.detection_interval = self.interval_var.get()
        self.log_message(f"Detection interval set to {self.detection_interval}")
    
    def video_loop(self):
        self.log_message("Video processing started...")
        
        while self.running:
            start_time = time.time()
            
            # Capture frame
            ret, frame = self.cap.read()
            
            if not ret:
                self.log_message("Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Store current frame
            display_frame = frame.copy()
            
            # Run face detection periodically
            if self.frame_count % self.detection_interval == 0:
                detection_start = time.time()
                
                # Convert to grayscale for faster detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                self.faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                detection_time = time.time() - detection_start
                self.detection_times.append(detection_time)
                
                # Keep only the last 20 measurements
                if len(self.detection_times) > 20:
                    self.detection_times.pop(0)
            
            # Draw rectangles around faces
            for (x, y, w, h) in self.faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Face", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate and display FPS
            self.frame_times.append(time.time() - start_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            
            # Convert to format for Tkinter
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            tk_image = ImageTk.PhotoImage(image=pil_image)
            
            # Update UI
            self.video_label.configure(image=tk_image)
            self.video_label.image = tk_image
            
            # Update stats every 10 frames
            if self.frame_count % 10 == 0:
                fps = 1.0 / (sum(self.frame_times) / max(1, len(self.frame_times)))
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                
                if self.detection_times:
                    avg_detection = sum(self.detection_times) / len(self.detection_times) * 1000
                    self.detection_label.config(text=f"Detection: {avg_detection:.1f}ms")
                
                self.face_count_label.config(text=f"Faces: {len(self.faces)}")
            
            self.frame_count += 1
            
            # Control loop speed
            elapsed = time.time() - start_time
            if elapsed < 0.033:  # ~30fps
                time.sleep(0.033 - elapsed)
    
    def release(self):
        """Release resources"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        
        self.log_message("Resources released")

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    
    def on_closing():
        app.release()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop() 