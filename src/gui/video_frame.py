import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import queue

from ..utils.error_handler import ErrorHandler
from ..utils.performance import PerformanceMonitor, Timer
from ..utils.config import config

class VideoFrame:
    def __init__(self, parent, detector, recognizer, tracker, error_recovery):
        """Initialize video frame component"""
        self.parent = parent
        self.detector = detector
        self.recognizer = recognizer
        self.tracker = tracker
        self.error_recovery = error_recovery
        
        # Create frame and components
        self.setup_ui()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, config.target_fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.video_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.video_height)
        
        # Processing state
        self.running = True
        self.current_frame = None
        self.last_frame_time = time.time()
        self.last_detection_time = 0
        self.frame_interval = 1.0 / config.target_fps
        
        # Store last known detections
        self.current_detections = []
        self.current_names = []
        
        # Create frame queue
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor(
            window_size=config.performance_window,
            target_fps=config.target_fps
        )
        
        # Start processing threads
        self.process_thread = threading.Thread(target=self.process_video)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Schedule first frame update
        self.schedule_next_frame()

    def setup_ui(self):
        """Set up video frame UI components"""
        self.frame = ttk.LabelFrame(self.parent, text="Video Feed")
        self.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = ttk.Label(self.frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.status_frame = ttk.Frame(self.frame)
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.fps_label = ttk.Label(self.status_frame, text="FPS: 0")
        self.fps_label.pack(side=tk.LEFT, padx=5)
        
        self.faces_label = ttk.Label(self.status_frame, text="Faces: 0")
        self.faces_label.pack(side=tk.LEFT, padx=5)

    def schedule_next_frame(self):
        """Schedule next frame update at precise interval"""
        if self.running:
            next_frame = self.last_frame_time + self.frame_interval
            now = time.time()
            delay = int(max(0, (next_frame - now) * 1000))
            self.parent.after(delay, self.update_frame)

    def update_frame(self):
        """Update frame display"""
        try:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                if self.error_recovery.log_error("Frame Capture", "Failed to grab frame"):
                    self.reset_camera()
                self.schedule_next_frame()
                return
            
            # Store current frame
            self.current_frame = frame.copy()
            
            # Add to processing queue
            try:
                self.frame_queue.put_nowait((frame, time.time()))
            except queue.Full:
                pass
            
            # Update timing
            now = time.time()
            frame_time = now - self.last_frame_time
            self.perf_monitor.update_frame_time(frame_time)
            self.last_frame_time = now
            
            # Schedule next frame
            self.schedule_next_frame()
            
        except Exception as e:
            self.error_recovery.log_error("Frame Update", str(e))
            self.schedule_next_frame()

    def process_video(self):
        """Process frames for face detection"""
        while self.running:
            try:
                # Get frame from queue
                frame, frame_time = self.frame_queue.get()
                display_frame = frame.copy()
                
                # Check if it's time for detection
                current_time = time.time()
                should_detect = (current_time - self.last_detection_time) >= (config.detection_interval / 1000.0)
                
                if should_detect:
                    self.last_detection_time = current_time
                    
                    # Detect and recognize faces
                    detections, used_insightface = self.detector.detect_faces(frame)
                    
                    if detections:
                        if used_insightface:
                            embeddings = self.detector.get_face_embeddings(frame, detections)
                            if embeddings:
                                recognized = self.recognizer.recognize_faces(embeddings)
                                names = [name for name, _ in recognized]
                            else:
                                names = ["Unknown"] * len(detections)
                        else:
                            names = ["Unknown"] * len(detections)
                            
                        # Update current detections
                        self.current_detections = detections
                        self.current_names = names
                
                # Always draw current detections
                if self.current_detections:
                    display_frame = self.detector.draw_detections(
                        display_frame, 
                        self.current_detections,
                        self.current_names
                    )
                
                # Update display
                self.update_display(display_frame)
                
                # Update status
                fps = 1.0 / (time.time() - frame_time)
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                self.faces_label.config(text=f"Faces: {len(self.current_detections)}")
                
            except Exception as e:
                self.error_recovery.log_error("Frame Processing", str(e))
                time.sleep(0.01)

    def update_display(self, frame):
        """Update video display with new frame"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=pil_image)
            self.video_label.configure(image=photo)
            self.video_label.image = photo
        except Exception as e:
            self.error_recovery.log_error("Display Update", str(e))

    def reset_camera(self):
        """Reset camera connection"""
        self.cap.release()
        time.sleep(1)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, config.target_fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.video_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.video_height)

    def get_current_frame(self):
        """Get copy of current frame"""
        return self.current_frame.copy() if self.current_frame is not None else None

    def stop(self):
        """Stop video processing"""
        self.running = False
        if self.cap is not None:
            self.cap.release()