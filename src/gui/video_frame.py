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
        self.next_frame_time = 0  # For frame timing
        
        # Create queues for thread-safe updates
        self.frame_queue = queue.Queue(maxsize=2)  # Frame buffer
        self.ui_queue = queue.Queue()  # UI updates
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor(
            window_size=config.performance_window,
            target_fps=config.target_fps
        )
        
        # Start processing threads
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)
        self.capture_thread.daemon = True
        self.process_thread.daemon = True
        self.capture_thread.start()
        self.process_thread.start()
        
        # Start UI update loop
        self.update_ui()

    def setup_ui(self):
        """Set up video frame UI components"""
        # Create main frame
        self.frame = ttk.LabelFrame(self.parent, text="Video Feed")
        self.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create video label
        self.video_label = ttk.Label(self.frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create status labels
        self.status_frame = ttk.Frame(self.frame)
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.fps_label = ttk.Label(self.status_frame, text="FPS: 0")
        self.fps_label.pack(side=tk.LEFT, padx=5)
        
        self.faces_label = ttk.Label(self.status_frame, text="Faces: 0")
        self.faces_label.pack(side=tk.LEFT, padx=5)

    def capture_frames(self):
        """Capture frames at target FPS"""
        while self.running:
            try:
                # Wait until next frame time
                now = time.time()
                if now < self.next_frame_time:
                    time.sleep(max(0, self.next_frame_time - now))
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    if self.error_recovery.log_error("Frame Capture", "Failed to grab frame"):
                        self.reset_camera()
                    time.sleep(0.1)
                    continue
                
                # Store frame and update timing
                self.current_frame = frame.copy()
                
                # Try to add frame to queue without blocking
                try:
                    self.frame_queue.put_nowait((frame, time.time()))
                except queue.Full:
                    # Skip frame if queue is full
                    pass
                
                # Calculate next frame time
                self.next_frame_time = time.time() + config.frame_interval
                
            except Exception as e:
                self.error_recovery.log_error("Frame Capture", str(e))
                time.sleep(0.1)

    def process_frames(self):
        """Process frames and perform detection"""
        while self.running:
            try:
                # Get frame from queue
                frame, frame_time = self.frame_queue.get()
                
                with Timer("Frame Processing"):
                    display_frame = frame.copy()
                    
                    # Update tracker if active
                    if self.tracker.is_tracking():
                        success, tracking_box = self.tracker.update(frame)
                        if success:
                            display_frame = self.tracker.draw_tracking(display_frame)
                    
                    current_time = time.time()
                    
                    # Check if it's time for detection using performance monitor
                    should_detect = self.perf_monitor.should_process_frame(
                        current_time,
                        self.last_detection_time,
                        config.detection_interval
                    )
                    face_count = 0
                    
                    if should_detect or not self.tracker.is_tracking():
                        self.last_detection_time = current_time
                        
                        # Detect faces
                        detections, used_insightface = self.detector.detect_faces(frame)
                        face_count = len(detections)
                        
                        # Get embeddings and recognize faces if using InsightFace
                        if used_insightface and detections:
                            embeddings = self.detector.get_face_embeddings(frame, detections)
                            if embeddings:
                                recognized = self.recognizer.recognize_faces(embeddings)
                                names = [name for name, _ in recognized]
                            else:
                                names = None
                        else:
                            names = None
                        
                        # Draw detections
                        display_frame = self.detector.draw_detections(
                            display_frame, detections, names)
                        
                        # Initialize tracking for first detected face
                        if detections and not self.tracker.is_tracking():
                            self.tracker.init_tracker(frame, detections[0]['bbox'])
                    
                    # Calculate frame time and FPS
                    process_time = time.time() - frame_time
                    fps = 1.0 / process_time if process_time > 0 else 0
                    
                    # Queue UI updates
                    self.ui_queue.put({
                        'frame': display_frame,
                        'fps': fps,
                        'face_count': face_count,
                        'interval': config.detection_interval
                    })
                    
                    # Update performance monitoring
                    self.perf_monitor.update_frame_time(process_time)
                    
            except Exception as e:
                self.error_recovery.log_error("Frame Processing", str(e))
                time.sleep(0.1)

    def update_ui(self):
        """Update UI from main thread"""
        try:
            # Process all pending updates
            while not self.ui_queue.empty():
                update = self.ui_queue.get_nowait()
                
                # Update frame display
                self.update_display(update['frame'])
                
                # Update status labels
                self.fps_label.config(
                    text=f"FPS: {update['fps']:.1f} | Int: {update['interval']}ms")
                self.faces_label.config(
                    text=f"Faces: {update['face_count']}")
        except:
            pass
            
        # Schedule next update
        if self.running:
            self.parent.after(int(1000/config.target_fps), self.update_ui)

    def update_display(self, frame):
        """Update video display with new frame"""
        try:
            # Convert frame to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL image
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update label
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