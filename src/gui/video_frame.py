import cv2
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
import time
import queue
import requests
import numpy as np

from ..utils.error_handler import ErrorHandler
from ..utils.performance import PerformanceMonitor, Timer
from ..utils.config import config

class VideoFrame:
    def preprocess_frame(self, frame):
        """Preprocess frame for face detection
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            tuple: (preprocessed_frame, scale_factor)
        """
        try:
            # Ensure frame is valid
            if frame is None or frame.size == 0:
                print("Invalid frame received in preprocessing")
                return None, 1.0
                
            print(f"Original frame: shape={frame.shape}, dtype={frame.dtype}")
            
            # Get target size from config
            target_width = config.processing_width
            height, width = frame.shape[:2]
            scale = target_width / width
            target_height = int(height * scale)
            
            # Resize frame
            small_frame = cv2.resize(frame, (target_width, target_height))
            print(f"Resized frame: shape={small_frame.shape}, scale={scale}")
            
            # Ensure correct color format
            if len(small_frame.shape) == 2:
                print("Converting grayscale to BGR")
                small_frame = cv2.cvtColor(small_frame, cv2.COLOR_GRAY2BGR)
            elif small_frame.shape[2] == 4:
                print("Converting RGBA to BGR")
                small_frame = cv2.cvtColor(small_frame, cv2.COLOR_RGBA2BGR)
                
            # Ensure uint8 data type
            if small_frame.dtype != np.uint8:
                print(f"Converting {small_frame.dtype} to uint8")
                small_frame = (small_frame * 255).astype(np.uint8)
                
            return small_frame, scale
            
        except Exception as e:
            print(f"Error in frame preprocessing: {str(e)}")
            return None, 1.0
        
    def __init__(self, parent, detector, recognizer, tracker, error_recovery):
        """Initialize video frame component"""
        self.parent = parent
        self.detector = detector
        self.recognizer = recognizer
        self.tracker = tracker
        self.error_recovery = error_recovery
        
        # Initialize feature detection from config
        self.detector.set_feature_detection(
            detect_eyes=config.detect_eyes,
            detect_mouth=config.detect_mouth
        )
        self.detector.set_feature_color(config.feature_color)
        
        # Create frame and components
        self.setup_ui()
        
        # Processing state
        self.running = True
        self.current_frame = None
        self.last_frame_time = time.time()
        self.last_detection_time = 0
        self.frame_interval = 1.0 / config.target_fps
        self.cap = None
        self.frame_count = 0
        
        # Store last known detections and tracking info
        self.current_detections = []
        self.current_names = []
        self.active_trackers = {}  # {tracking_id: (tracker, last_bbox)}
        
        # Create frame queue
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor(
            window_size=config.performance_window,
            target_fps=config.target_fps
        )
        
        # Try to initialize camera
        if not self.init_camera():
            # If camera init fails, show error but continue with UI
            self.error_recovery.log_error("Camera Init", "Failed to initialize camera")
            self.camera_label.config(text="Camera: Not Connected")
        
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
        
        # Create controls container
        controls_container = ttk.Frame(self.frame)
        controls_container.pack(fill=tk.X, padx=5, pady=5)
        
        # Create camera controls
        camera_frame = ttk.LabelFrame(controls_container, text="Camera")
        camera_frame.pack(side=tk.LEFT, fill=tk.X, padx=5)
        
        ttk.Button(
            camera_frame,
            text="Use Webcam",
            command=self.use_webcam
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(
            camera_frame,
            text="Use Phone Camera",
            command=self.setup_ip_camera
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Create feature detection controls
        feature_frame = ttk.LabelFrame(controls_container, text="Face Features")
        feature_frame.pack(side=tk.LEFT, fill=tk.X, padx=5)
        
        # Feature detection checkboxes
        self.detect_eyes_var = tk.BooleanVar(value=config.detect_eyes)
        ttk.Checkbutton(
            feature_frame,
            text="Detect Eyes",
            variable=self.detect_eyes_var,
            command=self.update_feature_detection
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        self.detect_mouth_var = tk.BooleanVar(value=config.detect_mouth)
        ttk.Checkbutton(
            feature_frame,
            text="Detect Mouth",
            variable=self.detect_mouth_var,
            command=self.update_feature_detection
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Color selection
        color_frame = ttk.LabelFrame(controls_container, text="Feature Color")
        color_frame.pack(side=tk.LEFT, fill=tk.X, padx=5)
        
        self.color_var = tk.StringVar(value=config.feature_color)
        for color, text in [("green", "Green"), ("red", "Red"),
                          ("blue", "Blue"), ("yellow", "Yellow")]:
            ttk.Radiobutton(
                color_frame,
                text=text,
                value=color,
                variable=self.color_var,
                command=self.update_feature_color
            ).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Create video display
        self.video_label = ttk.Label(self.frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create status bar
        self.status_frame = ttk.Frame(self.frame)
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.fps_label = ttk.Label(self.status_frame, text="FPS: 0")
        self.fps_label.pack(side=tk.LEFT, padx=5)
        
        self.faces_label = ttk.Label(self.status_frame, text="Faces: 0")
        self.faces_label.pack(side=tk.LEFT, padx=5)
        
        self.camera_label = ttk.Label(self.status_frame, text="Camera: Initializing...")
        self.camera_label.pack(side=tk.RIGHT, padx=5)

    def init_camera(self):
        """Initialize camera capture with optimized settings"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            camera_source = config.get_camera_url()
            self.cap = cv2.VideoCapture(camera_source)
            
            # Set optimized camera properties
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            self.cap.set(cv2.CAP_PROP_FPS, config.target_fps)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Higher resolution capture
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Test connection for IP camera
            if isinstance(camera_source, str):
                if not self.test_ip_camera():
                    return False
            
            if not self.cap.isOpened():
                return False
            
            # Update camera label
            if config.use_ip_camera:
                self.camera_label.config(text="Camera: Phone")
            else:
                self.camera_label.config(text="Camera: Webcam")
            
            return True
                
        except Exception as e:
            self.error_recovery.log_error("Camera Init", str(e))
            return False

    def test_ip_camera(self):
        """Test IP camera connection"""
        try:
            if config.use_ip_camera:
                test_url = config.ip_camera_url.replace('/video', '/shot.jpg')
                response = requests.get(test_url, timeout=3)
                return response.status_code == 200
            return True
        except:
            return False

    def setup_ip_camera(self):
        """Set up IP camera connection"""
        # Ask for IP camera URL
        url = simpledialog.askstring(
            "IP Camera Setup",
            "Enter IP Webcam URL (e.g., http://192.168.1.100:8080/video)",
            initialvalue=config.ip_camera_url
        )
        
        if url:
            # Save settings
            config.set_ip_camera(url)
            
            # Try to initialize camera
            if not self.init_camera():
                messagebox.showerror(
                    "Connection Error",
                    "Failed to connect to IP camera. Check the URL and make sure the phone app is running."
                )
                # Revert to webcam
                self.use_webcam()

    def use_webcam(self):
        """Switch to webcam"""
        config.use_webcam()
        if not self.init_camera():
            messagebox.showerror(
                "Camera Error",
                "Failed to initialize webcam. Please check the connection."
            )
            self.camera_label.config(text="Camera: Not Connected")

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
            if self.cap is None or not self.cap.isOpened():
                # Try to reinitialize camera
                if not self.init_camera():
                    # Show blank frame
                    blank = np.zeros((config.video_height, config.video_width, 3), np.uint8)
                    self.update_display(blank)
                    self.schedule_next_frame()
                    return
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                if self.error_recovery.log_error("Frame Capture", "Failed to grab frame"):
                    self.init_camera()
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
        """Process frames for face detection with tracking"""
        while self.running:
            try:
                # Get frame from queue
                frame, frame_time = self.frame_queue.get()
                if frame is None:
                    print("Received empty frame")
                    continue
                    
                print(f"Processing frame: shape={frame.shape}, dtype={frame.dtype}")
                display_frame = frame.copy()
                
                # Preprocess frame for detection
                small_frame, scale = self.preprocess_frame(frame)
                print(f"Preprocessed frame: shape={small_frame.shape}, scale={scale}")
                
                # Update existing trackers
                tracked_detections = []
                if config.enable_tracking:
                    for track_id, (tracker, last_bbox) in list(self.active_trackers.items()):
                        bbox, success = self.detector.update_tracking(small_frame, tracker, last_bbox)
                        if success:
                            # Scale bbox back to original size
                            scaled_bbox = tuple(int(v/scale) for v in bbox)
                            tracked_detections.append({
                                'bbox': scaled_bbox,
                                'tracking_id': track_id,
                                'confidence': 0.8,  # Tracking confidence
                                'facial_features': self.current_detections[track_id].get('facial_features', {}) if track_id < len(self.current_detections) else {}
                            })
                        else:
                            del self.active_trackers[track_id]
                
                # Check if detection is needed
                should_detect = (
                    self.frame_count % config.detection_interval == 0 or  # Regular interval
                    len(self.active_trackers) < len(self.current_detections) or  # Lost tracks
                    not self.active_trackers  # No active trackers
                )
                
                if should_detect:
                    self.last_detection_time = time.time()
                    
                    # Detect and recognize faces
                    print("Running face detection...")
                    detections, used_insightface = self.detector.detect_faces(small_frame)
                    print(f"Detection result: {len(detections)} faces, used_insightface: {used_insightface}")
                    
                    # Scale detections back to original size
                    if detections:
                        for detection in detections:
                            bbox = detection['bbox']
                            detection['bbox'] = tuple(int(v/scale) for v in bbox)
                        
                        if used_insightface:
                            embeddings = self.detector.get_face_embeddings(small_frame, detections)
                            if embeddings:
                                recognized = self.recognizer.recognize_faces(embeddings)
                                names = [name for name, _ in recognized]
                            else:
                                names = ["Unknown"] * len(detections)
                        else:
                            names = ["Unknown"] * len(detections)
                        
                        # Initialize/update trackers
                        if config.enable_tracking:
                            self.active_trackers.clear()  # Reset trackers
                            for i, detection in enumerate(detections):
                                tracker = self.detector.init_tracking(small_frame, detection)
                                if tracker:
                                    self.active_trackers[i] = (tracker, detection['bbox'])
                        
                        # Update current detections and names
                        self.current_detections = detections
                        self.current_names = names
                
                # Use tracked detections if available, otherwise use last known detections
                display_detections = tracked_detections if tracked_detections else self.current_detections
                
                # Draw detections and tracking info
                if display_detections:
                    # Draw tracking boxes in yellow for tracked faces
                    if config.enable_tracking and tracked_detections:
                        for detection in tracked_detections:
                            x, y, w, h = detection['bbox']
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    
                    # Draw full detection info
                    display_frame = self.detector.draw_detections(
                        display_frame,
                        display_detections,
                        self.current_names
                    )
                
                # Update display
                self.update_display(display_frame)
                
                # Update status
                fps = 1.0 / (time.time() - frame_time)
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                self.faces_label.config(text=f"Faces: {len(display_detections)}")
                
                # Increment frame counter
                self.frame_count += 1
                
            except Exception as e:
                self.error_recovery.log_error("Frame Processing", str(e))
                time.sleep(0.01)

    def update_display(self, frame):
        """Update video display with new frame"""
        try:
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # print(f"Display frame shape: {frame_rgb.shape}, dtype: {frame_rgb.dtype}")
            
            # Convert to PIL image
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update label
            self.video_label.configure(image=photo)
            self.video_label.image = photo
            
        except Exception as e:
            self.error_recovery.log_error("Display Update", str(e))

    def get_current_frame(self):
        """Get copy of current frame"""
        return self.current_frame.copy() if self.current_frame is not None else None

    def stop(self):
        """Stop video processing"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
    def update_feature_detection(self):
        """Update face feature detection settings"""
        detect_eyes = self.detect_eyes_var.get()
        detect_mouth = self.detect_mouth_var.get()
        
        # Update detector
        self.detector.set_feature_detection(
            detect_eyes=detect_eyes,
            detect_mouth=detect_mouth
        )
        
        # Save to config
        config.detect_eyes = detect_eyes
        config.detect_mouth = detect_mouth
        config.save_custom_settings({
            'detect_eyes': detect_eyes,
            'detect_mouth': detect_mouth
        })
    
    def update_feature_color(self):
        """Update face feature highlight color"""
        color = self.color_var.get()
        
        # Update detector
        self.detector.set_feature_color(color)
        
        # Save to config
        config.feature_color = color
        config.save_custom_settings({
            'feature_color': color
        })