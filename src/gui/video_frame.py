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
        
        # Initialize video capture with optimized settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, config.target_fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.video_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.video_height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        # Processing state
        self.running = True
        self.current_frame = None
        self.last_frame_time = time.time()
        self.last_detection_time = 0
        self.frame_interval = 1.0 / config.target_fps
        self.frame_count = 0  # Count processed frames
        
        # Store last known detections
        self.current_detections = []
        self.current_names = []
        
        # Create frame queues with increased capacity for better buffering
        self.frame_queue = queue.Queue(maxsize=5)
        self.display_queue = queue.Queue(maxsize=2)
        
        # Add processing lock to prevent race conditions
        self.processing_lock = threading.Lock()
        
        # Add tracking support
        self.enable_tracking = config.enable_tracking if hasattr(config, 'enable_tracking') else True
        self.tracker_initialized = False
        self.tracked_box = None
        self.tracked_faces = {}  # Store face tracking info by ID
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor(
            window_size=config.performance_window,
            target_fps=config.target_fps
        )
        
        # Track specialized timing metrics
        self.detection_times = []
        self.recognition_times = []
        
        # Start processing threads
        self.capture_thread = threading.Thread(target=self.capture_video)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.process_thread = threading.Thread(target=self.process_video)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        self.display_thread = threading.Thread(target=self.display_video)
        self.display_thread.daemon = True
        self.display_thread.start()
        
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
        
        # Add detection time label
        self.detection_label = ttk.Label(self.status_frame, text="Detection: 0ms")
        self.detection_label.pack(side=tk.LEFT, padx=5)

    def capture_video(self):
        """Capture video frames continuously in a separate thread"""
        last_frame_time = time.time()
        
        while self.running:
            try:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    if self.error_recovery.log_error("Frame Capture", "Failed to grab frame"):
                        self.reset_camera()
                    time.sleep(0.1)
                    continue
                
                # Calculate capture rate
                current_time = time.time()
                capture_interval = current_time - last_frame_time
                last_frame_time = current_time
                
                # Store current frame
                with self.processing_lock:
                    self.current_frame = frame.copy()
                
                # Add to processing queue, skip if full to avoid lag
                try:
                    self.frame_queue.put_nowait((frame, current_time))
                except queue.Full:
                    pass  # Skip frame if queue is full to prevent backlog
                
                # Slight delay to control capture rate - aim for target fps
                sleep_time = max(0, self.frame_interval - (time.time() - current_time) - 0.001)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.error_recovery.log_error("Frame Capture", str(e))
                time.sleep(0.1)

    def schedule_next_frame(self):
        """Schedule UI updates at a reasonable rate"""
        if self.running:
            # Schedule at a reduced rate for better UI responsiveness
            self.parent.after(20, self.update_ui)

    def update_ui(self):
        """Update UI elements and schedule next update"""
        try:
            # Update status labels with thread safety
            with self.processing_lock:
                faces_count = len(self.current_detections)
            
            # Get performance metrics
            fps = self.perf_monitor.get_average_fps()
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            self.faces_label.config(text=f"Faces: {faces_count}")
            
            # Update detection time if available
            if self.detection_times:
                avg_detection = sum(self.detection_times[-5:]) / min(len(self.detection_times), 5) * 1000
                self.detection_label.config(text=f"Detection: {avg_detection:.1f}ms")
            
            # Schedule next update
            self.schedule_next_frame()
            
        except Exception as e:
            self.error_recovery.log_error("UI Update", str(e))
            self.schedule_next_frame()

    def create_tracker(self):
        """Create appropriate tracker based on OpenCV version"""
        try:
            if int(cv2.__version__.split('.')[0]) >= 4:
                try:
                    return cv2.TrackerKCF_create()
                except:
                    return cv2.legacy.TrackerKCF_create()
            else:
                return cv2.Tracker_create('KCF')
        except:
            return None

    def process_video(self):
        """Process frames for face detection in a separate thread"""
        detection_interval = config.detection_interval
        
        while self.running:
            try:
                # Get frame from queue with timeout to avoid blocking forever
                try:
                    frame, frame_time = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Calculate processing delay - skip detection if we're falling behind
                processing_delay = time.time() - frame_time
                skip_detection = processing_delay > (self.frame_interval * 3)
                
                display_frame = frame.copy()
                current_time = time.time()
                
                # Try tracking update first if enabled
                tracking_success = False
                if self.enable_tracking and self.tracker_initialized and hasattr(self, 'tracker') and self.tracker:
                    try:
                        tracking_success, tracking_box = self.tracker.update(frame)
                        if tracking_success:
                            x, y, w, h = [int(v) for v in tracking_box]
                            # Draw tracking box with different color
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                            cv2.putText(display_frame, "Tracking", (x, y-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            self.tracked_box = (x, y, w, h)
                    except Exception as e:
                        self.error_recovery.log_error("Tracking", str(e))
                        tracking_success = False
                
                # Run detection periodically or when tracking fails
                should_detect = ((self.frame_count % detection_interval == 0) or 
                                not tracking_success or 
                                not self.tracker_initialized)
                
                if should_detect and not skip_detection:
                    detection_start = time.time()
                    
                    # Detect and recognize faces
                    detections, used_insightface = self.detector.detect_faces(frame)
                    
                    if detections:
                        # Detect facial features (eyes and mouth)
                        detections = self.detector.detect_facial_features(frame, detections)
                        
                        if used_insightface:
                            embeddings = self.detector.get_face_embeddings(frame, detections)
                            
                            # Measure recognition time
                            recog_start = time.time()
                            if embeddings:
                                recognized = self.recognizer.recognize_faces(embeddings)
                                names = [name for name, _ in recognized]
                            else:
                                names = ["Unknown"] * len(detections)
                            
                            # Store recognition time
                            self.recognition_times.append(time.time() - recog_start)
                        else:
                            names = ["Unknown"] * len(detections)
                        
                        # Update tracker with best detection if tracking enabled
                        if self.enable_tracking and detections:
                            # Find highest confidence detection
                            best_detection = max(detections, key=lambda x: x['confidence'])
                            
                            # Create tracker
                            self.tracker = self.create_tracker()
                            if self.tracker:
                                # Initialize with bounding box
                                x, y, w, h = best_detection['bbox']
                                self.tracker.init(frame, (x, y, w, h))
                                self.tracker_initialized = True
                        
                        # Update current detections with lock
                        with self.processing_lock:
                            self.current_detections = detections
                            self.current_names = names
                    
                    # Store detection time
                    self.detection_times.append(time.time() - detection_start)
                
                # Always draw current detections
                with self.processing_lock:
                    detections_to_draw = self.current_detections.copy()
                    names_to_draw = self.current_names.copy()
                
                if detections_to_draw:
                    display_frame = self.detector.draw_detections(
                        display_frame, 
                        detections_to_draw,
                        names_to_draw
                    )
                
                # Update performance metrics
                process_time = time.time() - frame_time
                self.perf_monitor.update_frame_time(process_time)
                
                # Add to display queue, skip if full to avoid lag
                try:
                    self.display_queue.put_nowait(display_frame)
                except queue.Full:
                    pass
                
                # Increment frame counter
                self.frame_count += 1
                
            except Exception as e:
                self.error_recovery.log_error("Frame Processing", str(e))
                time.sleep(0.01)

    def display_video(self):
        """Display processed frames in a separate thread"""
        while self.running:
            try:
                # Get processed frame from queue with timeout
                try:
                    display_frame = self.display_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Update display
                self.update_display(display_frame)
                
            except Exception as e:
                self.error_recovery.log_error("Display Update", str(e))
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