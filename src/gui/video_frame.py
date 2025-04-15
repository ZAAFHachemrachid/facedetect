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
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Increased buffer size
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Set MJPG format for better performance
        
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
        
        # Create enhanced frame queues with increased capacity
        self.frame_queue = queue.Queue(maxsize=8)
        self.detection_queue = queue.Queue(maxsize=3)  # For detection worker
        self.recognition_queue = queue.Queue(maxsize=3)  # For recognition worker
        self.display_queue = queue.Queue(maxsize=3)
        
        # Add processing locks to prevent race conditions
        self.processing_lock = threading.Lock()
        self.detection_lock = threading.Lock()
        
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
        
        # Create multiple processing threads for better parallelization
        self.process_thread = threading.Thread(target=self.process_video)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Dedicated detection thread
        self.detection_thread = threading.Thread(target=self.detection_worker)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # Dedicated recognition thread
        self.recognition_thread = threading.Thread(target=self.recognition_worker)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()
        
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
        frame_count = 0
        
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
                
                # Skip frames more aggressively if we're falling behind
                frame_count += 1
                if self.frame_queue.qsize() >= 1 and frame_count % config.min_skip_frames != 0:
                    continue
                
                # Store current frame
                with self.processing_lock:
                    self.current_frame = frame.copy()
                
                # Add to processing queue, skip if full to avoid lag
                try:
                    self.frame_queue.put_nowait((frame, current_time))
                except queue.Full:
                    pass  # Skip frame if queue is full to prevent backlog
                
                # Control capture rate with adaptive sleep
                process_time = time.time() - current_time
                target_interval = 1.0 / config.target_fps
                sleep_time = max(0, target_interval - process_time - 0.001)
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
        """Process frames for detection in a separate thread"""
        while self.running:
            try:
                # Get frame from queue with timeout to avoid blocking forever
                try:
                    frame, frame_time = self.frame_queue.get(timeout=0.1)  # Reduced timeout
                except queue.Empty:
                    continue
                
                # Simple brightness and contrast enhancement
                frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=5)
                
                # Determine if we should detect on this frame
                should_detect = self.frame_count % config.detection_interval == 0 or not self.tracker_initialized
                
                if should_detect:
                    # Send to detection queue instead of processing here
                    try:
                        self.detection_queue.put_nowait((frame, frame_time))
                    except queue.Full:
                        pass  # Skip if queue is full
                
                # Update tracking if initialized and not detecting
                elif self.enable_tracking and self.tracker_initialized:
                    tracking_success, bbox = self.tracker.update(frame)
                    
                    if tracking_success:
                        # Create a face detection from tracking result
                        tracked_detection = {
                            'bbox': bbox,
                            'confidence': 0.8,
                            'tracked': True
                        }
                        
                        # Update with just the tracked face
                        with self.processing_lock:
                            self.current_detections = [tracked_detection]
                    else:
                        # Reset tracking if it fails
                        self.tracker_initialized = False
                
                # Draw face detections on display frame
                display_frame = frame.copy()
                
                # Draw detected/tracked faces
                with self.processing_lock:
                    detections_to_draw = self.current_detections.copy()
                    names_to_draw = self.current_names.copy()
                
                # Draw all detections
                for i, detection in enumerate(detections_to_draw):
                    bbox = detection['bbox']
                    confidence = detection.get('confidence', 0)
                    
                    # Choose color based on recognition status
                    color = (0, 255, 0)  # Default green
                    name = "Unknown"
                    
                    # Show name if available from recognition
                    if i < len(names_to_draw):
                        name, name_confidence = names_to_draw[i]
                        if name != "Unknown":
                            # Recognized face - green
                            label = f"{name} ({name_confidence:.2f})"
                            color = (0, 255, 0)
                        else:
                            # Unknown face - red
                            label = f"Unknown ({confidence:.2f})"
                            color = (0, 0, 255)
                    else:
                        # Detection without recognition data
                        label = f"Face ({confidence:.2f})"
                        color = (255, 0, 0)  # Blue
                    
                    # Draw bounding box and name
                    x, y, w, h = bbox
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Improved text visibility with background
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display_frame, (x, y-text_size[1]-10), (x+text_size[0]+10, y), color, -1)
                    cv2.putText(display_frame, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add timestamp and FPS
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(display_frame, timestamp, (10, display_frame.shape[0] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                fps = self.perf_monitor.get_average_fps()
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                face_count = len(detections_to_draw)
                cv2.putText(display_frame, f"Faces: {face_count}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Send to display queue
                try:
                    self.display_queue.put_nowait((display_frame, frame_time))
                except queue.Full:
                    pass  # Skip frame if display queue is full
                
                self.frame_count += 1
                
                # Update processing FPS metrics
                process_time = time.time() - frame_time
                self.perf_monitor.update_frame_time(process_time)
                
            except Exception as e:
                self.error_recovery.log_error("Video Processing", str(e))

    def detection_worker(self):
        """Dedicated thread for face detection"""
        while self.running:
            try:
                # Get frame from detection queue
                try:
                    frame, frame_time = self.detection_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                with Timer("Face Detection"):
                    detection_start = time.time()
                    
                    # Perform face detection
                    detections, detector_used = self.detector.detect_faces(frame)
                    
                    # Update performance metrics
                    self.detection_times.append(time.time() - detection_start)
                    if len(self.detection_times) > 10:
                        self.detection_times.pop(0)
                
                # Update face detection info
                with self.processing_lock:
                    self.current_detections = detections
                
                # Only try recognition with InsightFace successful detections
                if detector_used and detections:
                    # Update tracker with best detection
                    if self.enable_tracking and detections and self.tracker:
                        # Find face with highest confidence for tracking
                        best_detection = max(detections, key=lambda x: x.get('confidence', 0))
                        bbox = best_detection['bbox']
                        
                        # Reinitialize tracker
                        self.tracker.init_tracker(frame, bbox)
                        self.tracker_initialized = True
                    
                    # Send to recognition queue
                    try:
                        self.recognition_queue.put_nowait((frame, detections))
                    except queue.Full:
                        pass  # Skip if queue is full
            
            except Exception as e:
                self.error_recovery.log_error("Face Detection Worker", str(e))
                time.sleep(0.01)
    
    def recognition_worker(self):
        """Dedicated thread for face recognition"""
        while self.running:
            try:
                # Get frame and detections from recognition queue
                try:
                    frame, detections = self.recognition_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                with Timer("Face Recognition"):
                    recognition_start = time.time()
                    
                    # Extract embeddings
                    embeddings = self.detector.get_face_embeddings(frame, detections)
                    if embeddings:
                        # Recognize faces
                        names = self.recognizer.recognize_faces(embeddings)
                        
                        # Update recognized names
                        with self.processing_lock:
                            self.current_names = names
                    
                    # Update performance metrics
                    self.recognition_times.append(time.time() - recognition_start)
                    if len(self.recognition_times) > 10:
                        self.recognition_times.pop(0)
            
            except Exception as e:
                self.error_recovery.log_error("Face Recognition Worker", str(e))
                time.sleep(0.01)

    def enhance_image_quality(self, frame):
        """Enhance image brightness and contrast for better visibility"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L-channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            
            # Increase brightness
            brightness_factor = 30  # Adjust as needed
            cl = cv2.add(cl, brightness_factor)
            
            # Merge channels back
            enhanced_lab = cv2.merge((cl, a_channel, b_channel))
            
            # Convert back to BGR
            enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Increase contrast
            alpha = 1.2  # Contrast control (1.0-3.0)
            beta = 10    # Brightness control (0-100)
            enhanced_frame = cv2.convertScaleAbs(enhanced_frame, alpha=alpha, beta=beta)
            
            return enhanced_frame
        except Exception as e:
            self.error_recovery.log_error("Image Enhancement", str(e))
            return frame

    def display_video(self):
        """Display processed frames in a separate thread"""
        while self.running:
            try:
                # Get processed frame from queue with timeout
                try:
                    display_frame, frame_time = self.display_queue.get(timeout=0.5)
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
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Create PhotoImage
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update label
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # Keep reference to avoid garbage collection
        except Exception as e:
            self.error_recovery.log_error("Display Update", str(e))

    def reset_camera(self):
        """Reset camera connection"""
        try:
            self.cap.release()
            time.sleep(1)
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FPS, config.target_fps)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.video_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.video_height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Increased buffer size
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Set MJPG format for better performance
            print("Camera reset completed")
        except Exception as e:
            self.error_recovery.log_error("Camera Reset", str(e))
            print(f"Camera reset failed: {e}")

    def get_current_frame(self):
        """Get copy of current frame"""
        return self.current_frame.copy() if self.current_frame is not None else None

    def stop(self):
        """Stop video processing"""
        self.running = False
        if self.cap is not None:
            self.cap.release()