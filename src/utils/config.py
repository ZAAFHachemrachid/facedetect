import os
import json
from pathlib import Path

class Config:
    def __init__(self):
        # Base paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.base_dir, "face_data")
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Database settings
        self.db_path = os.path.join(self.data_dir, "face_database.db")
        self.old_db_path = os.path.join(self.data_dir, "face_database.pkl")
        
        # Video capture settings
        self.video_width = 640  # Standard resolution
        self.video_height = 480  # Standard 4:3 resolution
        self.target_fps = 24    # Reduced for better processing
        self.frame_interval = 1.0 / self.target_fps
        
        # Detection settings
        self.detection_size = 320  # Size for face detection (single integer value)
        self.detection_size_tuple = (320, 320)  # Reduced for better performance
        self.min_confidence = 0.6  # Minimum confidence for face recognition
        self.detection_interval = 15  # Increased interval between detections for better performance
        self.enable_tracking = True  # Enable object tracking between detections
        self.max_faces = 10  # Maximum number of faces to detect and track
        
        # Performance settings
        self.performance_window = 10  # Reduced window size for more responsive metrics
        self.min_skip_frames = 1  # Skip at least 1 frame to reduce processing load
        self.max_skip_frames = 3  # Skip more frames for smoother performance
        self.frame_queue_size = 2  # Reduced queue size to minimize latency
        
        # Threading settings
        self.use_multithreading = True  # Enable multithreaded processing
        self.num_detection_threads = 1  # Number of detection worker threads
        self.num_recognition_threads = 1  # Number of recognition worker threads
        
        # Error recovery settings
        self.error_threshold = 5
        self.recovery_timeout = 60  # seconds
        
        # GUI settings
        self.window_size = (1200, 700)
        self.window_title = "Face Recognition System"
        
        # Load custom settings if exists
        self.load_custom_settings()

    def load_custom_settings(self):
        """Load custom settings from config file if it exists"""
        config_path = os.path.join(self.base_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_settings = json.load(f)
                for key, value in custom_settings.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            except Exception as e:
                print(f"Error loading custom settings: {e}")

    def save_custom_settings(self, settings):
        """Save custom settings to config file"""
        config_path = os.path.join(self.base_dir, "config.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(settings, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving custom settings: {e}")
            return False

    def get_detector_settings(self):
        """Get settings specific to face detection"""
        return {
            'detection_size': self.detection_size,
            'min_confidence': self.min_confidence,
            'detection_interval': self.detection_interval
        }

    def get_performance_settings(self):
        """Get settings specific to performance monitoring"""
        return {
            'target_fps': self.target_fps,
            'window_size': self.performance_window,
            'min_skip_frames': self.min_skip_frames,
            'max_skip_frames': self.max_skip_frames,
            'frame_interval': self.frame_interval
        }

    def get_error_settings(self):
        """Get settings specific to error handling"""
        return {
            'error_threshold': self.error_threshold,
            'recovery_timeout': self.recovery_timeout
        }

# Global configuration instance
config = Config()