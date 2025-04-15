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
        self.video_width = 1000
        self.video_height = 600
        self.frame_interval = 1.0 / 30.0  # Target 30 FPS (0.0333s per frame)
        
        # Detection settings
        self.detection_size = (320, 320)  # Size for face detection
        self.min_confidence = 0.6  # Minimum confidence for face recognition
        self.detection_interval = 400  # Milliseconds between detections (2.5 FPS)
        
        # Performance settings
        self.target_fps = 30
        self.performance_window = 30  # Window size for performance monitoring
        self.min_skip_frames = 1
        self.max_skip_frames = 5
        
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