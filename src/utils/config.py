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
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps  # 33.33ms between frames
        
        # Camera settings
        self.camera_source = 0  # Default webcam
        self.use_ip_camera = False  # Whether to use IP camera
        self.ip_camera_url = "http://192.168.1.100:8080/video"  # IP Webcam URL
        
        # Detection settings
        self.detection_size = (640, 640)  # Size for face detection (increased)
        self.min_confidence = 0.6  # Minimum confidence for face recognition
        self.detection_interval = 3  # Detection interval in frames (reduced)
        
        # Video processing settings
        self.processing_width = 1280  # Width for processing frames (increased)
        self.min_face_size = 30  # Minimum face size in pixels
        self.max_face_size = 300  # Maximum face size in pixels
        self.enable_tracking = True  # Enable face tracking
        self.optimize_for_distance = True  # Optimize detection for varying distances
        
        # Feature detection settings
        self.detect_eyes = True  # Enable eye detection
        self.detect_mouth = True  # Enable mouth detection
        self.feature_color = "green"  # Feature highlight color
        
        # Performance settings
        self.performance_window = 30  # Window size for performance monitoring
        self.min_skip_frames = 1
        self.max_skip_frames = 5
        self.frame_queue_size = 2  # Maximum frames to buffer
        
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

    def get_camera_url(self):
        """Get the camera URL based on settings"""
        if self.use_ip_camera:
            return self.ip_camera_url
        return self.camera_source

    def set_ip_camera(self, url):
        """Set IP camera URL and enable it"""
        self.ip_camera_url = url
        self.use_ip_camera = True
        self.save_custom_settings({
            'ip_camera_url': url,
            'use_ip_camera': True
        })

    def use_webcam(self):
        """Switch to webcam"""
        self.use_ip_camera = False
        self.save_custom_settings({
            'use_ip_camera': False
        })

    def get_detector_settings(self):
        """Get settings specific to face detection"""
        return {
            'detection_size': self.detection_size,
            'min_confidence': self.min_confidence,
            'detection_interval': self.detection_interval,
            'min_face_size': self.min_face_size,
            'max_face_size': self.max_face_size,
            'enable_tracking': self.enable_tracking,
            'optimize_for_distance': self.optimize_for_distance,
            'detect_eyes': self.detect_eyes,
            'detect_mouth': self.detect_mouth,
            'feature_color': self.feature_color
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

# Default configuration - can be overridden in config.json
default_config = {
    'processing_width': 800,
    'min_face_size': 30,
    'max_face_size': 300,
    'enable_tracking': True,
    'optimize_for_distance': True,
    'detection_interval': 5,
    'detect_eyes': True,
    'detect_mouth': True,
    'feature_color': "green"
}