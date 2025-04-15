import tkinter as tk
from tkinter import ttk, messagebox

from ..utils.config import config
from ..utils.error_handler import ErrorRecovery
from ..database.database import DatabaseManager
from ..detection.detector import FaceDetector
from ..detection.recognizer import FaceRecognizer
from ..detection.tracker import ObjectTracker
from .video_frame import VideoFrame
from .controls import ControlPanel

class MainWindow:
    def __init__(self):
        """Initialize main application window"""
        # Create main window
        self.root = tk.Tk()
        self.root.title(config.window_title)
        self.root.geometry(f"{config.window_size[0]}x{config.window_size[1]}")
        
        # Create loading label
        self.loading_label = ttk.Label(
            self.root, 
            text="Initializing components...",
            font=("Helvetica", 14)
        )
        self.loading_label.pack(expand=True)
        
        # Initialize components in background
        self.initialization_failed = False
        self.error_message = None
        
        # Initialize components
        self.setup_components()
        
        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_components(self):
        """Initialize and set up application components"""
        try:
            # Create error recovery system
            self.error_recovery = ErrorRecovery(
                error_threshold=config.error_threshold,
                recovery_timeout=config.recovery_timeout
            )
            
            # Initialize database
            try:
                self.db_manager = DatabaseManager(config.base_dir)
            except Exception as e:
                self.error_recovery.log_error("Database Init", str(e))
                raise
            
            # Initialize detection components
            self.detector = FaceDetector(self.error_recovery)
            self.recognizer = FaceRecognizer(
                self.db_manager.session,
                self.error_recovery
            )
            self.tracker = ObjectTracker(self.error_recovery)
            
            # Create main container
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create video frame
            self.video_frame = VideoFrame(
                main_frame,
                self.detector,
                self.recognizer,
                self.tracker,
                self.error_recovery
            )
            
            # Create control panel
            self.control_panel = ControlPanel(
                main_frame,
                self.video_frame,
                self.detector,
                self.recognizer,
                self.error_recovery
            )
            
            # Remove loading label
            self.loading_label.destroy()
            
        except Exception as e:
            self.initialization_failed = True
            self.error_message = str(e)
            self.error_recovery.log_error("Initialization", str(e))

    def check_initialization(self):
        """Check if initialization completed successfully"""
        if self.initialization_failed:
            messagebox.showerror(
                "Initialization Error",
                f"Failed to initialize application:\n{self.error_message}"
            )
            self.root.destroy()
            return False
        return True

    def run(self):
        """Start the application"""
        try:
            if self.check_initialization():
                self.root.mainloop()
        except Exception as e:
            self.error_recovery.log_error("Application", str(e))
            raise
        finally:
            self.cleanup()

    def on_closing(self):
        """Handle window closing"""
        self.cleanup()
        self.root.destroy()

    def cleanup(self):
        """Clean up resources"""
        # Stop video processing
        if hasattr(self, 'video_frame'):
            self.video_frame.stop()
        
        # Close database connection
        if hasattr(self, 'db_manager'):
            self.db_manager.session.close()

    def show_error(self, title, message):
        """Show error message"""
        try:
            messagebox.showerror(title, message)
        except:
            print(f"Error: {title} - {message}")