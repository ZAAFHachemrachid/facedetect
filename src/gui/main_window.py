import tkinter as tk
from tkinter import ttk, messagebox
import threading

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
        self.init_thread = threading.Thread(target=self._init_components)
        self.init_thread.daemon = True
        self.init_thread.start()
        
        # Check initialization status periodically
        self.root.after(100, self._check_init)
        
        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _init_components(self):
        """Initialize components in background thread"""
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
            
            # Signal successful initialization
            self.init_success = True
            
        except Exception as e:
            self.init_error = str(e)
            self.init_success = False

    def _check_init(self):
        """Check initialization status and create UI when ready"""
        if hasattr(self, 'init_success'):
            # Remove loading label
            self.loading_label.destroy()
            
            if self.init_success:
                # Create main UI
                self._create_ui()
            else:
                # Show error and close
                messagebox.showerror(
                    "Initialization Error",
                    f"Failed to initialize application: {self.init_error}"
                )
                self.root.destroy()
        else:
            # Check again after 100ms
            self.root.after(100, self._check_init)

    def _create_ui(self):
        """Create main UI components"""
        try:
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
            
        except Exception as e:
            self.error_recovery.log_error("UI Creation", str(e))
            messagebox.showerror(
                "UI Error",
                f"Failed to create user interface: {str(e)}"
            )
            self.root.destroy()

    def run(self):
        """Start the application"""
        try:
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