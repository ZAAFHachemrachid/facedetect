import tkinter as tk
from tkinter import ttk
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from modular structure
from src.gui.main_window import MainWindow
from src.detection.detector import FaceDetector
from src.detection.recognizer import FaceRecognizer
from src.detection.tracker import ObjectTracker
from src.utils.error_handler import ErrorRecovery
from src.database.models import create_session
from src.database.database import DatabaseManager
from src.utils.config import config

def main():
    """Main application entry point"""
    # Simply create and run MainWindow which handles its own initialization
    app = MainWindow()
    app.run()  # This method handles the mainloop and cleanup

if __name__ == "__main__":
    main()