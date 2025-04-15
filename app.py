import sys
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.gui.main_window import MainWindow
from src.utils.error_handler import ErrorRecovery

def main():
    """Main application entry point"""
    try:
        # Create and run main window
        app = MainWindow()
        app.run()
    except Exception as e:
        # Create error recovery just for logging fatal errors
        error_recovery = ErrorRecovery()
        error_recovery.log_error("Fatal", str(e))
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()