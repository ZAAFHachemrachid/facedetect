import time
import logging
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_detection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('FaceDetection')

class ErrorRecovery:
    def __init__(self, error_threshold=5, recovery_timeout=60):
        """Initialize error recovery system
        
        Args:
            error_threshold (int): Number of errors before recovery action
            recovery_timeout (int): Timeout in seconds to reset error count
        """
        self.error_count = 0
        self.last_error_time = 0
        self.error_threshold = error_threshold
        self.recovery_timeout = recovery_timeout
        
    def log_error(self, error_type, error_msg):
        """Log an error and check if recovery is needed
        
        Returns:
            bool: True if error threshold reached and recovery needed
        """
        current_time = time.time()
        self.error_count += 1
        
        if current_time - self.last_error_time > self.recovery_timeout:
            self.error_count = 1
            
        self.last_error_time = current_time
        logger.error(f"{error_type}: {error_msg}")
        
        return self.error_count >= self.error_threshold
        
    def reset(self):
        """Reset error counter"""
        self.error_count = 0
        self.last_error_time = 0

class ErrorHandler:
    """Context manager for error handling"""
    def __init__(self, error_recovery, error_type="Operation"):
        self.error_recovery = error_recovery
        self.error_type = error_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            if self.error_recovery.log_error(self.error_type, str(exc_val)):
                logger.warning(f"Error threshold reached for {self.error_type}")
            return False  # Re-raise the exception
        return True