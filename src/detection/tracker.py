import cv2
from ..utils.error_handler import ErrorHandler

class ObjectTracker:
    def __init__(self, error_recovery):
        """Initialize object tracker
        
        Args:
            error_recovery: ErrorRecovery instance for handling errors
        """
        self.error_recovery = error_recovery
        self.tracker = None
        self.tracking_initialized = False
        self.tracking_box = None

    def create_tracker(self):
        """Create appropriate tracker based on OpenCV version"""
        try:
            # Try legacy trackers first
            if hasattr(cv2, 'legacy'):
                try:
                    tracker = cv2.legacy.TrackerKCF_create()
                    print("Using legacy KCF tracker")
                    return tracker
                except:
                    print("Legacy KCF tracker not available")

            # Try older OpenCV 3.x syntax
            major_ver = int(cv2.__version__.split('.')[0])
            if major_ver < 4:
                try:
                    tracker = cv2.Tracker_create('KCF')
                    print("Using OpenCV 3.x KCF tracker")
                    return tracker
                except:
                    print("OpenCV 3.x tracker not available")

        except Exception as e:
            self.error_recovery.log_error("Tracker Creation", str(e))

        # Fall back to simple tracker
        print("Using basic tracking")
        return SimpleTracker()

    def init_tracker(self, frame, bbox):
        """Initialize tracker with a bounding box
        
        Args:
            frame: Initial frame
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            bool: True if initialization successful
        """
        with ErrorHandler(self.error_recovery, "Tracker Initialization"):
            self.tracker = self.create_tracker()
            if self.tracker is None:
                return False

            if isinstance(self.tracker, SimpleTracker):
                success = self.tracker.init(frame, bbox)
            else:
                success = self.tracker.init(frame, bbox)
                
            if success:
                self.tracking_initialized = True
                self.tracking_box = bbox
            return success

    def update(self, frame):
        """Update tracker with new frame
        
        Args:
            frame: New frame to track in
            
        Returns:
            tuple: (success, bbox) if successful, (False, None) otherwise
        """
        with ErrorHandler(self.error_recovery, "Tracking Update"):
            if not self.tracking_initialized or self.tracker is None:
                return False, None
                
            try:
                if isinstance(self.tracker, SimpleTracker):
                    success, bbox = self.tracker.update(frame)
                else:
                    success, bbox = self.tracker.update(frame)
                    
                if success:
                    self.tracking_box = tuple(int(v) for v in bbox)
                    return True, self.tracking_box
                else:
                    self.reset()
                    return False, None
                    
            except Exception as e:
                self.error_recovery.log_error("Tracking Update", str(e))
                self.reset()
                return False, None

    def reset(self):
        """Reset tracker state"""
        self.tracking_initialized = False
        self.tracking_box = None
        self.tracker = None

    def is_tracking(self):
        """Check if tracker is currently active"""
        return self.tracking_initialized and self.tracker is not None

    def draw_tracking(self, frame):
        """Draw current tracking box on frame"""
        if not self.is_tracking() or self.tracking_box is None:
            return frame
            
        x, y, w, h = self.tracking_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, "Tracking", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        return frame

class SimpleTracker:
    """Basic implementation of a tracker using template matching"""
    def __init__(self):
        self.bbox = None
        self.template = None
        self.search_scale = 1.2  # Search area scale factor
        
    def init(self, frame, bbox):
        """Initialize tracker with frame and bounding box"""
        try:
            self.bbox = tuple(map(int, bbox))
            x, y, w, h = self.bbox
            if w <= 0 or h <= 0:
                return False
            self.template = frame[y:y+h, x:x+w].copy()
            return True
        except:
            return False
        
    def update(self, frame):
        """Update tracker with new frame"""
        if self.template is None or self.bbox is None:
            return False, None
            
        try:
            # Get current tracking region
            x, y, w, h = self.bbox
            
            # Calculate search area
            search_w = int(w * self.search_scale)
            search_h = int(h * self.search_scale)
            
            # Calculate search bounds
            center_x = x + w // 2
            center_y = y + h // 2
            
            search_x = max(0, center_x - search_w // 2)
            search_y = max(0, center_y - search_h // 2)
            search_w = min(frame.shape[1] - search_x, search_w)
            search_h = min(frame.shape[0] - search_y, search_h)
            
            if search_w <= w or search_h <= h:
                return False, None
                
            # Extract search area
            search_area = frame[search_y:search_y+search_h, 
                              search_x:search_x+search_w]
            
            # Perform template matching
            result = cv2.matchTemplate(search_area, self.template, 
                                     cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            
            # Calculate new position
            new_x = search_x + max_loc[0]
            new_y = search_y + max_loc[1]
            
            # Update template and bounding box
            self.bbox = (new_x, new_y, w, h)
            self.template = frame[new_y:new_y+h, new_x:new_x+w].copy()
            
            return True, self.bbox
            
        except Exception:
            return False, None