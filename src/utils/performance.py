from collections import deque
import time

class PerformanceMonitor:
    def __init__(self, window_size=30, target_fps=30):
        """Initialize performance monitoring
        
        Args:
            window_size (int): Number of frames to keep for averaging
            target_fps (int): Target frames per second
        """
        self.frame_times = deque(maxlen=window_size)
        self.detection_times = deque(maxlen=window_size)
        self.skip_threshold = 1.0/target_fps  # Target FPS threshold
        self.min_skip_frames = 1
        self.max_skip_frames = 5
        self.target_fps = target_fps

    def update_frame_time(self, frame_time):
        """Add new frame processing time"""
        self.frame_times.append(frame_time)
        
    def update_detection_time(self, detection_time):
        """Add new detection processing time"""
        self.detection_times.append(detection_time)
        
    def get_optimal_skip_frames(self):
        """Calculate optimal number of frames to skip based on performance"""
        if not self.frame_times:
            return self.min_skip_frames
            
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        if avg_frame_time <= self.skip_threshold:
            return self.min_skip_frames
            
        skip_frames = int(avg_frame_time / self.skip_threshold)
        return min(max(skip_frames, self.min_skip_frames), self.max_skip_frames)

    def get_fps(self):
        """Calculate current FPS"""
        if not self.frame_times:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
    def get_average_fps(self):
        """Get average FPS over the window period"""
        return self.get_fps()

    def get_stats(self):
        """Get current performance statistics"""
        stats = {
            'fps': self.get_fps(),
            'frame_time': sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0,
            'detection_time': sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0,
            'skip_frames': self.get_optimal_skip_frames()
        }
        return stats

    def should_process_frame(self, current_time, last_process_time, interval):
        """Check if enough time has passed to process a new frame
        
        Args:
            current_time (float): Current timestamp
            last_process_time (float): Last processing timestamp
            interval (float): Desired interval in milliseconds
            
        Returns:
            bool: True if frame should be processed
        """
        return (current_time - last_process_time) * 1000 >= interval

class Timer:
    """Context manager for timing code blocks"""
    def __init__(self, name=None):
        self.name = name
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.name:
            print(f"{self.name}: {self.interval:.3f}s")