import cv2
import time
import numpy as np

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Camera initialized successfully.")
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera properties: {width}x{height} at {fps} FPS")
    
    # Performance tracking
    frame_times = []
    frame_count = 0
    max_frames = 100  # Stop after this many frames
    
    print("Starting video stream reading (no display)...")
    print("Press Ctrl+C to stop")
    
    try:
        while frame_count < max_frames:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Calculate basic frame statistics (no display)
            frame_avg_color = np.mean(frame, axis=(0, 1))
            frame_brightness = np.mean(frame)
            
            # Calculate fps
            frame_times.append(time.time() - start_time)
            if len(frame_times) > 10:
                frame_times.pop(0)
                
            current_fps = 1.0 / (sum(frame_times) / len(frame_times))
            
            # Print frame info
            if frame_count % 10 == 0:  # Only print every 10th frame
                print(f"Frame {frame_count}: Size={frame.shape}, " +
                      f"Avg RGB=[{frame_avg_color[0]:.1f}, {frame_avg_color[1]:.1f}, {frame_avg_color[2]:.1f}], " +
                      f"Brightness={frame_brightness:.1f}, FPS={current_fps:.1f}")
            
            frame_count += 1
            
            # Maintain frame rate without sleeping (might be slower than camera fps)
            
    except KeyboardInterrupt:
        print("\nVideo stream reading interrupted by user")
    
    finally:
        # Release the camera
        cap.release()
        print(f"Video stream closed. Processed {frame_count} frames.")

if __name__ == "__main__":
    main() 