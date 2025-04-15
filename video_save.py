import cv2
import numpy as np
import time
import os

def enhance_frame(frame):
    """Enhance video frame quality"""
    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    
    # Replace enhanced luminance channel
    lab[:, :, 0] = enhanced_l
    enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_frame

def main():
    # Create output directory
    output_dir = "captured_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize video capture with optimized settings
    cap = cv2.VideoCapture(0)
    
    # Set optimal camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    # Performance tracking
    frame_times = []
    frame_count = 0
    total_frames = 30  # Total frames to capture
    save_interval = 5  # Save every Nth frame
    
    try:
        print("Starting video capture...")
        while frame_count < total_frames:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Optional: Apply video enhancements
            display_frame = enhance_frame(frame)
            
            # Add frame count and timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, f"Frame: {frame_count} | {timestamp}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Calculate and display FPS
            frame_times.append(time.time() - start_time)
            if len(frame_times) > 10:
                frame_times.pop(0)
                
            fps = 1.0 / (sum(frame_times) / len(frame_times))
            cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Save frame periodically
            if frame_count % save_interval == 0:
                frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, display_frame)
                print(f"Saved frame {frame_count} to {frame_filename}")
            
            frame_count += 1
            
            # Control loop speed for consistent frame rate
            elapsed = time.time() - start_time
            if elapsed < 0.033:  # Target ~30fps
                time.sleep(0.033 - elapsed)
            
            print(f"Processed frame {frame_count}/{total_frames}, FPS: {fps:.1f}", end="\r")
    
    except KeyboardInterrupt:
        print("\nCapture interrupted by user.")
    finally:
        # Release resources
        cap.release()
        print(f"\nCapture complete. {frame_count} frames processed.")
        print(f"Frames saved to {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main() 