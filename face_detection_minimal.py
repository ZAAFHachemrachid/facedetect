import cv2
import time
import numpy as np
import os

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
    
    # Initialize face detector (using Haar cascade for simplicity)
    # Use the default cascade that comes with OpenCV
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        print(f"Warning: Could not find cascade file at {cascade_path}")
        print("Will attempt to use a generic path")
        cascade_path = 'haarcascade_frontalface_default.xml'
    
    face_detector = cv2.CascadeClassifier(cascade_path)
    
    # Performance tracking
    frame_times = []
    detection_times = []
    frame_count = 0
    max_frames = 100  # Stop after this many frames
    detection_interval = 5  # Only run detection every N frames
    
    print("Starting face detection...")
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
            
            # Run face detection periodically
            faces = []
            if frame_count % detection_interval == 0:
                detection_start = time.time()
                
                # Convert to grayscale for faster detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                detection_times.append(time.time() - detection_start)
            
            # Calculate fps
            frame_times.append(time.time() - start_time)
            if len(frame_times) > 10:
                frame_times.pop(0)
            
            current_fps = 1.0 / (sum(frame_times) / max(1, len(frame_times)))
            
            # Print frame and detection info
            if frame_count % 10 == 0:  # Print every 10th frame
                print(f"Frame {frame_count}: Size={frame.shape}, FPS={current_fps:.1f}")
            
            if len(faces) > 0:
                avg_detection_time = sum(detection_times) / len(detection_times) * 1000
                print(f"Detected {len(faces)} faces: {faces}")
                print(f"Detection time: {avg_detection_time:.1f}ms")
                
                # Extract face details
                for i, (x, y, w, h) in enumerate(faces):
                    face_img = frame[y:y+h, x:x+w]
                    face_brightness = np.mean(face_img)
                    print(f"  Face {i+1}: Position=({x},{y}), Size={w}x{h}, Brightness={face_brightness:.1f}")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nFace detection interrupted by user")
    
    except Exception as e:
        print(f"Error during face detection: {e}")
    
    finally:
        # Release the camera
        cap.release()
        print(f"Video stream closed. Processed {frame_count} frames.")
        
        # Print performance stats
        if detection_times:
            avg_detection = sum(detection_times) / len(detection_times) * 1000
            print(f"Average detection time: {avg_detection:.1f}ms")
        
        if frame_times:
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
            print(f"Average FPS: {avg_fps:.1f}")

if __name__ == "__main__":
    main() 