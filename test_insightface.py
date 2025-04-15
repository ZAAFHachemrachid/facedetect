import insightface
from insightface.app import FaceAnalysis
import cv2
import os

# Create models directory
os.makedirs("face_data/models", exist_ok=True)

print("Initializing InsightFace...")
try:
    # Initialize with explicit model download
    face_app = FaceAnalysis(
        name='buffalo_sc',
        root='face_data/models',  # Specify model directory
        providers=['CPUExecutionProvider'],
        allowed_modules=['detection', 'recognition']
    )
    
    print("Preparing face analyzer...")
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Test with webcam
    print("Testing webcam capture...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    if ret:
        print("Frame captured, testing face detection...")
        faces = face_app.get(frame)
        print(f"Detected {len(faces)} faces")
        
        # Draw detection results
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Save test image
        cv2.imwrite("face_detection_test.jpg", frame)
        print("Test image saved as face_detection_test.jpg")
    else:
        print("Failed to capture frame from webcam")
    
    cap.release()
    print("Test completed successfully!")
    
except Exception as e:
    print(f"Error during initialization: {e}") 