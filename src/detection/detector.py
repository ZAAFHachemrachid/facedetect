import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from ..utils.config import config
from ..utils.error_handler import ErrorHandler
from ..utils.performance import Timer

class FaceDetector:
    def __init__(self, error_recovery):
        """Initialize face detection system with fallback options"""
        self.error_recovery = error_recovery
        self.use_insightface = True
        self.initialize_detectors()

    def initialize_detectors(self):
        """Initialize both InsightFace and HOG detectors"""
        # Initialize InsightFace
        try:
            self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0, det_size=config.detection_size)
            print("InsightFace initialized successfully")
        except Exception as e:
            self.error_recovery.log_error("InsightFace Init", str(e))
            self.use_insightface = False
            print("Falling back to OpenCV HOG detector")

        # Initialize HOG detector as fallback
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Try to enable OpenCL for Intel GPU acceleration
        try:
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                print("Using OpenCL acceleration")
            else:
                print("OpenCL is not available, using CPU")
        except:
            print("Failed to set OpenCL, using CPU")

    def detect_faces(self, frame):
        """Detect faces in frame using available methods"""
        with ErrorHandler(self.error_recovery, "Face Detection"):
            if self.use_insightface:
                try:
                    with Timer("InsightFace Detection"):
                        faces = self.face_app.get(frame)
                    return self._process_insightface_results(faces), True
                except Exception as e:
                    self.error_recovery.log_error("InsightFace Detection", str(e))
                    self.use_insightface = False

            # Fallback to HOG detector
            return self._detect_with_hog(frame), False

    def _process_insightface_results(self, faces):
        """Process InsightFace detection results"""
        processed_faces = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x, y, x2, y2 = bbox
            processed_faces.append({
                'bbox': (int(x), int(y), int(x2 - x), int(y2 - y)),
                'embedding': face.embedding if hasattr(face, 'embedding') else None,
                'confidence': float(face.det_score),
                'gender': int(face.gender) if hasattr(face, 'gender') else None,
                'age': float(face.age) if hasattr(face, 'age') else None
            })
        return processed_faces

    def _detect_with_hog(self, frame):
        """Detect people using HOG detector"""
        with Timer("HOG Detection"):
            # Resize for faster detection
            small_frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Run detector
            boxes, weights = self.hog.detectMultiScale(
                gray,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05
            )
            
            processed_boxes = []
            if len(boxes) > 0:
                # Scale boxes to original size
                scale_x = frame.shape[1] / gray.shape[1]
                scale_y = frame.shape[0] / gray.shape[0]
                
                for (x, y, w, h), weight in zip(boxes, weights):
                    x_orig = int(x * scale_x)
                    y_orig = int(y * scale_y)
                    w_orig = int(w * scale_x)
                    h_orig = int(h * scale_y)
                    
                    processed_boxes.append({
                        'bbox': (x_orig, y_orig, w_orig, h_orig),
                        'embedding': None,
                        'confidence': float(weight[0]),
                        'gender': None,
                        'age': None
                    })
            
            return processed_boxes

    def draw_detections(self, frame, detections, recognized_names=None):
        """Draw detection boxes and labels on frame"""
        display_frame = frame.copy()
        
        if not recognized_names:
            recognized_names = ["Unknown"] * len(detections)
        
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            name = recognized_names[i] if i < len(recognized_names) else "Unknown"
            
            # Choose color based on recognition (BGR format)
            if name and name != "Unknown":
                color = (0, 255, 0)  # Green for recognized faces
            else:
                color = (0, 0, 255)  # Red for unknown faces
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label
            label_parts = []
            label_parts.append(name)
            if detection['confidence'] > 0:
                label_parts.append(f"({detection['confidence']:.2f})")
            if detection['gender'] is not None:
                gender = 'M' if detection['gender'] == 1 else 'F'
                label_parts.append(f"({gender})")
            if detection['age'] is not None:
                label_parts.append(f"Age: {int(detection['age'])}")
            
            label = " ".join(label_parts)
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                display_frame, 
                (x, y - label_h - 10), 
                (x + label_w, y), 
                color, 
                -1
            )
            
            # Draw label text in white
            cv2.putText(
                display_frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return display_frame

    def get_face_embeddings(self, frame, detections):
        """Extract face embeddings for recognition"""
        if not self.use_insightface:
            return None
        
        embeddings = []
        for detection in detections:
            if detection['embedding'] is not None:
                embeddings.append(detection['embedding'])
        
        return embeddings if embeddings else None