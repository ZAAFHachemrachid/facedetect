import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import time
from ..utils.config import config
from ..utils.error_handler import ErrorHandler
from ..utils.performance import Timer

class FaceDetector:
    def __init__(self, error_recovery):
        """Initialize face detection system with fallback options"""
        self.error_recovery = error_recovery
        self.use_insightface = True
        self.detect_eyes = True
        self.detect_mouth = True
        self.feature_color = (0, 255, 0)  # Default green
        self.initialize_detectors()

    def initialize_detectors(self):
        """Initialize both InsightFace and HOG detectors"""
        # Initialize InsightFace
        try:
            print("Initializing InsightFace...")
            self.face_app = FaceAnalysis(
                name='buffalo_sc',  # Use more accurate model
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition']  # Ensure all modules are loaded
            )
            print("Created FaceAnalysis instance")
            
            # Prepare with larger detection size for better accuracy
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            
            # Test face detection
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            test_result = self.face_app.get(test_img)
            print("InsightFace test successful")
            
            self.use_insightface = True
            print("InsightFace initialized successfully")
        except Exception as e:
            self.error_recovery.log_error("InsightFace Init", str(e))
            self.use_insightface = False
            print("Falling back to OpenCV HOG detector")

        # Initialize HOG detector as fallback
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_faces(self, frame):
        """Detect faces in frame using available methods"""
        with ErrorHandler(self.error_recovery, "Face Detection"):
            if self.use_insightface:
                try:
                    with Timer("InsightFace Detection"):
                        # Debug frame info
                        # print(f"Input frame shape: {frame.shape}, dtype: {frame.dtype}")
                        
                        # Ensure frame is not empty
                        if frame is None or frame.size == 0:
                            print("Empty frame received")
                            return [], False
                        
                        # InsightFace expects RGB format in uint8
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            # Convert to uint8 if float
                            if frame.dtype != np.uint8:
                                frame = (frame * 255).astype(np.uint8)
                            
                            # Convert BGR to RGB
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            print("Running face detection...")
                            faces = self.face_app.get(rgb_frame)
                            print(f"InsightFace detected {len(faces)} faces")
                            for i, face in enumerate(faces):
                                print(f"Face {i}: score={face.det_score}, bbox={face.bbox}")
                            
                            processed = self._process_insightface_results(faces)
                            print(f"Processed {len(processed)} faces after filtering")
                            return processed, True
                        else:
                            print(f"Unexpected frame format: {frame.shape}")
                            return [], False
                except Exception as e:
                    print(f"Error in face detection: {str(e)}")
                    self.error_recovery.log_error("InsightFace Detection", str(e))
                    self.use_insightface = False

            # Fallback to HOG detector
            return self._detect_with_hog(frame), False

    def _process_insightface_results(self, faces):
        """Process InsightFace detection results with size filtering and optimizations"""
        processed_faces = []
        min_size = 30  # Minimum face size (pixels)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y
            
            print(f"Processing face: size={w}x{h}, score={face.det_score}")
            print(f"Has embedding: {hasattr(face, 'embedding')}")
            print(f"Has landmarks: {hasattr(face, 'kps')}")
            
            # Only filter out very small faces
            if w < min_size or h < min_size:
                print(f"Face too small: {w}x{h} < {min_size}")
                continue
            
            # Create face object with tracking info
            face_data = {
                'bbox': (int(x), int(y), int(x2 - x), int(y2 - y)),
                'embedding': face.embedding if hasattr(face, 'embedding') else None,
                'confidence': float(face.det_score),
                'gender': int(face.gender) if hasattr(face, 'gender') else None,
                'age': float(face.age) if hasattr(face, 'age') else None,
                'tracking_id': None,  # Will be set when tracking is initialized
                'last_seen': time.time(),
                'facial_features': {}  # Initialize facial features dict
            }
            
            # Extract facial features if available
            if hasattr(face, 'kps') and face.kps is not None:
                landmarks = face.kps.astype(int)
                print(f"Processing landmarks: {landmarks.shape}")
                
                # Process eyes
                face_data['facial_features']['eyes'] = [
                    (landmarks[0][0]-15, landmarks[0][1]-10, 30, 20),  # right eye
                    (landmarks[1][0]-15, landmarks[1][1]-10, 30, 20)   # left eye
                ]
                print("Added eye features")
                
                # Process mouth
                mouth_right = landmarks[3]
                mouth_left = landmarks[4]
                mouth_center_x = (mouth_right[0] + mouth_left[0]) // 2
                mouth_center_y = (mouth_right[1] + mouth_left[1]) // 2
                mouth_width = int(abs(mouth_right[0] - mouth_left[0]) * 1.5)
                mouth_height = int(mouth_width * 0.4)
                face_data['facial_features']['mouth'] = (
                    mouth_center_x - mouth_width//2,
                    mouth_center_y - mouth_height//2,
                    mouth_width,
                    mouth_height
                )
                print("Added mouth features")
            
            # Early exit if high confidence match and not optimizing for distance
            if face_data['confidence'] > 0.85 and not config.optimize_for_distance:
                return [face_data]
            
            processed_faces.append(face_data)
            
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
        """Draw detection boxes, labels, and facial features on frame"""
        display_frame = frame.copy()
        
        if not recognized_names:
            recognized_names = ["Unknown"] * len(detections)
        
        # Draw shadow effect for better visibility
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            name = recognized_names[i] if i < len(recognized_names) else "Unknown"
            
            # Draw black outline for better visibility
            cv2.rectangle(display_frame, (x-1, y-1), (x + w+1, y + h+1), (0, 0, 0), 4)
            
            # Choose color based on recognition
            if name and name != "Unknown":
                color = (0, 255, 0)  # Green for recognized faces
            else:
                color = (0, 0, 255)  # Red for unknown faces
            
            # Draw colored box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw facial features if available
            if 'facial_features' in detection:
                features = detection['facial_features']
                
                # Draw eyes
                if self.detect_eyes and 'eyes' in features:
                    for eye in features['eyes']:
                        ex, ey, ew, eh = eye
                        cv2.rectangle(display_frame, (ex, ey),
                                    (ex+ew, ey+eh), self.feature_color, 2)
                        cv2.putText(display_frame, "Eye", (ex, ey-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.feature_color, 1)
                
                # Draw mouth
                if self.detect_mouth and 'mouth' in features:
                    mx, my, mw, mh = features['mouth']
                    cv2.rectangle(display_frame, (mx, my),
                                (mx+mw, my+mh), self.feature_color, 2)
                    cv2.putText(display_frame, "Mouth", (mx, my-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.feature_color, 1)
            
            # Prepare label
            label_parts = [name]
            if detection['confidence'] > 0:
                label_parts.append(f"{detection['confidence']:.2f}")
            label = " - ".join(label_parts)
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            cv2.rectangle(display_frame,
                        (x, y - label_h - 10),
                        (x + label_w + 10, y),
                        color, -1)
            
            # Draw label text in white
            cv2.putText(display_frame, label,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6, (255, 255, 255), 1)
        
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
        
    def init_tracking(self, frame, detection):
        """Initialize tracking for a detected face"""
        if not config.enable_tracking:
            return None
            
        try:
            bbox = detection['bbox']
            tracker = cv2.TrackerKCF_create()
            success = tracker.init(frame, bbox)
            if success:
                return tracker
        except Exception as e:
            self.error_recovery.log_error("Tracker Init", str(e))
        return None
        
    def update_tracking(self, frame, tracker, last_bbox):
        """Update tracking for a face"""
        if not config.enable_tracking:
            return None, False
            
        try:
            success, bbox = tracker.update(frame)
            if success:
                return bbox, True
        except Exception as e:
            self.error_recovery.log_error("Tracker Update", str(e))
        return last_bbox, False

    def set_feature_detection(self, detect_eyes=True, detect_mouth=True):
        """Configure which facial features to detect"""
        self.detect_eyes = detect_eyes
        self.detect_mouth = detect_mouth
    
    def set_feature_color(self, color):
        """Set the color for facial feature highlighting"""
        color_map = {
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255)
        }
        self.feature_color = color_map.get(color, (0, 255, 0))