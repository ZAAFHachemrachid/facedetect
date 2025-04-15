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
        
        # Initialize eye detector
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.track_eyes = True  # Can be toggled on/off
        self.track_mouth = True  # Can be toggled on/off
        self.feature_color = (0, 255, 255)  # Default to yellow
        
        # Detection distance optimization
        self.min_face_size = 15  # Reduced for greater distance detection
        self.max_face_size = 500  # Increased for wider range
        self.detection_upscale = True  # Enable upscaling for small faces
        
        # Multi-face optimization
        self.max_faces = 10  # Allow detection of multiple faces
        self.detection_batch_size = 4  # Process faces in batches for better performance

    def initialize_detectors(self):
        """Initialize both InsightFace and HOG detectors"""
        # Initialize InsightFace
        try:
            self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            # Increase detection size for better distance performance
            self.face_app.prepare(ctx_id=0, det_size=config.detection_size_tuple, det_thresh=0.35)
            
            # Configure for multi-face detection
            if hasattr(self.face_app, 'det_model'):
                # Set max number of faces to detect
                if hasattr(self.face_app.det_model, 'max_num'):
                    self.face_app.det_model.max_num = self.max_faces
            
            print("InsightFace initialized successfully")
        except Exception as e:
            self.error_recovery.log_error("InsightFace Init", str(e))
            self.use_insightface = False
            print("Falling back to OpenCV HOG detector")

        # Initialize HOG detector as fallback
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def preprocess_frame(self, frame):
        """Enhanced frame preprocessing for maximum detection distance"""
        # Initial resize if needed
        if frame.shape[1] > config.detection_size:
            scale = config.detection_size / frame.shape[1]
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            small_frame = frame.copy()
            scale = 1.0

        # Convert to RGB if needed (InsightFace expects RGB)
        if len(small_frame.shape) == 2:
            small_frame = cv2.cvtColor(small_frame, cv2.COLOR_GRAY2RGB)
        elif small_frame.shape[2] == 4:
            small_frame = small_frame[:, :, :3]
        elif small_frame.shape[2] == 1:
            small_frame = cv2.cvtColor(small_frame, cv2.COLOR_GRAY2RGB)

        # Performance optimization - skip additional processing if the frame is not for detection
        if not self.detection_upscale:
            return small_frame, scale

        # Simplified enhancement for better performance
        # Just convert to LAB and apply CLAHE to L channel
        lab = cv2.cvtColor(small_frame, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        lab[:, :, 0] = enhanced_l
        small_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return small_frame, scale

    def detect_faces(self, frame):
        """Detect faces in frame using available methods"""
        with ErrorHandler(self.error_recovery, "Face Detection"):
            # Apply enhanced preprocessing
            small_frame, scale = self.preprocess_frame(frame)
            
            if self.use_insightface:
                try:
                    with Timer("InsightFace Detection"):
                        faces = self.face_app.get(small_frame)
                    return self._process_insightface_results(faces, scale), True
                except Exception as e:
                    self.error_recovery.log_error("InsightFace Detection", str(e))
                    self.use_insightface = False

            # Fallback to HOG detector
            return self._detect_with_hog(small_frame, scale), False

    def _process_insightface_results(self, faces, scale):
        """Process InsightFace detection results with optimizations for distance detection"""
        processed_faces = []
        
        # Process all detected faces, up to max limit
        for face in faces[:self.max_faces]:
            bbox = face.bbox.astype(int)
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y
            
            # Dynamic size filtering based on position - safe calculation avoiding division by zero
            # Avoid tuple comparison by making sure we're using a scalar value
            position_factor = 1.0
            if y > 0:  # Avoid division by zero
                position_factor = 1.0 + (y / max(1, y))  # Adjust based on vertical position
            
            adjusted_min = self.min_face_size / position_factor
            adjusted_max = self.max_face_size * position_factor
            
            # Filter by adjusted size
            if not (adjusted_min <= w <= adjusted_max and adjusted_min <= h <= adjusted_max):
                continue
                
            # Adjust confidence based on size and position
            size_confidence = min(1.0, w / self.min_face_size) * min(1.0, h / self.min_face_size)
            
            # Scale back to original coordinates
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            orig_w = int(w / scale)
            orig_h = int(h / scale)
            
            face_data = {
                'bbox': (orig_x, orig_y, orig_w, orig_h),
                'embedding': face.embedding if hasattr(face, 'embedding') else None,
                'confidence': float(face.det_score) * size_confidence,
                'gender': int(face.gender) if hasattr(face, 'gender') else None,
                'age': float(face.age) if hasattr(face, 'age') else None,
                'eye_regions': None,  # Will be populated if eye detection is enabled
                'mouth_region': None, # Will be populated if mouth detection is enabled
                'landmarks': face.kps.astype(int) if hasattr(face, 'kps') else None
            }
            
            processed_faces.append(face_data)
            
        return processed_faces

    def _detect_with_hog(self, frame, scale):
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
                    x_orig = int(x * scale_x / scale)
                    y_orig = int(y * scale_y / scale)
                    w_orig = int(w * scale_x / scale)
                    h_orig = int(h * scale_y / scale)
                    
                    processed_boxes.append({
                        'bbox': (x_orig, y_orig, w_orig, h_orig),
                        'embedding': None,
                        'confidence': float(weight[0]),
                        'gender': None,
                        'age': None,
                        'eye_regions': None,  # Will be populated if eye detection is enabled
                        'mouth_region': None, # Will be populated if mouth detection is enabled
                        'landmarks': None
                    })
            
            return processed_boxes

    def detect_facial_features(self, frame, detections):
        """Detect eyes and mouth using facial landmarks or Haar cascades"""
        if not self.track_eyes and not self.track_mouth:
            return detections
            
        # Convert to grayscale for better feature detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for detection in detections:
            # Check if we have landmarks from InsightFace
            if detection['landmarks'] is not None:
                landmarks = detection['landmarks']
                
                # InsightFace facial landmarks:
                # 0: right eye, 1: left eye, 2: nose, 3: right mouth, 4: left mouth
                
                # Calculate face dimensions for proportional feature sizing
                x, y, w, h = detection['bbox']
                eye_size_w = int(max(w * 0.15, 15))  # Proportional to face width
                eye_size_h = int(max(h * 0.08, 10))  # Proportional to face height
                
                # Calculate eye regions with proportional sizing
                right_eye = landmarks[0]
                left_eye = landmarks[1]
                
                # Create bounding boxes for eyes centered on landmarks
                right_eye_bbox = (
                    int(right_eye[0] - eye_size_w/2),
                    int(right_eye[1] - eye_size_h/2),
                    eye_size_w,
                    eye_size_h
                )
                
                left_eye_bbox = (
                    int(left_eye[0] - eye_size_w/2),
                    int(left_eye[1] - eye_size_h/2),
                    eye_size_w,
                    eye_size_h
                )
                
                # Calculate mouth region from right and left mouth corners
                mouth_right = landmarks[3]
                mouth_left = landmarks[4]
                
                # Calculate proportional mouth size
                mouth_width = int(abs(mouth_right[0] - mouth_left[0]) * 1.2)
                mouth_height = int(mouth_width * 0.5)  # Improved height proportion
                
                # Center coordinates
                mouth_center_x = int((mouth_right[0] + mouth_left[0]) / 2)
                mouth_center_y = int((mouth_right[1] + mouth_left[1]) / 2)
                
                # Create mouth bounding box
                mouth_bbox = (
                    mouth_center_x - mouth_width//2,
                    mouth_center_y - mouth_height//3,  # Shifted up slightly
                    mouth_width,
                    mouth_height
                )
                
                if self.track_eyes:
                    detection['eye_regions'] = [right_eye_bbox, left_eye_bbox]
                if self.track_mouth:
                    detection['mouth_region'] = mouth_bbox
                
            else:
                # Use Haar cascades when landmarks are not available
                x, y, w, h = detection['bbox']
                
                # Extract face region with padding
                # Add padding to increase detection accuracy
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.1)
                
                # Calculate padded face region with boundary checks
                face_x = max(0, x - pad_x)
                face_y = max(0, y - pad_y)
                face_w = min(frame.shape[1] - face_x, w + 2*pad_x)
                face_h = min(frame.shape[0] - face_y, h + 2*pad_y)
                
                # Extract the face region
                face_region = gray[face_y:face_y+face_h, face_x:face_x+face_w]
                
                # Skip if face region is too small
                if face_region.size == 0 or face_w < 30 or face_h < 30:
                    continue
                    
                # Apply histogram equalization to improve contrast
                face_region = cv2.equalizeHist(face_region)
                
                # Detect eyes in the face region
                if self.track_eyes:
                    # Focus on upper region for eyes (40% of face)
                    eyes_region_h = int(face_h * 0.4)
                    eyes_region = face_region[:eyes_region_h, :]
                    
                    eyes = self.eye_cascade.detectMultiScale(
                        eyes_region,
                        scaleFactor=1.05,  # More precise scale factor
                        minNeighbors=6,    # More neighbor requirements for accuracy
                        minSize=(int(face_w * 0.08), int(face_h * 0.05))  # Proportional min size
                    )
                    
                    # Store detected eye regions
                    eye_regions = []
                    for (ex, ey, ew, eh) in eyes[:2]:  # Limit to 2 eyes
                        # Convert coordinates relative to whole frame
                        eye_x = face_x + ex
                        eye_y = face_y + ey
                        eye_regions.append((eye_x, eye_y, ew, eh))
                    
                    detection['eye_regions'] = eye_regions
                
                # Detect mouth in the face region
                if self.track_mouth:
                    # Focus on lower 40% of face for mouth detection
                    mouth_region_y = int(face_h * 0.6)
                    mouth_region_h = face_h - mouth_region_y
                    mouth_region = face_region[mouth_region_y:, :]
                    
                    mouths = self.mouth_cascade.detectMultiScale(
                        mouth_region,
                        scaleFactor=1.1,
                        minNeighbors=12,  # Increased for better accuracy
                        minSize=(int(face_w * 0.3), int(face_h * 0.1))  # Proportional size
                    )
                    
                    # Store detected mouth region (just one)
                    if len(mouths) > 0:
                        mx, my, mw, mh = mouths[0]
                        # Convert coordinates relative to whole frame
                        mouth_x = face_x + mx
                        mouth_y = face_y + mouth_region_y + my
                        detection['mouth_region'] = (mouth_x, mouth_y, mw, mh)
        
        return detections

    def draw_detections(self, frame, detections, recognized_names=None):
        """Draw detection boxes and labels on frame"""
        display_frame = frame.copy()
        
        if not recognized_names:
            recognized_names = ["Unknown"] * len(detections)
        
        # Count unknown faces for tagging
        unknown_count = 0
        
        # Draw shadow effect for better visibility
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            name = recognized_names[i] if i < len(recognized_names) else "Unknown"
            
            # Add numbers to unknown faces
            if name == "Unknown":
                unknown_count += 1
                name = f"Unknown {unknown_count}"
            
            # Draw black outline for better visibility
            cv2.rectangle(display_frame, (x-1, y-1), (x + w+1, y + h+1), (0, 0, 0), 4)
            
            # Choose color based on recognition
            if name and "Unknown" not in name:
                color = (0, 255, 0)  # Green for recognized faces
            else:
                color = (0, 0, 255)  # Red for unknown faces
            
            # Draw colored box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw facial features
            if self.track_eyes and detection['eye_regions']:
                for ex, ey, ew, eh in detection['eye_regions']:
                    # Draw eye regions
                    cv2.rectangle(display_frame, (ex, ey), (ex + ew, ey + eh), self.feature_color, 2)
                    
                    # Draw pupil center point
                    center_x = ex + ew // 2
                    center_y = ey + eh // 2
                    cv2.circle(display_frame, (center_x, center_y), 2, (255, 0, 0), -1)
                    # Add label
                    cv2.putText(display_frame, "Eye", (ex, ey-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.feature_color, 1)
            
            # Draw mouth region
            if self.track_mouth and detection['mouth_region']:
                mx, my, mw, mh = detection['mouth_region']
                cv2.rectangle(display_frame, (mx, my), (mx + mw, my + mh), self.feature_color, 2)
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
        
        # Process embeddings in batches for better performance
        embeddings = []
        batch_size = min(self.detection_batch_size, len(detections))
        
        # Process faces in batches
        for i in range(0, len(detections), batch_size):
            batch = detections[i:i+batch_size]
            
            for detection in batch:
                if detection['embedding'] is not None:
                    embeddings.append(detection['embedding'])
        
        return embeddings if embeddings else None
        
    def set_feature_color(self, color_name):
        """Set color for facial feature highlighting"""
        color_map = {
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255)
        }
        self.feature_color = color_map.get(color_name, (0, 255, 255))
        
    def toggle_mouth_tracking(self, enabled):
        """Toggle mouth tracking on/off"""
        self.track_mouth = enabled

    def enhance_frame_preprocessing(self, frame):
        """Fast version of frame preprocessing for video display"""
        # Simple brightness and contrast adjustment for better visibility
        alpha = 1.1  # Contrast control
        beta = 5     # Brightness control
        enhanced_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return enhanced_frame