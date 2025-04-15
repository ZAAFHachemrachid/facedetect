import cv2
import time
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import threading
import pickle

class FaceDetectionEnhancedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Face Detection System")
        self.root.geometry("1000x700")
        
        # Initialize variables before UI setup
        self.frame_times = []
        self.detection_times = []
        self.recognition_times = []
        self.frame_count = 0
        self.detection_interval = 5  # Only run detection every N frames
        self.faces = []
        self.face_names = []
        self.face_database = {}
        self.running = True
        self.show_landmarks = True
        self.register_mode = False
        self.new_face_name = ""
        self.new_face_samples = []
        self.samples_needed = 5
        self.recognition_threshold = 0.6  # Recognition confidence threshold
        self.face_color = (0, 255, 0)  # Default color for face rectangles
        
        # Setup face database file
        self.db_path = "face_recognition_db.pkl"
        self.load_face_database()
        
        # Setup UI
        self.setup_ui()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.log_message("Error: Could not open camera.")
            return
            
        self.log_message("Camera initialized successfully.")
        
        # Get camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.log_message(f"Camera properties: {width}x{height} at {fps} FPS")
        
        # Initialize face detector and recognizer
        self.initialize_face_detection()
        
        # Start processing thread
        self.thread = threading.Thread(target=self.video_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def initialize_face_detection(self):
        # Initialize face detector (using Haar cascade for simplicity)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            self.log_message(f"Warning: Could not find cascade file at {cascade_path}")
            self.log_message("Will attempt to use a generic path")
            cascade_path = 'haarcascade_frontalface_default.xml'
        
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        
        # Initialize facial landmark detector
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        if os.path.exists(eye_cascade_path):
            self.eye_detector = cv2.CascadeClassifier(eye_cascade_path)
            self.log_message("Eye detector initialized")
        else:
            self.eye_detector = None
            self.log_message("Eye detector not available")
        
        # Initialize face recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.has_recognizer_data = False
        
        # If we have saved recognizer data, load it
        if os.path.exists(self.db_path) and len(self.face_database) > 0:
            try:
                self.train_recognizer()
                self.log_message("Face recognizer initialized with existing data")
            except Exception as e:
                self.log_message(f"Error initializing recognizer: {e}")
    
    def load_face_database(self):
        """Load face database from file"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    self.face_database = pickle.load(f)
                self.log_message(f"Loaded {len(self.face_database)} profiles from database")
            except Exception as e:
                self.log_message(f"Error loading face database: {e}")
                self.face_database = {}
    
    def save_face_database(self):
        """Save face database to file"""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.face_database, f)
            self.log_message("Face database saved successfully")
        except Exception as e:
            self.log_message(f"Error saving face database: {e}")
    
    def train_recognizer(self):
        """Train face recognizer with current database"""
        if not self.face_database:
            self.log_message("No faces in database to train recognizer")
            self.has_recognizer_data = False
            return False
        
        try:
            # Prepare training data
            faces = []
            labels = []
            label_map = {}
            current_label = 0
            
            for name, face_data in self.face_database.items():
                label_map[current_label] = name
                for face_img in face_data:
                    # Convert to grayscale if needed
                    if len(face_img.shape) > 2:
                        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_face = face_img
                    
                    # Resize to standard size
                    resized_face = cv2.resize(gray_face, (100, 100))
                    
                    faces.append(resized_face)
                    labels.append(current_label)
                
                current_label += 1
            
            if not faces:
                self.log_message("No valid face samples for training")
                return False
            
            # Train the recognizer
            self.face_recognizer.train(faces, np.array(labels))
            self.label_map = label_map
            self.has_recognizer_data = True
            
            self.log_message(f"Recognizer trained with {len(faces)} samples from {len(label_map)} persons")
            return True
            
        except Exception as e:
            self.log_message(f"Error training recognizer: {e}")
            self.has_recognizer_data = False
            return False
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Video display
        self.video_frame = ttk.LabelFrame(main_frame, text="Video Feed")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel - Controls and info
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(control_frame, text="Performance")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.fps_label = ttk.Label(stats_frame, text="FPS: 0")
        self.fps_label.pack(padx=5, pady=2)
        
        self.detection_label = ttk.Label(stats_frame, text="Detection: 0ms")
        self.detection_label.pack(padx=5, pady=2)
        
        self.recognition_label = ttk.Label(stats_frame, text="Recognition: 0ms")
        self.recognition_label.pack(padx=5, pady=2)
        
        self.face_count_label = ttk.Label(stats_frame, text="Faces: 0")
        self.face_count_label.pack(padx=5, pady=2)
        
        # Control buttons
        button_frame = ttk.LabelFrame(control_frame, text="Controls")
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_stop_btn = ttk.Button(button_frame, text="Stop", command=self.toggle_capture)
        self.start_stop_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.register_btn = ttk.Button(button_frame, text="Register Face", command=self.start_registration)
        self.register_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.landmarks_var = tk.BooleanVar(value=self.show_landmarks)
        ttk.Checkbutton(button_frame, text="Show Facial Features", 
                       variable=self.landmarks_var,
                       command=self.toggle_landmarks).pack(fill=tk.X, padx=5, pady=5)
        
        # Detection interval controls
        interval_frame = ttk.Frame(button_frame)
        interval_frame.pack(padx=5, pady=5, fill=tk.X)
        
        ttk.Label(interval_frame, text="Detection Interval:").pack(side=tk.LEFT, padx=5)
        
        interval_values = [1, 2, 5, 10, 15]
        self.interval_var = tk.IntVar(value=self.detection_interval)
        
        for val in interval_values:
            ttk.Radiobutton(interval_frame, text=str(val), variable=self.interval_var, 
                           value=val, command=self.update_interval).pack(side=tk.LEFT, padx=2)
        
        # Face database frame
        db_frame = ttk.LabelFrame(control_frame, text="Saved Faces")
        db_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.face_listbox = tk.Listbox(db_frame)
        self.face_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        db_scrollbar = ttk.Scrollbar(db_frame, orient="vertical", command=self.face_listbox.yview)
        db_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.face_listbox.configure(yscrollcommand=db_scrollbar.set)
        
        db_button_frame = ttk.Frame(db_frame)
        db_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(db_button_frame, text="Delete", 
                  command=self.delete_face).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(db_button_frame, text="Retrain", 
                  command=self.train_recognizer).pack(side=tk.LEFT, padx=5)
        
        # Log frame
        log_frame = ttk.LabelFrame(control_frame, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_frame, height=10, width=30)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # Update face list
        self.update_face_list()
        
    def update_face_list(self):
        """Update face list in the UI"""
        self.face_listbox.delete(0, tk.END)
        for name in self.face_database.keys():
            samples = len(self.face_database[name])
            self.face_listbox.insert(tk.END, f"{name} ({samples} samples)")
    
    def delete_face(self):
        """Delete selected face from database"""
        selected = self.face_listbox.curselection()
        if not selected:
            messagebox.showerror("Error", "No face selected")
            return
            
        selected_item = self.face_listbox.get(selected[0])
        name = selected_item.split(" (")[0]  # Extract name from listbox string
        
        if messagebox.askyesno("Confirm", f"Delete face data for {name}?"):
            if name in self.face_database:
                del self.face_database[name]
                self.save_face_database()
                self.update_face_list()
                self.train_recognizer()  # Retrain the recognizer
                messagebox.showinfo("Success", f"Face data for {name} deleted")
    
    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
    
    def toggle_capture(self):
        if self.running:
            self.running = False
            self.start_stop_btn.config(text="Start")
            self.log_message("Video capture stopped")
        else:
            self.running = True
            self.start_stop_btn.config(text="Stop")
            self.log_message("Video capture started")
            if not self.thread.is_alive():
                self.thread = threading.Thread(target=self.video_loop)
                self.thread.daemon = True
                self.thread.start()
    
    def toggle_landmarks(self):
        self.show_landmarks = self.landmarks_var.get()
        self.log_message(f"Facial landmarks display: {'On' if self.show_landmarks else 'Off'}")
    
    def update_interval(self):
        self.detection_interval = self.interval_var.get()
        self.log_message(f"Detection interval set to {self.detection_interval}")
    
    def start_registration(self):
        """Start face registration process"""
        if self.register_mode:
            self.register_mode = False
            self.register_btn.config(text="Register Face")
            self.log_message("Registration mode cancelled")
            return
            
        # Ask for the name
        name = simpledialog.askstring("Register Face", "Enter name for the new face:")
        if not name:
            return
            
        # Check if name already exists
        if name in self.face_database:
            if not messagebox.askyesno("Name exists", 
                                       f"{name} already exists. Add more samples?"):
                return
        
        self.new_face_name = name
        self.new_face_samples = []
        self.register_mode = True
        self.register_btn.config(text="Cancel Registration")
        
        self.log_message(f"Registration started for {name}")
        self.log_message(f"Move your face slightly to capture different angles")
        self.log_message(f"Need {self.samples_needed} samples")
    
    def video_loop(self):
        self.log_message("Video processing started...")
        
        while self.running:
            start_time = time.time()
            
            # Capture frame
            ret, frame = self.cap.read()
            
            if not ret:
                self.log_message("Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Store current frame
            display_frame = frame.copy()
            
            # Run face detection periodically
            if self.frame_count % self.detection_interval == 0:
                detection_start = time.time()
                
                # Convert to grayscale for faster detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                self.faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                detection_time = time.time() - detection_start
                self.detection_times.append(detection_time)
                
                # Keep only the last 20 measurements
                if len(self.detection_times) > 20:
                    self.detection_times.pop(0)
                    
                # Recognize faces if not in registration mode
                if not self.register_mode and self.has_recognizer_data and len(self.faces) > 0:
                    self.face_names = []
                    rec_start = time.time()
                    
                    for (x, y, w, h) in self.faces:
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (100, 100))
                        
                        try:
                            label, confidence = self.face_recognizer.predict(face_roi)
                            # Convert confidence (lower is better in LBPH) to a percentage (higher is better)
                            confidence_score = max(0, min(100, 100 - confidence))
                            
                            if confidence_score >= self.recognition_threshold * 100:
                                name = self.label_map[label]
                            else:
                                name = "Unknown"
                                
                            self.face_names.append((name, confidence_score))
                            
                        except Exception as e:
                            self.face_names.append(("Error", 0))
                            self.log_message(f"Recognition error: {e}")
                    
                    self.recognition_times.append(time.time() - rec_start)
                    if len(self.recognition_times) > 20:
                        self.recognition_times.pop(0)
            
            # Registration mode processing
            if self.register_mode and len(self.faces) == 1:
                (x, y, w, h) = self.faces[0]
                face_roi = frame[y:y+h, x:x+w].copy()
                
                # Add visual indicator that we're capturing face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
                
                # Only add a sample every 10 frames to get variation
                if self.frame_count % 10 == 0 and len(self.new_face_samples) < self.samples_needed:
                    self.new_face_samples.append(face_roi)
                    self.log_message(f"Sample {len(self.new_face_samples)}/{self.samples_needed} captured")
                    
                    # Show progress on screen
                    progress_text = f"Capturing: {len(self.new_face_samples)}/{self.samples_needed}"
                    cv2.putText(display_frame, progress_text, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # If we've collected enough samples
                if len(self.new_face_samples) >= self.samples_needed:
                    # Add to database
                    if self.new_face_name in self.face_database:
                        self.face_database[self.new_face_name].extend(self.new_face_samples)
                    else:
                        self.face_database[self.new_face_name] = self.new_face_samples
                    
                    # Save database and retrain
                    self.save_face_database()
                    self.train_recognizer()
                    self.update_face_list()
                    
                    # Reset registration mode
                    self.register_mode = False
                    self.register_btn.config(text="Register Face")
                    messagebox.showinfo("Success", 
                                       f"Successfully registered {len(self.new_face_samples)} samples for {self.new_face_name}")
                    self.log_message(f"Registration complete for {self.new_face_name}")
            
            # Draw face rectangles and names
            for i, (x, y, w, h) in enumerate(self.faces):
                # Choose color based on recognition or registration mode
                if self.register_mode:
                    color = (0, 255, 255)  # Yellow for registration
                elif len(self.face_names) > i:
                    name, confidence = self.face_names[i]
                    if name != "Unknown" and name != "Error":
                        color = (0, 255, 0)  # Green for recognized
                    else:
                        color = (0, 0, 255)  # Red for unknown
                else:
                    color = (255, 0, 0)  # Blue for default
                
                # Draw face rectangle
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw name and confidence if available
                if len(self.face_names) > i:
                    name, confidence = self.face_names[i]
                    label_text = f"{name} ({confidence:.1f}%)" if confidence > 0 else name
                    cv2.putText(display_frame, label_text, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw facial landmarks (eyes) if enabled
                if self.show_landmarks and self.eye_detector:
                    face_roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                    eyes = self.eye_detector.detectMultiScale(face_roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(display_frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 255), 1)
            
            # Calculate and display FPS
            self.frame_times.append(time.time() - start_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            
            # Convert to format for Tkinter
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            tk_image = ImageTk.PhotoImage(image=pil_image)
            
            # Update UI
            self.video_label.configure(image=tk_image)
            self.video_label.image = tk_image
            
            # Update stats every 10 frames
            if self.frame_count % 10 == 0:
                # Calculate FPS
                if self.frame_times:
                    fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                    self.fps_label.config(text=f"FPS: {fps:.1f}")
                
                # Calculate detection time
                if self.detection_times:
                    avg_detection = sum(self.detection_times) / len(self.detection_times) * 1000
                    self.detection_label.config(text=f"Detection: {avg_detection:.1f}ms")
                
                # Calculate recognition time
                if self.recognition_times:
                    avg_recognition = sum(self.recognition_times) / len(self.recognition_times) * 1000
                    self.recognition_label.config(text=f"Recognition: {avg_recognition:.1f}ms")
                
                # Update face count
                self.face_count_label.config(text=f"Faces: {len(self.faces)}")
            
            self.frame_count += 1
            
            # Control loop speed
            elapsed = time.time() - start_time
            if elapsed < 0.033:  # ~30fps
                time.sleep(0.033 - elapsed)
    
    def release(self):
        """Release resources"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        
        self.log_message("Resources released")

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionEnhancedApp(root)
    
    def on_closing():
        app.release()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop() 