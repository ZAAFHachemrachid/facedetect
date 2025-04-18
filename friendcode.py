# Imports
import cv2
import numpy as np
import time
import os
import pickle
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import threading
import insightface
from insightface.app import FaceAnalysis
from datetime import datetime

# Print OpenCV version
print("OpenCV version:", cv2.__version__)

# Configuration and global variables
CONFIG = {
    'detection_interval': 5,
    'min_face_size': 30,
    'max_face_size': 300,
    'recognition_threshold': 0.55,
    'tracking_confidence_threshold': 0.65,
    'processing_width': 800,
    'enable_tracking': True,
    'use_face_landmarks': True,
    'optimize_for_distance': True
}

# Directory setup
DATA_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "face_data")
os.makedirs(DATA_DIR, exist_ok=True)
FACE_DB_PATH = os.path.join(DATA_DIR, "face_database.pkl")
ATTENDANCE_LOG = os.path.join(DATA_DIR, "attendance_log.csv")

# Initialize InsightFace
try:
    face_app = FaceAnalysis(
        name='buffalo_sc',
        providers=['CPUExecutionProvider'],
        root=os.path.join(DATA_DIR, "models")
    )
    face_app.prepare(
        ctx_id=0,
        det_size=(320, 320),
        det_thresh=0.5
    )
    print("InsightFace initialized with optimized settings")
    use_insightface = True
except Exception as e:
    print(f"Failed to initialize InsightFace: {e}")
    use_insightface = False

# Initialize HOG detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load face database
face_database = {}
if os.path.exists(FACE_DB_PATH):
    try:
        with open(FACE_DB_PATH, 'rb') as f:
            face_database = pickle.load(f)
        print(f"Loaded {len(face_database)} faces from database")
    except Exception as e:
        print(f"Error loading face database: {e}")

# Create attendance log if not exists
if not os.path.exists(ATTENDANCE_LOG):
    with open(ATTENDANCE_LOG, 'w') as f:
        f.write("Name,Date,Arrival Time,Departure Time\n")

# Helper functions


def create_tracker():
    try:
        if int(cv2.__version__.split('.')[0]) >= 4:
            try:
                return cv2.TrackerKCF_create()
            except:
                return cv2.legacy.TrackerKCF_create()
        else:
            return cv2.Tracker_create('KCF')
    except:
        return None


def save_face_database():
    try:
        with open(FACE_DB_PATH, 'wb') as f:
            pickle.dump(face_database, f)
    except Exception as e:
        print(f"Error saving face database: {e}")


def recognize_face(face_embedding):
    """Optimized face recognition with early termination"""
    if not face_database:
        return "Unknown", 0.0

    best_match = ("Unknown", 0.0)

    for name, embeddings in face_database.items():
        for embedding in embeddings:
            # Use cosine similarity
            similarity = np.dot(face_embedding, embedding) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(embedding))

            # Early termination if high confidence match
            if similarity > 0.85:
                return name, similarity

            if similarity > best_match[1]:
                best_match = (name, similarity)

    return best_match if best_match[1] >= CONFIG['recognition_threshold'] else ("Unknown", best_match[1])


def log_attendance(name, action):
    """Optimized attendance logging"""
    now = datetime.now()
    entry = f"{name},{now.strftime('%Y-%m-%d')},{now.strftime('%H:%M:%S')}"

    if action == "arrival":
        entry += ",\n"
    else:  # departure
        entry = ""
        with open(ATTENDANCE_LOG, 'r') as f:
            lines = f.readlines()

        found = False
        for i, line in enumerate(lines):
            if line.startswith(name) and line.strip().endswith(','):
                entry = f"{line.strip()}{now.strftime('%H:%M:%S')}\n"
                lines[i] = entry
                found = True
                break

        if not found:
            return

        with open(ATTENDANCE_LOG, 'w') as f:
            f.writelines(lines)
        return

    with open(ATTENDANCE_LOG, 'a') as f:
        f.write(entry)


def preprocess_frame(frame):
    """Optimized frame preprocessing"""
    # Downsample for faster processing while maintaining aspect ratio
    height, width = frame.shape[:2]
    scale = CONFIG['processing_width'] / width
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    # Convert to RGB if needed (InsightFace expects RGB)
    if len(small_frame.shape) == 2:
        small_frame = cv2.cvtColor(small_frame, cv2.COLOR_GRAY2RGB)
    elif small_frame.shape[2] == 4:
        small_frame = small_frame[:, :, :3]
    elif small_frame.shape[2] == 1:
        small_frame = cv2.cvtColor(small_frame, cv2.COLOR_GRAY2RGB)

    return small_frame, scale

# Main Application Class


class FaceRecognitionApp:
    def __init__(self, root):
        self.detect_eyes_var = tk.BooleanVar(value=True)
        self.detect_mouth_var = tk.BooleanVar(value=True)
        self.color_var = tk.StringVar(value="green")
        self.root = root
        self.root.title("Optimized Face Recognition System")
        self.root.geometry("1200x700")

        # Performance tracking
        self.frame_times = []
        self.detection_times = []
        self.recognition_times = []  # Add this line to initialize recognition_times

        self.setup_ui()

        # Initialize video capture with optimized settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

        # Tracking variables
        self.tracker = create_tracker() if CONFIG['enable_tracking'] else None
        self.tracking_initialized = False
        self.tracking_box = None
        self.tracked_faces = {}
        self.last_detected_boxes = []

        # Control variables
        self.running = True
        self.current_frame = None
        self.currently_present = set()
        self.frame_count = 0

        # Start processing thread
        self.thread = threading.Thread(target=self.optimized_video_loop)
        self.thread.daemon = True
        self.thread.start()

    def detect_facial_features(self, frame, face):
        """Detect eyes and mouth from face landmarks"""
        if not hasattr(face, 'kps') or face.kps is None:
            return None, None

        # InsightFace facial landmarks:
        # 0: right eye, 1: left eye, 2: nose, 3: right mouth, 4: left mouth
        landmarks = face.kps.astype(int)

        # Calculate eye regions
        right_eye = landmarks[0]
        left_eye = landmarks[1]

        # Calculate mouth region from right and left mouth corners
        mouth_right = landmarks[3]
        mouth_left = landmarks[4]

        # Create bounding boxes for eyes (expand by 10 pixels around landmark)
        right_eye_bbox = (right_eye[0]-15, right_eye[1]-10, 30, 20)
        left_eye_bbox = (left_eye[0]-15, left_eye[1]-10, 30, 20)

        # Create bounding box for mouth (expand around landmarks)
        mouth_center_x = (mouth_right[0] + mouth_left[0]) // 2
        mouth_center_y = (mouth_right[1] + mouth_left[1]) // 2
        mouth_width = int(abs(mouth_right[0] - mouth_left[0]) * 1.5)
        mouth_height = int(mouth_width * 0.4)  # Proportional height
        mouth_bbox = (mouth_center_x - mouth_width//2,
                      mouth_center_y - mouth_height//2,
                      mouth_width, mouth_height)

        eyes = [right_eye_bbox, left_eye_bbox]
        mouth = mouth_bbox

        return eyes, mouth

    def select_facial_feature(self):
        """Allow user to select which facial features to detect"""
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Select Facial Features")
        selection_window.geometry("300x200")

        # Variables to store selection
        if not hasattr(self, 'detect_eyes_var'):
            self.detect_eyes_var = tk.BooleanVar(value=True)
        if not hasattr(self, 'detect_mouth_var'):
            self.detect_mouth_var = tk.BooleanVar(value=True)
        if not hasattr(self, 'color_var'):
            self.color_var = tk.StringVar(value="green")

        ttk.Checkbutton(selection_window, text="Detect Eyes",
                        variable=self.detect_eyes_var).pack(padx=20, pady=10)
        ttk.Checkbutton(selection_window, text="Detect Mouth",
                        variable=self.detect_mouth_var).pack(padx=20, pady=10)

        # Color selection
        ttk.Label(selection_window, text="Highlight Color:").pack(
            padx=20, pady=5)
        color_frame = ttk.Frame(selection_window)
        color_frame.pack(padx=20, pady=5)

        colors = [("Green", "green"), ("Red", "red"),
                  ("Blue", "blue"), ("Yellow", "yellow")]
        for text, color in colors:
            ttk.Radiobutton(color_frame, text=text, value=color,
                            variable=self.color_var).pack(side=tk.LEFT, padx=5)

        ttk.Button(selection_window, text="Apply",
                   command=selection_window.destroy).pack(pady=20)

    def setup_ui(self):
        # Main UI setup (similar to before but with performance stats)
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Video display
        self.video_frame = ttk.LabelFrame(main_frame, text="Video Feed")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH,
                              expand=True, padx=5, pady=5)
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls & Stats")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Stats frame
        stats_frame = ttk.LabelFrame(control_frame, text="Performance")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)

        self.fps_label = ttk.Label(stats_frame, text="FPS: 0")
        self.fps_label.pack(padx=5, pady=2)

        self.detection_time_label = ttk.Label(
            stats_frame, text="Detection: 0ms")
        self.detection_time_label.pack(padx=5, pady=2)

        self.recognition_label = ttk.Label(
            stats_frame, text="Recognition: 0ms")
        self.recognition_label.pack(padx=5, pady=2)

        # Status frame
        status_frame = ttk.LabelFrame(control_frame, text="Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)

        self.status_label = ttk.Label(status_frame, text="Initializing...")
        self.status_label.pack(padx=5, pady=2)

        self.face_count_label = ttk.Label(status_frame, text="Faces: 0")
        self.face_count_label.pack(padx=5, pady=2)

        self.presence_label = ttk.Label(status_frame, text="Present: None")
        self.presence_label.pack(padx=5, pady=2)

        # Buttons frame (same as before)
        buttons_frame = ttk.LabelFrame(control_frame, text="Actions")
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(buttons_frame, text="Register New Face",
                   command=self.register_new_face).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(buttons_frame, text="View Saved Faces",
                   command=self.view_saved_faces).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(buttons_frame, text="View Attendance Logs",
                   command=self.view_attendance_logs).pack(fill=tk.X, padx=5, pady=2)

        # Face list (same as before)
        faces_frame = ttk.LabelFrame(control_frame, text="Saved Faces")
        faces_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.face_listbox = tk.Listbox(faces_frame)
        self.face_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Button(faces_frame, text="Delete Selected Face",
                   command=self.delete_face).pack(fill=tk.X, padx=5, pady=2)

        self.update_face_list()

    def optimized_video_loop(self):
        """Main processing loop with optimizations"""
        last_time = time.time()

        while self.running:
            start_time = time.time()

            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Store current frame
            self.current_frame = frame.copy()

            # Preprocess frame (downsample, convert colorspace)
            small_frame, scale = preprocess_frame(frame)
            display_frame = frame.copy()

            # Tracking update
            tracking_success = False
            if self.tracking_initialized and self.tracker:
                tracking_success, tracking_box = self.tracker.update(
                    small_frame)
                if tracking_success:
                    # Scale box back to original size
                    x, y, w, h = [int(v/scale) for v in tracking_box]
                    cv2.rectangle(display_frame, (x, y),
                                  (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(display_frame, "Tracking", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Run detection periodically or when tracking fails
            if (self.frame_count % CONFIG['detection_interval'] == 0 or
                not tracking_success or
                    not self.tracking_initialized):

                detection_start = time.time()
                faces = []
                recognized_names = []

                if use_insightface:
                    # Run face detection
                    faces = face_app.get(small_frame)
                    self.last_detected_boxes = []

                    # Process detected faces
                    for face in faces:
                        bbox = face.bbox.astype(int)
                        x, y, x2, y2 = bbox
                        w, h = x2 - x, y2 - y

                        # Filter by size
                        if not (CONFIG['min_face_size'] <= w <= CONFIG['max_face_size'] and
                                CONFIG['min_face_size'] <= h <= CONFIG['max_face_size']):
                            continue

                        # Scale back to original coordinates
                        orig_bbox = [int(v/scale) for v in [x, y, w, h]]
                        self.last_detected_boxes.append(orig_bbox)

                        # Recognize face
                        name = "Unknown"
                        confidence = 0.0

                        if hasattr(face, 'embedding') and face.embedding is not None:
                            rec_start = time.time()
                            name, confidence = recognize_face(face.embedding)
                            self.recognition_times.append(
                                time.time() - rec_start)
                            if name != "Unknown":
                                recognized_names.append(name)

                        # Store face data
                        face_id = f"{x}_{y}_{w}_{h}"
                        self.tracked_faces[face_id] = {
                            'bbox': orig_bbox,
                            'name': name,
                            'confidence': confidence,
                            'last_seen': self.frame_count
                        }

                        # Draw on display frame
                        color = (0, 255, 0) if name != "Unknown" else (
                            0, 0, 255)
                        cv2.rectangle(display_frame,
                                      (orig_bbox[0], orig_bbox[1]),
                                      (orig_bbox[0]+orig_bbox[2],
                                          orig_bbox[1]+orig_bbox[3]),
                                      color, 2)

                        label = f"{name}"
                        if confidence > 0:
                            label += f" ({confidence:.2f})"

                        cv2.putText(display_frame, label,
                                    (orig_bbox[0], orig_bbox[1]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Add eye and mouth detection here (where we process each face)
                        if hasattr(self, 'detect_eyes_var') and face.det_score > 0.5:
                            # Convert color name to BGR
                            color_map = {
                                "green": (0, 255, 0),
                                "red": (0, 0, 255),
                                "blue": (255, 0, 0),
                                "yellow": (0, 255, 255)
                            }
                            feature_color = color_map.get(
                                self.color_var.get(), (0, 255, 0))

                            # Detect facial features
                            eyes, mouth = self.detect_facial_features(
                                small_frame, face)

                            # Draw features on original frame
                            if eyes and self.detect_eyes_var.get():
                                for eye in eyes:
                                    ex, ey, ew, eh = [int(v/scale)
                                                      for v in eye]
                                    cv2.rectangle(
                                        display_frame, (ex, ey), (ex+ew, ey+eh), feature_color, 2)
                                    cv2.putText(display_frame, "Eye", (ex, ey-5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, feature_color, 1)

                            if mouth and self.detect_mouth_var.get():
                                mx, my, mw, mh = [int(v/scale) for v in mouth]
                                cv2.rectangle(
                                    display_frame, (mx, my), (mx+mw, my+mh), feature_color, 2)
                                cv2.putText(display_frame, "Mouth", (mx, my-5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, feature_color, 1)

                # Update tracker with the most confident detection
                if faces and self.tracker:
                    # Find face with highest detection score
                    best_face = max(faces, key=lambda x: x.det_score)
                    bbox = best_face.bbox.astype(int)
                    x, y, x2, y2 = bbox
                    w, h = x2 - x, y2 - y

                    # Reinitialize tracker
                    self.tracker = create_tracker()
                    if self.tracker:
                        self.tracker.init(small_frame, (x, y, w, h))
                        self.tracking_initialized = True

                # Update presence tracking
                self.update_presence(recognized_names)

                # Update detection time stats
                self.detection_times.append(time.time() - detection_start)
                self.status_label.config(text=f"Detection: {len(faces)} faces")
                self.face_count_label.config(text=f"Faces: {len(faces)}")

            # Display FPS and performance stats
            self.frame_times.append(time.time() - start_time)
            if len(self.frame_times) > 10:
                fps = 1 / np.mean(self.frame_times[-10:])
                self.fps_label.config(text=f"FPS: {fps:.1f}")

            if len(self.detection_times) > 0:
                avg_det = np.mean(self.detection_times[-10:]) * 1000
                self.detection_time_label.config(
                    text=f"Detection: {avg_det:.1f}ms")

            if len(self.recognition_times) > 0:
                avg_rec = np.mean(self.recognition_times[-10:]) * 1000
                self.recognition_label.config(
                    text=f"Recognition: {avg_rec:.1f}ms")

            # Convert and display frame
            cv_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            tk_image = ImageTk.PhotoImage(image=pil_image)

            self.video_label.configure(image=tk_image)
            self.video_label.image = tk_image

            self.frame_count += 1

            # Control loop speed
            elapsed = time.time() - start_time
            if elapsed < 0.033:  # ~30fps
                time.sleep(0.033 - elapsed)

    def update_presence(self, recognized_names):
        """Optimized presence tracking"""
        current_names = set(n for n in recognized_names if n != "Unknown")

        # Check arrivals
        new_arrivals = current_names - self.currently_present
        for name in new_arrivals:
            log_attendance(name, "arrival")

        # Check departures
        new_departures = self.currently_present - current_names
        for name in new_departures:
            log_attendance(name, "departure")

        # Update current presence
        self.currently_present = current_names

        # Update UI
        present_text = ", ".join(
            self.currently_present) if self.currently_present else "None"
        self.presence_label.config(text=f"Present: {present_text}")

    def register_new_face(self):
        """Register a new face in the database"""
        if not use_insightface:
            messagebox.showerror(
                "Error", "InsightFace is required for face registration")
            return

        if self.current_frame is None:
            messagebox.showerror("Error", "No video frame available")
            return

        # Check if face is detected
        frame = self.current_frame.copy()
        small_frame, scale = preprocess_frame(frame)
        faces = face_app.get(small_frame)

        if not faces:
            messagebox.showerror("Error", "No face detected in current frame")
            return

        # Ask for person's name
        name = simpledialog.askstring("Register Face", "Enter person's name:")
        if not name:
            return

        # Get face with highest detection score
        best_face = max(faces, key=lambda x: x.det_score)

        if not hasattr(best_face, 'embedding') or best_face.embedding is None:
            messagebox.showerror("Error", "Failed to extract face features")
            return

        # Add to database
        if name in face_database:
            face_database[name].append(best_face.embedding)
        else:
            face_database[name] = [best_face.embedding]

        # Save database
        save_face_database()
        self.update_face_list()
        messagebox.showinfo("Success", f"Face registered for {name}")

    def view_saved_faces(self):
        """View list of saved faces"""
        if not face_database:
            messagebox.showinfo("Info", "No faces saved in database")
            return

        names = "\n".join(face_database.keys())
        messagebox.showinfo("Saved Faces", f"Saved faces:\n{names}")

    def delete_face(self):
        """Delete selected face from database"""
        selected = self.face_listbox.curselection()
        if not selected:
            messagebox.showerror("Error", "No face selected")
            return

        name = self.face_listbox.get(selected[0])
        if messagebox.askyesno("Confirm", f"Delete face data for {name}?"):
            if name in face_database:
                del face_database[name]
                save_face_database()
                self.update_face_list()
                messagebox.showinfo("Success", f"Face data for {name} deleted")

    def view_attendance_logs(self):
        """View attendance logs"""
        if not os.path.exists(ATTENDANCE_LOG):
            messagebox.showinfo("Info", "No attendance records found")
            return

        # Create popup window
        log_window = tk.Toplevel(self.root)
        log_window.title("Attendance Logs")
        log_window.geometry("600x400")

        # Create text widget
        text_widget = tk.Text(log_window, wrap=tk.NONE)
        text_widget.pack(fill=tk.BOTH, expand=True)

        # Add scrollbars
        y_scrollbar = ttk.Scrollbar(
            log_window, orient=tk.VERTICAL, command=text_widget.yview)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        x_scrollbar = ttk.Scrollbar(
            log_window, orient=tk.HORIZONTAL, command=text_widget.xview)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        text_widget.configure(yscrollcommand=y_scrollbar.set,
                              xscrollcommand=x_scrollbar.set)

        # Load and display logs
        with open(ATTENDANCE_LOG, 'r') as f:
            text_widget.insert(tk.END, f.read())

        text_widget.configure(state='disabled')  # Make read-only

    def update_face_list(self):
        """Update the face list display"""
        self.face_listbox.delete(0, tk.END)
        for name in face_database.keys():
            self.face_listbox.insert(tk.END, name)


# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()