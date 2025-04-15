import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import json
from ..database.models import FaceData

class ControlPanel:
    def __init__(self, parent, video_frame, detector, recognizer, error_recovery):
        """Initialize control panel
        
        Args:
            parent: Parent tkinter widget
            video_frame: VideoFrame instance
            detector: FaceDetector instance
            recognizer: FaceRecognizer instance
            error_recovery: ErrorRecovery instance
        """
        self.parent = parent
        self.video_frame = video_frame
        self.detector = detector
        self.recognizer = recognizer
        self.error_recovery = error_recovery
        
        self.setup_ui()
        self.update_face_list()

    def setup_ui(self):
        """Set up control panel UI components"""
        # Create main frame
        self.frame = ttk.LabelFrame(self.parent, text="Controls")
        self.frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Status display
        status_frame = ttk.LabelFrame(self.frame, text="Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(padx=5, pady=5)
        
        # Control buttons
        buttons_frame = ttk.LabelFrame(self.frame, text="Actions")
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.register_button = ttk.Button(
            buttons_frame,
            text="Register New Face",
            command=self.register_face
        )
        self.register_button.pack(fill=tk.X, padx=5, pady=5)
        
        self.view_faces_button = ttk.Button(
            buttons_frame,
            text="View Saved Faces",
            command=self.view_faces
        )
        self.view_faces_button.pack(fill=tk.X, padx=5, pady=5)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(self.frame, text="Settings")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Eye tracking toggle
        self.eye_tracking_var = tk.BooleanVar(value=True)
        self.eye_tracking_check = ttk.Checkbutton(
            settings_frame,
            text="Track Eyes",
            variable=self.eye_tracking_var,
            command=self.toggle_eye_tracking
        )
        self.eye_tracking_check.pack(fill=tk.X, padx=5, pady=5)
        
        # Mouth tracking toggle
        self.mouth_tracking_var = tk.BooleanVar(value=True)
        self.mouth_tracking_check = ttk.Checkbutton(
            settings_frame,
            text="Track Mouth",
            variable=self.mouth_tracking_var,
            command=self.toggle_mouth_tracking
        )
        self.mouth_tracking_check.pack(fill=tk.X, padx=5, pady=5)
        
        # Feature color selection
        color_frame = ttk.Frame(settings_frame)
        color_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(color_frame, text="Feature Color:").pack(side=tk.LEFT, padx=5)
        
        self.color_var = tk.StringVar(value="yellow")
        color_options = ["Green", "Red", "Blue", "Yellow"]
        self.color_dropdown = ttk.Combobox(
            color_frame,
            textvariable=self.color_var,
            values=color_options,
            width=10,
            state="readonly"
        )
        self.color_dropdown.pack(side=tk.LEFT, padx=5)
        self.color_dropdown.bind("<<ComboboxSelected>>", self.change_feature_color)
        
        # Advanced settings button
        self.advanced_button = ttk.Button(
            settings_frame,
            text="Facial Feature Settings",
            command=self.open_feature_settings
        )
        self.advanced_button.pack(fill=tk.X, padx=5, pady=5)
        
        # Face list
        faces_frame = ttk.LabelFrame(self.frame, text="Saved Faces")
        faces_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.face_listbox = tk.Listbox(faces_frame)
        self.face_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.delete_face_button = ttk.Button(
            faces_frame,
            text="Delete Selected Face",
            command=self.delete_face
        )
        self.delete_face_button.pack(fill=tk.X, padx=5, pady=5)

    def register_face(self):
        """Handle face registration"""
        try:
            # Get current frame
            frame = self.video_frame.get_current_frame()
            if frame is None:
                messagebox.showerror("Error", "No video frame available")
                return
            
            # Detect faces
            detections, used_insightface = self.detector.detect_faces(frame)
            if not used_insightface:
                messagebox.showerror("Error", "InsightFace is required for face registration")
                return
            
            if not detections:
                messagebox.showerror("Error", "No face detected in current frame")
                return
            
            # Handle face selection when multiple faces are detected
            selected_face = None
            if len(detections) > 1:
                # Get current face names from video frame
                current_names = self.video_frame.current_names
                
                # Add numbers to unknown faces for selection
                face_labels = []
                unknown_count = 0
                for i, name in enumerate(current_names):
                    if name == "Unknown":
                        unknown_count += 1
                        face_labels.append(f"Unknown {unknown_count}")
                    else:
                        face_labels.append(name)
                
                # Create a selection dialog
                selection = simpledialog.askstring(
                    "Select Face", 
                    f"Multiple faces detected. Enter the number of the face to register (1-{len(detections)}):",
                    initialvalue="1"
                )
                
                if not selection:
                    return
                
                try:
                    idx = int(selection) - 1
                    if 0 <= idx < len(detections):
                        selected_face = detections[idx]
                    else:
                        messagebox.showerror("Error", "Invalid selection")
                        return
                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid number")
                    return
            else:
                # Only one face, use it directly
                selected_face = detections[0]
            
            # Check for embedding
            if selected_face['embedding'] is None:
                messagebox.showerror("Error", "Failed to extract face features")
                return
            
            # Get person's name
            name = simpledialog.askstring("Register Face", "Enter person's name:")
            if not name:
                return
            
            # Create metadata
            metadata = {
                'gender': selected_face['gender'],
                'age': selected_face['age']
            }
            
            # Add to database
            if self.recognizer.add_face(name, selected_face['embedding'], json.dumps(metadata)):
                self.update_face_list()
                messagebox.showinfo("Success", f"Face registered for {name}")
            else:
                messagebox.showerror("Error", "Failed to register face")
                
        except Exception as e:
            self.error_recovery.log_error("Face Registration", str(e))
            messagebox.showerror("Error", f"Failed to register face: {str(e)}")

    def view_faces(self):
        """Show list of saved faces"""
        faces = self.face_listbox.get(0, tk.END)
        if not faces:
            messagebox.showinfo("Info", "No faces saved in database")
            return
        
        face_list = "\n".join(faces)
        messagebox.showinfo("Saved Faces", f"Saved faces:\n{face_list}")

    def delete_face(self):
        """Delete selected face from database"""
        selected = self.face_listbox.curselection()
        if not selected:
            messagebox.showerror("Error", "No face selected")
            return
        
        name = self.face_listbox.get(selected[0])
        confirm = messagebox.askyesno(
            "Confirm",
            f"Delete face data for {name}?"
        )
        
        if confirm:
            if self.recognizer.remove_face(name):
                self.update_face_list()
                messagebox.showinfo("Success", f"Face data for {name} deleted")
            else:
                messagebox.showerror("Error", "Failed to delete face data")

    def update_face_list(self):
        """Update list of saved faces"""
        self.face_listbox.delete(0, tk.END)
        # Query unique names directly from FaceData
        names = (self.recognizer.db_session.query(FaceData.name)
                .distinct()
                .all())
        for (name,) in names:
            self.face_listbox.insert(tk.END, name)

    def set_status(self, text):
        """Update status label"""
        self.status_label.config(text=text)

    def toggle_eye_tracking(self):
        """Toggle eye tracking on/off"""
        is_enabled = self.eye_tracking_var.get()
        self.detector.track_eyes = is_enabled
        status = "enabled" if is_enabled else "disabled"
        self.set_status(f"Eye tracking {status}")
        
    def toggle_mouth_tracking(self):
        """Toggle mouth tracking on/off"""
        is_enabled = self.mouth_tracking_var.get()
        self.detector.toggle_mouth_tracking(is_enabled)
        status = "enabled" if is_enabled else "disabled"
        self.set_status(f"Mouth tracking {status}")
        
    def change_feature_color(self, event=None):
        """Change the color used for facial feature highlighting"""
        color = self.color_var.get().lower()
        self.detector.set_feature_color(color)
        self.set_status(f"Feature color set to {color}")
        
    def open_feature_settings(self):
        """Open advanced facial feature detection settings"""
        settings_window = tk.Toplevel(self.parent)
        settings_window.title("Facial Feature Settings")
        settings_window.geometry("350x250")
        settings_window.resizable(False, False)
        
        # Create a frame for the settings
        frame = ttk.Frame(settings_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Feature toggles
        ttk.Label(frame, text="Facial Features to Detect:", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Eyes settings
        eye_var = tk.BooleanVar(value=self.detector.track_eyes)
        ttk.Checkbutton(
            frame,
            text="Track Eyes",
            variable=eye_var,
            command=lambda: self.detector.__setattr__("track_eyes", eye_var.get())
        ).grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # Mouth settings  
        mouth_var = tk.BooleanVar(value=self.detector.track_mouth)
        ttk.Checkbutton(
            frame,
            text="Track Mouth",
            variable=mouth_var,
            command=lambda: self.detector.toggle_mouth_tracking(mouth_var.get())
        ).grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # Color selection
        ttk.Label(frame, text="Color Options:", font=("", 10, "bold")).grid(
            row=3, column=0, columnspan=2, sticky=tk.W, pady=(15, 5))
        
        # Color radio buttons
        color_var = tk.StringVar(value=self.color_var.get())
        colors = [("Green", "green"), ("Red", "red"), ("Blue", "blue"), ("Yellow", "yellow")]
        
        for i, (text, value) in enumerate(colors):
            ttk.Radiobutton(
                frame,
                text=text,
                value=value,
                variable=color_var,
                command=lambda: self.detector.set_feature_color(color_var.get())
            ).grid(row=4+i, column=0, sticky=tk.W, pady=2)
        
        # Apply button
        ttk.Button(
            frame,
            text="Apply & Close",
            command=lambda: [
                self.eye_tracking_var.set(eye_var.get()),
                self.mouth_tracking_var.set(mouth_var.get()),
                self.color_var.set(color_var.get()),
                settings_window.destroy()
            ]
        ).grid(row=8, column=0, columnspan=2, sticky=tk.E, pady=(15, 0))