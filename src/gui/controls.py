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
            
            # Get face with highest confidence
            best_face = max(detections, key=lambda x: x['confidence'])
            
            # Check for embedding
            if best_face['embedding'] is None:
                messagebox.showerror("Error", "Failed to extract face features")
                return
            
            # Get person's name
            name = simpledialog.askstring("Register Face", "Enter person's name:")
            if not name:
                return
            
            # Create metadata
            metadata = {
                'gender': best_face['gender'],
                'age': best_face['age']
            }
            
            # Add to database
            if self.recognizer.add_face(name, best_face['embedding'], json.dumps(metadata)):
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