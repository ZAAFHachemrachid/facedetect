import numpy as np
import time
from ..utils.config import config
from ..utils.error_handler import ErrorHandler
from ..database.models import FaceData

class FaceRecognizer:
    def __init__(self, db_session, error_recovery):
        """Initialize face recognizer
        
        Args:
            db_session: Database session for face data
            error_recovery: ErrorRecovery instance for handling errors
        """
        self.db_session = db_session
        self.error_recovery = error_recovery
        self.min_confidence = config.min_confidence

    def recognize_faces(self, embeddings):
        """Recognize faces from embeddings
        
        Args:
            embeddings: List of face embeddings to recognize
            
        Returns:
            list: List of (name, confidence) tuples for each face
        """
        with ErrorHandler(self.error_recovery, "Face Recognition"):
            if not embeddings:
                return []
            
            results = []
            faces = self.db_session.query(FaceData).all()
            
            for embedding in embeddings:
                name, confidence = self._find_best_match(embedding, faces)
                results.append((name, confidence))
            
            return results

    def _find_best_match(self, embedding, faces):
        """Find best matching face for embedding
        
        Args:
            embedding: Face embedding to match
            faces: List of FaceData objects to compare against
            
        Returns:
            tuple: (name, confidence) of best match
        """
        max_similarity = 0.0
        best_match_name = "Unknown"
        
        for face in faces:
            stored_embedding = face.get_embedding_array()
            similarity = self._compute_similarity(embedding, stored_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_name = face.name
        
        # Return unknown if confidence is below threshold
        if max_similarity < self.min_confidence:
            return "Unknown", max_similarity
        
        # Update last seen timestamp
        try:
            self.db_session.query(FaceData)\
                .filter(FaceData.name == best_match_name)\
                .update({"last_seen": int(time.time())})
            self.db_session.commit()
        except Exception as e:
            self.db_session.rollback()
            self.error_recovery.log_error("Database Update", str(e))
        
        return best_match_name, max_similarity

    def _compute_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between two embeddings"""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

    def add_face(self, name, embedding, metadata=None):
        """Add a new face to the database
        
        Args:
            name: Name of the person
            embedding: Face embedding
            metadata: Optional dictionary of additional data
            
        Returns:
            bool: True if successful
        """
        with ErrorHandler(self.error_recovery, "Face Addition"):
            face = FaceData(
                name=name,
                confidence=1.0,
                last_seen=int(time.time()),
                extra_data=metadata
            )
            face.set_embedding_array(embedding)
            
            self.db_session.add(face)
            self.db_session.commit()
            return True

    def remove_face(self, name):
        """Remove all faces for a person from database
        
        Args:
            name: Name of person to remove
            
        Returns:
            bool: True if successful
        """
        with ErrorHandler(self.error_recovery, "Face Removal"):
            self.db_session.query(FaceData)\
                .filter(FaceData.name == name)\
                .delete()
            self.db_session.commit()
            return True