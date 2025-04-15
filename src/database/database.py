from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import time
import json
from .models import Base, FaceData, create_session

class DatabaseManager:
    def __init__(self, base_dir):
        """Initialize database connection and create tables"""
        self.data_dir = os.path.join(base_dir, "face_data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Use the create_session function instead of creating our own
        self.session = create_session(base_dir)

    def add_face(self, name, embedding, confidence, metadata=None):
        """Add a new face to the database"""
        try:
            face = FaceData(
                name=name,
                confidence=float(confidence),
                last_seen=int(time.time()),
                extra_data=json.dumps(metadata) if metadata else None
            )
            face.set_embedding_array(embedding)
            
            self.session.add(face)
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            raise e

    def get_all_faces(self):
        """Get all faces from database"""
        return self.session.query(FaceData).all()

    def get_faces_by_name(self, name):
        """Get all face entries for a specific name"""
        return self.session.query(FaceData).filter(FaceData.name == name).all()

    def update_last_seen(self, face_id):
        """Update the last_seen timestamp for a face"""
        try:
            self.session.query(FaceData)\
                .filter(FaceData.id == face_id)\
                .update({"last_seen": int(time.time())})
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            raise e

    def delete_face(self, name):
        """Delete all entries for a specific name"""
        try:
            self.session.query(FaceData).filter(FaceData.name == name).delete()
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            raise e

    def migrate_from_pickle(self, pickle_data):
        """Migrate data from old pickle format"""
        try:
            # Check if database is empty
            if self.session.query(FaceData).count() == 0:
                for name, embeddings in pickle_data.items():
                    for embedding in embeddings:
                        face = FaceData(
                            name=name,
                            confidence=1.0,  # Default confidence for old data
                            last_seen=int(time.time())
                        )
                        face.set_embedding_array(embedding)
                        self.session.add(face)
                self.session.commit()
                return True
            return False
        except Exception as e:
            self.session.rollback()
            raise e

    def close(self):
        """Close the database session"""
        self.session.close()