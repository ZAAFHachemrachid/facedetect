from sqlalchemy import Column, Integer, String, Float, LargeBinary, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import numpy as np

Base = declarative_base()

class FaceData(Base):
    __tablename__ = 'faces'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # Stored as binary
    confidence = Column(Float)
    last_seen = Column(Integer)  # Unix timestamp
    extra_data = Column(String(500))  # JSON string for additional data

    def get_embedding_array(self):
        """Convert stored binary back to numpy array"""
        return np.frombuffer(self.embedding, dtype=np.float32)

    def set_embedding_array(self, arr):
        """Convert numpy array to binary for storage"""
        self.embedding = arr.astype(np.float32).tobytes()

# Database setup
def init_db(base_dir):
    """Initialize the database"""
    db_path = os.path.join(base_dir, "face_data", "face_database.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    
    # Create session factory
    Session = sessionmaker(bind=engine)
    return Session()

def migrate_pickle_to_sql(pickle_data, session):
    """Migrate existing pickle data to SQL database"""
    for name, embeddings in pickle_data.items():
        for embedding in embeddings:
            face = FaceData(
                name=name,
                confidence=1.0,  # Default confidence for existing data
                last_seen=int(os.path.getctime(FACE_DB_PATH))  # Use file creation time as last seen
            )
            face.set_embedding_array(embedding)
            session.add(face)
    
    session.commit()