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