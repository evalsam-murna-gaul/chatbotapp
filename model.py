from sqlalchemy import Column,Integer, String,DateTime ,VARCHAR, ForeignKey, text
from database import Base
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime 

Base = declarative_base()

class Student(Base):
    __tablename__ = 'STUDENT'
    user_id = Column(Integer, primary_key=True, index=True)
    studentID = Column(String, unique=True, nullable=False)  # Changed Integer to String
    password = Column(VARCHAR)
    email= Column(String, unique=True, index=True)
    token = Column(String, nullable=True)
    chats = relationship("ChatHistory", back_populates="user")

class BlacklistedToken(Base):
    __tablename__ = "blacklisted_token"
    token = Column(String, primary_key=True)  # Store JWT token
    created_at = Column(DateTime, default=datetime.utcnow().date) 

class OTPRequest(Base):
    __tablename__ = "otp_requests"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False)
    otp = Column(String, nullable=False)
    expires_at = Column(DateTime, nullable=False)

class ChatHistory(Base):
    __tablename__ = "chat_history2"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("STUDENT.user_id"), nullable=False)
    question = Column(String, nullable=False)
    response = Column(String, nullable=False)
    timestamp = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'), nullable=False)# Stores when the question was asked
    user = relationship("Student", back_populates="chats")

class UniversityDocument(Base):
    __tablename__ = "university_documents"
    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    timestamp = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'), nullable=False)  # Stores when the document was uploaded
    