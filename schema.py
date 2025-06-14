from pydantic import BaseModel, EmailStr
from typing import List
from datetime import datetime

class UserResponseModel(BaseModel):
    studentID: str
    password: str

class UserList(BaseModel):
    studentID: str
    password: str
    email: EmailStr
    

class Login(BaseModel):
    studentID: str
    password: str

class Token(BaseModel):
    user_id: int
    access_token: str
    token_type: str

class RefreshToken(BaseModel):
    refresh_token: str

class TokenData(BaseModel):
    studentID: str 
    token: str

class PasswordReset(BaseModel):
    studentID: str
    old_password: str
    new_password: str    

class ChatHistoryBase(BaseModel):
    question: str
    response: str
    timestamp: datetime

class ChatHistoryResponse(BaseModel):
    user_id: int
    chats: List[ChatHistoryBase]

    class Config:
        from_attributes = True 

class RequestResetPassword(BaseModel):
    StudentID: str
    email: EmailStr

class VerifyOTPReset(BaseModel):
    StudentID: str
    email: EmailStr
    otp: str
    new_password: str