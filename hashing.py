from passlib.context import CryptContext
from sqlalchemy.orm import Session
from model import Student
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class Hash():
    def bcrypt(password: str): # type: ignore
        return pwd_context.hash(password)
    
    def verify(hashed_password, plain_password):
        return pwd_context.verify(plain_password,hashed_password)