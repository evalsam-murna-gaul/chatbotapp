from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, HTTPException, status
from passlib.context import CryptContext
from sqlalchemy.orm import Session
import model, database

SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_MINUTES = 5

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
 
def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    refresh_token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    #refresh_token_storage[refresh_token] = data['username']  # Store refresh token in database
    return refresh_token

def get_user(db, email:EmailStr):
    return db.query(model.Student).filter(model.Student.studentID == int).first()

blacklisted_tokens = set()

def blacklist_token(token: str, db: Session):
    """
    Stores a token in the blacklist database table.
    """
    blacklisted = model.BlacklistedToken(token=token)
    db.add(blacklisted)
    db.commit()

def is_token_blacklisted(token: str, db: Session) -> bool:
    """
    Checks if a token is in the blacklist database table.
    """
    return db.query(model.BlacklistedToken).filter(model.BlacklistedToken.token == token).first() is not None

from jose import jwt, JWTError
from JWToken import SECRET_KEY, ALGORITHM

def decode_token(token: str):
    """
    Decodes and verifies a JWT token.
    Returns the payload if valid, or None if invalid/expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None  # Return None if token is invalid or expired


