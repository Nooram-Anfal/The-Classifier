"""
auth.py — JWT Authentication System
User registration, login, and protected route handling
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# ── Configuration ─────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# ── Pydantic Models ───────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    """Schema for user registration."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)

class UserResponse(BaseModel):
    """Schema for user response (no password)."""
    username: str
    email: str
    created_at: datetime

class Token(BaseModel):
    """Schema for JWT token response."""
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    """Schema for JWT token payload."""
    username: Optional[str] = None

# ── Helper Functions ──────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# ── Dependencies ──────────────────────────────────────────────────────────────

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db = None
) -> dict:
    """
    Dependency to get the current authenticated user.
    Decodes JWT, validates it, and looks up user in MongoDB.
    """
    credential_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credential_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credential_exception
    
    # db passed from app context
    if db is None:
        raise credential_exception
    
    users_collection = db["users"]
    user = await users_collection.find_one({"username": token_data.username})
    
    if user is None:
        raise credential_exception
    
    return user

# ── Service Functions ─────────────────────────────────────────────────────────

async def register_user(
    user_data: UserCreate,
    db
) -> dict:
    """
    Register a new user.
    Validates uniqueness of username and email.
    Hashes password before storing.
    """
    users_collection = db["users"]
    
    # Check if username exists
    existing_username = await users_collection.find_one({"username": user_data.username})
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    # Check if email exists
    existing_email = await users_collection.find_one({"email": user_data.email})
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists"
        )
    
    # Create user document
    user_doc = {
        "username": user_data.username,
        "email": user_data.email,
        "hashed_password": hash_password(user_data.password),
        "created_at": datetime.now(timezone.utc),
    }
    
    result = await users_collection.insert_one(user_doc)
    user_doc["_id"] = result.inserted_id
    return user_doc

async def authenticate_user(
    username: str,
    password: str,
    db
) -> Optional[dict]:
    """
    Authenticate user by username and password.
    Returns user document if valid, None otherwise.
    """
    users_collection = db["users"]
    user = await users_collection.find_one({"username": username})
    
    if not user:
        return None
    
    if not verify_password(password, user["hashed_password"]):
        return None
    
    return user
