"""
app.py — FastAPI backend, multi-model News Classifier
Migrated from Flask to FastAPI with async MongoDB support
"""
import os
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
import motor.motor_asyncio
from dotenv import load_dotenv

from model import (load_data, train_all, load_all_models, models_exist,
                   predict_one, predict_batch, MODEL_DEFS, EVAL_PATH)
from fake_news_model import (load_fake_model, fake_models_exist, predict_fake)
from auth import (
    UserCreate, UserResponse, Token,
    register_user, authenticate_user, create_access_token,
    get_current_user, oauth2_scheme
)

# ── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = "news_classifier_db"
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Global state for models
app_state = {
    "MODELS": None,
    "VECTORIZER": None,
    "EVAL_DATA": None,
    "FAKE_MODEL": None,
    "FAKE_VECTORIZER": None,
    "FAKE_EVAL": None,
    "mongodb_client": None,
    "db": None,
}

# ── Pydantic Models ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str
    model: str = "naive_bayes"

class BatchRequest(BaseModel):
    text: str
    model: str = "naive_bayes"

class FakeNewsRequest(BaseModel):
    text: str

# ── Startup/Shutdown ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up News Classifier...")
    
    # Load models
    if models_exist():
        print("Loading saved models...")
        MODELS, VECTORIZER, EVAL_DATA = load_all_models()
    else:
        print("Training all models from scratch...")
        df = load_data("data/test.csv")
        MODELS, VECTORIZER, EVAL_DATA = train_all(df)
    
    app_state["MODELS"] = MODELS
    app_state["VECTORIZER"] = VECTORIZER
    app_state["EVAL_DATA"] = EVAL_DATA
    
    # Load fake news model
    if fake_models_exist():
        try:
            print("Loading fake news model...")
            FAKE_MODEL, FAKE_VECTORIZER, FAKE_EVAL = load_fake_model()
            app_state["FAKE_MODEL"] = FAKE_MODEL
            app_state["FAKE_VECTORIZER"] = FAKE_VECTORIZER
            app_state["FAKE_EVAL"] = FAKE_EVAL
            print("✓ Fake news model loaded.")
        except Exception as e:
            print(f"⚠ Failed to load fake news model: {str(e)}")
            app_state["FAKE_MODEL"] = None
            app_state["FAKE_VECTORIZER"] = None
            app_state["FAKE_EVAL"] = None
    else:
        print("⚠ Fake news model files not found. Skipping.")
        app_state["FAKE_MODEL"] = None
        app_state["FAKE_VECTORIZER"] = None
        app_state["FAKE_EVAL"] = None
    
    # Connect to MongoDB
    print(f"Connecting to MongoDB at {MONGODB_URI}...")
    try:
        # Create MongoDB client with short connection timeout
        mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000
        )
        # Try to ping MongoDB with asyncio timeout
        try:
            await asyncio.wait_for(
                mongodb_client.admin.command("ping"),
                timeout=10.0
            )
            db = mongodb_client[DATABASE_NAME]
            app_state["mongodb_client"] = mongodb_client
            app_state["db"] = db
            print("✓ MongoDB connection confirmed.")
        except asyncio.TimeoutError:
            print("⚠ MongoDB connection timed out (not running).")
            app_state["mongodb_client"] = None
            app_state["db"] = None
    except Exception as e:
        print(f"⚠ MongoDB unavailable: {str(e)[:50]}")
        app_state["mongodb_client"] = None
        app_state["db"] = None
    
    print("Ready.\n")
    
    yield
    
    # Shutdown
    print("Shutting down...")
    if app_state["mongodb_client"]:
        app_state["mongodb_client"].close()
    print("Goodbye.")

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="News Classifier API",
    description="Multi-model news classification service",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount Jinja2 templates
templates = Jinja2Templates(directory="templates")

# ── Helper Dependencies ───────────────────────────────────────────────────────

def get_db():
    """Dependency to inject db into auth functions."""
    return app_state["db"]

async def get_current_user_with_db(
    token: str = Depends(oauth2_scheme),
    db = Depends(get_db)
):
    """Get current user with db injected."""
    return await get_current_user(token, db)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the index page with authentication context."""
    # Always render the HTML; JS reads token from localStorage client-side.
    # The is_authenticated variable is injected as a JS constant for server-side hint.
    is_authenticated = False
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            db = app_state["db"]
            if db:
                await get_current_user(token, db)
                is_authenticated = True
        except Exception:
            is_authenticated = False

    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    # Inject is_authenticated into template context (JS constant placeholder)
    html_content = html_content.replace(
        "const isAuthenticated = false; // SERVER_INJECT",
        f"const isAuthenticated = {str(is_authenticated).lower()};"
    )

    return html_content

@app.get("/api/models")
async def get_models():
    """Get model definitions."""
    return MODEL_DEFS

@app.get("/api/evaluation")
async def get_evaluation():
    """Get evaluation metrics for all models."""
    return app_state["EVAL_DATA"]

@app.post("/api/predict")
async def predict_route(payload: PredictRequest):
    """Predict category for a single text."""
    text = payload.text.strip()
    model_key = payload.model
    
    MODELS = app_state["MODELS"]
    VECTORIZER = app_state["VECTORIZER"]
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Please enter at least 10 characters.")
    if model_key not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}")
    
    result = predict_one(text, model_key, MODELS, VECTORIZER)
    return result

@app.post("/api/batch")
async def batch_route(payload: BatchRequest):
    """Predict categories for multiple texts (max 50)."""
    raw = payload.text.strip()
    model_key = payload.model
    
    MODELS = app_state["MODELS"]
    VECTORIZER = app_state["VECTORIZER"]
    
    if not raw:
        raise HTTPException(status_code=400, detail="No text provided.")
    if model_key not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}")
    
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if len(lines) > 50:
        raise HTTPException(status_code=400, detail="Batch limit is 50 lines.")
    
    results = predict_batch(lines, model_key, MODELS, VECTORIZER)
    return results

@app.post("/api/retrain")
async def retrain(current_user: dict = Depends(get_current_user_with_db)):
    """Retrain all models from the data file (protected route)."""
    try:
        df = load_data("data/test.csv")
        MODELS, VECTORIZER, EVAL_DATA = train_all(df)
        app_state["MODELS"] = MODELS
        app_state["VECTORIZER"] = VECTORIZER
        app_state["EVAL_DATA"] = EVAL_DATA
        return {"status": "Retrained successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Authentication Routes ─────────────────────────────────────────────────────

@app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, db = Depends(get_db)):
    """Register a new user."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection not available")
    
    user = await register_user(user_data, db)
    return UserResponse(
        username=user["username"],
        email=user["email"],
        created_at=user["created_at"]
    )

@app.post("/auth/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db = Depends(get_db)):
    """
    OAuth2 compatible token login.
    Use the /docs endpoint to test, or send:
    POST /auth/token
    username=<username>&password=<password>
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection not available")
    
    user = await authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user["username"]})
    return Token(access_token=access_token, token_type="bearer")

@app.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user_with_db)):
    """Get current authenticated user (protected route)."""
    return UserResponse(
        username=current_user["username"],
        email=current_user["email"],
        created_at=current_user["created_at"]
    )

@app.get("/auth/login", response_class=HTMLResponse)
async def auth_login_page():
    path = os.path.join(os.path.dirname(__file__), "templates", "login.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ── Fake News Detection Routes (Protected) ────────────────────────────────────

@app.post("/api/fake-news/predict")
async def predict_fake_news(
    payload: FakeNewsRequest,
    current_user: dict = Depends(get_current_user_with_db)
):
    """
    Predict whether an article is fake news (protected route).
    Logs the inference to MongoDB.
    """
    text = payload.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Please enter at least 10 characters.")
    
    # Check if fake news model is available
    FAKE_MODEL = app_state["FAKE_MODEL"]
    FAKE_VECTORIZER = app_state["FAKE_VECTORIZER"]
    
    if not FAKE_MODEL or not FAKE_VECTORIZER:
        raise HTTPException(status_code=503, detail="Fake news model not available.")
    
    # Make prediction
    result = predict_fake(text, FAKE_MODEL, FAKE_VECTORIZER)
    
    # Log to MongoDB if available
    if app_state["db"] is not None:
        try:
            db = app_state["db"]
            log_entry = {
                "username": current_user["username"],
                "input_text": text[:200],  # First 200 chars
                "result_label": result["label_text"],
                "confidence": result["confidence"],
                "timestamp": datetime.utcnow()
            }
            await db["fake_news_logs"].insert_one(log_entry)
        except Exception as e:
            print(f"⚠ Failed to log fake news prediction: {str(e)}")
    
    return result


@app.get("/api/fake-news/metrics")
async def get_fake_news_metrics(
    current_user: dict = Depends(get_current_user_with_db)
):
    """
    Get fake news model evaluation metrics including ROC curve data (protected route).
    """
    if app_state["FAKE_EVAL"] is None:
        raise HTTPException(status_code=503, detail="Fake news metrics not available.")
    
    return app_state["FAKE_EVAL"]


@app.get("/api/fake-news/history")
async def get_fake_news_history(
    current_user: dict = Depends(get_current_user_with_db)
):
    """
    Get the last 20 fake news inference logs for the current user (protected route).
    """
    if app_state["db"] is None:
        raise HTTPException(status_code=503, detail="Database not available.")
    
    try:
        db = app_state["db"]
        username = current_user["username"]
        
        # Query last 20 logs for this user, sorted by timestamp descending
        cursor = db["fake_news_logs"].find(
            {"username": username}
        ).sort("timestamp", -1).limit(20)
        
        logs = await cursor.to_list(length=20)
        
        # Convert ObjectId to string and format timestamps
        for log in logs:
            log["_id"] = str(log["_id"])
            if "timestamp" in log:
                log["timestamp"] = log["timestamp"].isoformat()
        
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "mongodb_connected": app_state["db"] is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)