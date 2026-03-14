"""
main.py — FastAPI server for VitalVoice CrewAI + RAG backend.
Exposes a single /analyze endpoint that Android app calls.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from rag import initialize_vector_db
from crew import VitalVoiceCrew
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="VitalVoice AI Backend",
    description="Multi-agent healthcare AI with RAG pipeline",
    version="1.0.0"
)

# Allow requests from Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Request/Response models ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    message: str
    module: str = "general"  # symptom, mental_health, report, child, emergency, etc.

class AnalyzeResponse(BaseModel):
    response: str
    blocked: bool
    module: str

# ── Initialize vector DB on startup ───────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    print("Initializing VitalVoice AI Backend...")
    initialize_vector_db()
    print("Backend ready!")

# ── Health check endpoint ──────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "VitalVoice AI Backend is running",
        "version": "1.0.0",
        "agents": ["Intent Filter", "Medical Analyst", "Risk Detector", "Response Writer"],
        "vector_db": "ChromaDB with medical knowledge base"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ── Main analyze endpoint ──────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Main endpoint called by VitalVoice Android app.
    Runs query through CrewAI multi-agent pipeline with RAG.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if len(request.message) > 2000:
        raise HTTPException(status_code=400, detail="Message too long. Maximum 2000 characters.")

    try:
        crew = VitalVoiceCrew()
        result = crew.run(
            user_input=request.message.strip(),
            module=request.module
        )
        return AnalyzeResponse(
            response=result["response"],
            blocked=result["blocked"],
            module=result["module"]
        )
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"AI processing error: {str(e)}"
        )
