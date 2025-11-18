from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.models.pydantic_models import (
    AnalyzeRequest, AnalyzeResponse,
    ChatRequest, ChatResponse
)
from api.core.ai_services import AIService
from api.core.rag_services import RAGService
import uvicorn

app = FastAPI(
    title="LawGeeks API",
    description="API for demystifying legal documents using Gemini and RAG.",
    version="1.0.0"
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Service Initialization ---
try:
    ai_service = AIService()
    rag_service = RAGService()
except Exception as e:
    print(f"FATAL: Could not initialize AI services. {e}")
    ai_service = None
    rag_service = None

# --- API Endpoints ---

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_document(request: AnalyzeRequest):
    """
    Receives a full legal document and returns a structured
    markdown analysis.
    """
    if not ai_service:
        raise HTTPException(status_code=500, detail="AI Service not initialized.")
    try:
        # This now returns a single markdown string
        analysis_string = ai_service.get_document_overview(request.document_text)
        return AnalyzeResponse(analysis_text=analysis_string)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    """
    Receives a user's question and the document context,
    and returns a RAG-powered answer.
    """
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG Service not initialized.")
    try:
        # This is your RAG-powered chat
        answer = rag_service.answer_user_query(
            document_text=request.document_text,
            user_question=request.question
        )
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# --- Static Frontend Serving ---
# This serves index.html and app.html
app.mount("/", StaticFiles(directory="public", html=True), name="static")


if __name__ == "__main__":
    print("Starting local development server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)