from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    document_text: str = Field(..., min_length=100, description="The full text of the legal document to analyze.")

class AnalyzeResponse(BaseModel):
    analysis_text: str # The raw markdown string from the AI

class ChatRequest(BaseModel):
    document_text: str = Field(..., description="The context of the user's legal document.")
    question: str = Field(..., min_length=5, description="The user's specific question about the document.")

class ChatResponse(BaseModel):
    answer: str