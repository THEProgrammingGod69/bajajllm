# main.py - Create this file in your LLM folder
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import logging

# For now, let's create a simple version without LangChain to test first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document QA Webhook API",
    description="Upload documents and query them using natural language",
    version="1.0.0"
)

# Simple data store (in production, you'd use a database)
documents_store = []
qa_ready = False

# ========== PYDANTIC MODELS ==========
class TextUpload(BaseModel):
    text: str
    filename: Optional[str] = "uploaded_text"

class QueryRequest(BaseModel):
    query: str

class WebhookRequest(BaseModel):
    type: str
    query: Optional[str] = None
    text: Optional[str] = None

# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Document QA Webhook API is running!",
        "status": "healthy",
        "docs": "Go to /docs for interactive API documentation",
        "endpoints": {
            "upload_text": "/upload/text - POST - Upload text content",
            "query": "/query - POST - Ask questions",
            "webhook": "/webhook - POST - Generic webhook",
            "health": "/health - GET - Health check",
            "status": "/status - GET - System status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app_running": True,
        "documents_loaded": len(documents_store),
        "qa_ready": qa_ready
    }

@app.get("/status")
async def status():
    """Status endpoint."""
    return {
        "status": "running",
        "documents_count": len(documents_store),
        "qa_available": qa_ready,
        "version": "1.0.0"
    }

@app.post("/upload/text")
async def upload_text(request: TextUpload):
    """Upload text content."""
    global qa_ready
    
    try:
        # For now, just store the text (later we'll add vector processing)
        document_data = {
            "filename": request.filename,
            "text": request.text,
            "length": len(request.text)
        }
        
        documents_store.append(document_data)
        qa_ready = True
        
        return {
            "message": "Text uploaded successfully",
            "filename": request.filename,
            "text_length": len(request.text),
            "total_documents": len(documents_store),
            "status": "ready"
        }
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query the uploaded documents."""
    if not qa_ready or not documents_store:
        raise HTTPException(
            status_code=400, 
            detail="No documents loaded. Please upload documents first."
        )
    
    try:
        # Simple text search (later we'll add semantic search)
        query_lower = request.query.lower()
        relevant_docs = []
        
        for i, doc in enumerate(documents_store):
            # Simple keyword matching
            if any(word in doc["text"].lower() for word in query_lower.split()):
                relevant_docs.append({
                    "rank": i + 1,
                    "filename": doc["filename"],
                    "content": doc["text"][:300] + "..." if len(doc["text"]) > 300 else doc["text"],
                    "relevance": "keyword_match"
                })
        
        if not relevant_docs:
            relevant_docs = [{
                "rank": 1,
                "filename": documents_store[0]["filename"],
                "content": documents_store[0]["text"][:300] + "..." if len(documents_store[0]["text"]) > 300 else documents_store[0]["text"],
                "relevance": "fallback_first_document"
            }]
        
        return {
            "query": request.query,
            "answer": f"Based on the uploaded documents, here are the most relevant sections for your query: '{request.query}'",
            "relevant_documents": relevant_docs[:3],  # Top 3
            "total_documents_searched": len(documents_store),
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/webhook")
async def webhook_handler(request: WebhookRequest):
    """Generic webhook handler."""
    try:
        logger.info(f"Webhook received: {request.dict()}")
        
        if request.type == "upload" and request.text:
            # Handle text upload via webhook
            upload_request = TextUpload(text=request.text, filename="webhook_upload")
            return await upload_text(upload_request)
        
        elif request.type == "query" and request.query:
            # Handle query via webhook
            query_request = QueryRequest(query=request.query)
            return await query_documents(query_request)
        
        else:
            # Generic response
            return {
                "message": "Webhook received successfully",
                "type": request.type,
                "processed_at": "now",
                "status": "received",
                "data": request.dict()
            }
    
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")

# Add CORS middleware for web browsers
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== STARTUP ==========
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)