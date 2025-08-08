# main.py - FastAPI Version
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import tempfile
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document QA Webhook API",
    description="Upload documents and query them using natural language",
    version="1.0.0"
)

# ========== CONFIGURATION ==========
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Global variables
qa_retriever = None
vectorstore = None

# ========== PYDANTIC MODELS ==========
class TextUpload(BaseModel):
    text: str
    filename: Optional[str] = "uploaded_text"

class MultipleTexts(BaseModel):
    documents: List[TextUpload]

class QueryRequest(BaseModel):
    query: str

class WebhookRequest(BaseModel):
    type: str
    query: Optional[str] = None
    text: Optional[str] = None
    data: Optional[dict] = None

# ========== HELPER FUNCTIONS ==========
def load_documents_from_text(text_content: str, filename: str = "uploaded_text"):
    """Load documents from text content."""
    doc = Document(
        page_content=text_content,
        metadata={"source": filename}
    )
    return [doc]

def load_documents_from_files(file_paths: List[str]):
    """Load documents from file paths."""
    docs = []
    for file_path in file_paths:
        try:
            if file_path.endswith(".pdf"):
                docs.extend(PyPDFLoader(file_path).load())
            elif file_path.endswith(".docx"):
                docs.extend(Docx2txtLoader(file_path).load())
            elif file_path.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                docs.extend(load_documents_from_text(content, file_path))
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    return docs

def split_docs(docs):
    """Split documents into chunks."""
    if not docs:
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

def create_vector_store(splits):
    """Create FAISS vector store."""
    if not splits:
        return None
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return None

# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Document QA Webhook API is running!",
        "docs": "/docs",  # FastAPI auto-generates interactive docs
        "endpoints": {
            "upload_text": "/upload/text - POST - Upload text content",
            "upload_files": "/upload/files - POST - Upload files",
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
        "retriever_loaded": qa_retriever is not None,
        "vectorstore_loaded": vectorstore is not None
    }

@app.get("/status")
async def status():
    """Status endpoint."""
    global qa_retriever, vectorstore
    return {
        "status": "running",
        "retriever_available": qa_retriever is not None,
        "vectorstore_available": vectorstore is not None,
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "model": EMBED_MODEL
    }

@app.post("/upload/text")
async def upload_text(request: TextUpload):
    """Upload text content."""
    global qa_retriever, vectorstore
    
    try:
        # Load documents from text
        documents = load_documents_from_text(request.text, request.filename)
        
        # Process documents
        splits = split_docs(documents)
        if not splits:
            raise HTTPException(status_code=500, detail="Failed to split documents")
        
        # Create vector store
        vectorstore = create_vector_store(splits)
        if vectorstore is None:
            raise HTTPException(status_code=500, detail="Failed to create vector store")
        
        # Setup retriever
        qa_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        return {
            "message": "Text uploaded successfully",
            "filename": request.filename,
            "chunks_created": len(splits),
            "status": "ready"
        }
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/upload/multiple")
async def upload_multiple_texts(request: MultipleTexts):
    """Upload multiple text documents."""
    global qa_retriever, vectorstore
    
    try:
        all_documents = []
        for doc_request in request.documents:
            documents = load_documents_from_text(doc_request.text, doc_request.filename)
            all_documents.extend(documents)
        
        if not all_documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        # Process documents
        splits = split_docs(all_documents)
        if not splits:
            raise HTTPException(status_code=500, detail="Failed to split documents")
        
        # Create vector store
        vectorstore = create_vector_store(splits)
        if vectorstore is None:
            raise HTTPException(status_code=500, detail="Failed to create vector store")
        
        # Setup retriever
        qa_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        return {
            "message": "Multiple documents uploaded successfully",
            "documents_count": len(request.documents),
            "total_documents_loaded": len(all_documents),
            "chunks_created": len(splits),
            "status": "ready"
        }
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/upload/files")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload files."""
    global qa_retriever, vectorstore
    
    try:
        temp_files = []
        
        # Save uploaded files to temporary locations
        for file in files:
            if file.filename and any(file.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f".{file.filename.split('.')[-1]}"
                )
                content = await file.read()
                temp_file.write(content)
                temp_file.close()
                temp_files.append(temp_file.name)
        
        if not temp_files:
            raise HTTPException(status_code=400, detail="No valid files uploaded")
        
        # Load documents from files
        documents = load_documents_from_files(temp_files)
        
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents loaded from files")
        
        # Process documents
        splits = split_docs(documents)
        if not splits:
            raise HTTPException(status_code=500, detail="Failed to split documents")
        
        # Create vector store
        vectorstore = create_vector_store(splits)
        if vectorstore is None:
            raise HTTPException(status_code=500, detail="Failed to create vector store")
        
        # Setup retriever
        qa_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        return {
            "message": "Files uploaded successfully",
            "files_processed": len(temp_files),
            "documents_loaded": len(documents),
            "chunks_created": len(splits),
            "status": "ready"
        }
    
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query the uploaded documents."""
    global qa_retriever, vectorstore
    
    if qa_retriever is None or vectorstore is None:
        raise HTTPException(
            status_code=400, 
            detail="No documents loaded. Please upload documents first."
        )
    
    try:
        # Get relevant documents
        relevant_docs = qa_retriever.get_relevant_documents(request.query)
        
        # Format response
        response = {
            "query": request.query,
            "answer": f"Based on the uploaded documents, here are the most relevant sections for your query: '{request.query}'",
            "relevant_documents": [],
            "status": "success"
        }
        
        for i, doc in enumerate(relevant_docs[:3]):  # Top 3 results
            doc_info = {
                "rank": i + 1,
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "relevance_score": f"Rank {i + 1}"
            }
            response["relevant_documents"].append(doc_info)
        
        return response
    
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/webhook")
async def webhook_handler(request: WebhookRequest):
    """Generic webhook handler."""
    try:
        logger.info(f"Webhook received: {request.dict()}")
        
        # Handle different webhook types
        if request.type == "text_upload" and request.text:
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
                "status": "received"
            }
    
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")

# ========== STARTUP ==========
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)