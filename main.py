import os
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import urllib.request
import urllib.error
import tempfile
import logging
from dotenv import load_dotenv
import asyncio
# Optional OCR dependencies (uncomment if needed)
# from pdf2image import convert_from_path
# import pytesseract

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Recipe AI Chatbot", description="Multi-agent recipe chatbot with PDF processing")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount templates folder as static files
templates_dir = "templates"
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)
app.mount("/templates", StaticFiles(directory=templates_dir), name="templates")

# Initialize Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama3-8b-8192",
    temperature=0.7
)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
persist_directory = "./recipe_vector_db"
vector_store = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

# Define request/response models
class PDFUrl(BaseModel):
    url: str

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    pages: List[int]

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True
)

# Custom prompt template
prompt_template = """
You are a professional culinary assistant specializing in recipes. Provide accurate, detailed, and helpful responses based on the provided context from a recipe PDF. Include page numbers if available. If the query is unclear or no relevant information is found, suggest alternatives or clarify.

Context: {context}
Question: {question}

Answer in a professional tone with step-by-step instructions or relevant details. If applicable, cite the page number(s) from the source PDF.
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Agent Classes
class PDFProcessorAgent:
    async def process_pdf(self, pdf_path: str, source_url: str = None) -> List[Dict[str, Any]]:
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            if not documents:
                logger.warning(f"No text extracted from PDF: {pdf_path}")
                # Optional OCR (uncomment to enable)
                # images = convert_from_path(pdf_path)
                # documents = []
                # for i, image in enumerate(images):
                #     text = pytesseract.image_to_string(image)
                #     if text.strip():
                #         documents.append({"page_content": text, "metadata": {"page": i + 1, "source": source_url or pdf_path}})
                
                if not documents:
                    raise ValueError("No content extracted from PDF, may require OCR")

            logger.debug(f"Extracted {len(documents)} pages from PDF: {pdf_path}")
            for i, doc in enumerate(documents):
                doc.metadata["page"] = i + 1
                doc.metadata["source"] = source_url or pdf_path
                logger.debug(f"Page {i+1} content sample: {doc.page_content[:100]}...")

            texts = text_splitter.split_documents(documents)
            logger.info(f"PDFProcessorAgent: Split into {len(texts)} chunks for PDF {source_url or pdf_path}")
            return texts
        except Exception as e:
            logger.error(f"PDFProcessorAgent: Error processing PDF {source_url or pdf_path}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

class PDFDownloaderAgent:
    async def download_pdf(self, pdf_url: str) -> str:
        try:
            if not pdf_url.lower().endswith('.pdf'):
                raise ValueError("URL must point to a PDF file")
            
            req = urllib.request.Request(
                pdf_url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            )
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                with urllib.request.urlopen(req) as response:
                    tmp_file.write(response.read())
                logger.info(f"PDFDownloaderAgent: Downloaded PDF from {pdf_url} to {tmp_file.name}")
                return tmp_file.name
        except urllib.error.HTTPError as e:
            logger.error(f"PDFDownloaderAgent: HTTP Error {e.code}: {e.reason} for URL {pdf_url}")
            raise HTTPException(status_code=404, detail=f"Failed to download PDF: HTTP {e.code} {e.reason}")
        except Exception as e:
            logger.error(f"PDFDownloaderAgent: Error downloading PDF from {pdf_url}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to download PDF: {str(e)}")

class KnowledgeStoreAgent:
    async def store_documents(self, documents: List[Dict[str, Any]]) -> bool:
        try:
            if not documents:
                logger.warning("KnowledgeStoreAgent: No documents to store")
                return False
            vector_store.add_documents(documents)
            vector_store.persist()
            logger.info(f"KnowledgeStoreAgent: Stored {len(documents)} documents")
            return True
        except Exception as e:
            logger.error(f"KnowledgeStoreAgent: Error storing documents: {str(e)}")
            return False

class RetrievalAgent:
    async def retrieve(self, query: str) -> Dict[str, Any]:
        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            docs = retriever.get_relevant_documents(query)
            if not docs:
                logger.warning(f"RetrievalAgent: No documents found for query: {query}")
                return {"context": "", "sources": [], "pages": [], "documents": []}
            
            context = "\n".join([f"[Page {doc.metadata.get('page', 'Unknown')}]: {doc.page_content}" for doc in docs])
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            pages = [doc.metadata.get("page", 0) for doc in docs]
            logger.debug(f"RetrievalAgent: Retrieved {len(docs)} documents for query: {query}")
            return {"context": context, "sources": sources, "pages": pages, "documents": docs}
        except Exception as e:
            logger.error(f"RetrievalAgent: Error retrieving documents: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

class ResponseAgent:
    async def generate_response(self, query: str, context: str) -> str:
        try:
            if not context:
                logger.warning(f"ResponseAgent: Empty context for query: {query}")
                return "No relevant recipe information found. Please upload a recipe PDF or clarify your query."
            
            prompt = PROMPT.format(context=context, question=query)
            response = llm.invoke(prompt).content
            logger.info(f"ResponseAgent: Generated response for query: {query}")
            return response
        except Exception as e:
            logger.error(f"ResponseAgent: Error generating response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Response generation error: {str(e)}")

class CoordinatorAgent:
    def __init__(self):
        self.downloader = PDFDownloaderAgent()
        self.pdf_processor = PDFProcessorAgent()
        self.knowledge_store = KnowledgeStoreAgent()
        self.retrieval = RetrievalAgent()
        self.response = ResponseAgent()

    async def process_pdf_url(self, pdf_url: str) -> bool:
        try:
            pdf_path = await self.downloader.download_pdf(pdf_url)
            documents = await self.pdf_processor.process_pdf(pdf_path, source_url=pdf_url)
            success = await self.knowledge_store.store_documents(documents)
            logger.info(f"CoordinatorAgent: Processed PDF from {pdf_url} with success: {success}")
            return success
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"CoordinatorAgent: Error processing PDF from {pdf_url}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")
        finally:
            try:
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
            except Exception as e:
                logger.error(f"CoordinatorAgent: Error cleaning up temp file: {str(e)}")

    async def process_local_pdf(self, pdf_file: UploadFile) -> bool:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(await pdf_file.read())
                tmp_file_path = tmp_file.name
            
            documents = await self.pdf_processor.process_pdf(tmp_file_path, source_url=pdf_file.filename)
            success = await self.knowledge_store.store_documents(documents)
            logger.info(f"CoordinatorAgent: Processed local PDF {pdf_file.filename} with success: {success}")
            return success
        except Exception as e:
            logger.error(f"CoordinatorAgent: Error processing local PDF {pdf_file.filename}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to process local PDF: {str(e)}")
        finally:
            try:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
            except Exception as e:
                logger.error(f"CoordinatorAgent: Error cleaning up temp file: {str(e)}")

    async def handle_query(self, query: str) -> Dict[str, Any]:
        try:
            retrieval_result = await self.retrieval.retrieve(query)
            response = await self.response.generate_response(query, retrieval_result["context"])
            return {
                "response": response,
                "sources": retrieval_result["sources"],
                "pages": retrieval_result["pages"]
            }
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"CoordinatorAgent: Error handling query: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

# Initialize coordinator
coordinator = CoordinatorAgent()

# API Endpoints
@app.get("/")
async def serve_index():
    """Serve the index.html file at the root endpoint."""
    index_path = os.path.join(templates_dir, "index.html")
    if not os.path.exists(index_path):
        logger.error(f"Index file not found at {index_path}")
        raise HTTPException(status_code=404, detail="index.html not found. Please ensure templates/index.html exists.")
    return FileResponse(index_path)

@app.post("/upload_pdf", response_model=dict)
async def upload_pdf(pdf: PDFUrl):
    """Upload and process a recipe PDF from a URL."""
    try:
        success = await coordinator.process_pdf_url(pdf.url)
        if success:
            return {"message": "PDF processed successfully"}
        raise HTTPException(status_code=400, detail="Failed to process PDF: Unknown error")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Upload endpoint: Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/upload_local_pdf", response_model=dict)
async def upload_local_pdf(pdf_file: UploadFile = File(...)):
    """Upload and process a local recipe PDF file."""
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    try:
        success = await coordinator.process_local_pdf(pdf_file)
        if success:
            return {"message": "Local PDF processed successfully"}
        raise HTTPException(status_code=400, detail="Failed to process local PDF: Unknown error")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Upload local endpoint: Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle user queries about recipes."""
    try:
        result = await coordinator.handle_query(request.query)
        return ChatResponse(
            response=result["response"],
            sources=result["sources"],
            pages=result["pages"]
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Chat endpoint: Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup."""
    logger.info("Recipe Chatbot initialized with multi-agent system")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=8000)