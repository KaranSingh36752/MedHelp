import os
import torch
import uvicorn
import magic

from torch.amp import autocast
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from io import BytesIO
from PyPDF2 import PdfReader
from pydantic import BaseModel

from contextlib import asynccontextmanager
from pydantic import Field
from time import time
from dotenv import load_dotenv

from rag_pipeline import RAGPipeline

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Translation model and resources
translation_model = "facebook/mbart-large-50-many-to-one-mmt"
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize RAG pipeline
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV", "aws")
groq_api_key = os.getenv("GROQ_API_KEY")

rag_pipeline = RAGPipeline(
    embedding_model_name="all-MiniLM-L6-v2",
    pinecone_api_key=pinecone_api_key,
    pinecone_env=pinecone_env,
    groq_api_key=groq_api_key,
)


# Pydantic model for query request
class QueryRequest(BaseModel):
    user_query: str = Field(
        ...,
        min_length=5,
        title="User Query",
        description="The legal question to be analyzed",
    )


# Asynchronous context manager to load and release model resources
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    model = MBartForConditionalGeneration.from_pretrained(translation_model).to(device)
    tokenizer = MBart50TokenizerFast.from_pretrained(translation_model)
    print("Model and tokenizer loaded.")
    yield
    del model
    del tokenizer
    print("Model and tokenizer resources released.")


app.router.lifespan_context = lifespan


# Function to detect MIME type
def get_mime_type(file: UploadFile):
    mime = magic.Magic(mime=True)
    file_content = file.file.read(2048)
    mime_type = mime.from_buffer(file_content)
    file.file.seek(0)
    return mime_type


# Function to read PDF content
def read_pdf(file: UploadFile):
    pdf_content = BytesIO(file.file.read())
    reader = PdfReader(pdf_content)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()


# Function to split content into chunks
def create_chunks(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1
        if current_length + word_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            overlap_words = (
                current_chunk[-overlap:]
                if overlap <= len(current_chunk)
                else current_chunk
            )
            current_chunk = overlap_words
            current_length = sum(len(w) + 1 for w in current_chunk)

        current_chunk.append(word)
        current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Function to translate chunks
def translate_chunks(chunks, batch_size):
    translations = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        encoded_input = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(device)

        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            generated_tokens = model.generate(
                encoded_input["input_ids"], max_length=128, num_beams=1
            )

        translations.extend(
            tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        )
    return translations


# Function to store embeddings in Pinecone
def store_embeddings(translated_chunks):
    """
    Converts translated text into embeddings and stores them in Pinecone.

    Args:
        translated_chunks (list): List of translated text chunks.
    """
    try:
        # Retrieve the Pinecone API key from the .env file
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY not found in environment variables. Ensure it is set in the .env file."
            )

        # Initialize Pinecone connection
        pc = Pinecone(api_key=pinecone_api_key)

        index_name = "legal-llm-portal"

        # Check if the index exists and delete it for a clean slate (optional)
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)

        # Create a new index
        pc.create_index(
            name=index_name,
            dimension=384,  # Dimension size for the 'all-MiniLM-L6-v2' model
            metric="cosine",  # Using cosine similarity for comparisons
            spec=ServerlessSpec(
                cloud="aws", region="us-east-1"
            ),  # Adjust cloud and region as needed
        )

        # Connect to the index
        index = pc.Index(index_name)

        # Initialize the embedding model
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Generate embeddings for the translated chunks
        embeddings = embedding_model.encode(translated_chunks)

        # Upsert embeddings into Pinecone
        for i, chunk in enumerate(translated_chunks):
            index.upsert(vectors=[(f"doc_{i}", embeddings[i], {"text": chunk})])

        print("Translated chunks embedded and stored in Pinecone.")

    except Exception as e:
        print(f"Error storing embeddings in Pinecone: {str(e)}")

    return index_name


# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the LegalDoc-Translate-Query-Assistant portal!"}


@app.post("/translate-pdf/")
async def process_pdf(
    file: UploadFile,
    background_tasks: BackgroundTasks,  # Add background task support
    chunk_size: int = 500,
    overlap: int = 50,
    batch_size: int = 2,
):
    """
    Endpoint to process the PDF, translate chunks, and asynchronously store embeddings.
    """
    mime_type = get_mime_type(file)
    if mime_type != "application/pdf":
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a valid PDF file."
        )

    if file.size > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=400, detail="File size exceeds 10 MB.")

    try:
        # Step 1: Process the PDF and translate chunks
        start_time = time()
        text = read_pdf(file)
        chunks = create_chunks(text, chunk_size, overlap)
        translated_chunks = translate_chunks(chunks, batch_size)
        processing_time = time() - start_time

        # Step 2: Store embeddings in Pinecone in the background
        background_tasks.add_task(store_embeddings, translated_chunks)

        # Step 3: Return immediate response with translated chunks
        return {
            "mime_type": mime_type,
            "translated_chunks": translated_chunks,
            "processing_time": f"{processing_time:.2f} seconds",
            "pinecone_status": "Embedding storage initiated in the background.",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing the PDF: {str(e)}"
        )


@app.post("/query/")
async def query_llm(request: QueryRequest):
    """
    Endpoint to process user query, retrieve relevant context, and return LLM-generated responses.
    """
    try:
        # Retrieve relevant context from Pinecone
        context = rag_pipeline.get_context(request.user_query)

        if not context:
            raise HTTPException(
                status_code=404, detail="No relevant context found for the query."
            )

        # Generate response using the retrieved context
        response = rag_pipeline.generate_response(request.user_query, context)

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
