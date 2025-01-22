from fastapi import FastAPI, UploadFile, HTTPException
from io import BytesIO
from PyPDF2 import PdfReader
import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from torch.amp import autocast
import torch
import uvicorn
import magic
from time import time
from contextlib import asynccontextmanager

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize FastAPI app
app = FastAPI()

model_name = "facebook/mbart-large-50-many-to-one-mmt"
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
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


# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the LegalDoc-Translate-Query-Assistant portal!"}


@app.post("/process-pdf/")
async def process_pdf(
    file: UploadFile, chunk_size: int = 500, overlap: int = 50, batch_size: int = 2
):
    mime_type = get_mime_type(file)
    if mime_type != "application/pdf":
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a valid PDF file."
        )

    if file.size > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=400, detail="File size exceeds 10 MB.")

    try:
        start_time = time()
        text = read_pdf(file)
        chunks = create_chunks(text, chunk_size, overlap)
        translated_chunks = translate_chunks(chunks, batch_size)
        processing_time = time() - start_time

        return {
            "mime_type": mime_type,
            "translated_chunks": translated_chunks,
            "processing_time": f"{processing_time:.2f} seconds",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing the PDF: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
