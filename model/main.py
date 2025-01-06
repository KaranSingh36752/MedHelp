import os
from PyPDF2 import PdfReader
import math
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
)
from getpass import getpass
import torch
from torch.cuda.amp import autocast
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, pipeline
import re


# Function to read the PDF content
def read_pdf(file_path):
    """
    Reads text from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Combined text of all pages.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()


# Function to split content into chunks
def create_chunks(text, chunk_size=3000, overlap=500):
    """
    Splits text into chunks with overlapping context.

    Args:
        text (str): The input text.
        chunk_size (int): The maximum size of each chunk (characters).
        overlap (int): The number of overlapping characters between chunks.

    Returns:
        list: List of text chunks.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1
        if current_length + word_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            overlap_words = (
                current_chunk[-math.ceil(overlap / len(current_chunk)) :]
                if current_chunk
                else []
            )
            current_chunk = overlap_words
            current_length = sum(len(w) + 1 for w in current_chunk)

        current_chunk.append(word)
        current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


file_path = "./Documents/AFFAIRE C.P. ET M.N. c. FRANCE.pdf"
document_text = read_pdf(file_path)
chunks = create_chunks(document_text, chunk_size=3000, overlap=500)

print(f"Total number of chunks: {len(chunks)}")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass("Enter your Hugging Face API token: ")

model_name = "facebook/mbart-large-50-many-to-one-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def translate_chunks(chunks, src_lang="fr_XX", batch_size=8):
    """
    Translates a list of text chunks into English using MBart50.

    Args:
        chunks (list): List of text chunks in the source language.
        src_lang (str): Source language code.
        batch_size (int): Number of chunks to process in a single batch.

    Returns:
        list: Translated text chunks in English.
    """
    translations = []
    tokenizer.src_lang = src_lang

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        encoded_input = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with autocast():
            generated_tokens = model.generate(
                **encoded_input,
                max_length=128,
                num_beams=1,
            )

        batch_translations = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        translations.extend(batch_translations)

    return translations


# Chunk translation
translated_chunks = translate_chunks(chunks, src_lang="fr_XX", batch_size=8)

for i, translation in enumerate(translated_chunks, 1):
    print(f"Chunk {i} Translation:\n{translation}\n")

pc = Pinecone(api_key=getpass("Enter your Pinecone API token: "))

index_name = "legal-llm"
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

index = pc.Index(index_name)

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = embedding_model.encode(translated_chunks)

for i, chunk in enumerate(translated_chunks):
    index.upsert(vectors=[(f"doc_{i}", embeddings[i], {"text": chunk})])

print("Translated chunks embedded and stored in Pinecone.")

login(token=getpass("Enter your Hugging Face access token: "))

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Configure the model for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load the model with the BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Create a pipeline for text generation
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prompt template
PROMPT_TEMPLATE = """You are a highly skilled legal expert with deep knowledge of law and legal cases.
You provide accurate, well-reasoned, and reliable answers based strictly on the given legal documents.

Context:
{context}

Question:
{question}

Instructions:
- Answer the question strictly based on the given context.
- If the context does not contain enough information, state that explicitly.
- Provide detailed legal reasoning where necessary.
- Keep the response clear and professional.
"""


# Function to perform similarity search
def search_similar_documents(query, index, embedding_model, top_k=5):
    """
    Searches for the most relevant documents in the Pinecone index based on query similarity.

    Args:
        query (str): The input legal question.
        index (pinecone.Index): The Pinecone index instance.
        embedding_model (SentenceTransformer): The embedding model.
        top_k (int): Number of top similar documents to retrieve.

    Returns:
        list: Retrieved similar documents.
    """
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    return [match["metadata"]["text"] for match in results["matches"]]


# Function to generate response
def generate_legal_response(question, index, embedding_model, llm, top_k=5):
    """
    Generates a legal response by retrieving relevant documents and using the LLM.

    Args:
        question (str): The legal question asked by the user.
        index (pinecone.Index): The Pinecone index instance.
        embedding_model (SentenceTransformer): The embedding model.
        llm (Pipeline): The LLM text generation pipeline.
        top_k (int): Number of relevant document chunks to retrieve.

    Returns:
        str: The legal response generated by the LLM.
    """
    # Retrieve relevant context from the vector database
    relevant_docs = search_similar_documents(question, index, embedding_model, top_k)

    if not relevant_docs:
        return "I couldn't find relevant legal information in the provided documents."

    context = "\n\n".join(relevant_docs)

    # Prompt template
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # Generate response
    response = llm(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0][
        "generated_text"
    ]

    # Remove any parts of the prompt accidentally included in the output
    cleaned_response = response.replace(prompt, "").strip()

    # Remove "Legal Answer:", "Context:", and "Instructions:" titles and content if they appear in the original response
    cleaned_response = re.sub(
        r"(?i)(Legal Answer:|Context:|Instructions:)", "", cleaned_response
    ).strip()

    return cleaned_response


question = (
    "What were the key legal arguments in the case AFFAIRE C.P. ET M.N. c. FRANCE?"
)

response = generate_legal_response(question, index, embedding_model, llm, top_k=5)

print(response)
