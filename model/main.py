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
