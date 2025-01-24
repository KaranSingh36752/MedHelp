# Setting Up the Legal LLM Environment

This guide provides step-by-step instructions to set up the environment required for the Legal LLM project. The environment leverages Python 3.9 and includes necessary libraries for machine learning, natural language processing, and PDF handling.

## Prerequisites

Ensure you have the following installed on your system:
- [Anaconda](https://www.anaconda.com/products/distribution)
- A compatible NVIDIA GPU (for CUDA acceleration)
- Python 3.9

## Environment Setup Steps

Follow these commands to create and configure the required Conda environment.

### Step 1: Create and Activate the Conda Environment

```bash
conda create -n legal-llm-env python=3.9
conda activate legal-llm-env
```

This creates a new Conda environment named `legal-llm-env` with Python version 3.9.

### Step 2: Install PyTorch with CUDA Support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This installs PyTorch along with CUDA 11.8 support for GPU acceleration.

### Step 3: Install Required Python Libraries

```bash
pip install transformers fastapi uvicorn python-magic PyPDF2 sentence-transformers pinecone-client python-dotenv groq
```

These libraries are necessary for the following functionalities:
- **`transformers`**: For NLP model handling.
- **`fastapi`**: For building APIs.
- **`uvicorn`**: ASGI server for FastAPI.
- **`python-magic`**: File type detection.
- **`PyPDF2`**: PDF processing.
- **`sentence-transformers`**: Embedding-based models.
- **`pinecone-client`**: Vector database interaction.
- **`python-dotenv`**: Environment variable management.

### Step 4: Install Additional Dependencies

```bash
pip install tiktoken protobuf
pip install python-multipart
```

- **`tiktoken`**: Tokenizer library required for processing text efficiently.
- **`protobuf`**: Protocol Buffers support, often required for model serialization.

### Step 5: Upgrade Key Libraries

```bash
pip install --upgrade transformers sentencepiece
```

This ensures that the latest versions of `transformers` and `sentencepiece` (for tokenization) are installed.

## Verifying Installation

After completing the above steps, you can verify the environment setup by running:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

If the commands execute without errors and return version numbers, the setup is successful.

## Additional Notes

- If any issues arise, ensure you have the correct CUDA version installed for your GPU.
- Use `conda list` to check installed packages within the environment.

---
