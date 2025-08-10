"""
Enhanced Configuration settings for Defense Tender LLM System
"""
import os
from pathlib import Path

# Base Configuration
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Enhanced LLM Model Configuration
class ModelConfig:
    # Primary Text Generation LLM
    PRIMARY_LLM = "google/flan-t5-base"  # 250M parameters, instruction-tuned
    FALLBACK_LLM = "gpt2-medium"         # 345M parameters
    
    # Alternative LLMs you can experiment with:
    ALTERNATIVE_LLMS = {
        "small": "google/flan-t5-small",      # 80M parameters (faster)
        "large": "google/flan-t5-large",      # 780M parameters (better quality)
        "gpt2_large": "gpt2-large",           # 774M parameters
        "distilgpt2": "distilgpt2",           # 82M parameters (fastest)
    }
    
    # Question Answering Model
    QA_MODEL_NAME = "deepset/roberta-base-squad2"  # More capable than DistilBERT
    
    # Embedding Model
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    
    # Summarization Model
    SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
    
    # Classification Model
    CLASSIFICATION_MODEL = "facebook/bart-large-mnli"
    
    # Generation Parameters
    MAX_GENERATION_LENGTH = 512
    TEMPERATURE = 0.7
    TOP_P = 0.9
    TOP_K = 50
    DO_SAMPLE = True
    
    # Model Parameters
    MAX_SEQUENCE_LENGTH = 512
    EMBEDDING_DIMENSION = 384
    BATCH_SIZE = 16  # Reduced for LLMs
    TOP_K_RESULTS = 5

# Streamlit Configuration
class StreamlitConfig:
    PAGE_TITLE = "Defense Tender LLM Analyzer"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"
    SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", 8501))
    SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")

# Document Processing Configuration
class DocumentConfig:
    SUPPORTED_FORMATS = [".pdf", ".docx", ".txt"]
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB per file
    MAX_TOTAL_SIZE = 200 * 1024 * 1024  # 200MB total
    CHUNK_SIZE = 400  # Words per chunk (optimized for LLMs)
    CHUNK_OVERLAP = 50
    MAX_CHUNKS_PER_DOCUMENT = 50  # Prevent memory issues

# Environment Configuration
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
HUGGING_FACE_TOKEN = os.getenv("HF_TOKEN", None)  # For private models
