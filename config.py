"""
Configuration settings for Defense Tender LLM
"""
import os
from pathlib import Path

# Base Configuration
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# LLM Model Configuration
class ModelConfig:
    # Question Answering Model
    QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"
    
    # Embedding Model
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    
    # Text Generation Model
    GENERATION_MODEL_NAME = "distilgpt2"
    
    # Classification Model
    CLASSIFICATION_MODEL_NAME = "facebook/bart-large-mnli"
    
    # Model Parameters
    MAX_SEQUENCE_LENGTH = 512
    EMBEDDING_DIMENSION = 384
    BATCH_SIZE = 32
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
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_TOTAL_SIZE = 200 * 1024 * 1024  # 200MB
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

# Environment Configuration
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")