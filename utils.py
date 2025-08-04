"""
Utility functions for Defense Tender LLM
"""
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_file(uploaded_file, max_size: int = 50 * 1024 * 1024) -> bool:
    """Validate uploaded file"""
    if not uploaded_file:
        return False
    
    # Check file size
    if hasattr(uploaded_file, 'size') and uploaded_file.size > max_size:
        st.error(f"File {uploaded_file.name} is too large (max 50MB)")
        return False
    
    # Check file extension
    allowed_extensions = ['.pdf', '.docx', '.txt']
    file_ext = Path(uploaded_file.name).suffix.lower()
    if file_ext not in allowed_extensions:
        st.error(f"File type {file_ext} not supported")
        return False
    
    return True

def display_processing_time(start_time: float, operation: str = "Operation"):
    """Display processing time"""
    elapsed = time.time() - start_time
    st.success(f"✅ {operation} completed in {elapsed:.2f} seconds")

def format_confidence_score(score: float) -> str:
    """Format confidence score with color"""
    if score > 0.8:
        return f"🟢 High ({score:.1%})"
    elif score > 0.5:
        return f"🟡 Medium ({score:.1%})"
    else:
        return f"🔴 Low ({score:.1%})"

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_time = time.time()
        self.metrics[operation] = {"start": self.start_time}
    
    def end_timer(self, operation: str):
        """End timing an operation"""
        if operation in self.metrics:
            self.metrics[operation]["end"] = time.time()
            self.metrics[operation]["duration"] = (
                self.metrics[operation]["end"] - self.metrics[operation]["start"]
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics