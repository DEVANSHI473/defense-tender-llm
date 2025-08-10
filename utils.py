"""
Enhanced utility functions for Defense Tender LLM System
"""
import logging
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import streamlit as st
import torch
import psutil
import gc

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Enhanced logging setup"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/llm_app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_file(uploaded_file, max_size: int = 50 * 1024 * 1024) -> Tuple[bool, str]:
    """Enhanced file validation"""
    if not uploaded_file:
        return False, "No file provided"
    
    # Check file size
    if hasattr(uploaded_file, 'size') and uploaded_file.size > max_size:
        return False, f"File {uploaded_file.name} is too large (max {max_size/(1024*1024):.1f}MB)"
    
    # Check file extension
    allowed_extensions = ['.pdf', '.docx', '.txt']
    file_ext = Path(uploaded_file.name).suffix.lower()
    if file_ext not in allowed_extensions:
        return False, f"File type {file_ext} not supported"
    
    return True, "File is valid"

def display_processing_time(start_time: float, operation: str = "Operation"):
    """Display processing time"""
    elapsed = time.time() - start_time
    st.success(f"✅ {operation} completed in {elapsed:.2f} seconds")

def format_confidence_score(score: float, method: str = "unknown") -> str:
    """Format confidence score with method context"""
    method_weights = {
        "LLM_generation": 0.9,
        "extractive_QA": 1.0,
        "semantic_search": 0.8,
        "unknown": 0.7
    }
    
    adjusted_score = score * method_weights.get(method, 0.7)
    
    if adjusted_score > 0.8:
        return f"🟢 High ({adjusted_score:.1%})"
    elif adjusted_score > 0.5:
        return f"🟡 Medium ({adjusted_score:.1%})"
    else:
        return f"🔴 Low ({adjusted_score:.1%})"

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def optimize_memory():
    """Optimize memory usage"""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except Exception:
        return False

def get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage"""
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        resources = {
            "memory_total_gb": memory.total / (1024**3),
            "memory_used_gb": memory.used / (1024**3),
            "memory_percent": memory.percent,
            "cpu_percent": cpu_percent,
            "gpu_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_used = torch.cuda.memory_allocated(0) / (1024**3)
                resources.update({
                    "gpu_memory_total_gb": gpu_memory,
                    "gpu_memory_used_gb": gpu_used,
                    "gpu_memory_percent": (gpu_used / gpu_memory) * 100
                })
            except Exception:
                pass
        
        return resources
    except Exception:
        return {"error": "Could not retrieve system resources"}

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
