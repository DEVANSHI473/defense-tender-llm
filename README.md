
🚀 Quick Start
Prerequisites

Python 3.8+
8GB+ RAM (for model loading)
2GB+ disk space (for model cache)

Installation
bash# Clone or create project directory
mkdir defense-tender-llm && cd defense-tender-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the LLM application
streamlit run defense_tender_llm.py
First Run

Models will download automatically (~2GB)
Access at: http://localhost:8501
Upload test documents (PDF, DOCX, TXT)
Ask questions using natural language

🎯 Core Capabilities
1. Semantic Document Search
python# Uses sentence transformers to create dense embeddings
query = "What are the eligibility criteria?"
embeddings = model.encode(query)  # 384-dimensional vectors
similar_docs = faiss_index.search(embeddings, top_k=5)
2. Transformer-based QA
python# BERT-based question answering
result = qa_pipeline(
    question="What is the deadline?",
    context=document_context
)
3. Neural Text Classification
python# Zero-shot classification using BART
labels = ["RFP", "Technical Specs", "Financial Terms"]
classification = classifier(document_text, labels)
