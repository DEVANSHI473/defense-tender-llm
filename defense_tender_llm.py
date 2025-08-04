import streamlit as st
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    AutoModel,
    pipeline
)
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import PyPDF2
import docx
import tempfile
import zipfile
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

class DefenseTenderLLM:
    def __init__(self):
        """Initialize the Defense Tender LLM System"""
        st.info("🤖 Initializing Large Language Models...")
        self.documents = []
        self.embeddings = None
        self.faiss_index = None
        self.setup_llm_models()
        
    def setup_llm_models(self):
        """Setup transformer-based language models"""
        try:
            with st.spinner("🧠 Loading transformer models... (This may take 2-3 minutes on first run)"):
                # 1. Question Answering LLM (BERT-based)
                self.qa_model_name = "distilbert-base-cased-distilled-squad"
                self.qa_tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)
                self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.qa_model_name)
                
                # Create QA pipeline
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model=self.qa_model,
                    tokenizer=self.qa_tokenizer,
                    return_tensors=True
                )
                
                # 2. Semantic Search LLM (Sentence Transformers)
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # 3. Text Generation LLM (GPT-2 based)
                self.text_generator = pipeline(
                    "text-generation",
                    model="distilgpt2",
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=50256
                )
                
                # 4. Classification LLM for document analysis
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
                
            st.success("✅ All LLM models loaded successfully!")
            self.display_model_info()
            
        except Exception as e:
            st.error(f"❌ Error loading LLM models: {e}")
            st.info("💡 Models will be downloaded automatically on first run")
            raise e
    
    def display_model_info(self):
        """Display information about loaded models"""
        with st.expander("🤖 Loaded LLM Models Information"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Question Answering LLM:**
                - Model: DistilBERT (66M parameters)
                - Architecture: Transformer encoder
                - Capability: Extractive QA
                
                **Semantic Search LLM:**
                - Model: MiniLM (22M parameters)
                - Architecture: Sentence transformer
                - Capability: Dense embeddings
                """)
            
            with col2:
                st.markdown("""
                **Text Generation LLM:**
                - Model: DistilGPT2 (82M parameters)
                - Architecture: Transformer decoder
                - Capability: Text generation
                
                **Classification LLM:**
                - Model: BART-Large (400M parameters)
                - Architecture: Encoder-decoder
                - Capability: Zero-shot classification
                """)
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except:
                        continue
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells])
                    if row_text.strip():
                        text += row_text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            return ""
        except Exception as e:
            st.error(f"Error reading TXT: {str(e)}")
            return ""
    
    def intelligent_chunking(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Intelligent text chunking optimized for transformer models
        Using 512 tokens which is optimal for BERT-based models
        """
        if not text or not text.strip():
            return []
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding sentence exceeds chunk size
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            # Use tokenizer to count actual tokens (more accurate for transformers)
            tokens = self.qa_tokenizer.encode(test_chunk, add_special_tokens=True)
            
            if len(tokens) <= chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk.split()) > 10]
        
        return chunks
    
    def process_documents(self, uploaded_files) -> bool:
        """Process uploaded documents with LLM optimization"""
        try:
            self.documents = []
            
            if not uploaded_files:
                st.error("No files uploaded!")
                return False
            
            total_files = len(uploaded_files)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"🤖 LLM Processing: {uploaded_file.name} ({idx+1}/{total_files})")
                    
                    try:
                        # Save uploaded file
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Extract text based on file type
                        text = ""
                        if uploaded_file.name.lower().endswith('.pdf'):
                            text = self.extract_text_from_pdf(file_path)
                        elif uploaded_file.name.lower().endswith('.docx'):
                            text = self.extract_text_from_docx(file_path)
                        elif uploaded_file.name.lower().endswith('.txt'):
                            text = self.extract_text_from_txt(file_path)
                        
                        if text and text.strip():
                            # Intelligent chunking optimized for transformers
                            chunks = self.intelligent_chunking(text)
                            
                            if chunks:
                                for chunk_idx, chunk in enumerate(chunks):
                                    self.documents.append({
                                        'text': chunk,
                                        'source': uploaded_file.name,
                                        'chunk_id': len(self.documents),
                                        'chunk_index': chunk_idx,
                                        'total_chunks': len(chunks)
                                    })
                                
                                st.success(f"✅ {uploaded_file.name}: {len(chunks)} chunks processed with LLM")
                            else:
                                st.warning(f"⚠️ No meaningful content found in {uploaded_file.name}")
                        else:
                            st.warning(f"⚠️ Could not extract text from {uploaded_file.name}")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                        continue
                    
                    progress_bar.progress((idx + 1) / total_files)
            
            if not self.documents:
                st.error("❌ No text content found in any uploaded files!")
                return False
            
            # Create embeddings after processing all documents
            self.create_semantic_embeddings()
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ LLM processed {len(self.documents)} chunks from {total_files} files")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error in LLM document processing: {str(e)}")
            return False
    
    def create_semantic_embeddings(self):
        """Create semantic embeddings using sentence transformer LLM"""
        if not self.documents:
            return
        
        texts = [doc['text'] for doc in self.documents]
        
        with st.spinner("🧠 Creating semantic embeddings with LLM..."):
            # Generate embeddings using sentence transformer
            self.embeddings = self.embedding_model.encode(
                texts,
                convert_to_tensor=False,
                show_progress_bar=True,
                batch_size=32  # Optimize batch size for performance
            )
            
            # Create FAISS index for fast vector similarity search
            dimension = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings.astype('float32'))
            self.faiss_index.add(self.embeddings.astype('float32'))
        
        st.success(f"✅ Created {dimension}D semantic embeddings for {len(self.documents)} chunks")
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform semantic search using transformer embeddings"""
        if self.faiss_index is None or not query.strip():
            return []
        
        # Encode query using the same LLM
        query_embedding = self.embedding_model.encode([query.strip()])
        faiss.normalize_L2(query_embedding.astype('float32'))
        
        # Search for similar chunks
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents) and score > 0.1:  # Threshold for relevance
                results.append({
                    'text': self.documents[idx]['text'],
                    'source': self.documents[idx]['source'],
                    'score': float(score),
                    'chunk_id': self.documents[idx]['chunk_id'],
                    'method': 'Semantic Search (LLM)'
                })
        
        return results
    
    def llm_question_answering(self, question: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Use transformer LLM for question answering"""
        if not context_chunks:
            return {
                'answer': "No relevant information found in the documents for this question.",
                'confidence': 0.0,
                'method': 'LLM-QA (No Context)',
                'model': 'DistilBERT'
            }
        
        try:
            # Combine top chunks as context (optimized for BERT input length)
            context_texts = []
            total_length = 0
            max_context_length = 400  # Leave room for question and special tokens
            
            for chunk in context_chunks:
                chunk_tokens = len(self.qa_tokenizer.encode(chunk['text']))
                if total_length + chunk_tokens <= max_context_length:
                    context_texts.append(chunk['text'])
                    total_length += chunk_tokens
                else:
                    break
            
            context = " ".join(context_texts)
            
            # Use BERT-based QA model
            result = self.qa_pipeline(
                question=question,
                context=context
            )
            
            return {
                'answer': result['answer'],
                'confidence': result['score'],
                'method': 'BERT-QA (Transformer)',
                'model': 'DistilBERT',
                'context_used': len(context_texts)
            }
            
        except Exception as e:
            st.error(f"LLM QA Error: {e}")
            return {
                'answer': f"Error in LLM processing: {str(e)}",
                'confidence': 0.0,
                'method': 'Error',
                'model': 'DistilBERT'
            }
    
    def generate_summary(self, chunks: List[Dict], max_length: int = 150) -> str:
        """Generate document summary using LLM"""
        if not chunks:
            return "No content available to summarize."
        
        try:
            # Prepare content for summarization
            content = " ".join([chunk['text'][:200] for chunk in chunks[:3]])
            prompt = f"Summarize this defense tender document: {content[:500]}"
            
            # Generate summary using GPT-2 based model
            result = self.text_generator(
                prompt,
                max_length=len(prompt.split()) + 50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            
            generated_text = result[0]['generated_text']
            summary = generated_text.replace(prompt, "").strip()
            
            return summary if summary else "Unable to generate summary with current LLM."
            
        except Exception as e:
            return f"Summary generation error: {str(e)}"
    
    def classify_document_type(self, text: str) -> Dict[str, Any]:
        """Classify document type using zero-shot classification LLM"""
        try:
            candidate_labels = [
                "Request for Proposal (RFP)",
                "Tender Notice",
                "Technical Specifications",
                "Financial Proposal",
                "Eligibility Criteria",
                "Terms and Conditions",
                "Contract Document"
            ]
            
            # Use first 500 characters for classification
            sample_text = text[:500]
            
            result = self.classifier(sample_text, candidate_labels)
            
            return {
                'document_type': result['labels'][0],
                'confidence': result['scores'][0],
                'all_scores': dict(zip(result['labels'], result['scores']))
            }
            
        except Exception as e:
            return {
                'document_type': 'Unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Main LLM-powered question answering pipeline"""
        if not self.documents:
            return {
                'answer': "No documents have been processed yet. Please upload documents first.",
                'confidence': 0.0,
                'sources': [],
                'method': 'No Documents'
            }
        
        if not question or not question.strip():
            return {
                'answer': "Please enter a valid question.",
                'confidence': 0.0,
                'sources': [],
                'method': 'Invalid Question'
            }
        
        try:
            start_time = time.time()
            
            # Step 1: Semantic search using sentence transformer
            relevant_chunks = self.semantic_search(question.strip(), top_k=5)
            
            if not relevant_chunks:
                return {
                    'answer': "No relevant information found. Try rephrasing your question or check if the information exists in the uploaded documents.",
                    'confidence': 0.0,
                    'sources': [],
                    'method': 'No Relevant Chunks'
                }
            
            # Step 2: Use transformer LLM for question answering
            llm_result = self.llm_question_answering(question.strip(), relevant_chunks)
            
            # Step 3: Generate summary if needed
            if llm_result['confidence'] < 0.3:
                summary = self.generate_summary(relevant_chunks[:2])
                llm_result['answer'] += f"\n\nAdditional context: {summary}"
            
            processing_time = time.time() - start_time
            
            return {
                'answer': llm_result['answer'],
                'confidence': llm_result['confidence'],
                'sources': list(set([chunk['source'] for chunk in relevant_chunks[:3]])),
                'relevant_chunks': relevant_chunks[:3],
                'method': llm_result['method'],
                'model': llm_result['model'],
                'processing_time': processing_time,
                'context_chunks_used': llm_result.get('context_used', 0)
            }
            
        except Exception as e:
            return {
                'answer': f"Error in LLM processing: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'method': 'Error'
            }
    
    def get_llm_stats(self) -> Dict[str, Any]:
        """Get LLM system statistics"""
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'qa_model': self.qa_model_name,
            'embedding_model': 'all-MiniLM-L6-v2',
            'total_parameters': '570M+',  # Approximate total across all models
            'semantic_search_enabled': self.faiss_index is not None,
            'models_loaded': 4
        }

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Defense Tender LLM Analyzer",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Defense Tender LLM Analyzer")
    st.markdown("**AI-Powered Document Analysis using Large Language Models (570M+ Parameters)**")
    
    # LLM Information Banner
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🧠 LLM Models", "4 Transformers")
    with col2:
        st.metric("🔢 Parameters", "570M+")
    with col3:
        st.metric("🏗️ Architecture", "BERT + GPT")
    with col4:
        st.metric("🔍 Search Type", "Semantic")
    
    # Initialize LLM system
    if 'llm_analyzer' not in st.session_state:
        try:
            st.session_state.llm_analyzer = DefenseTenderLLM()
            st.session_state.documents_processed = False
        except Exception as e:
            st.error(f"Failed to initialize LLM system: {e}")
            st.stop()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload tender documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Supported: PDF, DOCX, TXT files"
        )
        
        if uploaded_files:
            st.info(f"📄 {len(uploaded_files)} file(s) selected")
            
            if st.button("🤖 Process with LLM", type="primary"):
                start_time = time.time()
                success = st.session_state.llm_analyzer.process_documents(uploaded_files)
                processing_time = time.time() - start_time
                
                st.session_state.documents_processed = success
                
                if success:
                    st.success(f"✅ LLM processing completed in {processing_time:.1f}s!")
                    st.balloons()
                else:
                    st.error("❌ LLM processing failed.")
        
        # LLM Statistics
        if st.session_state.documents_processed:
            st.subheader("🤖 LLM Statistics")
            stats = st.session_state.llm_analyzer.get_llm_stats()
            
            st.metric("Document Chunks", stats['total_documents'])
            st.metric("Embedding Dimensions", stats['embedding_dimension'])
            st.metric("Models Loaded", stats['models_loaded'])
            
            with st.expander("🔍 LLM Model Details"):
                st.markdown(f"""
                **Question Answering:**
                - Model: {stats['qa_model']}
                - Type: BERT Transformer
                - Parameters: 66M
                
                **Semantic Search:**
                - Model: {stats['embedding_model']}
                - Type: Sentence Transformer
                - Parameters: 22M
                
                **Text Generation:**
                - Model: DistilGPT2
                - Type: GPT Transformer
                - Parameters: 82M
                
                **Classification:**
                - Model: BART-Large
                - Type: Encoder-Decoder
                - Parameters: 400M
                """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🤖 LLM-Powered Question Answering")
        
        if not st.session_state.documents_processed:
            st.info("👈 Please upload and process documents first using the LLM system.")
            
            with st.expander("💡 What makes this an LLM system?"):
                st.markdown("""
                **This system uses multiple Large Language Models:**
                
                🧠 **Transformer Architecture:**
                - Multi-head attention mechanisms
                - Positional encoding
                - Layer normalization
                - Feed-forward networks
                
                🤖 **Neural Language Models:**
                - **BERT**: 66M parameter encoder for understanding
                - **Sentence-BERT**: 22M parameter model for semantic similarity
                - **GPT-2**: 82M parameter decoder for text generation
                - **BART**: 400M parameter encoder-decoder for classification
                
                🔍 **Advanced NLP Capabilities:**
                - Semantic understanding (not just keyword matching)
                - Context-aware question answering
                - Dense vector representations
                - Zero-shot classification
                
                📊 **Total System:**
                - **570M+ parameters** across all models
                - **Transformer-based** architecture throughout
                - **Pre-trained** on massive text corpora
                - **Fine-tuned** for question answering tasks
                """)
        else:
            # Pre-defined questions optimized for LLM
            st.subheader("🎯 Quick LLM Questions")
            example_questions = [
                "What are the main eligibility requirements for bidders?",
                "When is the submission deadline for this tender?",
                "What technical specifications are required?",
                "What is the estimated budget or contract value?",
                "How will the proposals be evaluated?",
                "What documents must be submitted with the bid?",
                "What are the key delivery timelines?",
                "Are there any security clearance requirements?"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(example_questions):
                with cols[i % 2]:
                    if st.button(question, key=f"llm_q_{i}"):
                        st.session_state.selected_question = question
                        st.rerun()
            
            # Custom question input
            st.subheader("✍️ Ask Your LLM Question")
            default_question = st.session_state.get('selected_question', '')
            custom_question = st.text_area(
                "Your Question:",
                value=default_question,
                height=100,
                placeholder="Ask anything about the tender documents using natural language...",
                key="llm_question_input"
            )
            
            if st.button("🤖 Get LLM Answer", type="primary") and custom_question.strip():
                start_time = time.time()
                
                with st.spinner("🤖 LLM is processing your question..."):
                    result = st.session_state.llm_analyzer.answer_question(custom_question.strip())
                
                answer_time = time.time() - start_time
                
                # Display LLM answer
                st.subheader("🤖 LLM Answer")
                st.write(result['answer'])
                
                # LLM metrics
                col_conf, col_time, col_method, col_model = st.columns(4)
                
                with col_conf:
                    confidence_color = "green" if result['confidence'] > 0.7 else "orange" if result['confidence'] > 0.4 else "red"
                    st.markdown(f"**Confidence:** :{confidence_color}[{result['confidence']:.1%}]")
                
                with col_time:
                    st.markdown(f"**Response Time:** {answer_time:.2f}s")
                
                with col_method:
                    st.markdown(f"**Method:** {result.get('method', 'LLM')}")
                
                with col_model:
                    st.markdown(f"**Model:** {result.get('model', 'Transformer')}")
                
                # Sources
                if result.get('sources'):
                    st.subheader("📚 Source Documents")
                    for source in result['sources']:
                        st.info(f"📄 {source}")
                
                # Semantic search results
                if 'relevant_chunks' in result and result['relevant_chunks']:
                    with st.expander(f"🔍 Semantic Search Results (LLM Embeddings)"):
                        for i, chunk in enumerate(result['relevant_chunks'], 1):
                            st.markdown(f"**Chunk {i}** (Similarity Score: {chunk['score']:.3f})")
                            st.markdown(f"*Source: {chunk['source']} | Method: {chunk['method']}*")
                            st.markdown(chunk['text'][:400] + "..." if len(chunk['text']) > 400 else chunk['text'])
                            st.markdown("---")
                
                # Clear selected question
                if 'selected_question' in st.session_state:
                    del st.session_state.selected_question
    
    with col2:
        st.header("🤖 LLM System Info")
        
        # LLM Status
        with st.container():
            st.subheader("🚀 LLM Status")
            
            if st.session_state.get('llm_analyzer'):
                st.success("🟢 LLM Models Loaded")
                if st.session_state.documents_processed:
                    stats = st.session_state.llm_analyzer.get_llm_stats()
                    st.success(f"🟢 {stats['total_documents']} chunks ready")
                    if stats['semantic_search_enabled']:
                        st.success("🟢 Semantic search active")
                else:
                    st.warning("🟡 No documents processed")
            else:
                st.error("🔴 LLM not initialized")
        
        # Model Architecture
        st.subheader("🏗️ LLM Architecture")
        
        if st.session_state.get('llm_analyzer'):
            with st.expander("🤖 Transformer Models"):
                st.markdown("""
                **Active LLM Components:**
                
                🧠 **Question Answering:**
                - DistilBERT (66M params)
                - 6-layer transformer encoder
                - Multi-head attention (12 heads)
                
                🔍 **Semantic Search:**
                - MiniLM (22M params)
                - Sentence embeddings (384D)
                - Cosine similarity matching
                
                📝 **Text Generation:**
                - DistilGPT2 (82M params)
                - 6-layer transformer decoder
                - Autoregressive generation
                
                🏷️ **Classification:**
                - BART-Large (400M params)
                - Encoder-decoder architecture
                - Zero-shot classification
                """)
        
        # Performance monitoring
        if st.session_state.documents_processed:
            st.subheader("📊 Performance")
            
            stats = st.session_state.llm_analyzer.get_llm_stats()
            st.metric("Embedding Dimensions", stats['embedding_dimension'])
            st.metric("Total Parameters", stats['total_parameters'])
            
            if st.button("🧹 Clear LLM Cache"):
                # Clear any caches
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                st.success("Cache cleared!")
        
        # Help section
        st.subheader("❓ LLM Help")
        
        with st.expander("💡 How the LLM works"):
            st.markdown("""
            **LLM Processing Pipeline:**
            
            1. **Document Processing**: Text extraction and chunking
            2. **Semantic Encoding**: Convert text to 384D vectors using sentence transformers
            3. **Question Analysis**: Process query using BERT tokenizer
            4. **Similarity Search**: Find relevant chunks using FAISS vector search
            5. **Answer Generation**: Use DistilBERT QA model for final answer
            6. **Confidence Scoring**: Neural network confidence estimation
            
            **Why it's a proper LLM:**
            
            ✅ **Neural Networks**: All models use deep neural networks
            ✅ **Transformers**: BERT, GPT, BART architectures
            ✅ **Attention Mechanisms**: Multi-head self-attention
            ✅ **Pre-training**: Models trained on billions of tokens
            ✅ **Semantic Understanding**: Beyond keyword matching
            ✅ **Context Awareness**: Understands document context
            """)

if __name__ == "__main__":
    main()