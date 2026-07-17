import os
import warnings
import logging
import sys
import time

# Suppress warnings and reduce log noise
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import uuid
import shutil
from werkzeug.utils import secure_filename

# Optional imports (handled gracefully)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except Exception as e:
    print(f"Warning: SentenceTransformer not available: {e}")
    SENTENCE_TRANSFORMER_AVAILABLE = False

# Load API keys
load_dotenv(dotenv_path='key.env')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or st.secrets.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")

# Configure APIs (Gemini)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class PDFAssistant:
    def __init__(self, session_id):
        self.upload_folder = os.path.join("uploaded_pdfs", session_id)
        os.makedirs(self.upload_folder, exist_ok=True)
        # core state
        self.documents_content = ""
        self.has_documents = False
        self.document_chunks = []
        self.embeddings = None
        self.index = None
        # load embedding model if available
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # lightweight model good for RAG
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
                self.model = None
        else:
            self.model = None

    def add_documents(self, pdf_files):
        """Process uploaded documents and build embeddings+FAISS index (if model available)."""
        all_text = ""
        self.document_chunks = []

        # save uploaded files (avoid overwriting existing same-named files)
        for pdf_file in pdf_files:
            safe_name = secure_filename(pdf_file.name)
            save_path = os.path.join(self.upload_folder, safe_name)
            if not os.path.exists(save_path):
                try:
                    with open(save_path, "wb") as f:
                        f.write(pdf_file.getbuffer())
                except Exception as e:
                    st.warning(f"Could not save {safe_name}: {e}")

        # extract text and chunk
        for filename in os.listdir(self.upload_folder):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(self.upload_folder, filename)
                text = self._extract_pdf_text(file_path)
                if text:
                    all_text += f"\n\n=== From {filename} ===\n{text}"
                    chunks = self._create_chunks(text, filename)
                    self.document_chunks.extend(chunks)

        if all_text:
            self.documents_content = all_text
            self.has_documents = True

            # Build embeddings + FAISS index if model is available and we have chunks
            if self.document_chunks and self.model:
                try:
                    chunk_texts = [chunk['text'] for chunk in self.document_chunks]
                    # ensure numpy float32 for FAISS
                    embeddings = self.model.encode(chunk_texts, show_progress_bar=False, convert_to_numpy=True)
                    embeddings = np.array(embeddings, dtype='float32')
                    # normalize and build FAISS index
                    faiss.normalize_L2(embeddings)
                    dimension = embeddings.shape[1]
                    index = faiss.IndexFlatIP(dimension)
                    index.add(embeddings)

                    self.embeddings = embeddings
                    self.index = index
                except Exception as e:
                    st.warning(f"Warning: Could not create embeddings/index: {e}")
                    self.index = None
                    self.embeddings = None

            return True
        return False

    def _extract_pdf_text(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                # get_text() may return '' for some pages; combine
                text += page.get_text() + "\n"
            doc.close()
            return text
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""

    def _create_chunks(self, text, filename):
        chunks = []
        words = text.split()
        chunk_size = 300
        overlap = 50

        for i in range(0, max(1, len(words)), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            if len(chunk_text.strip()) > 100:
                chunks.append({
                    'text': chunk_text,
                    'source': filename,
                    'chunk_id': len(chunks)
                })
            if i + chunk_size >= len(words):
                break

        return chunks

    def search_documents(self, question, top_k=3):
        """Search documents using FAISS. Returns combined text and status."""
        if not self.has_documents:
            return None, "No documents uploaded"

        if not self.index or self.model is None or self.embeddings is None:
            return None, "Search unavailable (embedding model or index not ready)"

        try:
            query_embedding = self.model.encode([question], convert_to_numpy=True)
            query_embedding = np.array(query_embedding, dtype='float32')
            faiss.normalize_L2(query_embedding)

            k = min(top_k, len(self.document_chunks))
            if k <= 0:
                return None, "No document chunks available for search"

            distances, indices = self.index.search(query_embedding, k)

            relevant_chunks = []
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                # thresholding similarity score (cosine via inner product after normalization)
                if float(score) > 0.25:
                    chunk = self.document_chunks[int(idx)]
                    relevant_chunks.append({
                        'text': chunk['text'],
                        'source': chunk['source'],
                        'score': float(score)
                    })

            if relevant_chunks:
                combined_text = "\n\n".join([f"From {c['source']} (score={c['score']:.3f}):\n{c['text']}" for c in relevant_chunks])
                return combined_text, f"Found {len(relevant_chunks)} relevant sections"
            else:
                return None, "No relevant information found in PDFs"
        except Exception as e:
            return None, f"Error searching documents: {str(e)}"

    def answer_question(self, question, conversation_history=None):
        """Top-level QA: search PDFs first, then generate answer via Gemini (or AI fallback)."""
        # search PDFs
        pdf_content, pdf_status = self.search_documents(question)

        # generate response (PDF-grounded if pdf_content present)
        return self._generate_response(question, pdf_content or "", "PDF" if pdf_content else "AI", pdf_status, conversation_history)

    def _generate_response(self, question, context, source_type="AI", pdf_status="", conversation_history=None):
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')

            # conversation context snippet
            context_prompt = ""
            if conversation_history:
                recent_context = conversation_history[-3:]
                context_prompt = "\n\nPREVIOUS CONVERSATION:\n"
                for i, chat in enumerate(recent_context):
                    context_prompt += f"Q{i+1}: {chat.get('question','')}\nA{i+1}: {chat.get('answer','')[:200]}...\n"
                context_prompt += "\nCURRENT QUESTION:\n"

            if source_type == "PDF" and context:
                prompt = f"""You are a helpful assistant answering questions about the user's uploaded documents.

{context_prompt}

DOCUMENT CONTENT:
{context}

Question: {question}

Answer using the document content first, then add clarifications if needed. Be concise and clear."""
            else:
                prompt = f"""You are a helpful assistant answering questions.

{context_prompt}

Question: {question}

Answer clearly and helpfully."""
            response = model.generate_content(prompt)
            return {
                'answer': response.text,
                'source_type': source_type,
                'pdf_status': pdf_status,
                'method': 'Document Search' if source_type == 'PDF' else 'General Knowledge'
            }
        except Exception as e:
            # fallback to simple local knowledge if LLM call fails
            fallback_text = f"Sorry, the remote LLM failed: {e}."
            return {
                'answer': fallback_text,
                'source_type': 'AI',
                'pdf_status': pdf_status,
                'method': 'Fallback'
            }

# ---------- Streamlit app UI ----------
def main():
    st.set_page_config(page_title="Document Chat Assistant", layout="wide", initial_sidebar_state="collapsed")

    st.markdown("""
    <style>
    .stContainer > div { background-color: transparent !important; }
    div[data-testid="stMarkdownContainer"] { background-color: transparent !important; }
    </style>
    """, unsafe_allow_html=True)

    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # API key check
    if not GEMINI_API_KEY:
        st.error("❌ GEMINI_API_KEY not found. Add it to key.env or Streamlit secrets.")
        st.code('GEMINI_API_KEY=your_api_key_here', language='bash')
        st.stop()

    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"❌ API configuration error: {e}")
        st.stop()

    st.title("Document Chat Assistant")
    st.markdown("*Upload PDFs and ask questions — I will search your documents first, then use AI knowledge.*")
    st.success("✅ System loaded")

    # initialize assistant in session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'assistant' not in st.session_state:
        st.session_state.assistant = PDFAssistant(st.session_state.session_id)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if st.session_state.assistant.model is None and SENTENCE_TRANSFORMER_AVAILABLE:
        st.warning("⚠ Embedding model not loaded - PDF search may be limited.")
    elif not SENTENCE_TRANSFORMER_AVAILABLE:
        st.warning("⚠ SentenceTransformer not installed - PDF search disabled.")

    # Upload PDFs
    st.header("📚 Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            success = st.session_state.assistant.add_documents(uploaded_files)
            if success:
                st.success(f"✅ Processed PDFs. Found {len(st.session_state.assistant.document_chunks)} chunks.")
            else:
                st.error("❌ Could not process PDFs.")

    # Chat interface
    st.header("💬 Chat with your Document Assistant")

    question = st.text_area(
        "Type your message:",
        value=st.session_state.get('current_q',''),
        placeholder="Ask questions about your PDFs...",
        height=120
    )

    if st.button("🚀 Get Answer") and question.strip():
        with st.spinner("Searching PDFs and generating answer..."):
            response = st.session_state.assistant.answer_question(
                question,
                conversation_history=st.session_state.chat_history
            )

            # Display sources / status
            if st.session_state.assistant.has_documents and response.get('source_type') == 'PDF':
                st.success("✅ Answer source: Your PDFs")
                st.info("The answer below is based on content found in your uploaded PDFs.")
                pdf_content, _ = st.session_state.assistant.search_documents(question)
                if pdf_content:
                    with st.expander("📋 PDF content used for this answer"):
                        st.text(pdf_content[:2000] + ("..." if len(pdf_content) > 2000 else ""))
            else:
                if not st.session_state.assistant.has_documents:
                    st.info("📝 No PDFs uploaded.")
                st.info("Answer source: AI general knowledge")

            # show answer
            if response.get('source_type') == 'PDF':
                st.subheader("🎯 Response (from your documents)")
            else:
                st.subheader("🎯 Response (general knowledge)")

            st.write(response['answer'])

            # store in history
            st.session_state.chat_history.append({
                'question': question,
                'answer': response['answer'],
                'source_type': response.get('source_type', 'AI'),
                'method': response.get('method', ''),
                'timestamp': time.time()
            })
            # clear current_q placeholder
            st.session_state.current_q = ""

    # Chat history UI
    if st.session_state.chat_history:
        st.header("💬 Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-8:])):
            idx = len(st.session_state.chat_history) - 1 - i
            with st.expander(f"❓ Question: {chat['question'][:80]}"):
                st.write(f"*You:* {chat['question']}")
                st.write(f"*AI:* {chat['answer']}")
                st.caption(f"Source: {chat.get('source_type','AI')} | Method: {chat.get('method', '')}")

    st.write("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑 Clear History"):
            st.session_state.chat_history = []
            st.experimental_rerun()
    with col2:
        if st.button("🗑 Clear Documents"):
            if os.path.exists(st.session_state.assistant.upload_folder):
                shutil.rmtree(st.session_state.assistant.upload_folder, ignore_errors=True)
            os.makedirs(st.session_state.assistant.upload_folder, exist_ok=True)
            st.session_state.assistant.documents_content = ""
            st.session_state.assistant.has_documents = False
            st.session_state.assistant.document_chunks = []
            st.session_state.assistant.embeddings = None
            st.session_state.assistant.index = None
            st.experimental_rerun()

if __name__ == "__main__":
    main()
