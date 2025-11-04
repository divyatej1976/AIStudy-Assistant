# PDF AI Assistant (RAG)

This is a Streamlit-based Retrieval-Augmented Generation (RAG) PDF study assistant.

What it does
- Upload PDFs, extract text, split into chunks.
- Create embeddings (SentenceTransformer) and store in a FAISS index.
- On user question: retrieve relevant chunks from FAISS, then call Gemini (via google-generative-ai) to produce grounded answers. Falls back to AI-only when no relevant PDF text is found.

Quick start (Windows PowerShell)
1. Create a virtual environment and activate it:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

Note: `faiss-cpu` can be tricky to install on Windows. If `pip install faiss-cpu` fails, consider using a compatible wheel or run inside WSL / Linux environment.

3. Add your Gemini API key to a `key.env` file at the project root:

```
GEMINI_API_KEY=your_api_key_here
```

4. Run the app:

```powershell
streamlit run "app.py"
```

Notes & caveats
- If `sentence-transformers` or `faiss` are not available, PDF semantic search will be disabled and the app will fall back to LLM-only answers.
- The included `requirements.txt` is a starting point. Pin versions as needed for reproducibility.

If you'd like, I can also:
- Add a small unit test for the chunking function.
- Attempt to run the app here to validate environment (may fail if packages are missing).
