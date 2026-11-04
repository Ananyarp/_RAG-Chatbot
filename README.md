# Hybrid RAG Document Q&A

A Retrieval-Augmented Generation app built with Streamlit, FAISS, BM25, and Groq.

## What makes this different from the basic classroom version

1. **Multi-format ingestion** — PDF, TXT, DOCX, and Markdown (not just PDF/TXT).
2. **Hybrid retrieval** — FAISS dense (semantic) search is fused with BM25
   keyword search using reciprocal rank fusion. A sidebar slider lets you
   blend between pure semantic and pure keyword search, so exact terms
   (names, numbers, acronyms) are found even when embeddings alone would miss them.
3. **Conversational follow-up rewriting** — vague follow-ups like *"what about
   that?"* get rewritten into standalone questions using recent chat history
   before retrieval runs.
4. **Inline citations + confidence score** — answers cite sources as `[1]`,
   `[2]`, matching the retrieved chunk list, and a confidence bar reflects how
   strong the semantic match was.
5. **Persistence** — save a processed knowledge base to disk by name, and
   reload it later from the sidebar without re-uploading files or
   re-computing embeddings.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # then paste your Groq API key into .env
```

Get a free Groq API key at https://console.groq.com/keys

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Usage

1. Upload one or more PDF/TXT/DOCX/MD files.
2. (Optional) give the knowledge base a name to save it for later.
3. Click **Process Documents**.
4. Ask questions in the chat box — answers are grounded in your documents,
   with citations and a confidence indicator.
5. Reload a previously saved knowledge base anytime from the sidebar.

## Project structure

```
app.py            Streamlit UI
rag_engine.py      Loading, chunking, hybrid retrieval, generation, persistence
requirements.txt    Dependencies
.env.example        Template for GROQ_API_KEY
saved_stores/       Created automatically to hold saved knowledge bases
```
