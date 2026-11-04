from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

import faiss
import numpy as np
from docx import Document as DocxDocument
from groq import Groq
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STORE_DIR = Path(__file__).parent / "saved_stores"
STORE_DIR.mkdir(exist_ok=True)


@dataclass
class Document:
    text: str
    source: str


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


@dataclass
class VectorStore:
    model: SentenceTransformer
    index: faiss.Index
    bm25: BM25Okapi
    embeddings: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# Extraction Logic
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    return "\n".join(
        " ".join(line.split())
        for line in text.splitlines()
        if line.strip()
    ).strip()


def extract_pdf(uploaded_file) -> list[Document]:
    uploaded_file.seek(0)
    reader = PdfReader(BytesIO(uploaded_file.read()))
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            documents.append(
                Document(text=text, source=f"{uploaded_file.name}, page {page_number}")
            )
    return documents


def extract_txt(uploaded_file) -> list[Document]:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    text = clean_text(text)
    return [Document(text=text, source=uploaded_file.name)] if text else []


def extract_docx(uploaded_file) -> list[Document]:
    uploaded_file.seek(0)
    doc = DocxDocument(BytesIO(uploaded_file.read()))
    text = clean_text("\n".join(p.text for p in doc.paragraphs))
    return [Document(text=text, source=uploaded_file.name)] if text else []


def extract_md(uploaded_file) -> list[Document]:
    return extract_txt(uploaded_file)


EXTRACTORS = {
    "pdf": extract_pdf,
    "txt": extract_txt,
    "docx": extract_docx,
    "md": extract_md,
}


def extract_documents(uploaded_files: Iterable) -> list[Document]:
    documents = []
    for uploaded_file in uploaded_files:
        extension = uploaded_file.name.rsplit(".", 1)[-1].lower()
        extractor = EXTRACTORS.get(extension)
        if extractor is None:
            raise ValueError(f"Unsupported file format: {uploaded_file.name}")
        documents.extend(extractor(uploaded_file))

    if not documents:
        raise ValueError("No readable text found. Scanned or image-only files require OCR.")
    return documents


# ---------------------------------------------------------------------------
# Chunking Logic
# ---------------------------------------------------------------------------

def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size.")

    chunks = []
    start = 0

    while start < len(text):
        proposed_end = min(start + chunk_size, len(text))
        end = proposed_end

        if proposed_end < len(text):
            search_start = start + chunk_size // 2
            boundaries = [
                text.rfind("\n", search_start, proposed_end),
                text.rfind(". ", search_start, proposed_end),
                text.rfind(" ", search_start, proposed_end),
            ]
            boundary = max(boundaries)
            if boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def create_chunks(
    documents: Sequence[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks = []
    chunk_id = 1

    for document in documents:
        for text in split_text(document.text, chunk_size, chunk_overlap):
            chunks.append(Chunk(text=text, source=document.source, chunk_id=chunk_id))
            chunk_id += 1

    if not chunks:
        raise ValueError("No valid text chunks could be created.")
    return chunks


# ---------------------------------------------------------------------------
# Vector Store Setup
# ---------------------------------------------------------------------------

def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(DEFAULT_EMBEDDING_MODEL)


def build_vector_store(
    documents: Sequence[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    model: SentenceTransformer | None = None,
) -> tuple[VectorStore, list[Chunk]]:
    chunks = create_chunks(documents, chunk_size, chunk_overlap)
    if model is None:
        model = get_embedding_model()

    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    return VectorStore(model=model, index=index, bm25=bm25, embeddings=embeddings), chunks


def save_vector_store(name: str, vector_store: VectorStore, chunks: list[Chunk]) -> None:
    safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")) or "kb"
    path = STORE_DIR / safe_name
    path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(vector_store.index, str(path / "index.faiss"))
    with open(path / "meta.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": vector_store.embeddings}, f)


def load_vector_store(
    name: str, model: SentenceTransformer | None = None
) -> tuple[VectorStore, list[Chunk]]:
    path = STORE_DIR / name
    index = faiss.read_index(str(path / "index.faiss"))
    with open(path / "meta.pkl", "rb") as f:
        meta = pickle.load(f)

    if model is None:
        model = get_embedding_model()

    tokenized_corpus = [chunk.text.lower().split() for chunk in meta["chunks"]]
    bm25 = BM25Okapi(tokenized_corpus)

    return (
        VectorStore(model=model, index=index, bm25=bm25, embeddings=meta["embeddings"]),
        meta["chunks"],
    )


def list_saved_stores() -> list[str]:
    if not STORE_DIR.exists():
        return []
    return sorted(p.name for p in STORE_DIR.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Search and Retrieval
# ---------------------------------------------------------------------------

def _dense_search(query: str, vector_store: VectorStore, k: int) -> dict[int, float]:
    query_vector = vector_store.model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    scores, indices = vector_store.index.search(query_vector, k)
    return {int(i): float(s) for s, i in zip(scores[0], indices[0]) if i >= 0}


def _sparse_search(query: str, vector_store: VectorStore, k: int) -> dict[int, float]:
    tokenized_query = query.lower().split()
    scores = vector_store.bm25.get_scores(tokenized_query)
    top_idx = np.argsort(scores)[::-1][:k]
    return {int(i): float(scores[i]) for i in top_idx}


def retrieve_chunks(
    question: str,
    vector_store: VectorStore,
    chunks: Sequence[Chunk],
    top_k: int = 4,
    alpha: float = 0.5,
) -> list[dict]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    k = min(max(top_k, 1) * 3, len(chunks))
    dense_scores = _dense_search(question, vector_store, k)
    sparse_scores = _sparse_search(question, vector_store, k)

    def to_ranks(score_map: dict[int, float]) -> dict[int, int]:
        ordered = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return {idx: rank for rank, (idx, _) in enumerate(ordered, start=1)}

    dense_ranks = to_ranks(dense_scores)
    sparse_ranks = to_ranks(sparse_scores)

    rrf_k = 60
    fused: dict[int, float] = {}
    for idx in set(dense_ranks) | set(sparse_ranks):
        dense_component = alpha * (1 / (rrf_k + dense_ranks.get(idx, rrf_k * 2)))
        sparse_component = (1 - alpha) * (1 / (rrf_k + sparse_ranks.get(idx, rrf_k * 2)))
        fused[idx] = dense_component + sparse_component

    top_positions = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for position, fused_score in top_positions:
        chunk = chunks[position]
        results.append(
            {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "score": fused_score,
                "dense_score": dense_scores.get(position, 0.0),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Context Query Rewriting & Generation
# ---------------------------------------------------------------------------

def rewrite_followup_question(
    question: str,
    history: Sequence[dict],
    api_key: str,
    model_name: str,
) -> str:
    if not history or not api_key:
        return question

    recent = history[-4:]
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
    client = Groq(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's latest question into a fully "
                        "standalone question using conversation history. "
                        "Return ONLY the rewritten question, nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": f"CONVERSATION:\n{convo}\n\nLATEST QUESTION:\n{question}",
                },
            ],
            temperature=0.0,
            max_completion_tokens=100,
        )
        rewritten = completion.choices[0].message.content
        return rewritten.strip() if rewritten else question
    except Exception:
        return question


def generate_answer(
    question: str,
    retrieved_chunks: Sequence[dict],
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
) -> tuple[str, float]:
    if not api_key:
        raise ValueError("Please provide a Groq API key in the sidebar.")

    if not retrieved_chunks:
        return "I could not find that information in the uploaded documents.", 0.0

    context = "\n\n---\n\n".join(
        f"[{i + 1}] Source: {item['source']} | Chunk: {item['chunk_id']}\n{item['text']}"
        for i, item in enumerate(retrieved_chunks)
    )

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer strictly using the document context provided. Cite "
                    "sources inline using numbers in brackets such as [1], [2]. "
                    "If the information is not present, respond with: 'I could "
                    "not find that information in the uploaded documents.'"
                ),
            },
            {
                "role": "user",
                "content": f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}",
            },
        ],
        temperature=0.1,
        max_completion_tokens=700,
    )

    answer = completion.choices[0].message.content
    if not answer:
        raise RuntimeError("Model did not return a valid answer.")

    avg_dense_score = sum(c["dense_score"] for c in retrieved_chunks) / len(retrieved_chunks)
    confidence = max(0.0, min(1.0, avg_dense_score))

    return answer.strip(), confidence