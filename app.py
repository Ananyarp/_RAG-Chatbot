from __future__ import annotations

import streamlit as st
from sentence_transformers import SentenceTransformer

# Import logic components from rag_engine
from rag_engine import (
    DEFAULT_EMBEDDING_MODEL,
    build_vector_store,
    extract_documents,
    generate_answer,
    list_saved_stores,
    load_vector_store,
    retrieve_chunks,
    rewrite_followup_question,
    save_vector_store,
)


@st.cache_resource
def load_cached_embedding_model():
    return SentenceTransformer(DEFAULT_EMBEDDING_MODEL)


def apply_custom_theme():
    st.markdown(
        """
        <style>
        /* Force App Background */
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #f8fafc !important;
        }

        /* FORCE Chat Message Cards to have high contrast background */
        [data-testid="stChatMessage"] {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
            padding: 12px 18px !important;
            margin-bottom: 10px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
        }

        /* FORCE all text inside chat messages to be dark slate/black */
        [data-testid="stChatMessage"] *, 
        [data-testid="stChatMessage"] p, 
        [data-testid="stChatMessage"] span, 
        [data-testid="stChatMessage"] li, 
        [data-testid="stChatMessage"] h1, 
        [data-testid="stChatMessage"] h2, 
        [data-testid="stChatMessage"] h3, 
        [data-testid="stChatMessage"] div {
            color: #0f172a !important;
        }

        /* Header Title Styling */
        .main-header {
            color: #4f46e5 !important;
            font-weight: 800;
            font-size: 2.2rem;
            margin-bottom: 0.2rem;
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: #ffffff !important;
            border-radius: 12px;
            padding: 1.25rem;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
            color: #0f172a !important;
        }
        .glass-card * {
            color: #0f172a !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a !important;
        }
        section[data-testid="stSidebar"] * {
            color: #f1f5f9 !important;
        }

        /* Metric Badge Styling */
        .badge-confidence {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 600;
            text-align: center;
        }
        .conf-high { background-color: #d1fae5 !important; color: #065f46 !important; }
        .conf-med { background-color: #fef3c7 !important; color: #92400e !important; }
        .conf-low { background-color: #fee2e2 !important; color: #991b1b !important; }

        /* Custom Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #6366f1 0%, #4f46e5 100%) !important;
            color: #ffffff !important;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.set_page_config(page_title="Hybrid RAG Search", page_icon="✨", layout="wide")
    apply_custom_theme()

    embedding_model = load_cached_embedding_model()

    # Header Section
    st.markdown('<h1 class="main-header">✨ Hybrid RAG Intelligence Platform</h1>', unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; margin-bottom: 1.5rem;'>Combine dense semantic search with sparse BM25 keyword precision powered by Groq LLMs.</p>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Sidebar Setup
    # ---------------------------------------------------------------------------
    with st.sidebar:
        st.title("⚙️ Engine Control")
        
        api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key to enable LLM generation")
        model_name = st.selectbox(
            "Language Model",
            ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
        )

        st.divider()
        st.subheader("🎛️ Search Parameters")
        alpha = st.slider(
            "Dense / Sparse Weight (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="1.0 = Pure Semantic Search, 0.0 = Pure BM25 Keyword Search, 0.5 = Balanced Hybrid",
        )
        top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=10, value=4)

        st.divider()
        st.subheader("📚 Saved Knowledge Stores")
        saved_stores = list_saved_stores()
        if saved_stores:
            selected_store = st.selectbox("Load Saved Store", ["None"] + saved_stores)
            if selected_store != "None" and st.button("Load Knowledge Base"):
                with st.spinner("Loading store..."):
                    vector_store, chunks = load_vector_store(selected_store, model=embedding_model)
                    st.session_state["vector_store"] = vector_store
                    st.session_state["chunks"] = chunks
                    st.success(f"Loaded '{selected_store}'!")

        if st.session_state.get("history") and st.button("🗑️ Clear Chat History"):
            st.session_state["history"] = []
            st.rerun()

    # ---------------------------------------------------------------------------
    # Main Content Area
    # ---------------------------------------------------------------------------
    
    # Metrics Dashboard
    if "chunks" in st.session_state:
        chunks_count = len(st.session_state["chunks"])
        total_words = sum(len(c.text.split()) for c in st.session_state["chunks"])
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Indexed Chunks", f"{chunks_count:,}")
        m2.metric("Total Corpus Words", f"{total_words:,}")
        m3.metric("Selected Model", model_name.split("-")[0].upper())
        m4.metric("Hybrid Alpha (α)", f"{alpha:.2f}")
        st.divider()

    # File Upload Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("1. Index New Documents")
    
    uploaded_files = st.file_uploader(
        "Upload document files to convert into vector embeddings",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        store_name = st.text_input("Save Store Name (Optional)", value="my_knowledge_base")

    if uploaded_files and st.button("🚀 Process & Build Index"):
        with st.spinner("Parsing text and building FAISS + BM25 index..."):
            try:
                documents = extract_documents(uploaded_files)
                vector_store, chunks = build_vector_store(documents, model=embedding_model)
                st.session_state["vector_store"] = vector_store
                st.session_state["chunks"] = chunks

                if store_name:
                    save_vector_store(store_name, vector_store, chunks)
                    st.success(f"Indexed {len(documents)} file(s) into {len(chunks)} chunks and saved as '{store_name}'!")
                else:
                    st.success(f"Indexed {len(documents)} file(s) into {len(chunks)} chunks!")
                st.rerun()
            except Exception as e:
                st.error(f"Error processing files: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat / Search Section
    st.subheader("2. Conversational Search & Q&A")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Display chat history across reruns
    for idx, msg in enumerate(st.session_state["history"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # If the stored assistant message has metadata (confidence & retrieved chunks), render them
            if msg["role"] == "assistant" and "confidence" in msg:
                confidence = msg["confidence"]
                retrieved = msg.get("retrieved", [])

                # Confidence Badge
                conf_class = "conf-high" if confidence > 0.65 else ("conf-med" if confidence > 0.45 else "conf-low")
                conf_label = "High" if confidence > 0.65 else ("Medium" if confidence > 0.45 else "Low")
                
                st.markdown(
                    f"""<div style="margin-top: 10px;">
                        <span class="badge-confidence {conf_class}">Retrieval Confidence: {conf_label} ({confidence:.1%})</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # Context Sources Expander
                if retrieved:
                    with st.expander("🔎 View Retrieved Document Context"):
                        tabs = st.tabs([f"Chunk #{r['chunk_id']}" for r in retrieved])
                        for tab_idx, (tab, r) in enumerate(zip(tabs, retrieved)):
                            with tab:
                                st.markdown(f"**Source:** `{r['source']}`")
                                st.markdown(f"**Dense Similarity Score:** `{r['dense_score']:.4f}`")
                                st.text_area(
                                    "Content",
                                    value=r["text"],
                                    height=150,
                                    disabled=True,
                                    key=f"hist_{idx}_{tab_idx}_{r['chunk_id']}"
                                )

    # User Query Input
    question = st.chat_input("Ask anything based on your indexed documents...")

    if question:
        if "vector_store" not in st.session_state or "chunks" not in st.session_state:
            st.warning("⚠️ Please upload and index documents or load a saved store first.")
            return

        # 1. Add user message to history
        st.session_state["history"].append({"role": "user", "content": question})

        with st.spinner("Synthesizing context and querying Groq..."):
            try:
                # 2. Rewrite followup question if chat history exists
                standalone_query = rewrite_followup_question(
                    question=question,
                    history=st.session_state["history"][:-1],
                    api_key=api_key,
                    model_name=model_name,
                )

                # 3. Retrieve top chunks
                retrieved = retrieve_chunks(
                    question=standalone_query,
                    vector_store=st.session_state["vector_store"],
                    chunks=st.session_state["chunks"],
                    top_k=top_k,
                    alpha=alpha,
                )

                # 4. Generate response
                answer, confidence = generate_answer(
                    question=standalone_query,
                    retrieved_chunks=retrieved,
                    api_key=api_key,
                    model_name=model_name,
                )

                # 5. Append full assistant response with metadata to history
                st.session_state["history"].append({
                    "role": "assistant",
                    "content": f"**Answer:**\n\n{answer}",
                    "confidence": confidence,
                    "retrieved": retrieved
                })

                # 6. Rerun page so history renders cleanly
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()