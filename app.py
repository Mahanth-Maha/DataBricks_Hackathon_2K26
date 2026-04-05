import streamlit as st
import os
import json
import re
import faiss
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Nyaya Deepam — Indian Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark, premium legal aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0d0f14;
    --bg-secondary: #13161e;
    --bg-card: #1a1e28;
    --bg-input: #1f2433;
    --accent-gold: #c9a84c;
    --accent-gold-light: #e8c97a;
    --accent-blue: #4a7fa5;
    --text-primary: #e8e6df;
    --text-secondary: #9a9590;
    --text-muted: #5a5750;
    --border: #2a2e3d;
    --user-bubble: #1e3a5f;
    --bot-bubble: #1a1e28;
    --success: #3d7a5e;
    --radius: 12px;
}

/* Reset Streamlit defaults */
.stApp { background-color: var(--bg-primary) !important; }
.stApp > header { background: transparent !important; }
section[data-testid="stSidebar"] { 
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Main container */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Sidebar Styles ── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 1rem 1.5rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.sidebar-logo-icon {
    font-size: 2rem;
    line-height: 1;
}
.sidebar-logo-text h2 {
    font-family: 'Playfair Display', serif;
    color: var(--accent-gold);
    font-size: 1.1rem;
    margin: 0;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.sidebar-logo-text p {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-muted);
    font-size: 0.65rem;
    margin: 0;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.sidebar-section-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    padding: 0 1rem;
    margin-bottom: 0.5rem;
}
.sidebar-new-chat {
    margin: 0 0.5rem 1.5rem 0.5rem;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent-gold) 0%, #a07830 100%) !important;
    color: #0d0f14 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(201, 168, 76, 0.25) !important;
}
.history-item {
    padding: 0.6rem 1rem;
    margin: 0 0.25rem;
    border-radius: 8px;
    cursor: pointer;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: all 0.15s ease;
}
.history-item:hover {
    background: var(--bg-card);
    color: var(--text-primary);
}

/* ── Chat Area ── */
.chat-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: var(--bg-primary);
}
.chat-header {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem 2rem;
    border-bottom: 1px solid var(--border);
    background: var(--bg-secondary);
    position: sticky;
    top: 0;
    z-index: 10;
}
.chat-header-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--text-primary);
    font-weight: 600;
}
.chat-header-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    color: var(--text-muted);
    text-align: center;
    margin-top: 2px;
}
.mode-badge {
    display: inline-block;
    background: rgba(74, 127, 165, 0.15);
    border: 1px solid rgba(74, 127, 165, 0.3);
    color: var(--accent-blue);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}

/* ── Messages ── */
.messages-area {
    flex: 1;
    overflow-y: auto;
    padding: 2rem;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}
.message-row {
    display: flex;
    gap: 1rem;
    animation: fadeSlideIn 0.3s ease;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.message-row.user { flex-direction: row-reverse; }

.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.avatar.bot {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a8a 100%);
    border: 1px solid rgba(74, 127, 165, 0.3);
}
.avatar.user {
    background: linear-gradient(135deg, #3d2800 0%, #6b4a00 100%);
    border: 1px solid rgba(201, 168, 76, 0.3);
}

.bubble {
    max-width: 72%;
    padding: 1rem 1.25rem;
    border-radius: var(--radius);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    line-height: 1.65;
}
.bubble.user {
    background: var(--user-bubble);
    color: var(--text-primary);
    border: 1px solid rgba(74, 127, 165, 0.2);
    border-top-right-radius: 3px;
}
.bubble.bot {
    background: var(--bot-bubble);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-top-left-radius: 3px;
}
.bubble.bot .answer-text {
    margin-bottom: 0.75rem;
    color: var(--text-primary);
}
.citation {
    display: inline-block;
    background: rgba(201, 168, 76, 0.08);
    border: 1px solid rgba(201, 168, 76, 0.2);
    color: var(--accent-gold-light);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 5px;
    margin-top: 4px;
}
.retrieved-toggle {
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border);
}
.retrieved-toggle summary {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    color: var(--text-muted);
    cursor: pointer;
    user-select: none;
    letter-spacing: 0.5px;
}
.retrieved-toggle summary:hover { color: var(--text-secondary); }
.retrieved-chunk {
    margin-top: 0.5rem;
    padding: 0.6rem 0.8rem;
    background: rgba(255,255,255,0.02);
    border-radius: 6px;
    border-left: 2px solid var(--accent-blue);
}
.retrieved-chunk .chunk-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent-blue);
    margin-bottom: 4px;
}
.retrieved-chunk .chunk-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    color: var(--text-muted);
    line-height: 1.5;
}

/* ── Welcome Screen ── */
.welcome-screen {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 2rem;
    text-align: center;
}
.welcome-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    filter: drop-shadow(0 0 20px rgba(201,168,76,0.3));
}
.welcome-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: var(--accent-gold);
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.welcome-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: var(--text-secondary);
    max-width: 480px;
    line-height: 1.7;
    margin-bottom: 2.5rem;
}
.example-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    max-width: 580px;
    width: 100%;
}
.example-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    text-align: left;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    color: var(--text-secondary);
}
.example-card:hover {
    border-color: var(--accent-gold);
    color: var(--text-primary);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}
.example-card-icon {
    font-size: 1.1rem;
    margin-bottom: 0.4rem;
    display: block;
}

/* ── Input Area ── */
.input-area {
    padding: 1.25rem 2rem 1.5rem;
    border-top: 1px solid var(--border);
    background: var(--bg-secondary);
}
.input-disclaimer {
    text-align: center;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
    letter-spacing: 0.2px;
}
.stTextInput > div > div > input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent-gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.1) !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--text-muted) !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--accent-gold) !important; }

/* Selectbox */
.stSelectbox > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
label { color: var(--text-secondary) !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.8rem !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* Thinking indicator */
.thinking-dots {
    display: flex; gap: 4px; align-items: center; padding: 0.5rem 0;
}
.thinking-dot {
    width: 7px; height: 7px;
    background: var(--accent-gold);
    border-radius: 50%;
    animation: bounce 1.2s infinite;
}
.thinking-dot:nth-child(2) { animation-delay: 0.15s; }
.thinking-dot:nth-child(3) { animation-delay: 0.3s; }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
}

/* Exact match badge */
.exact-badge {
    display: inline-block;
    background: rgba(61, 122, 94, 0.15);
    border: 1px solid rgba(61, 122, 94, 0.3);
    color: #5db88a;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    padding: 1px 7px;
    border-radius: 20px;
    margin-left: 6px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ─────────────────────────────────────────────
# CONFIG — Paths use Unity Catalog Volume
# ─────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")

# VOLUME_ROOT = "/Volumes/workspace/default/nyaya_deepam_volume"
VOLUME_ROOT = "/nyaya_deepam"
FULL_DIR    = f"{VOLUME_ROOT}/artifacts/full"
ATOMIC_DIR  = f"{VOLUME_ROOT}/artifacts/atomic"
BNS_JSONL     = f"{VOLUME_ROOT}/raw/bns_rag.jsonl"
IPC_JSONL     = f"{VOLUME_ROOT}/raw/ipc_rag.jsonl"
MAPPING_JSONL = f"{VOLUME_ROOT}/raw/bns_ipc_rag.jsonl"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
LLM_MODEL_NAME   = "Qwen/Qwen2.5-3B-Instruct"

# Fine-tuned adapter — place it inside the volume or alongside app.py
FINETUNED_CHECKPOINT_PATH = os.getenv(
    "FINETUNED_CHECKPOINT_PATH",
    f"{VOLUME_ROOT}/checkpoints"
)
USE_FINETUNED = os.path.isdir(FINETUNED_CHECKPOINT_PATH)

# ─────────────────────────────────────────────
# HELPERS (same logic as your notebook)
# ─────────────────────────────────────────────
def normalize_act_name(act_text):
    if not act_text: return None
    t = act_text.lower().strip()
    if "bns" in t or "bharatiya nyaya sanhita" in t: return "BNS"
    if "ipc" in t or "indian penal code" in t: return "IPC"
    return None

def normalize_section_number(section_text):
    if not section_text: return None
    return str(section_text).strip().lower()

def load_jsonl_records(jsonl_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def build_section_lookup(records):
    lookup = {}
    for row in records:
        metadata = row.get("metadata", {})
        text = row.get("text", "")
        act_name_raw = metadata.get("act_name", "") or text
        act = normalize_act_name(act_name_raw)
        section_raw = metadata.get("section", "")
        section = normalize_section_number(section_raw)
        if act and section:
            lookup[(act, section)] = row
    return lookup

def build_mapping_lookup(records):
    mapping_lookup = {}
    for row in records:
        meta = row.get("metadata", {})
        ipc = str(meta.get("ipc_section", "")).strip().lower()
        bns = str(meta.get("bns_section_subsection", "")).strip().lower()
        if ipc and bns:
            mapping_lookup[ipc] = row
    return mapping_lookup

def extract_act_and_section_from_query(query):
    q = query.lower().strip()
    bns_match = re.search(r'\b(?:section|sec\.?)\s+(\d+[a-z]?(?:\(\d+\))?)\s*(?:of\s*)?bns\b', q)
    ipc_match = re.search(r'\b(?:section|sec\.?)\s+(\d+[a-z]?(?:\(\d+\))?)\s*(?:of\s*)?ipc\b', q)
    if bns_match and ipc_match: return None, None
    if bns_match: return "BNS", bns_match.group(1).lower()
    if ipc_match: return "IPC", ipc_match.group(1).lower()
    return None, None

def smart_lookup(query, section_lookup, mapping_lookup):
    act, section = extract_act_and_section_from_query(query)
    if act is None or section is None: return None
    if act == "BNS":
        direct = section_lookup.get(("BNS", section))
        if direct:
            return {"type": "direct_bns", "act": "BNS", "section": section, "text": direct["text"], "raw": direct}
    if act == "IPC":
        direct = section_lookup.get(("IPC", section))
        if direct:
            return {"type": "direct_ipc", "act": "IPC", "section": section, "text": direct["text"], "raw": direct}
        mapping = mapping_lookup.get(section)
        if mapping:
            return {"type": "mapping", "act": "IPC", "section": section,
                    "bns_equivalent": mapping["metadata"].get("bns_section_subsection", ""),
                    "text": mapping["text"], "raw": mapping}
    return None

def build_exact_context(exact_match):
    if exact_match is None: return ""
    if exact_match["type"] == "direct_bns":
        return f"[Exact Match]\nAct: BNS\nSection: {exact_match['section']}\n{exact_match['text']}\n"
    if exact_match["type"] == "direct_ipc":
        return f"[Exact Match]\nAct: IPC\nSection: {exact_match['section']}\n{exact_match['text']}\n"
    if exact_match["type"] == "mapping":
        return (f"[Exact Mapping Match]\nIPC Section: {exact_match['section']}\n"
                f"BNS Equivalent: {exact_match['bns_equivalent']}\n{exact_match['text']}\n")
    return ""

def choose_mode(query):
    q = query.lower()
    keywords = ["equivalent", "mapping", "mapped", "difference", "changed", "compare", "ipc", "bns"]
    return "full" if any(k in q for k in keywords) else "atomic"

# ─────────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_models():
    if HF_TOKEN:
        login(HF_TOKEN)

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    tokenizer_src = FINETUNED_CHECKPOINT_PATH if USE_FINETUNED else LLM_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16
    )

    if USE_FINETUNED:
        llm_model = PeftModel.from_pretrained(base_model, FINETUNED_CHECKPOINT_PATH)
    else:
        llm_model = base_model

    llm_model.eval()
    return embed_model, tokenizer, llm_model

@st.cache_resource(show_spinner=False)
def load_indices():
    def _load(index_dir):
        index = faiss.read_index(f"{index_dir}/corpus.faiss")
        rows = []
        with open(f"{index_dir}/chunks.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        chunks_df = pd.DataFrame(rows).reset_index(drop=True)
        texts = chunks_df["text"].fillna("").astype(str).tolist()
        bm25 = BM25Okapi([t.lower().split() for t in texts])
        return {"index": index, "chunks_df": chunks_df, "bm25": bm25}

    return _load(FULL_DIR), _load(ATOMIC_DIR)

@st.cache_resource(show_spinner=False)
def load_lookup_data():
    bns_records = load_jsonl_records(BNS_JSONL)
    ipc_records = load_jsonl_records(IPC_JSONL)
    mapping_records = load_jsonl_records(MAPPING_JSONL)
    section_lookup = build_section_lookup(bns_records + ipc_records)
    mapping_lookup = build_mapping_lookup(mapping_records)
    return section_lookup, mapping_lookup

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def embed_query(query, embed_model):
    return embed_model.encode(["query: " + query], normalize_embeddings=True)

def hybrid_search(query, bundle, embed_model, top_k=5, alpha=0.65):
    index = bundle["index"]
    chunks_df = bundle["chunks_df"]
    bm25 = bundle["bm25"]

    q_vec = embed_query(query, embed_model)
    vec_scores, vec_idx = index.search(np.array(q_vec, dtype=np.float32), top_k * 3)
    vec_scores = vec_scores[0]; vec_idx = vec_idx[0]

    if len(vec_scores) > 0:
        vmin, vmax = vec_scores.min(), vec_scores.max()
        vec_scores = (vec_scores - vmin) / (vmax - vmin + 1e-8)

    q_tokens = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(q_tokens), dtype=np.float32)
    if len(bm25_scores) > 0:
        bmin, bmax = bm25_scores.min(), bm25_scores.max()
        bm25_scores = (bm25_scores - bmin) / (bmax - bmin + 1e-8)

    combined = {}
    for i, idx in enumerate(vec_idx):
        idx = int(idx)
        if 0 <= idx < len(chunks_df):
            combined[idx] = combined.get(idx, 0.0) + alpha * float(vec_scores[i])
    for i, score in enumerate(bm25_scores):
        if 0 <= i < len(chunks_df):
            combined[i] = combined.get(i, 0.0) + (1 - alpha) * float(score)

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, score in ranked:
        row = chunks_df.iloc[idx]
        results.append({
            "score": round(float(score), 4), "chunk_id": row["chunk_id"],
            "source": row["source"], "act_name": row["act_name"],
            "section": row["section"], "section_name": row["section_name"],
            "text": row["text"],
        })
    return results

def answer_with_rag(user_query, top_k=4, mode="auto", max_new_tokens=300,
                    embed_model=None, tokenizer=None, llm_model=None,
                    bundle_full=None, bundle_atomic=None,
                    section_lookup=None, mapping_lookup=None):
    if mode == "auto":
        mode = choose_mode(user_query)

    bundle = bundle_full if mode == "full" else bundle_atomic

    exact_match = smart_lookup(user_query, section_lookup, mapping_lookup)
    exact_context = build_exact_context(exact_match)

    results = hybrid_search(user_query, bundle, embed_model, top_k=top_k)

    rag_context = "\n\n".join(
        [f"[Source {i+1}] {r['source']} | {r['act_name']} | Section {r['section']} | {r['section_name']}\n{r['text']}"
         for i, r in enumerate(results)]
    )
    context = (exact_context + "\n\n" + rag_context).strip() if exact_context.strip() else rag_context

    messages = [
        {"role": "system", "content": (
            "You are an expert Indian Legal Assistant. Your goal is to provide direct, authoritative, and concise answers "
            "based on the Bharatiya Nyaya Sanhita (BNS) and the Indian Penal Code (IPC).\n\n"
            "### OPERATIONAL GUIDELINES:\n"
            "1. **Tone**: Professional, direct, and helpful. Avoid meta-talk like 'Based on the context provided'.\n"
            "2. **Structure**: Respond in a single coherent paragraph. Start with the direct answer, followed by a brief explanation.\n"
            "3. **Citation**: Always end the response with a formal citation in the format: '- [Act Name] Section [Number]'.\n"
            "4. **Strict Matching**: If the user asks for a specific section (e.g., 1(3)), and the context only contains a different section, do NOT substitute it.\n"
            "5. **Handling Failure**: If the retrieved context does not contain the answer, say: 'I'm sorry, I couldn't find the specific details regarding that section.'\n"
            "6. **No Notes**: Never include 'Note:', 'Section Used:', or 'Explanation:' headers.\n"
            "7. **MAPPING**: If user asked about IPC and there is information about BNS, provide BNS section and all details.\n"
            "8. **More details**: Provide complete details about the requested IPC or BNS section.\n"
        )},
        {"role": "user", "content": f"Use the following legal context to answer the question:\n---\n{context}\n---\nQuestion: {user_query}"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(next(llm_model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=0.0, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return {"mode": mode, "answer": answer, "retrieved": results,
            "exact_match": exact_match, "used_exact_match": exact_match is not None}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">⚖️</div>
        <div class="sidebar-logo-text">
            <h2>Nyaya Deepam</h2>
            <p>Indian Legal Intelligence</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("+ New Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown('<div class="sidebar-section-title">Settings</div>', unsafe_allow_html=True)

    mode_option = st.selectbox(
        "Search Mode",
        ["Auto", "Full (with mappings)", "Atomic (BNS/IPC only)"],
        help="Auto detects mapping queries automatically"
    )
    top_k = st.slider("Retrieved chunks", 2, 8, 4)
    max_tokens = st.slider("Max response tokens", 128, 512, 300)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">Recent</div>', unsafe_allow_html=True)

    for past_q in st.session_state.history[-8:][::-1]:
        st.markdown(f'<div class="history-item">💬 {past_q[:42]}{"…" if len(past_q)>42 else ""}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    # Model status
    model_status = "🟢 Fine-tuned LoRA" if USE_FINETUNED else "🔵 Base Qwen2.5-3B"
    st.markdown(f"""
    <div style="font-family:'DM Sans',sans-serif;font-size:0.72rem;color:#5a5750;padding:0 0.5rem;">
        <strong style="color:#9a9590">Model</strong><br>
        {model_status}<br><br>
        <strong style="color:#9a9590">Embeddings</strong><br>
        multilingual-e5-small<br><br>
        <strong style="color:#9a9590">Volume</strong><br>
        /Volumes/workspace/default/<br>nyaya_deepam_volume
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
models_ready = False
with st.spinner(""):
    try:
        embed_model, tokenizer, llm_model = load_all_models()
        bundle_full, bundle_atomic = load_indices()
        section_lookup, mapping_lookup = load_lookup_data()
        models_ready = True
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {e}\n\nCheck your paths and HF_TOKEN in .env")

# ─────────────────────────────────────────────
# MAIN CHAT UI
# ─────────────────────────────────────────────
st.markdown('<div class="chat-header"><div><div class="chat-header-title">⚖️ Nyaya Deepam</div><div class="chat-header-subtitle">Ask anything about BNS 2023 or IPC 1860</div></div></div>', unsafe_allow_html=True)

# EXAMPLE QUERIES
EXAMPLES = [
    ("📋", "What is the punishment for murder under BNS?"),
    ("🔄", "What is BNS Section for IPC 420?"),
    ("📖", "Explain Section 96 BNS in simple terms"),
    ("⚖️", "Difference between IPC Section 95 and BNS equivalent"),
]

# Show welcome screen if no messages
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-screen">
        <div class="welcome-icon">⚖️</div>
        <div class="welcome-title">Nyaya Deepam</div>
        <div class="welcome-subtitle">
            Your intelligent guide to Indian criminal law.<br>
            Ask about BNS 2023 provisions, IPC 1860 sections, punishments, mappings, and more.
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    for i, (icon, example) in enumerate(EXAMPLES):
        with cols[i % 2]:
            if st.button(f"{icon} {example}", key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_query = example
                st.rerun()

# Render chat messages
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        st.markdown(f"""
        <div class="message-row user">
            <div class="avatar user">👤</div>
            <div class="bubble user">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        retrieved = msg.get("retrieved", [])
        exact = msg.get("exact_match")
        mode = msg.get("mode", "")
        exact_badge = '<span class="exact-badge">⚡ exact match</span>' if msg.get("used_exact") else ""

        retrieved_html = ""
        if retrieved:
            chunks_html = "".join([
                f"""<div class="retrieved-chunk">
                    <div class="chunk-meta">{r['act_name']} § {r['section']} — {r['source']} (score: {r['score']})</div>
                    <div class="chunk-text">{r['text'][:200]}{"…" if len(r['text'])>200 else ""}</div>
                </div>"""
                for r in retrieved
            ])
            retrieved_html = f"""
            <details class="retrieved-toggle">
                <summary>📚 {len(retrieved)} sources retrieved · mode: {mode}</summary>
                {chunks_html}
            </details>"""

        st.markdown(f"""
        <div class="message-row bot">
            <div class="avatar bot">⚖️</div>
            <div class="bubble bot">
                <div class="answer-text">{content} {exact_badge}</div>
                {retrieved_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
col1, col2 = st.columns([8, 1])
with col1:
    user_input = st.text_input(
        "Enter your legal Question", placeholder="Ask about any BNS or IPC section, punishment, or legal provision…",
        key="chat_input", label_visibility="collapsed"
    )
with col2:
    send_btn = st.button("Send ➤", use_container_width=True)

st.markdown('<div class="input-disclaimer">Nyaya Deepam may make mistakes. Always verify with official legal sources.</div>',
            unsafe_allow_html=True)

# Resolve query (from input box OR example card click)
query_to_process = None
if send_btn and user_input.strip():
    query_to_process = user_input.strip()
elif st.session_state.pending_query:
    query_to_process = st.session_state.pending_query
    st.session_state.pending_query = None

# ─────────────────────────────────────────────
# PROCESS QUERY
# ─────────────────────────────────────────────
if query_to_process and models_ready:
    st.session_state.messages.append({"role": "user", "content": query_to_process})
    st.session_state.history.append(query_to_process)

    mode_map = {"Auto": "auto", "Full (with mappings)": "full", "Atomic (BNS/IPC only)": "atomic"}
    selected_mode = mode_map[mode_option]

    with st.spinner("Searching legal corpus…"):
        try:
            result = answer_with_rag(
                query_to_process, top_k=top_k, mode=selected_mode,
                max_new_tokens=max_tokens,
                embed_model=embed_model, tokenizer=tokenizer, llm_model=llm_model,
                bundle_full=bundle_full, bundle_atomic=bundle_atomic,
                section_lookup=section_lookup, mapping_lookup=mapping_lookup,
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "retrieved": result["retrieved"],
                "exact_match": result["exact_match"],
                "used_exact": result["used_exact_match"],
                "mode": result["mode"],
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Error generating response: {str(e)}",
                "retrieved": [], "exact_match": None, "used_exact": False, "mode": "error"
            })

    st.rerun()