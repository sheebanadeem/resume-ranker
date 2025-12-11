# resume_ranker.py
"""
Advanced resume ranking utilities.

Features:
- text extraction from pdf / txt / docx
- light resume parsing: emails, phones, name/title snippet
- chunking of resume into passages (for passage-level scoring / explainability)
- embeddings with caching (numpy .npy)
- FAISS index build (if faiss available) with cosine fallback
- ranking + explanation: top passages that matched JD
- TF-IDF keyword extraction for JD and resumes
"""
from __future__ import annotations
import io
import os
import re
import json
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import hashlib
import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try optional imports
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except Exception:
    _HAS_SBERT = False

# PDF / docx extractors
try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore

try:
    import docx  # python-docx
except Exception:
    docx = None  # type: ignore

# -------------------------
# Utilities
# -------------------------
def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def normalize_text(t: str) -> str:
    t = t or ""
    t = t.replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

# -------------------------
# Text extraction
# -------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    except Exception:
        return ""

def extract_text_from_docx(file_bytes: bytes) -> str:
    if docx is None:
        return ""
    try:
        bio = io.BytesIO(file_bytes)
        document = docx.Document(bio)
        return "\n".join(p.text for p in document.paragraphs)
    except Exception:
        return ""

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def extract_text(filename: str, file_bytes: bytes) -> str:
    fname = (filename or "").lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if fname.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    return extract_text_from_txt(file_bytes)

# -------------------------
# Resume parsing helpers
# -------------------------
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}")

def extract_email(text: str) -> Optional[str]:
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else None

def extract_phone(text: str) -> Optional[str]:
    m = _PHONE_RE.search(text)
    if not m:
        return None
    return re.sub(r"[^\d+]", "", m.group(0))

def guess_name_snippet(text: str, max_chars: int = 120) -> str:
    # Take the first non-empty line(s) as candidate
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    header = " | ".join(lines[:3])
    return header[:max_chars]

# -------------------------
# Text chunking for passages
# -------------------------
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Chunk text into overlapping windows of approx chunk_size words.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# -------------------------
# Embeddings & caching
# -------------------------
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_EMBED_CACHE_DIR = Path(".emb_cache")
_EMBED_CACHE_DIR.mkdir(exist_ok=True)

def _embed_cache_path(model_name: str, key: str) -> Path:
    fn = f"{model_name.replace('/', '_')}_{key}.npy"
    return _EMBED_CACHE_DIR / fn

def get_sentence_transformer(model_name: str = _DEFAULT_MODEL):
    if not _HAS_SBERT:
        raise RuntimeError("sentence-transformers is not installed. Install it to use embeddings.")
    return SentenceTransformer(model_name)

def embed_texts(model, texts: List[str], model_name: str = _DEFAULT_MODEL, cache_prefix: Optional[str] = None) -> np.ndarray:
    """
    Embed a list of texts. If cache_prefix provided, it will try to load per-text cached embeddings
    and only compute missing ones. Returns numpy array (n_texts, dim).
    """
    texts = [normalize_text(t) for t in texts]
    key_parts = [cache_prefix] if cache_prefix else []
    key_parts.append(_sha256_text(model_name))
    cache_base = "_".join([p for p in key_parts if p])
    cached_embs = []
    to_compute = []
    compute_idx = []
    for i, t in enumerate(texts):
        key = _sha256_text(t)[:16]
        path = _embed_cache_path(cache_base, key)
        if path.exists():
            try:
                arr = np.load(path)
                cached_embs.append((i, arr))
                continue
            except Exception:
                pass
        # missing -> mark to compute
        to_compute.append(t)
        compute_idx.append(i)

    # if nothing to compute, return stacked cached
    dim = None
    if len(to_compute) == 0:
        results = []
        for i in range(len(texts)):
            key = _sha256_text(texts[i])[:16]
            arr = np.load(_embed_cache_path(cache_base, key))
            results.append(arr)
        return np.vstack(results)

    # compute embeddings for missing
    embed = get_sentence_transformer(model_name)
    computed = embed.encode(to_compute, convert_to_numpy=True, show_progress_bar=False)
    # save computed
    for j, idx in enumerate(compute_idx):
        key = _sha256_text(texts[idx])[:16]
        path = _embed_cache_path(cache_base, key)
        try:
            np.save(path, computed[j])
        except Exception:
            pass

    # now assemble final array
    results = [None] * len(texts)
    for (i, arr) in cached_embs:
        results[i] = arr
    for j, idx in enumerate(compute_idx):
        results[idx] = computed[j]
    stacked = np.vstack(results)
    return stacked

# -------------------------
# FAISS index building or fallback
# -------------------------
def build_faiss_index(embs: np.ndarray):
    if not _HAS_FAISS:
        raise RuntimeError("faiss not installed")
    d = embs.shape[1]
    # use IndexFlatIP (inner product) with normalization for cosine
    index = faiss.IndexFlatIP(d)
    # normalize
    faiss.normalize_L2(embs)
    index.add(embs)
    return index

def search_index(index, embs_query: np.ndarray, top_k: int = 5):
    # query must be normalized if using IP
    if _HAS_FAISS:
        faiss.normalize_L2(embs_query)
        D, I = index.search(embs_query, top_k)
        # D are inner products, equivalent to cosine if normalized
        return D, I
    else:
        # fallback: compute cosine similarity matrix
        sims = cosine_similarity(embs_query, index)  # here index is actually emb matrix
        I = np.argsort(-sims, axis=1)[:, :top_k]
        D = np.take_along_axis(sims, I, axis=1)
        return D, I

# -------------------------
# Keyword extraction (TF-IDF)
# -------------------------
def top_keywords_for_jd(jd_text: str, resume_texts: List[str], top_n: int = 10) -> List[str]:
    docs = [jd_text] + resume_texts
    vect = TfidfVectorizer(ngram_range=(1,2), stop_words="english", max_features=3000)
    try:
        X = vect.fit_transform([normalize_text(d) for d in docs])
    except Exception:
        return []
    jd_vec = X[0].toarray().ravel()
    if jd_vec.sum() == 0:
        return []
    feature_names = np.array(vect.get_feature_names_out())
    top_idx = jd_vec.argsort()[::-1][:top_n]
    keywords = [feature_names[i] for i in top_idx if jd_vec[i] > 0]
    return keywords

# -------------------------
# Produce ranking & explainability
# -------------------------
def produce_ranking_advanced(
    jd_text: str,
    resumes: List[Tuple[str, str]],
    model_name: str = _DEFAULT_MODEL,
    top_k_keywords: int = 12,
    passage_chunk_size: int = 80,
    passage_overlap: int = 20,
    top_k_matches: int = 5,
) -> List[Dict]:
    """
    resumes: list of (filename, text)
    returns list of dicts:
      filename, score (overall similarity), email, phone, header_snippet,
      top_jd_keywords, missing_skills, top_passages: list of (passage_text, passage_score)
    """
    # prepare resume texts
    resume_texts = [t for (_, t) in resumes]
    # extract keywords
    jd_keywords = top_keywords_for_jd(jd_text, resume_texts, top_n=top_k_keywords)

    # create passages and mapping: each resume's passages stored sequentially
    all_passages = []
    passage_map = []  # (resume_idx, passage_idx_within_resume)
    for ri, (_, text) in enumerate(resumes):
        chunks = chunk_text(text, chunk_size=passage_chunk_size, overlap=passage_overlap)
        if not chunks:
            chunks = [""]  # keep placeholder
        for pi, c in enumerate(chunks):
            all_passages.append(c)
            passage_map.append((ri, pi))

    # embed JD and passages (with caching using prefix from JD)
    cache_prefix = _sha256_text(jd_text)[:12]
    # embed jd as single
    if not _HAS_SBERT:
        raise RuntimeError("sentence-transformers not installed. Please install it.")
    model = model_name
    passage_embs = embed_texts(model=model, texts=all_passages, model_name=model, cache_prefix=cache_prefix)
    jd_emb = embed_texts(model=model, texts=[jd_text], model_name=model, cache_prefix=cache_prefix)[0].reshape(1, -1)

    # build/search
    if _HAS_FAISS:
        index = build_faiss_index(passage_embs.copy())
        D, I = search_index(index, jd_emb, top_k=top_k_matches)
        scores = D[0]  # top similarity scores
        indices = I[0]
    else:
        # fallback: return top matches across passage_embs
        sims = cosine_similarity(jd_emb, passage_embs)[0]
        indices = np.argsort(-sims)[:top_k_matches]
        scores = sims[indices]

    # compute aggregate score per resume: average top passage score for that resume
    per_resume_scores = [0.0] * len(resumes)
    per_resume_counts = [0] * len(resumes)
    # gather top passages per resume (we will also get top K per resume separately)
    resume_top_passages = [[] for _ in range(len(resumes))]
    # For explainability, compute cosine between jd and all passages to pick top for each resume
    sims_all = cosine_similarity(jd_emb, passage_embs)[0]
    for ri in range(len(resumes)):
        # indices of passages for this resume
        idxs = [i for i, (rj, _) in enumerate(passage_map) if rj == ri]
        if not idxs:
            per_resume_scores[ri] = 0.0
            continue
        # take top 3 passage sims
        vals = sims_all[idxs]
        if len(vals) == 0:
            continue
        top_vals = sorted(vals, reverse=True)[:3]
        per_resume_scores[ri] = float(sum(top_vals) / len(top_vals))
        per_resume_counts[ri] = len(idxs)
        # store the top passages (text + score)
        top_idx_rel = np.argsort(-np.array(vals))[:3]
        for j in top_idx_rel:
            global_idx = idxs[j]
            passage_text = all_passages[global_idx]
            passage_score = float(sims_all[global_idx])
            resume_top_passages[ri].append({"text": passage_text, "score": passage_score})

    # build result objects
    results = []
    for ri, (fname, text) in enumerate(resumes):
        email = extract_email(text) or ""
        phone = extract_phone(text) or ""
        header = guess_name_snippet(text)
        missing = [k for k in jd_keywords if k not in normalize_text(text)]
        results.append({
            "filename": fname,
            "score": float(per_resume_scores[ri]),
            "email": email,
            "phone": phone,
            "header_snippet": header,
            "top_jd_keywords": jd_keywords,
            "missing_skills": missing,
            "top_passages": resume_top_passages[ri],
        })

    # sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
