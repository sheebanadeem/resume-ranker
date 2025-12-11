# app.py — upgraded Streamlit UI for advanced resume ranker
import os
import csv
import io
from io import BytesIO
from pathlib import Path

import streamlit as st
import pandas as pd

from resume_ranker import extract_text, produce_ranking_advanced

st.set_page_config(page_title="Intelligent Resume Ranker — Advanced", layout="wide")
st.title(" Intelligent Resume Ranker — Advanced")

st.markdown("""
This upgraded resume ranker uses local sentence-transformer embeddings, passage-level scoring,
FAISS (if installed) for fast retrieval, and explains results by showing top matching passages.
""")

# Sidebar
with st.sidebar:
    st.header("Options")
    model_name = st.selectbox("Embedding model", options=[
        "all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"
    ], index=0)
    top_k_keywords = st.slider("Top JD keywords", 5, 20, 10)
    passage_chunk = st.slider("Passage chunk (words)", 40, 300, 120, step=20)
    passage_overlap = st.slider("Passage overlap (words)", 0, 120, 30, step=10)
    st.markdown("---")
    st.markdown("Tips:")
    st.markdown("- Use plain-text resumes for best extraction speed.")
    st.markdown("- If a model download fails, try changing the model to all-MiniLM-L6-v2.")

# Upload & JD
uploaded = st.file_uploader("Upload resumes (pdf/txt/docx) — multiple", accept_multiple_files=True, type=["pdf", "txt", "docx"])
jd_text = st.text_area("Paste job description (JD) here", height=220)

if st.button("Rank (Advanced)"):

    if not jd_text or len(jd_text.strip()) < 20:
        st.error("Please paste a more descriptive job description (20+ chars).")
    elif not uploaded:
        st.error("Please upload one or more resumes.")
    else:
        resumes = []
        pbar = st.progress(0)
        for i, f in enumerate(uploaded):
            raw = f.read()
            txt = extract_text(f.name, raw)
            if not txt.strip():
                txt = "(no extractable text)"
            resumes.append((f.name, txt))
            pbar.progress(int((i+1)/len(uploaded) * 100))
        pbar.empty()

        with st.spinner("Ranking (may download model first time)..."):
            try:
                results = produce_ranking_advanced(
                    jd_text,
                    resumes,
                    model_name=model_name,
                    top_k_keywords=top_k_keywords,
                    passage_chunk_size=passage_chunk,
                    passage_overlap=passage_overlap,
                    top_k_matches=5
                )
            except Exception as e:
                st.error(f"Error during ranking: {e}")
                raise

        # show summary table
        table = []
        for r in results:
            table.append({
                "filename": r["filename"],
                "score": round(r["score"], 4),
                "email": r["email"],
                "phone": r["phone"],
                "missing_keywords_count": len(r["missing_skills"])
            })
        df = pd.DataFrame(table)
        st.subheader("Ranked candidates")
        st.table(df)

        # Expanders for details
        for r in results:
            with st.expander(f"{r['filename']} — score {r['score']:.4f} — missing {len(r['missing_skills'])}"):
                st.markdown(f"**Header snippet:** {r.get('header_snippet','')}")
                st.markdown(f"- **Email:** {r.get('email','')}")
                st.markdown(f"- **Phone:** {r.get('phone','')}")
                st.markdown("**Top JD keywords:** " + ", ".join(r["top_jd_keywords"]))
                st.markdown("**Missing keywords:** " + ", ".join(r["missing_skills"][:50]) if r["missing_skills"] else "None")
                st.markdown("**Top matching passages (explainability):**")
                for p in r["top_passages"]:
                    st.markdown(f"- (score {p['score']:.3f}) {p['text'][:400]}")

        # CSV download
        csv_out = io.StringIO()
        csvw = csv.writer(csv_out)
        csvw.writerow(["filename","score","email","phone","missing_count","missing_keywords"])
        for r in results:
            csvw.writerow([r["filename"], r["score"], r["email"], r["phone"], len(r["missing_skills"]), ";".join(r["missing_skills"])])
        st.download_button("Download CSV", data=csv_out.getvalue().encode("utf-8"), file_name="resume_ranking_advanced.csv", mime="text/csv")
