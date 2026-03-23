"""
Student Name: [Eaint Taryar Linlat]
Task 5: Financial News Semantic Search with Sentence Embeddings and Gradio
In this task, I built a semantic search tool for financial news headlines using sentence embeddings and a Gradio UI.
What I did:

Cleaned the data — used regex (\s+(https?://\S+)\s*$) to extract trailing URLs from each row in the text column into a new URL column, then removed them from the original text
Built sentence embeddings — used all-MiniLM-L6-v2 from sentence-transformers to encode all headlines into 384-dimensional vectors, with normalize_embeddings=True so that cosine similarity reduces to a simple dot product
Built semantic search — for any query, the same model encodes it into a vector, then matrix multiplication (embeddings @ query_vec) computes similarity against all documents at once, and the top 5 are returned
Built a Gradio UI — text input box triggers the search on click or Enter, results display in a ranked table with similarity scores, article text, and URL

Key lessons:

Regex for URL extraction — anchoring the pattern to end-of-string ($) ensures only trailing URLs are captured, not any URLs embedded mid-sentence
Normalised embeddings make cosine similarity faster — when vectors are unit-length, dot product equals cosine similarity, which means no division needed and the entire corpus can be scored in one vectorised matrix multiplication
Semantic search vs keyword search — a query like "earnings surprise" returns articles about profit beats and analyst expectations even if those exact words are not present, because the model understands meaning rather than just matching strings
"""

# ═══════════════════════════════════════════════════════════════════════════
# 0.  Install dependencies  (run this cell first in Colab)
# ═══════════════════════════════════════════════════════════════════════════
# !pip install sentence-transformers gradio --quiet


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Imports
# ═══════════════════════════════════════════════════════════════════════════
import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import gradio as gr


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Load Data
# ═══════════════════════════════════════════════════════════════════════════
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "financial_news.csv")

df = pd.read_csv(CSV_PATH)

print("─" * 60)
print("STEP 1 — Raw Data")
print("─" * 60)
print(f"Shape  : {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Extract Trailing URL from `text` → new `URL` column
#
#     Pattern explanation
#     ───────────────────
#     \s+          one or more whitespace characters before the URL
#     (https?://   capture group: http or https
#      \S+)        one or more non-whitespace chars (the URL body)
#     \s*$         optional trailing whitespace, then end of string
#
#     This reliably strips the LAST token when it is a URL, regardless
#     of whether the URL contains query-strings (?q=1&b=2) or fragments (#section).
# ═══════════════════════════════════════════════════════════════════════════
URL_PATTERN = r'\s+(https?://\S+)\s*$'

# Extract URL into its own column (NaN if no URL found in that row)
df["URL"]  = df["text"].str.extract(URL_PATTERN, flags=re.IGNORECASE)[0]

# Remove the URL + preceding whitespace from text, then strip remaining whitespace
df["text"] = (df["text"]
              .str.replace(URL_PATTERN, "", regex=True, flags=re.IGNORECASE)
              .str.strip())

print("\n" + "─" * 60)
print("STEP 2 — After URL Extraction")
print("─" * 60)
print(df[["text", "URL"]].head(6).to_string())

# Sanity checks
urls_remaining = df["text"].str.contains(r'https?://', regex=True).sum()
urls_extracted = df["URL"].notna().sum()
print(f"\nURLs still embedded in text : {urls_remaining}  (should be 0)")
print(f"URLs successfully extracted : {urls_extracted} / {len(df)} rows")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Build Sentence Embeddings
#
#     Model: all-MiniLM-L6-v2
#       • Very fast (~14k sentences/sec on CPU)
#       • 384-dimensional embeddings
#       • Trained specifically for semantic similarity tasks
#       • Excellent for financial / news text out of the box
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("STEP 3 — Building Sentence Embeddings")
print("─" * 60)

MODEL_NAME = "all-MiniLM-L6-v2"
print(f"Loading model: {MODEL_NAME} …")
encoder = SentenceTransformer(MODEL_NAME)

texts = df["text"].fillna("").tolist()

print(f"Encoding {len(texts)} sentences …")
# encode() returns a numpy array of shape (n_sentences, embedding_dim)
embeddings = encoder.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True    # unit-normalise → dot product == cosine similarity
)

print(f"Embedding matrix shape: {embeddings.shape}")
print(f"  → {embeddings.shape[0]} sentences × {embeddings.shape[1]} dimensions")


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Semantic Search Function
#
#     Because embeddings are already L2-normalised, cosine similarity
#     reduces to a plain dot product:  sim(q, d) = q · d
#     Matrix multiplication (embeddings @ query_vec) computes all
#     similarities in one vectorised call — very fast even for 100k rows.
# ═══════════════════════════════════════════════════════════════════════════
TOP_K = 5


def semantic_search(query: str) -> pd.DataFrame:
    """
    Encode `query`, compute cosine similarity against every document,
    return the top-5 most similar rows as a DataFrame.
    """
    if not query or not query.strip():
        return pd.DataFrame({"Message": ["Please enter a search query."]})

    # Encode query with the same normalisation as the corpus
    query_vec = encoder.encode(
        [query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]                                    # shape: (384,)

    # Cosine similarities via dot product (vectors are already unit-length)
    scores = embeddings @ query_vec         # shape: (n_docs,)

    # Get indices of top-K highest scores
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    # Build results DataFrame
    results = df.iloc[top_indices].copy()
    results.insert(0, "Similarity", scores[top_indices].round(4))
    results = results.reset_index(drop=True)
    results.index += 1                      # 1-based rank

    # Select only the most useful columns for display
    display_cols = ["Similarity", "text", "URL"]
    # Add any other columns that exist (e.g. sentiment, date, source)
    extra = [c for c in df.columns if c not in ("text", "URL")]
    display_cols = ["Similarity"] + extra + ["text", "URL"]

    return results[display_cols]


# ─── Quick console test ────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("STEP 4 — Quick Search Test")
print("─" * 60)
test_results = semantic_search("earnings surprise")
print("Query: 'earnings surprise'")
print(test_results[["Similarity", "text"]].to_string())


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Gradio Interface
# ═══════════════════════════════════════════════════════════════════════════

# ── Custom CSS for a polished look ────────────────────────────────────────
CUSTOM_CSS = """
/* ── Page background ────────────────────────────────────────────────── */
body, .gradio-container {
    background: #0f172a !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Header banner ──────────────────────────────────────────────────── */
.header-box {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 60%, #162032 100%);
    border: 1px solid #2d5a8e;
    border-radius: 14px;
    padding: 28px 36px;
    margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
}
.header-box h1 {
    color: #60a5fa;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.header-box p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.5;
}

/* ── Input area ─────────────────────────────────────────────────────── */
.query-row {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
}
.query-row label {
    color: #93c5fd !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.query-row input, .query-row textarea {
    background: #0f172a !important;
    border: 1px solid #475569 !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
}
.query-row input:focus, .query-row textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.2) !important;
}

/* ── Search button ──────────────────────────────────────────────────── */
#search-btn {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 12px 28px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.4) !important;
    min-width: 160px !important;
}
#search-btn:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Results dataframe ──────────────────────────────────────────────── */
.results-section label {
    color: #93c5fd !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.results-section .svelte-1gfkn6j, .results-section table {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.results-section th {
    background: #0f172a !important;
    color: #60a5fa !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #2d5a8e !important;
    padding: 10px 14px !important;
}
.results-section td {
    border-bottom: 1px solid #1e3a5f !important;
    padding: 10px 14px !important;
    color: #cbd5e1 !important;
}
.results-section tr:hover td {
    background: #162032 !important;
    color: #f1f5f9 !important;
}

/* ── Example pills ──────────────────────────────────────────────────── */
.example-pills .gr-button {
    background: #1e3a5f !important;
    border: 1px solid #2d5a8e !important;
    color: #93c5fd !important;
    border-radius: 20px !important;
    font-size: 0.85rem !important;
    padding: 6px 16px !important;
    transition: all 0.2s !important;
}
.example-pills .gr-button:hover {
    background: #2d5a8e !important;
    color: #bfdbfe !important;
}

/* ── Stats bar ──────────────────────────────────────────────────────── */
.stats-bar {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}
.stat-chip {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px 18px;
    color: #94a3b8;
    font-size: 0.85rem;
}
.stat-chip span {
    color: #60a5fa;
    font-weight: 700;
    font-size: 1.1rem;
}
"""

# ── Compute stats for the header ─────────────────────────────────────────
n_docs   = len(df)
n_urls   = df["URL"].notna().sum()
emb_dim  = embeddings.shape[1]

HEADER_HTML = f"""
<div class="header-box">
  <h1>📰 Financial News Semantic Search</h1>
  <p>
    Powered by <strong>sentence-transformers</strong> (all-MiniLM-L6-v2) ·
    Returns the <strong>top 5</strong> most semantically similar headlines
    using cosine similarity on {emb_dim}-dimensional embeddings.
  </p>
  <div class="stats-bar" style="margin-top:14px;">
    <div class="stat-chip">🗞️ Articles &nbsp;<span>{n_docs:,}</span></div>
    <div class="stat-chip">🔗 URLs extracted &nbsp;<span>{n_urls:,}</span></div>
    <div class="stat-chip">📐 Embedding dim &nbsp;<span>{emb_dim}</span></div>
    <div class="stat-chip">🔍 Results shown &nbsp;<span>{TOP_K}</span></div>
  </div>
</div>
"""

# ── Build the Gradio app ──────────────────────────────────────────────────
with gr.Blocks(title="Financial News Semantic Search") as demo:

    gr.HTML(HEADER_HTML)

    # ── Input row ─────────────────────────────────────────────────────────
    with gr.Row():
        query_box = gr.Textbox(
            label="Search Query",
            placeholder='e.g. "earnings surprise", "regulatory fine", "interest rate hike" …',
            lines=1,
            scale=5,
        )
        search_btn = gr.Button("🔍  Search", elem_id="search-btn", scale=1)

    # ── Example queries ───────────────────────────────────────────────────
    gr.Examples(
        examples=[
            ["earnings surprise"],
            ["regulatory fine"],
            ["interest rate hike"],
            ["merger acquisition deal"],
            ["CEO resignation leadership change"],
            ["oil price surge energy"],
            ["quarterly revenue growth"],
            ["fraud investigation"],
        ],
        inputs=query_box,
        label="💡 Example queries — click to try",
    )

    # ── Results table ─────────────────────────────────────────────────────
    with gr.Column():
        results_table = gr.DataFrame(
            label=f"Top {TOP_K} Closest Results  (ranked by cosine similarity ↓)",
            wrap=True,
            interactive=False,
        )

    # ── Wire up events ────────────────────────────────────────────────────
    # Trigger on button click OR pressing Enter in the text box
    search_btn.click(
        fn=semantic_search,
        inputs=query_box,
        outputs=results_table,
    )
    query_box.submit(
        fn=semantic_search,
        inputs=query_box,
        outputs=results_table,
    )

# ── Launch ────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("Launching Gradio app …")
print("═" * 60)
demo.launch(
    server_name="0.0.0.0",   # accessible on local network
    server_port=7860,         # open http://localhost:7860 in your browser
    share=False,
    debug=False,
    show_error=True,
    css=CUSTOM_CSS,           # Gradio 6.0+: css goes here instead of gr.Blocks()
)