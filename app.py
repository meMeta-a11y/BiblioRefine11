# app.py
"""
BiblioRefine - Optimized single-file Streamlit app
- Sidebar navigation
- Leiden -> Louvain fallback
- Friendly logs
- <=10k record cap
- Exports: CSV, GEXF, VOSviewer (.txt/.clu), timeline JSON, ZIP
"""

from typing import List, Dict, Any
import streamlit as st
st.set_page_config(page_title="📚 BiblioRefine", layout="wide")

# Standard libs
import os, io, re, json, time, hashlib, datetime, subprocess, zipfile
from io import BytesIO
import logging

# Data libs
import pandas as pd
import numpy as np

# Optional visualization libs
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import plotly.graph_objects as go

# Parsing optional libs
try:
    import bibtexparser
except Exception:
    bibtexparser = None
try:
    import rispy
except Exception:
    rispy = None

# XML parse fallback
try:
    from lxml import etree as ET
except Exception:
    import xml.etree.ElementTree as ET

# NLP & stopwords
import nltk
try:
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('stopwords')
from nltk.corpus import stopwords

# Sparse & ML
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

# Graph & clustering
import networkx as nx
try:
    import community as community_louvain  # python-louvain
except Exception:
    community_louvain = None

# Leiden optional
LEIDEN_AVAILABLE = False
try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except Exception:
    LEIDEN_AVAILABLE = False

# Logging config (internal)
logger = logging.getLogger("biblio_refine")
logger.setLevel(logging.INFO)

# -------------------------
# App configuration
# -------------------------
MAX_RECORDS = 10000
DEFAULT_VOCAB = 5000
DEFAULT_MIN_DF = 3
DEFAULT_EDGE_THRESHOLD = 0.05
DEFAULT_WINDOW = 5
DEFAULT_STRIDE = 5
DEFAULT_OVERLAP = 0.05
DEFAULT_SEED = 42

# Stopwords
DEFAULT_STOPWORDS = set(stopwords.words("english"))
CUSTOM_STOPWORDS = {"study", "studies", "analysis", "based", "using", "approach"}
STOPWORDS = DEFAULT_STOPWORDS.union(CUSTOM_STOPWORDS)

# MASTER mapping for WoS/Scopus header normalisation
MASTER_MAP = {
    "Publication Type": "PT", "Document Type": "DT", "Language": "LA",
    "Publication Year": "PY", "Year": "PY", "DOI": "DI", "DOI Link": "DI",
    "Source": "SO", "Source title": "SO", "Source Title": "SO",
    "Authors": "AU", "Author full names": "AF", "Author Full Names": "AF",
    "Affiliations": "C1", "Addresses": "C1", "Title": "TI", "Article Title": "TI",
    "Abstract": "AB", "Author Keywords": "DE", "Keywords Plus": "ID",
    "Cited References": "CR", "References": "CR", "Cited Reference Count": "NR",
    "Volume": "VL", "Issue": "IS", "Page count": "PG", "Page Count": "PG",
    "Funding Texts": "FX", "Funding Text": "FX"
}

# -------------------------
# Utility helpers
# -------------------------
def dependency_report() -> Dict[str, bool]:
    return {
        "python-igraph+leidenalg (Leiden available)": LEIDEN_AVAILABLE,
        "python-louvain (Louvain available)": community_louvain is not None,
        "bibtexparser (.bib parser)": bibtexparser is not None,
        "rispy (.ris parser)": rispy is not None,
        "matplotlib (plotting)": plt is not None,
    }

def standardize_df(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    mapping = {c: MASTER_MAP[c] for c in df.columns if c in MASTER_MAP}
    df = df.rename(columns=mapping)
    df = df.loc[:, ~df.columns.duplicated()]
    df["Source"] = source_label
    return df

def normalize_doi(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().lower()
    s = s.replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "").strip()
    return s if s else pd.NA

def title_key(title, year):
    t = "" if pd.isna(title) else re.sub(r"\s+", " ", str(title).strip().lower())
    y = ""
    try:
        if not pd.isna(year):
            y = str(int(year))
    except Exception:
        y = ""
    return f"{t}___{y}"

# Parsers for extra input formats
def parse_bib_bytes(b: bytes) -> pd.DataFrame:
    if not bibtexparser:
        return pd.DataFrame()
    try:
        txt = b.decode("utf-8", errors="ignore")
        bib_db = bibtexparser.loads(txt)
        rows = []
        for e in bib_db.entries:
            r = {}
            r["TI"] = e.get("title")
            if e.get("author"):
                authors = [a.strip() for a in re.split(r"\s+and\s+", e.get("author"))]
                r["AU"] = "; ".join(authors); r["AF"] = r["AU"]
            r["PY"] = e.get("year")
            r["SO"] = e.get("journal") or e.get("booktitle")
            r["DI"] = e.get("doi") or e.get("url")
            r["AB"] = e.get("abstract")
            r["DE"] = e.get("keywords")
            rows.append(r)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def parse_ris_bytes(b: bytes) -> pd.DataFrame:
    if not rispy:
        return pd.DataFrame()
    try:
        txt = b.decode("utf-8", errors="ignore").splitlines()
        entries = rispy.load(txt)
        rows = []
        for e in entries:
            r = {
                "TI": e.get("title"),
                "AU": "; ".join(e.get("authors") or []),
                "AF": "; ".join(e.get("authors") or []),
                "PY": e.get("year") or e.get("date"),
                "SO": e.get("journal_name") or e.get("publication_name"),
                "DI": e.get("doi"),
                "AB": e.get("abstract"),
                "DE": "; ".join(e.get("keywords") or [])
            }
            rows.append(r)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def parse_endnote_xml_bytes(b: bytes) -> pd.DataFrame:
    try:
        txt = b.decode("utf-8", errors="ignore")
        root = ET.fromstring(txt)
        rows = []
        for record in root.findall(".//record"):
            r = {}
            title = record.find(".//titles/title")
            if title is not None and title.text:
                r["TI"] = title.text
            authors = [a.text for a in record.findall(".//contributors/authors/author") if a is not None and a.text]
            if authors:
                r["AU"] = "; ".join(authors); r["AF"] = r["AU"]
            year = record.find(".//dates/year")
            if year is not None and year.text:
                r["PY"] = year.text
            journal = record.find(".//periodical/full_title") or record.find(".//periodical/title")
            if journal is not None and journal.text:
                r["SO"] = journal.text
            doi = record.find(".//electronic-resource-num")
            if doi is not None and doi.text:
                r["DI"] = doi.text
            abstract = record.find(".//abstract")
            if abstract is not None and abstract.text:
                r["AB"] = abstract.text
            kw = [k.text for k in record.findall(".//keywords/keyword") if k.text]
            if kw:
                r["DE"] = "; ".join(kw)
            rows.append(r)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def parse_openalex_json_bytes(b: bytes) -> pd.DataFrame:
    try:
        txt = b.decode("utf-8", errors="ignore")
        data = json.loads(txt)
        rows = []
        items = data.get("results") if isinstance(data, dict) and "results" in data else (data if isinstance(data, list) else [data])
        for item in items:
            r = {}
            r["TI"] = item.get("title")
            ab = item.get("abstract_inverted_index") or item.get("abstract")
            if isinstance(ab, dict):
                words = []
                for w in sorted(ab.keys(), key=lambda k: min(ab[k]) if ab[k] else 0):
                    words.append(w)
                r["AB"] = " ".join(words)
            else:
                r["AB"] = ab
            authors = item.get("authorships") or []
            if authors:
                r["AU"] = "; ".join([a.get("author", {}).get("display_name") for a in authors if a.get("author")])
                r["AF"] = r["AU"]
            r["PY"] = item.get("publication_year")
            r["SO"] = item.get("host_venue", {}).get("display_name")
            r["DI"] = item.get("doi")
            concepts = item.get("concepts") or []
            if concepts:
                r["DE"] = "; ".join([c.get("display_name") for c in concepts if c.get("display_name")])
            rows.append(r)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# -------------------------
# Tokenization & keyword cleaning
# -------------------------
def tokenize(text: str) -> List[str]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    s = str(text).strip().lower()
    s = re.sub(r"[;/\|]", " ", s)
    toks = [t.strip() for t in re.split(r"[\s,;|]+", s) if t.strip()]
    toks = [t for t in toks if len(t) > 1 and t not in STOPWORDS]
    return toks

def combine_de_id(row, max_terms: int = 50) -> List[str]:
    parts = []
    for col in ["DE", "ID"]:
        if col in row and pd.notna(row.get(col)):
            parts.extend([t.strip() for t in re.split(r"[;,\|]+", str(row.get(col))) if t.strip()])
    seen = []
    out = []
    for p in parts:
        if p not in seen:
            seen.append(p); out.append(p)
        if len(seen) >= max_terms:
            break
    return out

# -------------------------
# Cached heavy computations
# -------------------------
@st.cache_data(show_spinner=False)
def build_term_doc_matrix(docs_tokens: List[List[str]], max_features:int, min_df:int):
    docs_txt = [" ".join([t.replace(" ", "_") for t in toks]) for toks in docs_tokens]
    vect = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", max_features=max_features, min_df=min_df)
    X = vect.fit_transform(docs_txt)
    terms = vect.get_feature_names_out()
    return vect, X, terms

@st.cache_data(show_spinner=False)
def compute_cooccurrence(X):
    return (X.T @ X).tocsr()

# -------------------------
# Normalization helpers
# -------------------------
def normalize_salton(C):
    diag = np.sqrt(C.diagonal()).astype(float)
    inv = np.zeros_like(diag)
    nz = diag > 0
    inv[nz] = 1.0 / diag[nz]
    D = sp.diags(inv)
    S = D.dot(C).dot(D)
    S.setdiag(0.0)
    return S

def normalize_association_strength(C, doc_count:int):
    f = C.diagonal().astype(float)
    coo = C.tocoo()
    rows, cols, data = coo.row, coo.col, coo.data
    vals = []
    for r,c,v in zip(rows, cols, data):
        denom = (f[r] * f[c]) / max(1, doc_count)
        vals.append(v / denom if denom != 0 else 0.0)
    S = sp.coo_matrix((vals, (rows, cols)), shape=C.shape).tocsr()
    S.setdiag(0.0)
    return S

# -------------------------
# Build Network & clustering
# -------------------------
def build_graph_from_similarity(S, terms, threshold=DEFAULT_EDGE_THRESHOLD, top_k=None):
    S = S.tocsr()
    n = S.shape[0]
    G = nx.Graph()
    for i,t in enumerate(terms):
        G.add_node(i, term=t)
    if top_k is None:
        S_triu = sp.triu(S, k=1).tocoo()
        for i,j,v in zip(S_triu.row, S_triu.col, S_triu.data):
            if v >= threshold:
                G.add_edge(i, j, weight=float(v))
    else:
        for i in range(n):
            row = S.getrow(i).toarray().ravel()
            row[i] = 0.0
            if np.count_nonzero(row) == 0:
                continue
            k = min(len(row)-1, top_k)
            idxs = np.argpartition(-row, k)[:k]
            for j in idxs:
                if row[j] >= threshold:
                    G.add_edge(i, j, weight=float(row[j]))
    return G

def cluster_leiden(G: nx.Graph, seed:int = DEFAULT_SEED) -> Dict[int,int]:
    mapping = {n:i for i,n in enumerate(G.nodes())}
    rev = {i:n for n,i in mapping.items()}
    if not LEIDEN_AVAILABLE:
        raise RuntimeError("Leiden libs not available.")
    g = ig.Graph()
    g.add_vertices(len(mapping))
    g.vs['name'] = [G.nodes[n]['term'] for n in G.nodes()]
    edges = []
    weights = []
    for u,v,d in G.edges(data=True):
        edges.append((mapping[u], mapping[v])); weights.append(d.get('weight',1.0))
    if edges:
        g.add_edges(edges)
        g.es['weight'] = weights
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights='weight', seed=seed)
        membership = part.membership
        return {rev[i]: membership[i] for i in range(len(membership))}
    else:
        return {n: 0 for n in G.nodes()}

def cluster_louvain(G: nx.Graph, seed:int = DEFAULT_SEED) -> Dict[int,int]:
    if community_louvain is None:
        return {n: 0 for n in G.nodes()}
    return community_louvain.best_partition(G, weight='weight', random_state=seed)

def run_clustering(G: nx.Graph, prefer_leiden:bool = True, seed:int = DEFAULT_SEED) -> Dict[int,int]:
    if prefer_leiden and LEIDEN_AVAILABLE:
        return cluster_leiden(G, seed=seed)
    else:
        if prefer_leiden and not LEIDEN_AVAILABLE:
            # user friendly warning - shown where run_clustering is called
            pass
        return cluster_louvain(G, seed=seed)

# -------------------------
# Callon metrics
# -------------------------
def compute_callon(G: nx.Graph, partition: Dict[int,int]) -> Dict[int,Dict[str,Any]]:
    from collections import defaultdict
    clusters = defaultdict(list)
    for n,c in partition.items():
        clusters[c].append(n)
    metrics = {}
    for cid, nodes in clusters.items():
        internal = 0.0; external = 0.0
        n = len(nodes)
        for i,u in enumerate(nodes):
            for v in nodes[i+1:]:
                if G.has_edge(u,v):
                    internal += G[u][v].get('weight',1.0)
        denom = (n*(n-1)/2) if n>1 else 1
        density = internal/denom if denom > 0 else 0.0
        for u in nodes:
            for nbr in G.neighbors(u):
                if nbr not in nodes:
                    external += G[u][nbr].get('weight',1.0)
        metrics[cid] = {'nodes': nodes, 'size': n, 'density': density, 'centrality': external}
    return metrics

# -------------------------
# Time windows & thematic evolution
# -------------------------
def make_time_windows(df: pd.DataFrame, year_col='PY', window_size=DEFAULT_WINDOW, stride=DEFAULT_STRIDE):
    if year_col not in df.columns:
        return []
    years = df[year_col].dropna().astype(int) if df[year_col].notna().any() else pd.Series(dtype=int)
    if years.empty:
        return []
    min_y = int(years.min()); max_y = int(years.max())
    windows = []
    start = min_y
    while start <= max_y:
        end = start + window_size - 1
        slice_df = df[(df[year_col].notna()) & (df[year_col].astype(int) >= start) & (df[year_col].astype(int) <= end)]
        windows.append((start, end, slice_df))
        start += stride
    return windows

def windowed_clustering(df: pd.DataFrame, window_size=DEFAULT_WINDOW, stride=DEFAULT_STRIDE, co_word_params: Dict[str,Any] = None, progress_cb=None) -> Dict[str,Any]:
    if co_word_params is None:
        co_word_params = {}
    windows = make_time_windows(df, year_col='PY', window_size=window_size, stride=stride)
    results = {}
    total = len(windows)
    for i, (start, end, slice_df) in enumerate(windows):
        if progress_cb:
            progress_cb(i, total, f"Processing window {start}-{end}")
        key = f"{start}_{end}"
        if slice_df.empty:
            results[key] = None
            continue
        kw_lists = []
        for _, row in slice_df.iterrows():
            kws = combine_de_id(row)
            if not kws and pd.notna(row.get('AB')):
                kws = tokenize(str(row.get('AB')))[:20]
            cleaned = []
            for t in kws:
                cleaned.extend(tokenize(t))
            # dedupe & limit
            seen = []; out = []
            for t in cleaned:
                if t not in seen:
                    seen.append(t); out.append(t)
                if len(seen) >= 50: break
            kw_lists.append(out)
        if not kw_lists:
            results[key] = None
            continue
        vect, X, terms = build_term_doc_matrix(kw_lists, max_features=co_word_params.get("vocab_size", DEFAULT_VOCAB), min_df=co_word_params.get("min_term_freq", DEFAULT_MIN_DF))
        doc_count = X.shape[0]
        C = compute_cooccurrence(X)
        if co_word_params.get("use_association", False):
            S = normalize_association_strength(C, doc_count)
        else:
            S = normalize_salton(C)
        G = build_graph_from_similarity(S, terms, threshold=co_word_params.get("edge_threshold", DEFAULT_EDGE_THRESHOLD))
        if G.number_of_nodes() == 0:
            results[key] = None
            continue
        partition = run_clustering(G, prefer_leiden=co_word_params.get("use_leiden", True), seed=co_word_params.get("seed", DEFAULT_SEED))
        metrics = compute_callon(G, partition)
        results[key] = {"start": start, "end": end, "graph": G, "terms": list(terms), "partition": partition, "metrics": metrics}
    if progress_cb:
        progress_cb(total, total, "Done")
    return results

def jaccard_between_clusters(ra, pa, rb, pb):
    # ra/rb are graphs, pa/pb are partitions mapping node->cluster
    from collections import defaultdict
    ca = defaultdict(set); cb = defaultdict(set)
    for n,lab in pa.items():
        ca[lab].add(ra.nodes[n]['term'])
    for n,lab in pb.items():
        cb[lab].add(rb.nodes[n]['term'])
    overlaps = []
    for a_id, a_terms in ca.items():
        for b_id, b_terms in cb.items():
            u = a_terms | b_terms
            j = 0.0 if not u else len(a_terms & b_terms) / len(u)
            overlaps.append((a_id, b_id, j))
    return overlaps

def build_timeline_json(window_results: Dict[str,Any], top_k_clusters:int = 20, overlap_threshold:float = DEFAULT_OVERLAP):
    ordered = sorted([k for k in window_results.keys()])
    node_meta = []; node_index = {}; idx = 0
    for w in ordered:
        res = window_results[w]
        if not res: continue
        metrics = res["metrics"]
        top_clusters = sorted(metrics.items(), key=lambda x: -x[1]["size"])[:top_k_clusters]
        for cid, m in top_clusters:
            name = f"{w}|C{cid}"
            node_index[name] = idx
            node_meta.append({"id": name, "window": w, "cluster": cid, "size": m["size"], "density": m["density"], "centrality": m["centrality"], "top_terms": [res["graph"].nodes[n]["term"] for n in m["nodes"][:10]]})
            idx += 1
    links = []
    for i in range(len(ordered)-1):
        a = ordered[i]; b = ordered[i+1]
        ra = window_results[a]; rb = window_results[b]
        if not ra or not rb: continue
        overlaps = jaccard_between_clusters(ra["graph"], ra["partition"], rb["graph"], rb["partition"])
        for ca, cb, j in overlaps:
            if j >= overlap_threshold:
                s = f"{a}|C{ca}"; t = f"{b}|C{cb}"
                if s in node_index and t in node_index:
                    links.append({"source": s, "target": t, "value": float(j)})
    return {"nodes": node_meta, "links": links, "windows": ordered}

def build_alluvial_figure(timeline_json):
    nodes = timeline_json["nodes"]; links = timeline_json["links"]; windows = timeline_json["windows"]
    from collections import defaultdict
    nodes_by_window = defaultdict(list)
    for n in nodes:
        nodes_by_window[n["window"]].append(n)
    positions = {}
    x_map = {w:i for i,w in enumerate(windows)}
    for w in windows:
        items = nodes_by_window.get(w, [])
        items = sorted(items, key=lambda x: -x.get("size",0))
        total = sum([max(1, it.get("size",1)) for it in items]) or 1
        y = 0.0
        for it in items:
            h = max(0.02, it.get("size",1)/total)
            positions[it["id"]] = (x_map[w], y + h/2.0, h)
            y += h
    flow_traces = []
    for l in links:
        s = l["source"]; t = l["target"]; v = l["value"]
        if s not in positions or t not in positions: continue
        x0,y0,h0 = positions[s]; x1,y1,h1 = positions[t]
        width = max(0.002, v)
        xs = [x0, x0 + 0.2*(x1-x0), x1 - 0.2*(x1-x0), x1, x1 - 0.2*(x1-x0), x0 + 0.2*(x1-x0)]
        ys = [y0 - width/2, y0 - width/2 + 0.02*(y1-y0), y1 - width/2 - 0.02*(y1-y0), y1 - width/2, y1 + width/2, y0 + width/2]
        flow_traces.append(go.Scatter(x=xs, y=ys, fill="toself", mode="none", opacity=0.6, hoverinfo="text", text=f"{s} → {t}<br>Jaccard={v:.3f}"))
    node_traces = []
    for w in windows:
        items = nodes_by_window.get(w, [])
        x=[]; y=[]; sizes=[]; hover=[]
        for it in items:
            xp, yp, h = positions[it["id"]]
            x.append(xp); y.append(yp); sizes.append(it.get("size",1))
            hover.append(f"{it['id']}<br>size:{it.get('size')}<br>top_terms:{', '.join(it.get('top_terms',[])[:6])}")
        if x:
            node_traces.append(go.Scatter(x=x, y=y, mode="markers+text", marker=dict(size=[8+6*np.log1p(s) for s in sizes]), text=[n.split("|")[-1] for n in [it["id"] for it in items]], textposition="middle right", hovertext=hover, hoverinfo="text", showlegend=False))
    fig = go.Figure()
    for t in flow_traces: fig.add_trace(t)
    for t in node_traces: fig.add_trace(t)
    fig.update_layout(title="Alluvial thematic evolution", xaxis=dict(tickmode="array", tickvals=list(range(len(windows))), ticktext=windows), yaxis=dict(showticklabels=False), height=700, margin=dict(l=80, r=40, t=80, b=80))
    fig.update_xaxes(range=[-0.5, len(windows)-0.5])
    return fig

# -------------------------
# VOSviewer exports
# -------------------------
def build_vosviewer_files(G: nx.Graph, partition: Dict[int,int]):
    nodes_lines = ["Id\tLabel"]
    for n,d in G.nodes(data=True):
        nodes_lines.append(f"{n}\t{d.get('term','')}")
    edges_lines = ["Source\tTarget\tWeight"]
    for u,v,d in G.edges(data=True):
        edges_lines.append(f"{u}\t{v}\t{d.get('weight',1.0)}")
    clu_lines = [str(partition.get(n,0)) for n in sorted(G.nodes())]
    return "\n".join(nodes_lines), "\n".join(edges_lines), "\n".join(clu_lines)

# -------------------------
# UI helpers: logs & progress
# -------------------------
def init_log():
    if "log_lines" not in st.session_state:
        st.session_state["log_lines"] = []
def log(msg: str):
    init_log()
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    st.session_state["log_lines"].append(f"[{timestamp}] {msg}")
    # keep last 200 lines
    if len(st.session_state["log_lines"]) > 200:
        st.session_state["log_lines"] = st.session_state["log_lines"][-200:]

def progress_callback_factory(st_progress, st_text):
    def cb(i, total, msg):
        if total > 0:
            st_progress.progress(min(1.0, float(i)/float(total)))
        st_text.text(msg)
        log(msg)
    return cb

# -------------------------
# Sidebar navigation
# -------------------------
st.sidebar.title("BiblioRefine")
st.sidebar.markdown("Pipeline: Upload → Enrich → Analyze → Evolve → Export")
deps = dependency_report()
st.sidebar.subheader("Dependency status")
for k,v in deps.items():
    st.sidebar.write(f"{k}: {'✅' if v else '⚠️'}")

nav = st.sidebar.radio("Navigation", ["Upload & Merge", "Enrichment", "Analysis", "Evolution", "Export & Logs"])

# Sidebar parameters for analysis
st.sidebar.subheader("Analysis parameters")
vocab_size = st.sidebar.number_input("Max vocabulary size", min_value=200, max_value=20000, value=DEFAULT_VOCAB, step=100)
min_df = st.sidebar.number_input("Min term doc frequency", min_value=1, max_value=50, value=DEFAULT_MIN_DF)
edge_threshold = st.sidebar.slider("Edge threshold (similarity)", 0.0, 1.0, DEFAULT_EDGE_THRESHOLD, 0.01)
use_association = st.sidebar.checkbox("Use Association Strength normalization (VOS)", value=False)
prefer_leiden = st.sidebar.checkbox("Prefer Leiden clustering if available", value=True)
seed = int(st.sidebar.number_input("Random seed", min_value=0, max_value=2**31-1, value=DEFAULT_SEED))
window_size = int(st.sidebar.number_input("Window size (years)", min_value=1, max_value=10, value=DEFAULT_WINDOW))
stride = int(st.sidebar.number_input("Window stride (years)", min_value=1, max_value=10, value=DEFAULT_STRIDE))
overlap_thresh = st.sidebar.slider("Jaccard threshold for evolution links", 0.0, 1.0, DEFAULT_OVERLAP, 0.01)

# -------------------------
# Main: Upload & Merge
# -------------------------
if nav == "Upload & Merge":
    st.header("Step 1 — Upload & Merge Data")
    st.markdown("Upload Web of Science (.xlsx) and/or Scopus (.csv). Optional extras: .bib, .ris, .xml, .json (OpenAlex).")
    col1, col2 = st.columns(2)
    with col1:
        wos_file = st.file_uploader("Upload WoS (.xlsx)", type="xlsx")
    with col2:
        scopus_file = st.file_uploader("Upload Scopus (.csv)", type="csv")
    extra_files = st.file_uploader("Optional extras (.bib/.ris/.xml/.json)", accept_multiple_files=True)

    if st.button("Load & Prepare (sample to 10k if larger)"):
        log("Starting load & prepare step.")
        dfs = []
        # read WoS
        if wos_file:
            try:
                dfw = pd.read_excel(wos_file)
                dfw = standardize_df(dfw, "WoSCC")
                dfs.append(dfw)
                log(f"Loaded WoS — {dfw.shape[0]} rows")
            except Exception as e:
                st.error(f"Failed to read WoS file: {e}")
                log(f"Failed to read WoS file: {e}")
        # read Scopus
        if scopus_file:
            try:
                dfsc = pd.read_csv(scopus_file, low_memory=False)
                dfsc = standardize_df(dfsc, "Scopus")
                dfs.append(dfsc)
                log(f"Loaded Scopus — {dfsc.shape[0]} rows")
            except Exception as e:
                st.error(f"Failed to read Scopus file: {e}")
                log(f"Failed to read Scopus file: {e}")
        # parse extras
        parsed = []
        if extra_files:
            for f in extra_files:
                name = f.name.lower()
                try:
                    b = f.read()
                    if name.endswith(".bib"):
                        dfx = parse_bib_bytes(b)
                    elif name.endswith(".ris"):
                        dfx = parse_ris_bytes(b)
                    elif name.endswith(".xml"):
                        dfx = parse_endnote_xml_bytes(b)
                    elif name.endswith(".json"):
                        dfx = parse_openalex_json_bytes(b)
                    else:
                        dfx = pd.DataFrame()
                    if not dfx.empty:
                        parsed.append(dfx)
                        log(f"Parsed {f.name}: {dfx.shape[0]} rows")
                except Exception as e:
                    st.warning(f"Failed to parse {f.name}: {e}")
                    log(f"Failed to parse {f.name}: {e}")
        if parsed:
            dfs.append(pd.concat(parsed, ignore_index=True, sort=False))
        if not dfs:
            st.error("No files loaded — upload at least one.")
            log("No files loaded.")
        else:
            merged = pd.concat(dfs, ignore_index=True, sort=False)
            if "DI" not in merged.columns:
                merged["DI"] = pd.NA
            merged["DI"] = merged["DI"].apply(normalize_doi)
            # ensure core fields
            CORE = ["TI","AU","AF","SO","PY","DE","AB","C1","FX","CR","NR","DI"]
            for c in CORE:
                if c not in merged.columns:
                    merged[c] = pd.NA
            # dedupe: prefer DI then title+year
            merged["title_key"] = merged.apply(lambda r: title_key(r.get("TI"), r.get("PY")), axis=1)
            # safe sort: put records with DI present first
            if "DI" in merged.columns:
                merged = merged.assign(_missing_doi = merged["DI"].isna().astype(int))
                merged = merged.sort_values(by="_missing_doi").drop(columns=["_missing_doi"])
            # drop duplicates by DI if DI exists
            if "DI" in merged.columns:
                merged = merged.drop_duplicates(subset=["DI"], keep="first")
            merged = merged.drop_duplicates(subset=["title_key"], keep="first").drop(columns=["title_key"]).reset_index(drop=True)
            # sample if larger than MAX_RECORDS
            if len(merged) > MAX_RECORDS:
                log(f"Dataset too large ({len(merged)} rows) — sampling {MAX_RECORDS} records for performance.")
                merged = merged.sample(n=MAX_RECORDS, random_state=seed).reset_index(drop=True)
            st.session_state["merged"] = merged
            st.success(f"Merged dataset prepared: {len(merged)} records")
            log(f"Merged dataset prepared: {len(merged)} records")
            st.dataframe(merged.head(5))

# -------------------------
# Enrichment
# -------------------------
if nav == "Enrichment":
    st.header("Step 2 — Optional Enrichment (OpenAlex)")
    st.markdown("Fill missing title/abstract/year/authors via OpenAlex (requires DOI). No key required but API-rate limited.")
    if "merged" not in st.session_state:
        st.warning("Please upload and prepare data in 'Upload & Merge' first.")
    else:
        if st.button("Run OpenAlex enrichment (by DOI)"):
            merged = st.session_state["merged"]
            API = "https://api.openalex.org/works/doi:"
            rows = list(merged.index)
            prog = st.progress(0); status = st.empty()
            updated = 0
            for i, idx in enumerate(rows):
                doi = merged.at[idx, "DI"]
                if pd.isna(doi) or doi == "":
                    prog.progress((i+1)/len(rows)); status.text(f"Skipped {i+1}/{len(rows)} (no DOI)"); continue
                try:
                    r = requests.get(API + doi, timeout=15)
                    if r.status_code == 200:
                        meta = r.json()
                        if pd.isna(merged.at[idx,"TI"]) and meta.get("title"):
                            merged.at[idx,"TI"] = meta.get("title")
                        if pd.isna(merged.at[idx,"AB"]) and meta.get("abstract"):
                            merged.at[idx,"AB"] = meta.get("abstract")
                        if pd.isna(merged.at[idx,"PY"]) and meta.get("publication_year"):
                            merged.at[idx,"PY"] = meta.get("publication_year")
                        if pd.isna(merged.at[idx,"AU"]) and "authorships" in meta:
                            au = "; ".join([a["author"]["display_name"] for a in meta["authorships"] if a.get("author")])
                            merged.at[idx,"AU"] = au; merged.at[idx,"AF"] = au
                        updated += 1
                except Exception as e:
                    log(f"OpenAlex fetch error for DOI {doi}: {e}")
                prog.progress((i+1)/len(rows)); status.text(f"Processed {i+1}/{len(rows)}")
                time.sleep(0.05)
            st.session_state["merged"] = merged
            st.success(f"OpenAlex enrichment attempted — approx {updated} records updated.")
            log(f"OpenAlex enrichment attempted — approx {updated} records updated.")

# -------------------------
# Analysis (co-word, clustering)
# -------------------------
if nav == "Analysis":
    st.header("Step 3 — Co-word analysis & clustering")
    st.markdown("Build co-word network from DE+ID, normalize, cluster (Leiden preferred) and compute cluster metrics.")
    if "merged" not in st.session_state:
        st.warning("Please upload & prepare dataset first.")
    else:
        if st.button("Run co-word pipeline & clustering"):
            merged = st.session_state["merged"]
            log("Building keyword lists (DE + ID) and tokenizing...")
            kw_lists = []
            for _, row in merged.iterrows():
                kws = combine_de_id(row)
                if (not kws) and pd.notna(row.get("AB")):
                    kws = tokenize(str(row.get("AB")))[:20]
                cleaned = []
                for t in kws:
                    cleaned.extend(tokenize(t))
                seen = []; out = []
                for t in cleaned:
                    if t not in seen:
                        seen.append(t); out.append(t)
                    if len(seen) >= 50: break
                kw_lists.append(out)
            st.write(f"Prepared keywords for {len(kw_lists)} documents.")
            log(f"Prepared keywords for {len(kw_lists)} documents.")
            # term-doc
            st.info("Building term-document matrix (sparse)...")
            vect, X, terms = build_term_doc_matrix(kw_lists, max_features=vocab_size, min_df=min_df)
            st.write(f"Matrix: {X.shape}; vocab: {len(terms)}")
            log(f"Term-doc matrix shape: {X.shape}; vocab: {len(terms)}")
            # co-occurrence
            st.info("Computing co-occurrence and normalization...")
            C = compute_cooccurrence(X)
            if use_association:
                S = normalize_association_strength(C, X.shape[0])
                norm_label = "Association Strength"
            else:
                S = normalize_salton(C)
                norm_label = "Salton (cosine)"
            st.write(f"Normalization: {norm_label}")
            log(f"Normalization used: {norm_label}")
            # graph
            st.info("Building NetworkX graph from similarity matrix...")
            G = build_graph_from_similarity(S, terms, threshold=edge_threshold)
            st.write(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
            log(f"Graph built: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
            # clustering
            st.info("Clustering (Leiden preferred; Louvain fallback)...")
            try:
                partition = run_clustering(G, prefer_leiden=prefer_leiden, seed=seed)
            except Exception as e:
                log(f"Clustering error: {e}")
                partition = cluster_louvain(G, seed=seed)
            if prefer_leiden and not LEIDEN_AVAILABLE:
                st.warning("Leiden not available — used Louvain instead. To enable Leiden, install python-igraph and leidenalg on Python 3.10/3.11.")
            st.session_state["global_graph"] = G
            st.session_state["global_terms"] = list(terms)
            st.session_state["global_partition"] = partition
            # metrics
            metrics = compute_callon(G, partition)
            st.session_state["global_metrics"] = metrics
            df_clusters = pd.DataFrame([{"cluster":cid,"size":m["size"],"density":m["density"],"centrality":m["centrality"],"top_terms":", ".join([G.nodes[n]["term"] for n in m["nodes"][:10]])} for cid,m in metrics.items()]).sort_values("size", ascending=False)
            st.subheader("Cluster summary")
            st.dataframe(df_clusters.head(200))
            log("Clustering done and metrics computed.")

# -------------------------
# Evolution (time-slicing & alluvial)
# -------------------------
if nav == "Evolution":
    st.header("Step 4 — Thematic evolution (time-sliced)")
    st.markdown("Slices dataset into windows (default 5-year), clusters each window, computes Jaccard overlaps and renders alluvial chart.")
    if "merged" not in st.session_state:
        st.warning("Please upload & prepare dataset and run analysis first.")
    else:
        if st.button("Run thematic evolution (windowed clustering + alluvial)"):
            merged = st.session_state["merged"]
            # windowed clustering
            st.info("Running windowed clustering across time windows...")
            prog = st.progress(0); status = st.empty()
            co_params = {"vocab_size": vocab_size, "min_term_freq": min_df, "edge_threshold": edge_threshold, "use_association": use_association, "use_leiden": prefer_leiden, "seed": seed}
            window_results = windowed_clustering(merged, window_size=window_size, stride=stride, co_word_params=co_params, progress_cb=progress_callback_factory(prog, status))
            # filter empty
            window_results = {k:v for k,v in window_results.items() if v is not None}
            if not window_results:
                st.warning("No windows produced — probably missing PY values or slices too narrow.")
                log("Windowed clustering produced no results.")
            else:
                st.success(f"Windowed clustering complete — {len(window_results)} windows.")
                log(f"Windowed clustering complete — {len(window_results)} windows.")
                # timeline json and alluvial
                timeline_json = build_timeline_json(window_results, top_k_clusters=20, overlap_threshold=overlap_thresh)
                st.session_state["timeline"] = timeline_json
                fig = build_alluvial_figure(timeline_json)
                st.plotly_chart(fig, use_container_width=True)
                log("Alluvial chart generated for thematic evolution.")

# -------------------------
# Export & Logs
# -------------------------
if nav == "Export & Logs":
    st.header("Step 5 — Export & Logs")
    st.markdown("Download CSVs, GEXF, VOSviewer files, timeline JSON, and ZIP package. Logs are shown below.")
    if "global_graph" in st.session_state:
        G = st.session_state["global_graph"]; partition = st.session_state.get("global_partition", {}); metrics = st.session_state.get("global_metrics", {})
        # term->cluster CSV
        rows = [{"term": G.nodes[n]["term"], "node_idx": n, "cluster": int(partition.get(n,0))} for n in G.nodes()]
        df_term_cluster = pd.DataFrame(rows)
        st.download_button("Download term->cluster (CSV)", df_term_cluster.to_csv(index=False).encode("utf-8"), "term_clusters.csv")
        # edges CSV
        edge_rows = [{"source": u, "target": v, "weight": d.get("weight",1.0), "source_term": G.nodes[u]["term"], "target_term": G.nodes[v]["term"]} for u,v,d in G.edges(data=True)]
        df_edges = pd.DataFrame(edge_rows)
        st.download_button("Download edges (CSV)", df_edges.to_csv(index=False).encode("utf-8"), "co_word_edges.csv")
        # cluster stats
        cs = [{"cluster": cid, "size": m["size"], "density": m["density"], "centrality": m["centrality"]} for cid,m in metrics.items()]
        df_cs = pd.DataFrame(cs).sort_values("size", ascending=False)
        st.download_button("Download cluster stats (CSV)", df_cs.to_csv(index=False).encode("utf-8"), "cluster_stats.csv")
        # gexf
        try:
            gexport = nx.Graph()
            for n,d in G.nodes(data=True):
                gexport.add_node(n, term=d.get("term",""), cluster=int(partition.get(n,0)))
            for u,v,d in G.edges(data=True):
                gexport.add_edge(u, v, weight=float(d.get("weight",1.0)))
            bio = BytesIO(); nx.write_gexf(gexport, bio); bio.seek(0)
            st.download_button("Download graph (GEXF)", bio, "co_word_graph.gexf")
        except Exception as e:
            st.warning(f"GEXF export failed: {e}")
        # VOSviewer files
        vn, ve, vc = build_vosviewer_files(G, partition)
        st.download_button("Download VOSviewer nodes (.txt)", vn.encode("utf-8"), "vos_nodes.txt")
        st.download_button("Download VOSviewer edges (.txt)", ve.encode("utf-8"), "vos_edges.txt")
        st.download_button("Download VOSviewer cluster (.clu)", vc.encode("utf-8"), "vos_clusters.clu")
    else:
        st.info("No global graph results available yet. Run 'Analysis' first.")

    # timeline JSON
    if "timeline" in st.session_state:
        st.download_button("Download timeline JSON (Cytoscape)", json.dumps(st.session_state["timeline"], indent=2), "timeline_cytoscape.json")
    # merged cleaned dataset
    if "merged" in st.session_state:
        st.download_button("Download merged_cleaned.csv", st.session_state["merged"].to_csv(index=False).encode("utf-8"), "merged_cleaned.csv")
    # ZIP package
    if st.button("Prepare ZIP package of all outputs"):
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            try:
                if "merged" in st.session_state:
                    zf.writestr("merged_cleaned.csv", st.session_state["merged"].to_csv(index=False))
            except Exception:
                pass
            try:
                if "global_graph" in st.session_state:
                    zf.writestr("term_clusters.csv", df_term_cluster.to_csv(index=False))
                    zf.writestr("co_word_edges.csv", df_edges.to_csv(index=False))
                    zf.writestr("cluster_stats.csv", df_cs.to_csv(index=False))
                    zf.writestr("vos_nodes.txt", vn)
                    zf.writestr("vos_edges.txt", ve)
                    zf.writestr("vos_clusters.clu", vc)
            except Exception:
                pass
            if "timeline" in st.session_state:
                zf.writestr("timeline_cytoscape.json", json.dumps(st.session_state["timeline"], indent=2))
            try:
                with open("run_metadata.json","r",encoding="utf-8") as fh:
                    zf.writestr("run_metadata.json", fh.read())
            except Exception:
                pass
        buf.seek(0)
        st.download_button("Download ZIP package", buf, "biblio_analysis_results.zip")

    # Logs display
    st.subheader("Processing logs")
    init_log()
    log_area = st.empty()
    log_text = "\n".join(st.session_state.get("log_lines", []))
    log_area.code(log_text, language="text")

# -------------------------
# End of app
# -------------------------
st.sidebar.markdown("---")
st.sidebar.caption("BiblioRefine — Streamlit optimized pipeline. For Leiden support install python-igraph and leidenalg on Python 3.10/3.11. This app caps processing to 10k records to keep UX responsive.")
