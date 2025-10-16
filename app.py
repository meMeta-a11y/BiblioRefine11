import streamlit as st
st.set_page_config(page_title="📚 BiblioRefine (Optimized, ≤10k)", layout="wide")

import pandas as pd, numpy as np, requests, time, json, zipfile, hashlib, datetime, subprocess, io, re, os
from io import BytesIO
from typing import List, Dict, Any

# plotting
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# parsing helpers
try:
    import bibtexparser
except Exception:
    bibtexparser = None
try:
    import rispy
except Exception:
    rispy = None
try:
    from lxml import etree as ET
except Exception:
    import xml.etree.ElementTree as ET

# NLP & tokens
import nltk
try:
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('stopwords')
from nltk.corpus import stopwords

# ML & sparse
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

# graph / clustering
import networkx as nx
try:
    import community as community_louvain
except Exception:
    community_louvain = None

LEIDEN_AVAILABLE = False
try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except Exception:
    LEIDEN_AVAILABLE = False

# Basic stopwords
DEFAULT_STOPWORDS = set(stopwords.words('english'))
CUSTOM_STOPWORDS = set(['study','studies','analysis','based','using','approach'])
STOPWORDS = DEFAULT_STOPWORDS.union(CUSTOM_STOPWORDS)

# -------------------------
# Constants & UI defaults
# -------------------------
MAX_RECORDS = 10000  # cap
DEFAULT_VOCAB = 5000
DEFAULT_MIN_DF = 3
DEFAULT_EDGE_THRESHOLD = 0.05
DEFAULT_WINDOW = 5  # years
DEFAULT_STRIDE = 5
DEFAULT_OVERLAP = 0.05
DEFAULT_SEED = 42

# -------------------------
# Utility functions
# -------------------------
MASTER_MAP = {
    'Publication Type': 'PT', 'Document Type': 'DT', 'Language': 'LA',
    'Publication Year': 'PY', 'Year': 'PY', 'DOI': 'DI', 'DOI Link': 'DI',
    'Source': 'SO', 'Source title': 'SO', 'Source Title': 'SO',
    'Authors': 'AU', 'Author full names': 'AF', 'Author Full Names': 'AF',
    'Affiliations': 'C1', 'Addresses': 'C1', 'Title': 'TI', 'Article Title': 'TI',
    'Abstract': 'AB', 'Author Keywords': 'DE', 'Keywords Plus': 'ID',
    'Cited References': 'CR', 'References': 'CR', 'Cited Reference Count': 'NR',
    'Volume': 'VL', 'Issue': 'IS', 'Page count': 'PG', 'Page Count': 'PG',
    'Funding Texts': 'FX', 'Funding Text': 'FX'
}

def standardize(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    df = df.rename(columns={c: MASTER_MAP[c] for c in df.columns if c in MASTER_MAP})
    df = df.loc[:, ~df.columns.duplicated()]
    df['Source'] = source_label
    return df

def normalize_doi(x: Any) -> Any:
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().lower()
    s = s.replace('https://doi.org/','').replace('http://doi.org/','').replace('doi:','').strip()
    return s if s else pd.NA

def title_key(title, year):
    if pd.isna(title):
        return ''
    t = re.sub(r'\s+', ' ', str(title).strip().lower())
    y = ''
    try:
        y = str(int(year))
    except Exception:
        y = ''
    return f"{t}___{y}"

# -------------------------
# Parse extra formats (cached)
# -------------------------
@st.cache_data(show_spinner=False)
def parse_bibtex_bytes(b: bytes):
    if bibtexparser is None:
        return pd.DataFrame()
    try:
        txt = b.decode('utf-8', errors='ignore')
        bib_db = bibtexparser.loads(txt)
        rows=[]
        for e in bib_db.entries:
            r={}
            r['TI']=e.get('title'); r['AU']='; '.join([a.strip() for a in re.split(r'\s+and\s+', e.get('author',''))]) if e.get('author') else pd.NA
            r['AF']=r['AU']; r['PY']=e.get('year'); r['SO']=e.get('journal') or e.get('booktitle'); r['DI']=e.get('doi') or e.get('url')
            r['AB']=e.get('abstract'); r['DE']=e.get('keywords')
            rows.append(r)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def parse_ris_bytes(b: bytes):
    if rispy is None:
        return pd.DataFrame()
    try:
        lines = b.decode('utf-8', errors='ignore').splitlines()
        entries = rispy.load(lines)
        rows=[]
        for e in entries:
            r={'TI': e.get('title'), 'AU': '; '.join(e.get('authors') or []), 'AF': '; '.join(e.get('authors') or []),
               'PY': e.get('year') or e.get('date'), 'SO': e.get('journal_name') or e.get('publication_name'),
               'DI': e.get('doi'), 'AB': e.get('abstract'), 'DE': '; '.join(e.get('keywords') or [])}
            rows.append(r)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def parse_endnote_xml_bytes(b: bytes):
    try:
        txt = b.decode('utf-8', errors='ignore')
        root = ET.fromstring(txt)
        rows=[]
        for record in root.findall('.//record'):
            r={}
            title = record.find('.//titles/title')
            if title is not None and title.text: r['TI']=title.text
            authors = [a.text for a in record.findall('.//contributors/authors/author') if a is not None and a.text]
            if authors: r['AU']= '; '.join(authors); r['AF']=r['AU']
            year = record.find('.//dates/year')
            if year is not None and year.text: r['PY']=year.text
            journal = record.find('.//periodical/full_title') or record.find('.//periodical/title')
            if journal is not None and journal.text: r['SO']=journal.text
            doi = record.find('.//electronic-resource-num')
            if doi is not None and doi.text: r['DI']=doi.text
            abstract = record.find('.//abstract'); 
            if abstract is not None and abstract.text: r['AB']=abstract.text
            kws = [k.text for k in record.findall('.//keywords/keyword') if k.text]; 
            if kws: r['DE']='; '.join(kws)
            rows.append(r)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def parse_openalex_json_bytes(b: bytes):
    try:
        txt = b.decode('utf-8', errors='ignore')
        data = json.loads(txt)
        rows=[]
        items = data.get('results') if isinstance(data, dict) and 'results' in data else (data if isinstance(data, list) else [data])
        for item in items:
            r={'TI': item.get('title')}
            ab = item.get('abstract_inverted_index') or item.get('abstract')
            if isinstance(ab, dict):
                words=[] 
                for w in sorted(ab.keys(), key=lambda k: min(ab[k]) if ab[k] else 0): words.append(w)
                r['AB']=' '.join(words)
            else:
                r['AB']=ab
            authors = item.get('authorships') or []
            if authors:
                r['AU']='; '.join([a.get('author',{}).get('display_name') for a in authors if a.get('author')])
                r['AF']=r['AU']
            r['PY']=item.get('publication_year'); r['SO']=item.get('host_venue',{}).get('display_name'); r['DI']=item.get('doi')
            concepts = item.get('concepts') or []
            if concepts: r['DE']='; '.join([c.get('display_name') for c in concepts if c.get('display_name')])
            rows.append(r)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# -------------------------
# Tokenization & keyword combination
# -------------------------
def tokenize(text: str) -> List[str]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    s = str(text).strip().lower()
    s = re.sub(r'[;/\|]', ' ', s)
    toks = [t.strip() for t in re.split(r'[\s,;|]+', s) if t.strip()]
    toks = [t for t in toks if len(t) > 1 and t not in STOPWORDS]
    return toks

def combine_DE_ID(row, max_terms=50) -> List[str]:
    parts=[]
    for col in ['DE','ID']:
        if col in row and pd.notna(row.get(col)):
            parts.extend([t.strip() for t in re.split(r'[;,\|]+', str(row.get(col))) if t.strip()])
    seen=[]
    out=[]
    for p in parts:
        if p not in seen:
            seen.append(p); out.append(p)
        if len(seen) >= max_terms: break
    return out

# -------------------------
# Cached heavy computations
# -------------------------
@st.cache_data(show_spinner=False)
def build_term_doc_matrix(docs_tokens: List[List[str]], max_features:int, min_df:int):
    # join tokens with underscore to keep multi-word as single token
    docs_txt = [' '.join([t.replace(' ','_') for t in toks]) for toks in docs_tokens]
    vect = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b', max_features=max_features, min_df=min_df)
    X = vect.fit_transform(docs_txt)
    terms = vect.get_feature_names_out()
    return vect, X, terms

@st.cache_data(show_spinner=False)
def compute_cooccurrence(X):
    return (X.T @ X).tocsr()

# -------------------------
# Normalization functions
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
    for r,c,v in zip(rows,cols,data):
        denom = (f[r] * f[c]) / max(1, doc_count)
        vals.append(v / denom if denom != 0 else 0.0)
    S = sp.coo_matrix((vals, (rows, cols)), shape=C.shape).tocsr()
    S.setdiag(0.0)
    return S

# -------------------------
# Build NetworkX graph from similarity matrix
# -------------------------
def build_graph_from_similarity(S, terms, threshold=0.05, top_k=None):
    S = S.tocsr()
    n = S.shape[0]
    G = nx.Graph()
    for i,t in enumerate(terms):
        G.add_node(i, term=t)
    if top_k is None:
        S_triu = sp.triu(S, k=1).tocoo()
        for i,j,v in zip(S_triu.row, S_triu.col, S_triu.data):
            if v >= threshold:
                G.add_edge(i,j,weight=float(v))
    else:
        for i in range(n):
            row = S.getrow(i).toarray().ravel()
            row[i]=0.0
            if np.count_nonzero(row)==0: continue
            k = min(len(row)-1, top_k)
            idxs = np.argpartition(-row, k)[:k]
            for j in idxs:
                if row[j] >= threshold:
                    G.add_edge(i,j,weight=float(row[j]))
    return G

# -------------------------
# Clustering: Leiden or Louvain fallback
# -------------------------
def cluster_leiden(G:nx.Graph, seed:int=42) -> Dict[int,int]:
    # convert to igraph
    mapping = {n:i for i,n in enumerate(G.nodes())}
    rev = {i:n for n,i in mapping.items()}
    g = ig.Graph()
    g.add_vertices(len(mapping))
    g.vs['name'] = [G.nodes[n]['term'] for n in G.nodes()]
    edges=[]
    weights=[]
    for u,v,d in G.edges(data=True):
        edges.append((mapping[u], mapping[v])); weights.append(d.get('weight',1.0))
    if not edges:
        return {n:0 for n in G.nodes()}
    g.add_edges(edges)
    g.es['weight'] = weights
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights='weight', seed=seed)
    membership = part.membership
    return {rev[i]: membership[i] for i in range(len(membership))}

def cluster_louvain(G:nx.Graph, seed:int=42) -> Dict[int,int]:
    if community_louvain is None:
        return {n:0 for n in G.nodes()}
    return community_louvain.best_partition(G, weight='weight', random_state=seed)

def run_clustering(G:nx.Graph, use_leiden:bool, seed:int=42):
    if use_leiden and LEIDEN_AVAILABLE:
        return cluster_leiden(G, seed=seed)
    else:
        return cluster_louvain(G, seed=seed)

# -------------------------
# Callon metrics (density & centrality)
# -------------------------
def compute_callon(G:nx.Graph, partition:Dict[int,int]):
    from collections import defaultdict
    clusters=defaultdict(list)
    for n,c in partition.items(): clusters[c].append(n)
    metrics={}
    for cid, nodes in clusters.items():
        internal=0.0; external=0.0; n=len(nodes)
        for i,u in enumerate(nodes):
            for v in nodes[i+1:]:
                if G.has_edge(u,v):
                    internal += G[u][v].get('weight',1.0)
        denom = (n*(n-1)/2) if n>1 else 1
        density = internal / denom if denom>0 else 0.0
        for u in nodes:
            for nbr in G.neighbors(u):
                if nbr not in nodes:
                    external += G[u][nbr].get('weight',1.0)
        metrics[cid] = {'nodes':nodes, 'size':n, 'density':density, 'centrality':external}
    return metrics

# -------------------------
# Time windows & windowed clustering
# -------------------------
def make_windows(df:pd.DataFrame, year_col='PY', window_size=5, stride=5):
    years = df[year_col].dropna().astype(int) if df[year_col].notna().any() else pd.Series([], dtype=int)
    if years.empty: return []
    min_y=int(years.min()); max_y=int(years.max())
    windows=[]
    start=min_y
    while start <= max_y:
        end = start + window_size -1
        slice_df = df[(df[year_col].notna()) & (df[year_col].astype(int) >= start) & (df[year_col].astype(int) <= end)]
        windows.append((start,end,slice_df))
        start += stride
    return windows

def windowed_clustering(df:pd.DataFrame, window_size=5, stride=5, co_word_params:dict=None, progress_callback=None):
    if co_word_params is None: co_word_params={}
    windows = make_windows(df, window_size=window_size, stride=stride)
    results={}
    for i,(start,end,slice_df) in enumerate(windows):
        if progress_callback:
            progress_callback(i, len(windows), f"Processing window {start}-{end}")
        key=f"{start}_{end}"
        if slice_df.empty:
            results[key]=None; continue
        # prepare kw lists
        kw_lists=[]
        for _,row in slice_df.iterrows():
            kws = combine_DE_ID(row)
            if (not kws) and pd.notna(row.get('AB')):
                kws = tokenize(str(row.get('AB')))[:20]
            cleaned=[]
            for t in kws:
                cleaned.extend(tokenize(t))
            # dedupe
            seen=[]; out=[]
            for t in cleaned:
                if t not in seen:
                    seen.append(t); out.append(t)
                if len(seen)>=50: break
            kw_lists.append(out)
        if not kw_lists:
            results[key]=None; continue
        vect, X, terms = build_term_doc_matrix(kw_lists, max_features=co_word_params.get('vocab_size',DEFAULT_VOCAB), min_df=co_word_params.get('min_term_freq',DEFAULT_MIN_DF))
        doc_count = X.shape[0]
        C = compute_cooccurrence(X)
        if co_word_params.get('use_association', False):
            S = normalize_association_strength(C, doc_count)
        else:
            S = normalize_salton(C)
        G = build_graph_from_similarity(S, terms, threshold=co_word_params.get('edge_threshold', DEFAULT_EDGE_THRESHOLD))
        if G.number_of_nodes() == 0:
            results[key]=None; continue
        partition = run_clustering(G, use_leiden=co_word_params.get('use_leiden', True), seed=co_word_params.get('seed', DEFAULT_SEED))
        metrics = compute_callon(G, partition)
        results[key] = {'start':start,'end':end,'graph':G,'terms':list(terms),'partition':partition,'metrics':metrics}
    if progress_callback: progress_callback(len(windows), len(windows), "Done")
    return results

# -------------------------
# Build timeline JSON & Alluvial figure
# -------------------------
def build_timeline(window_results:Dict[str, Any], top_k=20, overlap_threshold=0.05):
    ordered = sorted([k for k in window_results.keys()])
    node_meta=[]; node_index={}; idx=0
    for w in ordered:
        res = window_results[w]
        if not res: continue
        metrics = res['metrics']
        top_clusters = sorted(metrics.items(), key=lambda x: -x[1]['size'])[:top_k]
        for cid, m in top_clusters:
            nid = f"{w}|C{cid}"
            node_index[nid]=idx
            node_meta.append({'id':nid,'window':w,'cluster':cid,'size':m['size'],'density':m['density'],'centrality':m['centrality'],'top_terms':[res['graph'].nodes[n]['term'] for n in m['nodes'][:10]]})
            idx+=1
    links=[]
    for i in range(len(ordered)-1):
        a=ordered[i]; b=ordered[i+1]
        ra=window_results[a]; rb=window_results[b]
        if not ra or not rb: continue
        # compute jaccard overlaps
        # build cluster->term sets
        ca={}
        cb={}
        for n,l in ra['partition'].items():
            ca.setdefault(l,set()).add(ra['graph'].nodes[n]['term'])
        for n,l in rb['partition'].items():
            cb.setdefault(l,set()).add(rb['graph'].nodes[n]['term'])
        for a_id, a_terms in ca.items():
            for b_id, b_terms in cb.items():
                u = a_terms | b_terms
                if not u: j=0.0
                else: j = len(a_terms & b_terms)/len(u)
                if j >= overlap_threshold:
                    s=f"{a}|C{a_id}"; t=f"{b}|C{b_id}"
                    if s in node_index and t in node_index:
                        links.append({'source':s,'target':t,'value':float(j)})
    timeline={'nodes':node_meta,'links':links,'windows':ordered}
    return timeline

def build_alluvial_plot(timeline_json):
    nodes=timeline_json['nodes']; links=timeline_json['links']; windows=timeline_json['windows']
    from collections import defaultdict
    nodes_by_win=defaultdict(list)
    for n in nodes: nodes_by_win[n['window']].append(n)
    # positions
    positions={}
    x_map={w:i for i,w in enumerate(windows)}
    for w in windows:
        items = nodes_by_win.get(w,[])
        items = sorted(items, key=lambda x:-x.get('size',0))
        total = sum([max(1,it.get('size',1)) for it in items]) or 1
        y=0.0
        for it in items:
            h = max(0.02, it.get('size',1)/total)
            positions[it['id']] = (x_map[w], y + h/2.0, h)
            y += h
    flow_traces=[]
    for l in links:
        s=l['source']; t=l['target']; v=l['value']
        if s not in positions or t not in positions: continue
        x0,y0,h0 = positions[s]; x1,y1,h1 = positions[t]
        width = max(0.002, v)
        xs = [x0, x0 + 0.2*(x1-x0), x1 - 0.2*(x1-x0), x1, x1 - 0.2*(x1-x0), x0 + 0.2*(x1-x0)]
        ys = [y0 - width/2, y0 - width/2 + 0.02*(y1-y0), y1 - width/2 - 0.02*(y1-y0), y1 - width/2, y1 + width/2, y0 + width/2]
        flow_traces.append(go.Scatter(x=xs, y=ys, fill='toself', mode='none', opacity=0.6, hoverinfo='text', text=f"{s} → {t}<br>Jaccard={v:.3f}"))
    node_traces=[]
    for w in windows:
        items = nodes_by_win.get(w,[])
        x=[]; y=[]; text=[]; sizes=[]
        for it in items:
            nid = it['id']; xp,yp,h = positions[nid]
            x.append(xp); y.append(yp); sizes.append(it.get('size',1))
            text.append(f"{nid}<br>size:{it.get('size')}<br>top_terms:{', '.join(it.get('top_terms',[])[:6])}")
        if x:
            node_traces.append(go.Scatter(x=x, y=y, mode='markers+text', marker=dict(size=[8+6*np.log1p(s) for s in sizes]), text=[n.split('|')[-1] for n in [it['id'] for it in items]], textposition='middle right', hovertext=text, hoverinfo='text', showlegend=False))
    fig=go.Figure()
    for t in flow_traces: fig.add_trace(t)
    for t in node_traces: fig.add_trace(t)
    fig.update_layout(title='Alluvial thematic evolution', xaxis=dict(tickmode='array', tickvals=list(range(len(windows))), ticktext=windows), yaxis=dict(showticklabels=False), height=700, margin=dict(l=80,r=40,t=80,b=80))
    fig.update_xaxes(range=[-0.5, len(windows)-0.5])
    return fig

# -------------------------
# VOSviewer export helpers (simple node and edge files)
# -------------------------
def build_vosviewer_files(G:nx.Graph, partition:Dict[int,int], terms:List[str]):
    # nodes file: Id<TAB>Label
    nodes_lines = ["Id\tLabel"]
    for n,d in G.nodes(data=True):
        nodes_lines.append(f"{n}\t{d.get('term','')}")
    # edges file: Source<TAB>Target<TAB>Weight
    edges_lines = ["Source\tTarget\tWeight"]
    for u,v,d in G.edges(data=True):
        edges_lines.append(f"{u}\t{v}\t{d.get('weight',1.0)}")
    # cluster file: nodeId<TAB>clusterId (no header)
    clu_lines = []
    for n in sorted(G.nodes()):
        clu_lines.append(f"{partition.get(n,0)}")
    return '\n'.join(nodes_lines), '\n'.join(edges_lines), '\n'.join(clu_lines)

# -------------------------
# Progress callback for windowed
# -------------------------
def progress_callback_factory(st_progress, st_text):
    def cb(i, total, msg):
        if total>0:
            st_progress.progress(min(1.0, float(i)/float(total)))
        st_text.text(msg)
    return cb

# -------------------------
# Streamlit UI layout
# -------------------------
st.title("📚 BiblioRefine — Optimized (≤10k) with Thematic Evolution & VOSviewer export")

st.sidebar.header("Pipeline controls (optimized)")
vocab_size = st.sidebar.number_input("Max vocabulary size", min_value=200, max_value=20000, value=DEFAULT_VOCAB, step=100)
min_df = st.sidebar.number_input("Min term doc frequency", min_value=1, max_value=50, value=DEFAULT_MIN_DF)
edge_threshold = st.sidebar.slider("Edge weight threshold", 0.0, 1.0, DEFAULT_EDGE_THRESHOLD, 0.01)
use_association = st.sidebar.checkbox("Use Association Strength normalization", value=False)
use_leiden_opt = st.sidebar.checkbox("Prefer Leiden clustering if available", value=True)
seed = int(st.sidebar.number_input("Random seed", min_value=0, max_value=2**31-1, value=DEFAULT_SEED))
window_size = int(st.sidebar.number_input("Window size (years)", min_value=1, max_value=10, value=DEFAULT_WINDOW))
stride = int(st.sidebar.number_input("Window stride (years)", min_value=1, max_value=10, value=DEFAULT_STRIDE))
overlap_thresh = st.sidebar.slider("Jaccard threshold for evolution links", 0.0, 1.0, DEFAULT_OVERLAP, 0.01)

# Upload inputs
st.header("Step 1 — Upload files (WoS .xlsx, Scopus .csv, optional extras .bib/.ris/.xml/.json)")
col1,col2 = st.columns(2)
with col1:
    wos_file = st.file_uploader("Upload WoS .xlsx (optional)", type='xlsx')
with col2:
    scopus_file = st.file_uploader("Upload Scopus .csv (optional)", type='csv')
extra_files = st.file_uploader("Optional extras (.bib, .ris, .xml, .json) — multiple allowed", accept_multiple_files=True)

# Load & prepare
if st.button("Load & Prepare (limit to 10k)"):
    dfs=[]
    if wos_file:
        try:
            dfw = pd.read_excel(wos_file)
            dfw = standardize(dfw, 'WoSCC'); dfs.append(dfw); st.success(f"Loaded WoS: {dfw.shape}")
        except Exception as e:
            st.error(f"Failed to read WoS: {e}")
    if scopus_file:
        try:
            dfsc = pd.read_csv(scopus_file, low_memory=False)
            dfsc = standardize(dfsc, 'Scopus'); dfs.append(dfsc); st.success(f"Loaded Scopus: {dfsc.shape}")
        except Exception as e:
            st.error(f"Failed to read Scopus: {e}")
    parsed=[]
    if extra_files:
        for f in extra_files:
            name=f.name.lower()
            try:
                b = f.read()
                if name.endswith('.bib'):
                    dfx = parse_bibtex_bytes(b)
                elif name.endswith('.ris'):
                    dfx = parse_ris_bytes(b)
                elif name.endswith('.xml'):
                    dfx = parse_endnote_xml_bytes(b)
                elif name.endswith('.json'):
                    dfx = parse_openalex_json_bytes(b)
                else:
                    dfx = pd.DataFrame()
                if not dfx.empty:
                    parsed.append(dfx); st.info(f"Parsed {f.name}: {dfx.shape[0]} rows")
            except Exception as e:
                st.warning(f"Error parsing {f.name}: {e}")
    if parsed:
        dfs.append(pd.concat(parsed, ignore_index=True, sort=False))
    if not dfs:
        st.error("No inputs provided.")
    else:
        merged = pd.concat(dfs, ignore_index=True, sort=False)
        # ensure DI column
        if 'DI' not in merged.columns:
            merged['DI'] = pd.NA
        merged['DI'] = merged['DI'].apply(normalize_doi)
        # ensure core fields
        CORE = ['TI','AU','AF','SO','PY','DE','AB','C1','FX','CR','NR','DI']
        for c in CORE:
            if c not in merged.columns:
                merged[c] = pd.NA
        # dedupe: prefer DOI; fallback title+year
        merged['title_key'] = merged.apply(lambda r: title_key(r.get('TI'), r.get('PY')), axis=1)
        merged = merged.sort_values(by=merged['DI'].isna().astype(int))
        merged = merged.drop_duplicates(subset=['DI'], keep='first')
        merged = merged.drop_duplicates(subset=['title_key'], keep='first')
        merged = merged.drop(columns=['title_key']).reset_index(drop=True)
        # limit to MAX_RECORDS (sample if larger)
        if len(merged) > MAX_RECORDS:
            st.warning(f"Dataset has {len(merged)} records — sampling {MAX_RECORDS} randomly for performance.")
            merged = merged.sample(n=MAX_RECORDS, random_state=seed).reset_index(drop=True)
        st.session_state['merged'] = merged
        st.success(f"Prepared merged dataset: {len(merged)} records")
        st.write(merged.head(5))

# Optional enrichment
if 'merged' in st.session_state:
    if st.button("Enrich missing fields via OpenAlex (by DOI)"):
        merged = st.session_state['merged']
        API = 'https://api.openalex.org/works/doi:'
        rows = merged.index.tolist()
        prog = st.progress(0); status = st.empty()
        updated = 0
        for i, idx in enumerate(rows):
            doi = merged.at[idx,'DI']
            if pd.isna(doi) or doi == '': continue
            try:
                r = requests.get(API + doi, timeout=15)
                if r.status_code == 200:
                    meta = r.json()
                    if pd.isna(merged.at[idx,'TI']) and meta.get('title'): merged.at[idx,'TI'] = meta.get('title')
                    if pd.isna(merged.at[idx,'AB']) and meta.get('abstract'): merged.at[idx,'AB'] = meta.get('abstract')
                    if pd.isna(merged.at[idx,'PY']) and meta.get('publication_year'): merged.at[idx,'PY'] = meta.get('publication_year')
                    if pd.isna(merged.at[idx,'AU']) and 'authorships' in meta:
                        au = '; '.join([a['author']['display_name'] for a in meta['authorships'] if a.get('author')])
                        merged.at[idx,'AU'] = au; merged.at[idx,'AF'] = au
                    updated += 1
            except Exception:
                pass
            prog.progress((i+1)/len(rows)); status.text(f"Processed {i+1}/{len(rows)}")
            time.sleep(0.05)
        st.session_state['merged'] = merged
        st.success(f"OpenAlex enrichment attempted — ~{updated} records updated.")

# Analysis trigger
if st.button("Run analysis (co-word, clustering, evolution)"):
    if 'merged' not in st.session_state:
        st.error("Please load & prepare dataset first.")
    else:
        merged = st.session_state['merged']
        st.info("Building keyword lists (DE + ID), cleaning and tokenizing...")
        kw_lists=[]
        for i,row in merged.iterrows():
            kws = combine_DE_ID(row)
            if (not kws) and pd.notna(row.get('AB')):
                kws = tokenize(str(row.get('AB')))[:20]
            cleaned=[]
            for t in kws:
                cleaned.extend(tokenize(t))
            # dedupe & limit
            seen=[]; out=[]
            for t in cleaned:
                if t not in seen:
                    seen.append(t); out.append(t)
                if len(seen)>=50: break
            kw_lists.append(out)
        st.write(f"Keywords prepared for {len(kw_lists)} documents.")

        # build term-doc matrix (cached)
        st.info("Building term-document matrix (sparse)...")
        vect, X, terms = build_term_doc_matrix(kw_lists, max_features=vocab_size, min_df=min_df)
        st.write(f"Matrix shape: {X.shape}; vocab used: {len(terms)}")

        # compute co-occurrence
        st.info("Computing co-occurrence & normalization...")
        C = compute_cooccurrence(X)
        if use_association:
            S = normalize_association_strength(C, X.shape[0])
            norm_label = "Association Strength"
        else:
            S = normalize_salton(C)
            norm_label = "Salton (cosine)"
        st.write(f"Normalization: {norm_label}")

        # build graph
        st.info("Building graph (NetworkX) from similarity matrix...")
        G = build_graph_from_similarity(S, terms, threshold=edge_threshold, top_k=None)
        st.write(f"Graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        # clustering
        st.info("Clustering with Leiden (if available) else Louvain...")
        partition = run_clustering(G, use_leiden=use_leiden_opt, seed=seed)
        st.session_state['global_graph']=G; st.session_state['global_terms']=list(terms); st.session_state['global_partition']=partition

        # compute Callon metrics
        metrics = compute_callon(G, partition)
        st.session_state['global_metrics']=metrics

        # cluster summary
        df_clusters = pd.DataFrame([{'cluster':cid,'size':m['size'],'density':m['density'],'centrality':m['centrality'],'top_terms':', '.join([G.nodes[n]['term'] for n in m['nodes'][:10]])} for cid,m in metrics.items()]).sort_values('size',ascending=False)
        st.subheader("Cluster summary (top clusters)")
        st.dataframe(df_clusters.head(200))

        # windowed clustering (thematic evolution)
        st.info("Running windowed clustering for thematic evolution (this may take a bit)...")
        progress_bar = st.progress(0); status = st.empty()
        co_params = {'vocab_size':vocab_size,'min_term_freq':min_df,'edge_threshold':edge_threshold,'use_association':use_association,'use_leiden':use_leiden_opt,'seed':seed}
        window_results = windowed_clustering(merged, window_size=window_size, stride=stride, co_word_params=co_params, progress_callback=progress_callback_factory(progress_bar, status))
        # drop empty
        window_results = {k:v for k,v in window_results.items() if v is not None}
        if not window_results:
            st.warning("No valid windows produced (check PY/year fields).")
        else:
            st.success(f"Windowed clustering done: {len(window_results)} windows with results.")
            timeline = build_timeline(window_results, top_k=20, overlap_threshold=overlap_thresh)
            st.session_state['timeline']=timeline
            # alluvial plot
            fig = build_alluvial_plot(timeline)
            st.plotly_chart(fig, use_container_width=True)

        # write run metadata
        metadata = {'timestamp': datetime.datetime.utcnow().isoformat()+'Z', 'params':{'vocab_size':vocab_size,'min_df':min_df,'edge_threshold':edge_threshold,'use_association':use_association,'use_leiden':use_leiden_opt,'seed':seed,'window_size':window_size,'stride':stride,'overlap_thresh':overlap_thresh}}
        try:
            git_hash = subprocess.check_output(['git','rev-parse','HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            git_hash = None
        metadata['git_commit'] = git_hash
        try:
            doi_concat = ''.join(sorted([str(x) for x in merged['DI'].fillna('')]))
            metadata['dataset_hash'] = hashlib.sha1(doi_concat.encode('utf-8')).hexdigest()
        except Exception:
            metadata['dataset_hash'] = None
        with open('run_metadata.json','w', encoding='utf-8') as fh:
            json.dump(metadata, fh, indent=2)
        st.success("Analysis finished and run_metadata.json saved.")

# -------------------------
# Exports: CSV, ZIP, VOSviewer files
# -------------------------
if 'global_graph' in st.session_state:
    st.header("Exports")
    G = st.session_state['global_graph']; terms = st.session_state['global_terms']; partition = st.session_state['global_partition']; metrics = st.session_state.get('global_metrics',{})
    # term->cluster CSV
    rows = [{'term': G.nodes[n]['term'], 'node_idx': n, 'cluster': int(partition.get(n,0))} for n in G.nodes()]
    df_term_cluster = pd.DataFrame(rows)
    csv_tc = df_term_cluster.to_csv(index=False).encode('utf-8')
    st.download_button("Download term->cluster (CSV)", csv_tc, "term_clusters.csv")
    # edge list CSV
    edge_rows = [{'source':u, 'target':v, 'weight':d.get('weight',1.0), 'source_term':G.nodes[u]['term'], 'target_term':G.nodes[v]['term']} for u,v,d in G.edges(data=True)]
    df_edges = pd.DataFrame(edge_rows)
    st.download_button("Download edge list (CSV)", df_edges.to_csv(index=False).encode('utf-8'), "co_word_edges.csv")
    # cluster stats CSV
    cs_rows = [{'cluster':cid,'size':m['size'],'density':m['density'],'centrality':m['centrality']} for cid,m in metrics.items()]
    df_cs = pd.DataFrame(cs_rows).sort_values('size',ascending=False)
    st.download_button("Download cluster stats (CSV)", df_cs.to_csv(index=False).encode('utf-8'), "cluster_stats.csv")
    # GEXF
    try:
        gexport = nx.Graph()
        for n,d in G.nodes(data=True):
            gexport.add_node(n, term=d.get('term',''), cluster=int(partition.get(n,0)))
        for u,v,d in G.edges(data=True):
            gexport.add_edge(u,v,weight=float(d.get('weight',1.0)))
        bio = BytesIO(); nx.write_gexf(gexport, bio); bio.seek(0)
        st.download_button("Download graph (GEXF)", bio, "co_word_graph.gexf")
    except Exception as e:
        st.warning(f"GEXF export failed: {e}")

    # VOSviewer files
    vos_nodes_txt, vos_edges_txt, vos_clu_txt = build_vosviewer_files(G, partition, terms)
    st.download_button("Download VOSviewer nodes (txt)", vos_nodes_txt.encode('utf-8'), "vos_nodes.txt")
    st.download_button("Download VOSviewer edges (txt)", vos_edges_txt.encode('utf-8'), "vos_edges.txt")
    st.download_button("Download VOSviewer cluster file (clu)", vos_clu_txt.encode('utf-8'), "vos_clusters.clu")

    # Timeline JSON
    if 'timeline' in st.session_state:
        timeline = st.session_state['timeline']
        st.download_button("Download timeline JSON (Cytoscape)", json.dumps(timeline, indent=2), "timeline_cytoscape.json")

    # ZIP package
    if st.button("Prepare ZIP package of all outputs"):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # merged data
            try:
                merged_csv = st.session_state['merged'].to_csv(index=False).encode('utf-8')
                zf.writestr('merged_cleaned.csv', merged_csv)
            except Exception:
                pass
            zf.writestr('term_clusters.csv', df_term_cluster.to_csv(index=False))
            zf.writestr('co_word_edges.csv', df_edges.to_csv(index=False))
            zf.writestr('cluster_stats.csv', df_cs.to_csv(index=False))
            zf.writestr('vos_nodes.txt', vos_nodes_txt)
            zf.writestr('vos_edges.txt', vos_edges_txt)
            zf.writestr('vos_clusters.clu', vos_clu_txt)
            if 'timeline' in st.session_state:
                zf.writestr('timeline_cytoscape.json', json.dumps(st.session_state['timeline'], indent=2))
            # run metadata
            try:
                with open('run_metadata.json','r',encoding='utf-8') as fh:
                    zf.writestr('run_metadata.json', fh.read())
            except Exception:
                pass
        buffer.seek(0)
        st.download_button("Download full ZIP package", buffer, "biblio_analysis_results.zip")

st.caption("Notes: This optimized pipeline limits processing to 10k records for speed. For Leiden clustering install python-igraph and leidenalg on Python 3.10/3.11; otherwise Louvain is used.")
