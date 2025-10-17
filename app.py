from typing import List, Dict, Any
import streamlit as st
import pandas as pd
import requests
import time
import zipfile
import numpy as np
from io import BytesIO
import json, re, datetime

st.set_page_config(page_title="BiblioRefine", layout="wide")
st.title("BiblioRefine (WoSCC + Scopus)")

# -----------------------------
# Step 1: Upload & Merge 
# -----------------------------
st.header("Step 1: Upload and Prepare Data")
wos_file = st.file_uploader("Upload WoSCC (.xlsx)", type="xlsx")
scopus_file = st.file_uploader("Upload Scopus (.csv)", type="csv")

if wos_file and scopus_file:
    wos_df = pd.read_excel(wos_file)
    scopus_df = pd.read_csv(scopus_file)
    st.success(f"✅ Loaded WoSCC: {wos_df.shape}, Scopus: {scopus_df.shape}")

    # Standardize field names
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

    def standardize(df, source):
        df = df.rename(columns={c: MASTER_MAP[c] for c in df.columns if c in MASTER_MAP})
        df = df.loc[:, ~df.columns.duplicated()]
        df['Source'] = source
        return df

    wos_df = standardize(wos_df, 'WoSCC')
    scopus_df = standardize(scopus_df, 'Scopus')

    # Normalize DOI
    def normalize_doi(x):
        if pd.isna(x):
            return pd.NA
        x = str(x).strip().lower()
        return x.replace('https://doi.org/', '').replace('http://doi.org/', '').replace('doi:', '')

    for df in [wos_df, scopus_df]:
        if 'DI' not in df.columns:
            df['DI'] = pd.NA
        df['DI'] = df['DI'].apply(normalize_doi)

    # DOI missing stats
    st.info(
        f"📊 Missing DOIs: WoSCC: {wos_df['DI'].isna().sum()} / {len(wos_df)} | "
        f"Scopus: {scopus_df['DI'].isna().sum()} / {len(scopus_df)}"
    )

    # Merge & Deduplicate
    wos_df = wos_df.drop_duplicates(subset=['DI'])
    scopus_df = scopus_df.drop_duplicates(subset=['DI'])
    merged = pd.merge(wos_df, scopus_df, on='DI', how='outer', suffixes=('_wos', '_scopus'))

    FIELDS = ['TI', 'AU', 'AF', 'SO', 'PY', 'DE', 'AB', 'C1', 'FX', 'CR', 'NR']
    for f in FIELDS:
        col_wos = merged[f + '_wos'] if f + '_wos' in merged.columns else pd.Series([pd.NA] * len(merged))
        col_sco = merged[f + '_scopus'] if f + '_scopus' in merged.columns else pd.Series([pd.NA] * len(merged))
        merged[f] = col_wos.combine_first(col_sco)

    merged['Duplicate'] = merged.duplicated(subset=['DI'], keep='first')
    dedup_log = merged[merged['Duplicate']][['DI']].copy()
    merged = merged[~merged['Duplicate']]

    st.success("✅ Data cleaned and deduplicated.")

    # -----------------------------
    # Download Cleaned Merged Data
    # -----------------------------
    st.subheader("📥 Download Merged Data (Before Enrichment)")
    merged_csv = merged[FIELDS + ['DI']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Cleaned & Deduplicated Dataset (CSV)", merged_csv, "Merged_Cleaned.csv")

    # Download Duplicates Log
    if not dedup_log.empty:
        st.subheader("📄 Duplicates Removed Log")
        dup_csv = dedup_log.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Duplicates Log (CSV)", dup_csv, "Duplicates_Removed_Log.csv")
    else:
        st.info("✅ No duplicates were detected.")

    # -----------------------------
    # Step 2: Recover Missing DOIs from PubMed
    # -----------------------------
    st.header("Step 2: Recover Missing DOIs from PubMed")

    if st.button("🔍 Search PubMed for Missing DOIs"):
        pubmed_found = []
        pubmed_not_found = []
        pubmed_latencies = []

        API_PUBMED = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        API_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

        def fetch_pubmed_doi(title, year):
            try:
                params = {"db": "pubmed", "term": f"{title} AND {year}[dp]", "retmode": "json"}
                r = requests.get(API_PUBMED, params=params, timeout=15)
                if r.status_code != 200:
                    return None
                result = r.json()
                if "esearchresult" not in result or not result["esearchresult"]["idlist"]:
                    return None
                pmid = result["esearchresult"]["idlist"][0]
                r2 = requests.get(API_FETCH, params={"db": "pubmed", "id": pmid, "retmode": "json"}, timeout=15)
                if r2.status_code != 200:
                    return None
                doi = r2.json()["result"].get(pmid, {}).get("elocationid", "")
                if doi and "doi" in doi.lower():
                    return doi.replace("doi:", "").strip()
                return None
            except:
                return None

        progress = st.progress(0)
        missing_rows = merged[merged['DI'].isna()].index

        for i, idx in enumerate(missing_rows):
            title, year = merged.at[idx, 'TI'], merged.at[idx, 'PY']
            if pd.isna(title) or pd.isna(year):
                pubmed_not_found.append({"Index": idx, "Reason": "No title/year"})
                continue

            start = time.time()
            doi = fetch_pubmed_doi(title, year)
            end = time.time()

            latency = round(end - start, 3)
            pubmed_latencies.append({"Index": idx, "Latency_sec": latency})

            if doi:
                merged.at[idx, 'DI'] = doi
                pubmed_found.append({"Index": idx, "DOI": doi, "Latency_sec": latency})
            else:
                pubmed_not_found.append({"Index": idx, "Reason": "DOI not found"})

            progress.progress((i + 1) / len(missing_rows))
            time.sleep(0.3)

        st.success(f"✅ PubMed search complete — Found: {len(pubmed_found)}, Not found: {len(pubmed_not_found)}")

        # PubMed Latency Performance
        if pubmed_latencies:
            lat_df = pd.DataFrame(pubmed_latencies)
            median = np.median(lat_df["Latency_sec"])
            q1, q3 = np.percentile(lat_df["Latency_sec"], [25, 75])
            st.markdown(f"**📈 Median Latency per DOI:** {median:.2f}s (IQR: {q1:.2f} – {q3:.2f}s)")

            st.download_button("Download PubMed Performance CSV",
                               lat_df.to_csv(index=False).encode('utf-8'),
                               "PubMed_Performance.csv")

   # -----------------------------
# Step 3: Enrich with OpenAlex
st.header("Step 3: Enrich with OpenAlex")

# Initialize session state flag
if "openalex_done" not in st.session_state:
    st.session_state["openalex_done"] = False

if st.button("Run OpenAlex Enrichment"):
    no_match_log = []
    updated = 0
    openalex_latencies = []

    API_OPEN = 'https://api.openalex.org/works/doi:'

    def fetch_openalex(doi):
        try:
            r = requests.get(API_OPEN + doi, timeout=15)
            return r.json() if r.status_code == 200 else None
        except:
            return None

    progress = st.progress(0)
    for i, idx in enumerate(merged.index):
        doi = merged.at[idx, 'DI']
        if pd.isna(doi) or doi == '':
            continue

        start = time.time()
        meta = fetch_openalex(doi)
        end = time.time()

        openalex_latencies.append({"DOI": doi, "Latency_sec": round(end - start, 3)})

        if not meta:
            no_match_log.append(doi)
            continue

        if pd.isna(merged.at[idx, 'TI']): merged.at[idx, 'TI'] = meta.get('title')
        if pd.isna(merged.at[idx, 'AB']): merged.at[idx, 'AB'] = meta.get('abstract')
        if pd.isna(merged.at[idx, 'SO']): merged.at[idx, 'SO'] = meta.get('host_venue', {}).get('display_name')
        if pd.isna(merged.at[idx, 'PY']): merged.at[idx, 'PY'] = meta.get('publication_year')

        if 'authorships' in meta:
            au = '; '.join([a['author']['display_name'] for a in meta['authorships'] if a.get('author')])
            if pd.isna(merged.at[idx, 'AU']): merged.at[idx, 'AU'] = au
            if pd.isna(merged.at[idx, 'AF']): merged.at[idx, 'AF'] = au

        updated += 1
        progress.progress((i + 1) / len(merged))

    st.session_state["openalex_done"] = True
    st.session_state["merged_result"] = merged
    st.session_state["openalex_latencies"] = openalex_latencies
    st.session_state["no_match_log"] = no_match_log
    st.success(f"✅ OpenAlex enrichment complete — Records updated: {updated}")

# ✅ Show results if enrichment was already run
if st.session_state["openalex_done"]:
    merged = st.session_state["merged_result"]
    openalex_latencies = st.session_state["openalex_latencies"]
    no_match_log = st.session_state["no_match_log"]

    # 📊 OpenAlex Latency
    if openalex_latencies:
        oa_lat_df = pd.DataFrame(openalex_latencies)
        median = np.median(oa_lat_df["Latency_sec"])
        q1, q3 = np.percentile(oa_lat_df["Latency_sec"], [25, 75])
        st.markdown(f"**📈 Median Latency per DOI:** {median:.2f}s (IQR: {q1:.2f} – {q3:.2f}s)")

        st.download_button("Download OpenAlex Performance CSV",
                           oa_lat_df.to_csv(index=False).encode('utf-8'),
                           "OpenAlex_Performance.csv")

    # -----------------------------
    # Final ZIP Packaging
    # -----------------------------
    vos_name = 'VOSviewer_OpenAlex.txt'
    vos_content = merged[FIELDS + ['DI']].to_csv(sep='\t', index=False)

    with BytesIO() as buffer:
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr('Merged_Enriched.csv', merged[FIELDS + ['DI']].to_csv(index=False))
            zipf.writestr(vos_name, vos_content)
            zipf.writestr('No_DOI_Log.csv', merged[merged['DI'].isna()][['DI']].to_csv(index=False))
            zipf.writestr('Records_Deduplicated_Log.csv', dedup_log.to_csv(index=False) if not dedup_log.empty else "No duplicates found")
            zipf.writestr('No_OpenAlex_Match.csv', pd.DataFrame({'DI': no_match_log}).to_csv(index=False))
            if 'pubmed_found' in locals():
                zipf.writestr('PubMed_DOI_Found.csv', pd.DataFrame(pubmed_found).to_csv(index=False))
                zipf.writestr('PubMed_No_DOI.csv', pd.DataFrame(pubmed_not_found).to_csv(index=False))

        buffer.seek(0)
        st.download_button("Download Final ZIP Package", buffer, "Bibliography_Enriched_OpenAlex.zip")

# -----------------------------
# Step 4 Data Quality Dashboard (after enrichment)
# -----------------------------
st.sidebar.header("Data Quality")
if "merged_result" in st.session_state:
    dfq = st.session_state["merged_result"]
else:
    dfq = None

if st.sidebar.button("Show Data Quality Report"):
    if dfq is None:
        st.sidebar.warning("Run OpenAlex enrichment first (Step 3).")
    else:
        FIELDS = ['TI','AU','AF','SO','PY','DE','AB','C1','FX','CR','NR']
        rows = []
        total = len(dfq)
        for f in FIELDS:
            present = dfq[f].notna().sum() if f in dfq.columns else 0
            missing = total - present
            pct = 100.0 * missing / total if total>0 else 0.0
            if present == total:
                status = "Excellent"
            elif pct < 10:
                status = "Good"
            elif pct < 30:
                status = "Moderate"
            elif pct < 70:
                status = "Poor"
            else:
                status = "Critical"
            rows.append({"Field": f, "Present": int(present), "Missing": int(missing), "Missing%": round(pct,2), "Status": status})
        dqdf = pd.DataFrame(rows).sort_values("Missing%", ascending=True)
        st.subheader("Metadata completeness (post-OpenAlex)")
        # color map
        def color_status(s):
            if s=="Excellent": return "background-color: #d4f7d4"
            if s=="Good": return "background-color: #e6f8d1"
            if s=="Moderate": return "background-color: #fff2b2"
            if s=="Poor": return "background-color: #ffd2a6"
            return "background-color: #ffb3b3"
        styled = dqdf.style.applymap(lambda v: color_status(v) if v in ["Excellent","Good","Moderate","Poor","Critical"] else "", subset=["Status"])
        st.dataframe(dqdf)
        csv_bytes = dqdf.to_csv(index=False).encode("utf-8")
        st.download_button("Download Missing Data Report (CSV)", csv_bytes, "missing_data_report.csv")

# -----------------------------
# Step 5: Analysis - Co-word, Bibliographic coupling, Co-citation
# -----------------------------
st.header("Step 5: Analysis")
st.markdown("Run co-word (DE), bibliographic coupling and co-citation analyses.")

# helpers for keywords cleaning
def clean_keyword_simple(kw):
    if pd.isna(kw): return ""
    s = str(kw).lower()
    s = re.sub(r'[^a-z\\s\\-]', ' ', s)
    s = re.sub(r'\\s+', ' ', s).strip()
    toks = [t for t in s.split() if len(t)>2]
    return ' '.join([t for t in toks if t not in set(['the','and','for','with'])])

def extract_keywords_list(df_row):
    out = []
    if 'DE' in df_row and pd.notna(df_row['DE']):
        parts = re.split(r'[;|,]+', str(df_row['DE']))
        for p in parts:
            k = clean_keyword_simple(p)
            if k: out.append(k)
    return out

def build_doc_ref_matrix_from_CR(cr_series: pd.Series):
    # Parse CR strings into normalized reference tokens; each unique reference becomes a column
    docs = []
    ref_index = {}
    rows = []
    for i, cr in cr_series.fillna("").iteritems():
        items = re.split(r'[\;\|]+', str(cr)) if cr else []
        refs = []
        for it in items:
            it = it.strip()
            if not it: continue
            # normalize: take first author + year + short title fragment
            m = re.match(r'([^,]+),\s*([^,]+),\s*(\d{4})', it)
            token = it.lower()[:80]
            refs.append(token)
            if token not in ref_index:
                ref_index[token] = len(ref_index)
        rows.append(refs)
    # build sparse doc x ref matrix
    import scipy.sparse as sp
    from collections import defaultdict
    data = []
    r_idx = []
    c_idx = []
    for doc_i, refs in enumerate(rows):
        counts = defaultdict(int)
        for r in refs:
            counts[ref_index[r]] += 1
        for col, cnt in counts.items():
            r_idx.append(doc_i); c_idx.append(col); data.append(cnt)
    if len(data)==0:
        return None, [], []
    M = sp.csr_matrix((data, (r_idx, c_idx)), shape=(len(rows), len(ref_index)), dtype=int)
    terms = [k for k,v in sorted(ref_index.items(), key=lambda x:x[1])]
    return M, terms, ref_index

def normalize_salton_sparse(A):
    # For A (sparse similarity) output normalized Salton (cosine-like)
    import numpy as np
    from scipy.sparse import diags
    diag = np.sqrt(A.diagonal()).astype(float)
    inv = np.zeros_like(diag)
    nz = diag>0
    inv[nz]=1.0/diag[nz]
    D = diags(inv)
    S = D.dot(A).dot(D)
    S.setdiag(0)
    return S

if st.button("Run all analyses (co-word, bibliographic coupling, co-citation)"):
    if "merged_result" not in st.session_state:
        st.error("Run OpenAlex enrichment first (Step 3).")
    else:
        merged = st.session_state["merged_result"]
        if merged.empty:
            st.error("No records to analyze.")
        else:
            log_msgs = []
            st.info("Preparing keywords (DE) ...")
            kw_lists = [extract_keywords_list(row) for _, row in merged.iterrows()]
            # Term-doc matrix
            vect = None
            try:
                from sklearn.feature_extraction.text import CountVectorizer
                docs_txt = [" ".join([t.replace(" ", "_") for t in toks]) for toks in kw_lists]
                vect = CountVectorizer(lowercase=True, token_pattern=r'(?u)\\b\\w+\\b', max_features=5000, min_df=2)
                X = vect.fit_transform(docs_txt)
                terms = vect.get_feature_names_out()
                st.write(f"Co-word term-doc matrix: {X.shape}, vocab={len(terms)}")
            except Exception as e:
                st.error(f"Term-doc building failed: {e}"); X = None; terms = []

            # Co-word network
            if X is not None:
                st.info("Building co-occurrence and Salton normalization ...")
                C = (X.T @ X).tocsr()
                S = normalize_salton_sparse(C)
                # build graph
                import networkx as nx
                G_co = nx.Graph()
                for i,t in enumerate(terms):
                    G_co.add_node(i, term=t)
                S_triu = sp.triu(S, k=1).tocoo()
                for i,j,v in zip(S_triu.row, S_triu.col, S_triu.data):
                    if v>0.05:
                        G_co.add_edge(i,j, weight=float(v))
                st.write(f"Co-word graph nodes={G_co.number_of_nodes()}, edges={G_co.number_of_edges()}")
                # clustering
                try:
                    import community as community_louvain
                    part_co = community_louvain.best_partition(G_co, weight='weight', random_state=42)
                except Exception:
                    part_co = {n:0 for n in G_co.nodes()}
                # compute top terms per cluster
                from collections import defaultdict
                cluster_terms = defaultdict(list)
                for n,c in part_co.items():
                    cluster_terms[c].append(n)
                top_terms_co = {}
                for cid, nodes in cluster_terms.items():
                    freq = []
                    for n in nodes:
                        term = G_co.nodes[n]['term']
                        freq.append((n, term))
                    top_terms_co[cid] = [t for _,t in sorted(freq, key=lambda x: x[0])[:10]]

                st.session_state['G_co'] = G_co
                st.session_state['terms_co'] = list(terms)
                st.session_state['part_co'] = part_co
                st.success("Co-word pipeline complete.")

            # Bibliographic coupling (doc-doc similarity by shared refs)
            st.info("Building doc-ref incidence from CR ...")
            M, ref_terms, ref_index = build_doc_ref_matrix_from_CR(merged['CR'] if 'CR' in merged.columns else pd.Series(['']*len(merged)))
            if M is None:
                st.warning("No cited references found; bibliographic coupling skipped.")
            else:
                st.info("Computing bibliographic coupling matrix ...")
                BC = (M @ M.T).tocsr()
                # normalize Salton
                Sbc = normalize_salton_sparse(BC)
                # build graph of documents (use small threshold)
                G_bc = nx.Graph()
                for i in range(M.shape[0]): G_bc.add_node(i, title=str(merged.at[i,'TI']))
                S_triu = sp.triu(Sbc, k=1).tocoo()
                for i,j,v in zip(S_triu.row, S_triu.col, S_triu.data):
                    if v>0.05:
                        G_bc.add_edge(i,j, weight=float(v))
                st.write(f"Bibliographic coupling graph nodes={G_bc.number_of_nodes()}, edges={G_bc.number_of_edges()}")
                # clustering documents
                try:
                    import community as community_louvain
                    part_bc = community_louvain.best_partition(G_bc, weight='weight', random_state=42)
                except Exception:
                    part_bc = {n:0 for n in G_bc.nodes()}
                st.session_state['G_bc'] = G_bc
                st.session_state['part_bc'] = part_bc
                st.success("Bibliographic coupling complete.")

            # Co-citation: build cited-ref x cited-ref co-citation across papers
            st.info("Computing co-citation (reference x reference) ...")
            if M is None:
                st.warning("No cited references; co-citation skipped.")
            else:
                RC = (M.T @ M).tocsr()  # reference x reference co-citation counts
                SRC = normalize_salton_sparse(RC)
                # build graph of references
                G_cc = nx.Graph()
                for i, ref in enumerate(ref_terms):
                    G_cc.add_node(i, ref=ref)
                S_triu = sp.triu(SRC, k=1).tocoo()
                for i,j,v in zip(S_triu.row, S_triu.col, S_triu.data):
                    if v>0.05:
                        G_cc.add_edge(i,j, weight=float(v))
                st.session_state['G_cc'] = G_cc
                st.session_state['ref_terms'] = ref_terms
                st.success("Co-citation complete.")

# -----------------------------
# Step 6: Thematic evolution
# -----------------------------
st.header("Step 6: Thematic Evolution")
st.markdown("Run windowed co-word clustering and generate alluvial figure.")

def make_time_windows(df, year_col='PY', window_size=5, stride=5):
    if year_col not in df.columns:
        return []
    years = df[year_col].dropna().astype(int) if df[year_col].notna().any() else pd.Series(dtype=int)
    if years.empty: return []
    min_y = int(years.min()); max_y = int(years.max())
    windows = []
    start = min_y
    while start <= max_y:
        end = start + window_size - 1
        slice_df = df[(df[year_col].notna()) & (df[year_col].astype(int)>=start) & (df[year_col].astype(int)<=end)]
        windows.append((start, end, slice_df))
        start += stride
    return windows

def jaccard_between_clusters(ra, pa, rb, pb):
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
            j = 0.0 if not u else len(a_terms & b_terms)/len(u)
            overlaps.append((a_id, b_id, j))
    return overlaps

def build_timeline_json(window_results, top_k_clusters=20, overlap_threshold=0.05):
    ordered = sorted([k for k in window_results.keys()])
    node_meta=[]; node_index={}; idx=0
    for w in ordered:
        res = window_results.get(w)
        if not res: continue
        metrics = res.get('metrics', {})
        top_clusters = sorted(metrics.items(), key=lambda x: -x[1].get('size',0))[:top_k_clusters]
        for cid,m in top_clusters:
            uq_id = f"{w}|C{cid}"
            label = f"C{cid}"
            short_terms = []
            if res.get('top_terms') and cid in res['top_terms']:
                tlist = res['top_terms'][cid]
                short_terms = [tt['term'] for tt in tlist[:3]]
                if short_terms:
                    label = short_terms[0]
                    if len(short_terms)>1:
                        label += ' – ' + ', '.join(short_terms[1:])
            label = f"{label} (n={m.get('size',0)})"
            node_index[uq_id]=idx
            node_meta.append({'id':uq_id,'label':label,'window':w,'cluster':cid,'size':m.get('size',0),'density':m.get('density',0.0),'centrality':m.get('centrality',0.0),'top_terms':short_terms})
            idx+=1
    links=[]
    for i in range(len(ordered)-1):
        a=ordered[i]; b=ordered[i+1]
        ra=window_results.get(a); rb=window_results.get(b)
        if not ra or not rb: continue
        overlaps = jaccard_between_clusters(ra['graph'], ra['partition'], rb['graph'], rb['partition'])
        for ca,cb,j in overlaps:
            if j>=overlap_threshold:
                s=f"{a}|C{ca}"; t=f"{b}|C{cb}"
                if s in node_index and t in node_index:
                    links.append({'source':s,'target':t,'value':float(j)})
    return {'nodes':node_meta,'links':links,'windows':ordered}

def build_alluvial_figure(timeline_json):
    import plotly.graph_objects as go
    nodes = timeline_json.get('nodes',[]); links = timeline_json.get('links',[]); windows = timeline_json.get('windows',[])
    from collections import defaultdict
    nodes_by_window = defaultdict(list)
    for n in nodes: nodes_by_window[n['window']].append(n)
    positions={}; x_map={w:i for i,w in enumerate(windows)}
    for w in windows:
        items = nodes_by_window.get(w,[])
        items = sorted(items, key=lambda x:-x.get('size',0))
        total = max(1, sum([max(1,it.get('size',1)) for it in items]))
        y=0.0
        for it in items:
            h = max(0.02, it.get('size',1)/total)
            positions[it['id']] = (x_map[w], y + h/2.0, h); y += h
    flow_traces=[]
    for l in links:
        s=l['source']; t=l['target']; v=l['value']
        if s not in positions or t not in positions: continue
        x0,y0,h0 = positions[s]; x1,y1,h1 = positions[t]
        width = max(0.002, v)
        xs=[x0, x0+0.25*(x1-x0), x1-0.25*(x1-x0), x1, x1-0.25*(x1-x0), x0+0.25*(x1-x0)]
        ys=[y0-width/2, y0-width/2+0.15*(y1-y0), y1-width/2-0.15*(y1-y0), y1-width/2, y1+width/2, y0+width/2]
        src_label = next((n['label'] for n in nodes if n['id']==s), s)
        tgt_label = next((n['label'] for n in nodes if n['id']==t), t)
        hovertxt = f"{src_label} → {tgt_label}<br>Jaccard: {v:.3f}"
        flow_traces.append(go.Scatter(x=xs, y=ys, fill='toself', mode='none', opacity=0.45, hoverinfo='text', text=hovertxt, showlegend=False))
    node_traces=[]
    for w in windows:
        items = nodes_by_window.get(w,[])
        if not items: continue
        x=[]; y=[]; sizes=[]; labels=[]; hover=[]
        for it in items:
            xp,yp,h = positions[it['id']]; x.append(xp); y.append(yp); sizes.append(it.get('size',1))
            labels.append(it.get('label', it['id'])); tt = it.get('top_terms', [])
            hover.append(f"{it.get('label','')}<br>size: {it.get('size',0)}<br>density: {it.get('density',0):.3f}<br>centrality: {it.get('centrality',0):.2f}<br>top terms: {', '.join(tt[:6])}")
        marker_sizes = [8 + 6 * np.log1p(s) for s in sizes]
        node_traces.append(go.Scatter(x=x, y=y, mode='markers+text', marker=dict(size=marker_sizes), text=labels, textposition='middle right', hovertext=hover, hoverinfo='text', showlegend=False))
    fig = go.Figure()
    for t in flow_traces: fig.add_trace(t)
    for t in node_traces: fig.add_trace(t)
    fig.update_layout(title='Alluvial thematic evolution', xaxis=dict(tickmode='array', tickvals=list(range(len(windows))), ticktext=windows, title='Time windows'), yaxis=dict(showticklabels=False), height=700, margin=dict(l=80, r=40, t=80, b=80))
    fig.update_xaxes(range=[-0.5, len(windows)-0.5])
    return fig

if st.button("Run Thematic Evolution (Co-word per window)"):
    if "merged_result" not in st.session_state:
        st.error("Run OpenAlex enrichment first (Step 3).")
    else:
        merged = st.session_state["merged_result"]
        windows = make_time_windows(merged, year_col='PY', window_size=5, stride=5)
        window_results = {}
        prog = st.progress(0); status = st.empty()
        for i,(start,end,slice_df) in enumerate(windows):
            status.text(f"Window {start}-{end}")
            if slice_df.empty:
                window_results[f'{start}_{end}'] = None; prog.progress((i+1)/len(windows)); continue
            # build kw lists for this window
            kw_lists = []
            for _, row in slice_df.iterrows():
                if pd.notna(row.get('DE')):
                    parts = re.split(r'[;|,]+', str(row.get('DE')))
                    kws = [clean_keyword_simple(p) for p in parts if clean_keyword_simple(p)]
                else:
                    # fallback to abstract tokens
                    toks = [clean_keyword_simple(t) for t in re.split(r'[\\s,;:.()]+', str(row.get('AB') or ''))]
                    kws = [t for t in toks if t][:20]
                kw_lists.append(kws)
            # term-doc
            try:
                from sklearn.feature_extraction.text import CountVectorizer
                docs_txt = [" ".join([t.replace(" ", "_") for t in toks]) for toks in kw_lists]
                vect = CountVectorizer(lowercase=True, token_pattern=r'(?u)\\b\\w+\\b', max_features=5000, min_df=2)
                X = vect.fit_transform(docs_txt); terms = vect.get_feature_names_out()
            except Exception as e:
                window_results[f'{start}_{end}'] = None; prog.progress((i+1)/len(windows)); continue
            C = (X.T @ X).tocsr(); S = normalize_salton_sparse(C)
            G = nx.Graph()
            for i_t,t in enumerate(terms): G.add_node(i_t, term=t)
            S_triu = sp.triu(S, k=1).tocoo()
            for ii,jj,v in zip(S_triu.row, S_triu.col, S_triu.data):
                if v>0.05: G.add_edge(ii,jj, weight=float(v))
            if G.number_of_nodes()==0:
                window_results[f'{start}_{end}'] = None; prog.progress((i+1)/len(windows)); continue
            # clustering
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G, weight='weight', random_state=42)
            except Exception:
                partition = {n:0 for n in G.nodes()}
            # metrics and top terms
            metrics = {}
            from collections import defaultdict
            clusters = defaultdict(list)
            for n,c in partition.items(): clusters[c].append(n)
            for cid, nodes in clusters.items():
                internal=0.0; external=0.0
                for a_i, u in enumerate(nodes):
                    for v in nodes[a_i+1:]:
                        if G.has_edge(u,v): internal += G[u][v].get('weight',1.0)
                denom = (len(nodes)*(len(nodes)-1)/2) if len(nodes)>1 else 1
                density = internal/denom if denom>0 else 0.0
                for u in nodes:
                    for nbr in G.neighbors(u):
                        if nbr not in nodes: external += G[u][nbr].get('weight',1.0)
                metrics[cid] = {'nodes': nodes, 'size': len(nodes), 'density': density, 'centrality': external}
            top_terms = {}
            for cid, nodes in clusters.items():
                top = []
                for n in nodes:
                    top.append((n, G.nodes[n]['term']))
                top_terms[cid] = [{'node': n, 'term': t} for n,t in sorted(top, key=lambda x:-x[0])[:10]]
            window_results[f'{start}_{end}'] = {'start': start, 'end': end, 'graph': G, 'terms': list(terms), 'partition': partition, 'metrics': metrics, 'top_terms': top_terms}
            prog.progress((i+1)/len(windows))
        window_results = {k:v for k,v in window_results.items() if v is not None}
        if not window_results:
            st.warning("No valid windows produced.")
        else:
            timeline = build_timeline_json(window_results, top_k_clusters=20, overlap_threshold=0.05)
            st.session_state['timeline'] = timeline
            fig = build_alluvial_figure(timeline)
            st.plotly_chart(fig, use_container_width=True)
            st.success("Thematic evolution generated.")

# -----------------------------
# Step 7: Export & Logs
# -----------------------------
st.header("Step 7: Export & Logs")
st.markdown("Download CSVs, graphs, timeline JSON, and view processing logs.")

if 'G_co' in st.session_state:
    G = st.session_state['G_co']; part = st.session_state.get('part_co', {})
    rows = [{'term':G.nodes[n]['term'].replace('_',' '), 'node_idx':n, 'cluster': int(part.get(n,0))} for n in G.nodes()]
    df_term_cluster = pd.DataFrame(rows)
    st.download_button("Download term->cluster (CSV)", df_term_cluster.to_csv(index=False).encode('utf-8'), 'term_clusters.csv')
    edge_rows = [{'source':u,'target':v,'weight':d.get('weight',1.0),'source_term':G.nodes[u]['term'].replace('_',' '),'target_term':G.nodes[v]['term'].replace('_',' ')} for u,v,d in G.edges(data=True)]
    df_edges = pd.DataFrame(edge_rows)
    st.download_button("Download edges (CSV)", df_edges.to_csv(index=False).encode('utf-8'), 'co_word_edges.csv')
    cs = [{'cluster': cid, 'size': m['size'], 'density': m['density'], 'centrality': m['centrality']} for cid,m in st.session_state.get('global_metrics', {}).items()] if 'global_metrics' in st.session_state else []
    df_cs = pd.DataFrame(cs)
    if not df_cs.empty:
        st.download_button("Download cluster stats (CSV)", df_cs.to_csv(index=False).encode('utf-8'), 'cluster_stats.csv')

    # VOSviewer exports
    vn = '\\n'.join(['Id\\tLabel'] + [f"{n}\\t{d.get('term','')}" for n,d in G.nodes(data=True)])
    ve = '\\n'.join(['Source\\tTarget\\tWeight'] + [f"{u}\\t{v}\\t{d.get('weight',1.0)}" for u,v,d in G.edges(data=True)])
    vc = '\\n'.join([str(st.session_state.get('part_co', {}).get(n,0)) for n in sorted(G.nodes())])
    st.download_button("Download VOSviewer nodes (.txt)", vn.encode('utf-8'), 'vos_nodes.txt')
    st.download_button("Download VOSviewer edges (.txt)", ve.encode('utf-8'), 'vos_edges.txt')
    st.download_button("Download VOSviewer cluster (.clu)", vc.encode('utf-8'), 'vos_clusters.clu')

if 'G_bc' in st.session_state:
    try:
        bio = BytesIO(); nx.write_gexf(st.session_state['G_bc'], bio); bio.seek(0)
        st.download_button("Download bibliographic-coupling graph (GEXF)", bio, 'bibliographic_coupling.gexf')
    except Exception as e:
        st.warning(f"GEXF export failed: {e}")

if 'timeline' in st.session_state:
    st.download_button("Download timeline JSON (Cytoscape)", json.dumps(st.session_state['timeline'], indent=2), 'timeline_cytoscape.json')

if 'merged_result' in st.session_state:
    st.download_button("Download merged_enriched.csv", st.session_state['merged_result'][['TI','AU','AF','SO','PY','DE','AB','C1','FX','CR','NR','DI']].to_csv(index=False).encode('utf-8'), 'merged_enriched.csv')

if st.button("Prepare ZIP of all outputs"):
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        try:
            if 'merged_result' in st.session_state:
                zf.writestr('merged_enriched.csv', st.session_state['merged_result'][['TI','AU','AF','SO','PY','DE','AB','C1','FX','CR','NR','DI']].to_csv(index=False))
        except Exception:
            pass
        try:
            if 'G_co' in st.session_state:
                zf.writestr('vos_nodes.txt', vn)
                zf.writestr('vos_edges.txt', ve)
                zf.writestr('vos_clusters.clu', vc)
        except Exception:
            pass
        if 'timeline' in st.session_state:
            zf.writestr('timeline_cytoscape.json', json.dumps(st.session_state['timeline'], indent=2))
    buf.seek(0)
    st.download_button("Download results ZIP", buf, 'biblio_refine_results.zip')

# Logs & run metadata
st.header("Run Logs & Reproducibility")
if 'merged_result' in st.session_state:
    st.write("Records:", len(st.session_state['merged_result']))
if st.button("Show simple run metadata"):
    meta = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "seed": 42,
        "max_records": 10000,
        "note": "BiblioRefine combined run"
    }
    st.json(meta)

if 'log_lines' in st.session_state:
    st.subheader("Processing logs")
    st.code("\\n".join(st.session_state.get('log_lines', [])))

