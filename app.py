import streamlit as st
import pandas as pd
import requests
import time
import zipfile
import numpy as np
from io import BytesIO

st.set_page_config(page_title="BiblioRefine", layout="wide")
st.title("📚 BiblioRefine (WoSCC + Scopus)")

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
    # 📥 Download Cleaned Merged Data
    # -----------------------------
    st.subheader("📥 Download Merged Data (Before Enrichment)")
    merged_csv = merged[FIELDS + ['DI']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Cleaned & Deduplicated Dataset (CSV)", merged_csv, "Merged_Cleaned.csv")

    # 📄 Download Duplicates Log
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

        # 📊 PubMed Latency Performance
        if pubmed_latencies:
            lat_df = pd.DataFrame(pubmed_latencies)
            median = np.median(lat_df["Latency_sec"])
            q1, q3 = np.percentile(lat_df["Latency_sec"], [25, 75])
            st.markdown(f"**📈 Median Latency per DOI:** {median:.2f}s (IQR: {q1:.2f} – {q3:.2f}s)")

            st.download_button("📥 Download PubMed Performance CSV",
                               lat_df.to_csv(index=False).encode('utf-8'),
                               "PubMed_Performance.csv")

   # -----------------------------
# Step 3: Enrich with OpenAlex
# -----------------------------
st.header("Step 3: Enrich with OpenAlex")

# Initialize session state flag
if "openalex_done" not in st.session_state:
    st.session_state["openalex_done"] = False

if st.button("🚀 Run OpenAlex Enrichment"):
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

        st.download_button("📥 Download OpenAlex Performance CSV",
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
        st.download_button("📦 Download Final ZIP Package", buffer, "Bibliography_Enriched_OpenAlex.zip")
