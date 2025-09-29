import streamlit as st
import pandas as pd
import requests
import time
import zipfile
import numpy as np
from io import BytesIO

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="BiblioRefine", layout="wide")
st.title("📚 BiblioRefine (WoSCC + Scopus)")
st.caption("Enhanced bibliographic refinement with DOI recovery, enrichment & performance benchmarking")

# -----------------------------
# Step 1: Upload & Merge
# -----------------------------
st.header("Step 1: Upload and Prepare Data")
wos_file = st.file_uploader("📤 Upload WoSCC (.xlsx)", type="xlsx")
scopus_file = st.file_uploader("📤 Upload Scopus (.csv)", type="csv")

if wos_file and scopus_file:
    wos_df = pd.read_excel(wos_file)
    scopus_df = pd.read_csv(scopus_file)
    st.success(f"✅ Loaded WoSCC: {wos_df.shape}, Scopus: {scopus_df.shape}")

    # --- Standardize column names ---
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

    # --- Normalize DOI ---
    def normalize_doi(x):
        if pd.isna(x):
            return pd.NA
        x = str(x).strip().lower()
        return x.replace('https://doi.org/', '').replace('http://doi.org/', '').replace('doi:', '')

    for df in [wos_df, scopus_df]:
        if 'DI' not in df.columns:
            df['DI'] = pd.NA
        df['DI'] = df['DI'].apply(normalize_doi)

    # --- DOI missing stats ---
    wos_missing = wos_df['DI'].isna().sum()
    scopus_missing = scopus_df['DI'].isna().sum()
    st.info(f"📊 Missing DOIs: WoSCC: {wos_missing} / {len(wos_df)} | Scopus: {scopus_missing} / {len(scopus_df)}")

    # --- Merge & Deduplicate ---
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
    original_missing_mask = merged['DI'].isna()

    st.success("✅ Data cleaned and deduplicated.")

    # -----------------------------
    # Step 2: Recover Missing DOIs from PubMed
    # -----------------------------
    st.header("Step 2: Recover Missing DOIs from PubMed")

    if st.button("🔍 Search PubMed for Missing DOIs"):
        pubmed_found = []
        pubmed_not_found = []
        performance_records = []

        API_PUBMED = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        API_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

        def fetch_pubmed_doi(title, year):
            try:
                params = {
                    "db": "pubmed",
                    "term": f"{title} AND {year}[dp]",
                    "retmode": "json"
                }
                r = requests.get(API_PUBMED, params=params, timeout=15)
                if r.status_code != 200:
                    return None
                result = r.json()
                if "esearchresult" not in result or not result["esearchresult"]["idlist"]:
                    return None
                pmid = result["esearchresult"]["idlist"][0]
                # Fetch DOI
                r2 = requests.get(API_FETCH, params={"db": "pubmed", "id": pmid, "retmode": "json"}, timeout=15)
                if r2.status_code != 200:
                    return None
                summary = r2.json()
                doi = summary["result"].get(pmid, {}).get("elocationid", "")
                if doi and "doi" in doi.lower():
                    return doi.replace("doi:", "").strip()
                return None
            except:
                return None

        total_missing = original_missing_mask.sum()
        progress = st.progress(0)
        for i, idx in enumerate(merged[original_missing_mask].index):
            title = merged.at[idx, 'TI']
            year = merged.at[idx, 'PY']
            start = time.time()

            if pd.isna(title) or pd.isna(year):
                pubmed_not_found.append({"Index": idx, "Reason": "No title/year"})
                latency = time.time() - start
                performance_records.append({"Index": idx, "DOI": None, "Status": "No title/year", "Latency_sec": latency})
                continue

            doi = fetch_pubmed_doi(title, year)
            latency = time.time() - start
            if doi:
                merged.at[idx, 'DI'] = doi
                pubmed_found.append({"Index": idx, "DOI": doi})
                performance_records.append({"Index": idx, "DOI": doi, "Status": "Found", "Latency_sec": latency})
            else:
                pubmed_not_found.append({"Index": idx, "Reason": "DOI not found"})
                performance_records.append({"Index": idx, "DOI": None, "Status": "Not Found", "Latency_sec": latency})

            time.sleep(0.3)
            progress.progress((i + 1) / total_missing)

        pubmed_perf_df = pd.DataFrame(performance_records)
        median_latency = pubmed_perf_df['Latency_sec'].median()
        q1, q3 = pubmed_perf_df['Latency_sec'].quantile([0.25, 0.75])

        # Summary metrics
        total = len(pubmed_perf_df)
        success = (pubmed_perf_df['Status'] == "Found").sum()
        failure = total - success
        st.subheader("📊 PubMed DOI Recovery Performance")
        cols = st.columns(4)
        cols[0].metric("DOIs Submitted", total)
        cols[1].metric("Successful DOI Recovery", f"{success} ({success/total*100:.1f}%)")
        cols[2].metric("Failures", failure)
        cols[3].metric("Median Latency (IQR)", f"{median_latency:.2f}s ({q1:.2f}-{q3:.2f})")

        st.dataframe(pubmed_perf_df)
        st.download_button("📥 Download PubMed Performance CSV", pubmed_perf_df.to_csv(index=False), "PubMed_Performance.csv")

        st.success(f"✅ PubMed DOI search complete! Found: {len(pubmed_found)}, Not found: {len(pubmed_not_found)}")

    # -----------------------------
    # Step 3: Enrich with OpenAlex
    # -----------------------------
    st.header("Step 3: Enrich with OpenAlex")

    if st.button("🚀 Run OpenAlex Enrichment"):
        no_match_log = []
        performance_records = []

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
            latency = time.time() - start

            if not meta:
                no_match_log.append(doi)
                performance_records.append({"DOI": doi, "Status": "No Match", "Latency_sec": latency})
                continue

            status = "Success"
            if pd.isna(merged.at[idx, 'TI']): merged.at[idx, 'TI'] = meta.get('title')
            if pd.isna(merged.at[idx, 'AB']): merged.at[idx, 'AB'] = meta.get('abstract')
            if pd.isna(merged.at[idx, 'SO']): merged.at[idx, 'SO'] = meta.get('host_venue', {}).get('display_name')
            if pd.isna(merged.at[idx, 'PY']): merged.at[idx, 'PY'] = meta.get('publication_year')

            if 'authorships' in meta:
                au = '; '.join([a['author']['display_name'] for a in meta['authorships'] if a.get('author')])
                if pd.isna(merged.at[idx, 'AU']): merged.at[idx, 'AU'] = au
                if pd.isna(merged.at[idx, 'AF']): merged.at[idx, 'AF'] = au

            performance_records.append({"DOI": doi, "Status": status, "Latency_sec": latency})
            progress.progress((i + 1) / len(merged))

        openalex_perf_df = pd.DataFrame(performance_records)
        median_latency = openalex_perf_df['Latency_sec'].median()
        q1, q3 = openalex_perf_df['Latency_sec'].quantile([0.25, 0.75])

        total = len(openalex_perf_df)
        success = (openalex_perf_df['Status'] == "Success").sum()
        failure = total - success
        st.subheader("📊 OpenAlex Enrichment Performance")
        cols = st.columns(4)
        cols[0].metric("DOIs Submitted", total)
        cols[1].metric("Successful Enrichment", f"{success} ({success/total*100:.1f}%)")
        cols[2].metric("Failures", failure)
        cols[3].metric("Median Latency (IQR)", f"{median_latency:.2f}s ({q1:.2f}-{q3:.2f})")

        st.dataframe(openalex_perf_df)
        st.download_button("📥 Download OpenAlex Performance CSV", openalex_perf_df.to_csv(index=False), "OpenAlex_Performance.csv")

        st.success(f"✅ OpenAlex enrichment complete! Records updated: {success}")

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
                zipf.writestr('PubMed_Performance.csv', pubmed_perf_df.to_csv(index=False))
                zipf.writestr('OpenAlex_Performance.csv', openalex_perf_df.to_csv(index=False))

            buffer.seek(0)
            st.download_button("📦 Download All Results (ZIP)", buffer, "Bibliography_Enriched_Results.zip")
