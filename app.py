# app.py
import streamlit as st
import pandas as pd
import requests
import time
import zipfile
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

    # Standardize
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
    # Step 2: Enrich with OpenAlex
    # -----------------------------
    st.header("Step 2: Enrich with OpenAlex")

    if st.button("🚀 Run OpenAlex Enrichment"):
        no_match_log = []
        updated = 0

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
            meta = fetch_openalex(doi)
            time.sleep(0.2)
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

        st.success(f"✅ OpenAlex enrichment complete! Records updated: {updated}")

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

            buffer.seek(0)
            st.download_button("📦 Download Enriched ZIP", buffer, "Bibliography_Enriched_OpenAlex.zip")
