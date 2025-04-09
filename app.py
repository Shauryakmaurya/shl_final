import streamlit as st
import pandas as pd
from Hybrid_Rag_Ui_Table import hybrid_search, call_llama, format_table
import re
import requests

# Configure page
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🧠 SHL Assessment Recommender (Hybrid RAG + LLaMA)")

# Query input
query = st.text_area("✍️ Enter Job Description, Query or JD Link", height=200)

# Filter controls
col1, col2, col3 = st.columns(3)
max_duration = col1.slider("⏱️ Max Duration (min)", 0, 120, 60)
remote_only = col2.checkbox("🌐 Only show Remote Tests")
adaptive_only = col3.checkbox("🧠 Only show Adaptive/IRT Tests")
search_filter = st.text_input("🔎 Filter Results by Keyword (optional)")

if st.button("🔍 Recommend Assessments") and query.strip():
    with st.spinner("Running Hybrid Search and LLaMA..."):
        # Optional: fetch page content if URL in query
        url_match = re.search(r'https?://\S+', query)
        if url_match:
            try:
                page = requests.get(url_match.group(0), timeout=5)
                query += f"\n\n[Page snippet]:\n{page.text[:3000]}"
            except:
                st.warning("Could not fetch content from URL")

        top_meta, top_docs = hybrid_search(query)
        df = format_table(top_meta)  # now returns DataFrame

        # Apply filters
        df = df[df["Duration (min)"].apply(lambda x: int(x) if str(x).isdigit() else 999) <= max_duration]
        if remote_only:
            df = df[df["Remote"] == "✅"]
        if adaptive_only:
            df = df[df["Adaptive/IRT"] == "✅"]
        if search_filter:
            df = df[df.apply(lambda row: search_filter.lower() in row.astype(str).str.lower().str.cat(), axis=1)]

        # ✅ Display DataFrame normally (auto dark-mode support)
        st.subheader("🔝 Filtered Matching Assessments")
        st.dataframe(df, use_container_width=True)

        # Download
        csv = df.to_csv(index=False)
        st.download_button("⬇️ Download Results as CSV", data=csv, file_name="recommended_assessments.csv", mime="text/csv")

        # LLaMA output
        context = "\n\n".join(top_docs)
        prompt = f"Here is the context of available SHL tests:\n\n{context}\n\nBased on this, suggest the most relevant assessments for the following job description or query:\n{query}"
        llama_output = call_llama(prompt)

        st.subheader("💡 LLaMA Recommendation")
        st.write(llama_output)
