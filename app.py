import streamlit as st
import pandas as pd
from Hybrid_Rag_Ui_Table import query_rag_system, hybrid_search, format_table
import base64

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #0e1117; }
        .stButton>button { color: white; background-color: #0099ff; }
        .css-1cpxqw2 edgvbvh3 { background-color: #0e1117; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 SHL Assessment Recommender (Hybrid RAG + LLaMA)")

query = st.text_area("Enter job description, candidate profile, or query to get matching assessments:", height=200)

if st.button("🔍 Recommend Assessments"):
    if query.strip():
        with st.spinner("Searching and generating recommendations..."):
            results, _ = hybrid_search(query)
            if results:
                st.markdown("""
                <h2 style='margin-top: 40px;'>
                    🔺 Filtered Matching Assessments
                </h2>
                """, unsafe_allow_html=True)

                html_table = format_table(results)
                st.components.v1.html(html_table, height=500, scrolling=True)

                # Download as CSV
                df = pd.DataFrame(results)
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                st.markdown(f'<a href="data:file/csv;base64,{b64}" download="matching_assessments.csv" class="button">⬇️ Download Results as CSV</a>', unsafe_allow_html=True)
            else:
                st.warning("No matching assessments found. Please try a different query.")
    else:
        st.warning("Please enter a query to get recommendations.")
