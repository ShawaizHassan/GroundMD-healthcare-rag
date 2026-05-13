import streamlit as st
import requests
import time
import os

st.set_page_config(page_title="Medical Edge AI", layout="wide")

st.markdown("""
<style>
.chip-time {
    background-color: #e0e0e0;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-family: monospace;
    margin-right: 10px;
}
.query-history-item {
    padding: 8px;
    border-bottom: 1px solid #ddd;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏥 Medical Edge AI")
st.markdown("**PROCEDURE INTELLIGENCE SYSTEM**")

# API Base URL - works both in Docker and locally
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

with st.sidebar:
    st.header("Settings")
    top_k = st.selectbox("Top-K sources", [1, 2, 3, 5], index=2)
    show_raw = st.checkbox("Show raw JSON response")
    st.markdown("---")
    st.header("Query History")
    if "history" not in st.session_state:
        st.session_state.history = []
    for q in reversed(st.session_state.history[-10:]):
        st.markdown(f'<div class="query-history-item">📋 {q[:60]}...</div>', unsafe_allow_html=True)
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

st.markdown("### CLINICAL QUERY")
query = st.text_area("", height=100, placeholder="What is headache, how to get rid of it")

with st.expander("Example queries"):
    st.markdown("""
    - What is diabetes?
    - What are the HbA1c targets?
    - How to get rid of headache?
    """)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("Submit Query", type="primary")

if submit and query:
    st.session_state.history.append(query)
    start = time.time()
    with st.spinner("Processing..."):
        try:
            r = requests.post(f"{API_BASE}/api/query", json={"query": query}, timeout=60)
            sec = time.time() - start
            st.markdown(f'<span class="chip-time">{sec:.2f} s</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="chip-time">top_k = {top_k}</span>', unsafe_allow_html=True)
            if r.status_code == 200:
                data = r.json()
                st.success("Answer:")
                st.write(data.get("answer", "No answer"))
                if data.get("patient"):
                    st.info(f"👤 Patient detected: {data['patient']}")
                if show_raw:
                    with st.expander("Raw JSON Response"):
                        st.json(data)
            else:
                st.error(f"Error: {r.status_code}")
        except Exception as e:
            st.error("❌ Cannot reach backend. Is the server running?")