import streamlit as st
import requests
import json
import time
import os
from datetime import datetime

from components.chat import render_chat
from components.uploader import render_uploader
from components.sidebar import render_sidebar

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedQuery AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #fafbfe;
    --surface: #fafbfe;
    --border:  #1e2d45;
    --accent:  #00c9a7;
    --accent2: #0084ff;
    --warn:    #f5a623;
    --text:    #000;
    --muted:   #5a7190;
    --mono:    'IBM Plex Mono', monospace;
    --sans:    'IBM Plex Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg);
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label { color: var(--text) !important; font-size: 0.85rem; }

.med-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 24px;
    background: #fafbfe;
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 1.4rem;
}
.med-header .icon { font-size: 2.2rem; line-height: 1; }
.med-header h1 { margin: 0; font-size: 1.55rem; font-weight: 600; letter-spacing: -0.4px; color: #000; }
.med-header .sub { font-size: 0.78rem; color: var(--muted); font-family: var(--mono); letter-spacing: 0.5px; margin-top: 2px; }
.badge {
    margin-left: auto; font-family: var(--mono); font-size: 0.68rem;
    padding: 4px 10px; border-radius: 20px;
    background: rgba(0,201,167,0.12); color: var(--accent);
    border: 1px solid rgba(0,201,167,0.3); letter-spacing: 0.6px;
    text-transform: uppercase; white-space: nowrap;
}

.query-label { font-size: 0.72rem; font-family: var(--mono); color: var(--accent); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; }

[data-testid="stTextArea"] textarea {
    background: #fafbfe !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
    font-family: var(--sans) !important; font-size: 0.95rem !important;
    caret-color: var(--accent);
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,201,167,0.15) !important;
}
[data-testid="stTextArea"] textarea::placeholder {
    color:var(--muted);
}

.stButton > button {
    background: linear-gradient(135deg, #00c9a7, #0084ff) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    font-family: var(--mono) !important; font-size: 0.82rem !important;
    font-weight: 500 !important; letter-spacing: 0.8px !important;
    padding: 10px 24px !important; cursor: pointer !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stDownloadButton button{  padding: 9.3px 24px !important;}
.stDownloadButton button:hover,.stDownloadButton button:focus:not(:active){  border-color: #00c9a7 ;color: #00c9a7 }
.stDownloadButton button:focus-visible{box-shadow: rgba(255, 75, 75, 0.5) 0px 0px 0px 0.2rem;}
.stDownloadButton button:active{  border-color: #00c9a7 ;background:#00c9a7;color: white }
.answer-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 22px 24px; margin-bottom: 1.2rem; border-left: 3px solid var(--accent);
}
.card-label { font-size: 0.68rem; font-family: var(--mono); color: var(--accent); letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 12px; }
.answer-text { font-size: 0.96rem; line-height: 1.75; color: var(--text); }

.conf-row { display: flex; align-items: center; gap: 12px; margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--border); }
.conf-label { font-size: 0.72rem; font-family: var(--mono); color: var(--muted); }
.conf-bar { flex: 1; height: 5px; background: #1a2840; border-radius: 4px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 4px; }
.conf-pct { font-size: 0.8rem; font-family: var(--mono); font-weight: 500; color: var(--accent); min-width: 36px; text-align: right; }

.sources-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 22px; margin-bottom: 1.2rem; }
.source-item { display: flex; align-items: flex-start; gap: 12px; padding: 10px 0; border-bottom: 1px solid #111f33; }
.source-item:last-child { border-bottom: none; padding-bottom: 0; }
.src-num { font-family: var(--mono); font-size: 0.7rem; color: var(--accent); background: rgba(0,201,167,0.1); border: 1px solid rgba(0,201,167,0.25); border-radius: 4px; padding: 2px 7px; min-width: 28px; text-align: center; margin-top: 1px; }
.src-text { font-size: 0.86rem; color: var(--text); line-height: 1.5; }
.src-meta { font-size: 0.72rem; color: var(--muted); font-family: var(--mono); margin-top: 2px; }

.status-row { display: flex; gap: 10px; align-items: center; margin-bottom: 1.2rem; flex-wrap: wrap; }
.chip { font-family: var(--mono); font-size: 0.68rem; padding: 4px 10px; border-radius: 20px; letter-spacing: 0.5px; }
.chip-ok   { background: rgba(0,201,167,0.12); color: var(--accent); border: 1px solid rgba(0,201,167,0.3); }
.chip-err  { background: rgba(245,90,90,0.12);  color: #f55a5a;     border: 1px solid rgba(245,90,90,0.3); }
.chip-mock { background: rgba(245,166,35,0.12); color: var(--warn); border: 1px solid rgba(245,166,35,0.3); }
.chip-time { background: rgba(90,113,144,0.15); color: var(--muted); border: 1px solid var(--border); }

.hist-entry { padding: 10px 14px; border-radius: 8px; background:var(--bg); border: 1px solid var(--border); margin-bottom: 8px; }
.hist-q { font-size: 0.82rem; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hist-ts { font-size: 0.68rem; color: var(--muted); font-family: var(--mono); margin-top: 3px; }

.mock-notice { background: rgba(245,166,35,0.08); border: 1px solid rgba(245,166,35,0.25); border-radius: 8px; padding: 10px 14px; font-size: 0.78rem; color: var(--warn); font-family: var(--mono); margin-bottom: 1rem; }
[data-testid="stExpander"] summary:hover {color: #00c9a7 }
[data-testid="stExpander"] summary:hover svg {fill: #00c9a7 }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# ── Configuration ─────────────────────────────────────────────────────────────
MOCK_MODE = False
API_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8000")
# ── Mock data ─────────────────────────────────────────────────────────────────
MOCK_RESPONSES = [
    {
        "answer": (
            "Based on retrieved procedural guidelines, the standard pre-operative checklist "
            "for laparoscopic cholecystectomy includes NPO status (≥6 h solids, ≥2 h clear liquids), "
            "informed consent, Cefazolin 2 g IV prophylaxis, DVT prophylaxis assessment, and anaesthesia "
            "evaluation. Pneumoperitoneum is established at 12–15 mmHg CO₂."
        ),
        "confidence": 0.91,
        "sources": [
            {"title": "SAGES Laparoscopic Cholecystectomy Guidelines 2023", "page": "pp. 14–17", "relevance": 0.94},
            {"title": "WHO Surgical Safety Checklist v2 (2022)", "page": "Section 3", "relevance": 0.88},
            {"title": "ACS NSQIP Pre-operative Protocol", "page": "pp. 5–8", "relevance": 0.82},
        ],
        "status": "success",
    },
    {
        "answer": (
            "For central venous catheter (CVC) insertion, evidence-based protocol mandates ultrasound "
            "guidance (Level 1A), maximal sterile barrier precautions, chlorhexidine-alcohol skin antisepsis, "
            "and subclavian site preference for CLABSI reduction. Post-insertion CXR is required to confirm "
            "tip position at the SVC-RA junction before first use."
        ),
        "confidence": 0.87,
        "sources": [
            {"title": "CDC Guidelines for Intravascular Catheter Infections 2024", "page": "pp. 22–26", "relevance": 0.92},
            {"title": "NEJM CVC Best Practices Review", "page": "Vol 388 p.1123", "relevance": 0.85},
        ],
        "status": "success",
    },
]

def call_backend(query: str, top_k: int) -> dict:
    if MOCK_MODE:
        import random
        time.sleep(1.1)
        r = random.choice(MOCK_RESPONSES).copy()
        r["query"] = query
        return r

    endpoint = f"{API_BASE_URL}/api/query"
    try:
        resp = requests.post(endpoint, json={"query": query, "top_k": top_k}, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "error": "Cannot reach backend. Is the server running?"}
    except requests.exceptions.Timeout:
        return {"status": "error", "error": "Request timed out after 30 s."}
    except requests.exceptions.HTTPError:
        return {"status": "error", "error": f"HTTP {resp.status_code} from backend."}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# # ── Sidebar ───────────────────────────────────────────────────────────────────
top_k, show_raw = render_sidebar()

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="med-header">
    <div class="icon">🏥</div>
    <div>
        <h1>Medical Edge AI</h1>
        <div class="sub">PROCEDURE INTELLIGENCE SYSTEM </div>
    </div>
    <div class="badge">v1 · MVP</div>
</div>
""", unsafe_allow_html=True)

if MOCK_MODE:
    st.markdown("""
    <div class="mock-notice">
        ⚠ MOCK MODE ACTIVE — responses are simulated. Set <code>MOCK_MODE = False</code>
        and update <code>API_BASE_URL</code> to connect to the live backend.
    </div>
    """, unsafe_allow_html=True)

# left_col, right_col = st.columns([3, 1], gap="large")

# with left_col:
query, submit = render_chat()

# with right_col:
#     uploaded_files = render_uploader()

st.markdown("---")

# ── Execute query ─────────────────────────────────────────────────────────────
if submit:
    if not query.strip():
        st.warning("Please enter a clinical query before submitting.")
    else:
        t0 = time.time()
        with st.spinner("Retrieving context and generating answer…"):
            result = call_backend(query.strip(), top_k)
        elapsed = time.time() - t0

        st.session_state.last_result = result
        st.session_state.last_elapsed = elapsed
        st.session_state.history.append({
            "query": query.strip(),
            "ts": datetime.now().strftime("%H:%M:%S"),
            "status": result.get("status", "unknown"),
        })

# ── Render result ─────────────────────────────────────────────────────────────
result = st.session_state.get("last_result")

if result:
    status = result.get("status", "unknown")
    elapsed = st.session_state.get("last_elapsed", 0)

    chip_s = "chip-ok" if status == "success" else "chip-err"
    mock_chip = '<span class="chip chip-mock">⚠ MOCK</span>' if MOCK_MODE else ""

    st.markdown(f"""
    <div class="status-row">
        <span class="chip {chip_s}">● {status.upper()}</span>
        {mock_chip}
        <span class="chip chip-time">⏱ {elapsed:.2f} s</span>
        <span class="chip chip-time">top_k = {top_k}</span>
    </div>
    """, unsafe_allow_html=True)

    if status == "error":
        st.error(f"❌ {result.get('error', 'Unknown error from backend.')}")
    else:
        answer = result.get("answer", "No answer returned.")
        st.markdown(f"""
        <div class="answer-card">
            <div class="card-label">🤖 AI Answer</div>
            <div class="answer-text">{answer}</div>
        """, unsafe_allow_html=True)

        confidence = result.get("confidence")
        if confidence is not None:
            pct = int(confidence * 100)
            color = "#00c9a7" if pct >= 75 else "#f5a623" if pct >= 50 else "#f55a5a"
            st.markdown(f"""
            <div class="conf-row">
                <span class="conf-label">CONFIDENCE</span>
                <div class="conf-bar">
                    <div class="conf-fill" style="width:{pct}%; background:{color}"></div>
                </div>
                <span class="conf-pct">{pct}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        citations = result.get("citations", [])
        st.markdown(f"""
        <div class="sources-wrap">
            <div class="card-label">📚 Retrieved Sources ({len(citations)})</div>
        """, unsafe_allow_html=True)

        if citations:
            for i, citation in enumerate(citations):
                st.markdown(f"""
                <div class="source-item">
                    <span class="src-num">{i+1:02d}</span>
                    <div>
                        <div class="src-text">{citation}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='color:#5a7190;font-size:0.83rem'>No citations returned.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        if show_raw:
            with st.expander("🔧 Raw JSON response"):
                st.json(result)

        col_dl, col_new, _ = st.columns([1, 1,1])
        with col_dl:
            st.download_button(
                "⬇ Download JSON",
                data=json.dumps(result, indent=2),
                file_name=f"medquery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_new:
            if st.button("🔄 New query", use_container_width=True):
                st.session_state.last_result = None
                st.rerun()