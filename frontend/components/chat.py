import streamlit as st

def render_chat():
    st.markdown('<div class="query-label">Clinical Query</div>', unsafe_allow_html=True)

    col_q, col_btn = st.columns([5, 1], gap="small")

    with col_q:
        query = st.text_area(
            "query",
            label_visibility="collapsed",
            placeholder="e.g. What is the pre-operative checklist for laparoscopic cholecystectomy?",
            height=100,
        )

    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        submit = st.button("⚡ QUERY", use_container_width=True)

    with st.expander("💡 Example queries"):
        examples = [
            "Pre-operative checklist for laparoscopic cholecystectomy?",
            "Steps for central venous catheter insertion?",
            "Post-operative care protocol for CABG patients?",
            "Antibiotic prophylaxis for colorectal surgery?",
            "Emergency appendectomy checklist for paediatric patients?",
        ]

        c1, c2 = st.columns(2)
        for i, ex in enumerate(examples):
            with (c1 if i % 2 == 0 else c2):
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    st.session_state["prefill"] = ex
                    st.rerun()

    if "prefill" in st.session_state and st.session_state["prefill"]:
        query = st.session_state.pop("prefill")
        submit = True

    return query, submit