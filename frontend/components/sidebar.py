import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")

        top_k = st.slider(
            "Top-K sources",
            1,
            10,
            3,
            help="Number of documents retrieved from the vector store",
        )
        show_raw = st.checkbox("Show raw JSON response", value=False)

        st.markdown("---")
        st.markdown("### 📋 Query History")

        if st.session_state.history:
            if st.button("🗑 Clear history", use_container_width=True):
                st.session_state.history = []
                st.rerun()

            for h in reversed(st.session_state.history[-10:]):
                st.markdown(
                    f"""
                    <div class="hist-entry">
                        <div class="hist-q">🔍 {h['query'][:55]}{'…' if len(h['query']) > 55 else ''}</div>
                        <div class="hist-ts">{h['ts']} · {h['status']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<span style='color:#5a7190;font-size:0.8rem'>No queries yet.</span>",
                unsafe_allow_html=True,
            )

    return top_k, show_raw