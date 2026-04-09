import streamlit as st

def render_uploader():
    st.markdown("### 📤 Upload Files")

    uploaded_files = st.file_uploader(
        "Upload supporting files",
        type=["pdf", "txt", "docx", "json", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded")
        for file in uploaded_files:
            st.markdown(f"- {file.name}")

    return uploaded_files