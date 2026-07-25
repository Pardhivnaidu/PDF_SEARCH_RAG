import os
import tempfile
import streamlit as st

from rag_backend import create_vector_store, get_answer

st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📄"
)

st.title("📄 PDF Question Answering")
st.write("Upload a PDF and ask questions about it.")

# Read API key from environment variable (Render)
API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY is None:
    st.error("GOOGLE_API_KEY environment variable is not set.")
    st.stop()




import hashlib

st.write("Key loaded:", bool(api_key))
st.write("Key length:", len(api_key))
st.write("Key fingerprint:",hashlib.sha256(api_key.encode()).hexdigest()[:12])






@st.cache_resource(show_spinner=False)
def load_vector_store(pdf_path, api_key):
    return create_vector_store(pdf_path, api_key)


uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_pdf:

    if (
        "vector_store" not in st.session_state
        or st.session_state.get("pdf_name") != uploaded_pdf.name
    ):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_pdf.read())
            pdf_path = tmp.name

        with st.spinner("Reading PDF and creating embeddings..."):

            vector_store = load_vector_store(
                pdf_path,
                API_KEY
            )

            st.session_state.vector_store = vector_store
            st.session_state.pdf_name = uploaded_pdf.name

        st.success("PDF processed successfully!")

question = st.text_input(
    "Ask a question"
)

if st.button("Get Answer"):

    if "vector_store" not in st.session_state:

        st.warning("Please upload a PDF first.")

    elif question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Generating answer..."):

            answer = get_answer(
                st.session_state.vector_store,
                question,
                API_KEY
            )

        st.markdown("### Answer")
        st.write(answer)
