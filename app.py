
import os
import hashlib
import tempfile
import streamlit as st

from dotenv import load_dotenv


#load environment variables
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")


#page configuration
st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📄"
)


#check api key
if not API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file")
    st.stop()


#import backend
from rag_backend import create_vector_store, get_answer


#title
st.title("📄 PDF Question Answering")

st.write("Upload a PDF and ask questions about it.")


#upload pdf
uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


if uploaded_pdf:

    #read uploaded pdf
    pdf_bytes = uploaded_pdf.getvalue()

    #create unique pdf hash
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]

    collection_name = "pdf_" + pdf_hash


    #check if a new pdf is uploaded
    if (
        "vector_store" not in st.session_state
        or st.session_state.get("pdf_hash") != pdf_hash
    ):

        #temporary pdf file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(pdf_bytes)
            pdf_path = tmp.name


        try:

            #create vector store
            with st.spinner("Reading PDF and creating embeddings..."):

                vector_store = create_vector_store(
                    pdf_path,
                    collection_name
                )

                st.session_state.vector_store = vector_store
                st.session_state.pdf_hash = pdf_hash
                st.session_state.pdf_name = uploaded_pdf.name

            st.success("PDF processed successfully!")

        except Exception as e:

            st.error(f"Error processing PDF: {str(e)}")

            #remove old vector store state after failure
            st.session_state.pop("vector_store", None)
            st.session_state.pop("pdf_hash", None)
            st.session_state.pop("pdf_name", None)

        finally:

            #remove temporary file
            if os.path.exists(pdf_path):
                os.remove(pdf_path)


    #display uploaded file
    if st.session_state.get("pdf_hash") == pdf_hash:

        st.write("Current PDF:", uploaded_pdf.name)


        #question input
        question = st.text_input(
            "Enter your question"
        )


        #get answer
        if st.button("Get Answer"):

            if question.strip() == "":

                st.warning("Please enter a question.")

            else:

                try:

                    with st.spinner("Generating answer..."):

                        answer = get_answer(
                            st.session_state.vector_store,
                            question
                        )


                    #display answer
                    st.markdown("### Answer")

                    st.write(answer)

                except Exception as e:

                    st.error(f"Error generating answer: {str(e)}")
