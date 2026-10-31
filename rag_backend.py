
import os
import chromadb

from dotenv import load_dotenv

#doc loading
from langchain_community.document_loaders import PyPDFLoader

#text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

#embedding and vector store
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

#prompt and output parser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")


#initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)


#initialize gen ai
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=API_KEY
)

#load pdf and create vector store
def create_vector_store(pdf_path, collection_name):

    #loader
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    print("number of pages : ", len(documents))


    #split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print("number of chunks : ", len(chunks))


    #remove old collection if it exists
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


    #vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name=collection_name
    )

    print("documents stored in chroma db")

    return vectorstore


#convert vector store into retriever
def get_answer(vectorstore, question):

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )


    #retrieve documents
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return "I don't know."


    #create context
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )


    #prompt
    prompt = PromptTemplate(
        template="""You are a helpful PDF question answering AI assistant.

        Answer from below context only.
        If the answer is not present in the context, just say I don't know.

        context = {context}

        question = {question}
        """,
        input_variables=["context", "question"]
    )


    #parser
    parser = StrOutputParser()


    #chain
    chain = prompt | llm | parser


    #generate answer
    ans = chain.invoke({
        "context": context,
        "question": question
    })

    return ans
