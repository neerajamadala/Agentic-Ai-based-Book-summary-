import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.title("📚 AI Book Summary & Q/A (Groq + Llama3)")

uploaded_file = st.file_uploader("📂 Upload PDF Book", type=["pdf"])
question = st.text_input("🤖 Ask anything about the book")

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)




# Free embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(text)
    st.write(f"📄 Chunks: {len(chunks)}")

    db = Chroma.from_texts(chunks, embedding=embeddings)
    retriever = db.as_retriever()

    # --- Simple RAG Chain ---
    system_prompt = ChatPromptTemplate.from_template(
        "Use the retrieved context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | system_prompt
        | llm
    )

    if st.button("📘 Generate Summary"):
        prompt = f"Summarize the book into 10 bullet points:\n\n{text[:12000]}"
        st.write(llm.invoke(prompt))

    if question:
        st.subheader("💡 Answer")
        st.write(rag_chain.invoke(question))
