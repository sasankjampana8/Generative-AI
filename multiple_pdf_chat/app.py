import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain, ConversationalRetrievalChain
)
import pdfplumber
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from htmlTemplates import css, bot_template, user_template

# Set Streamlit page configuration as the first Streamlit command
st.set_page_config(page_title="Chat with multiple PDF's", page_icon=":books:")

def get_conversation_chain(vectorstore, model):
    """Create a conversational chain using a vector store and generative model."""
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = create_history_aware_retriever(model, vectorstore.as_retriever(), memory)
    return conversation_chain

def get_vectorstore(chunks):
    Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
    model_name = "hkunlp/instructor-large"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceInstructEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    documents = [Document(page_content=chunk) for chunk in chunks]

    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=hf,
        index_name='pdfs')
    
    return vectorstore

def extract_from_pdf(pdfs):
    text = ""
    for pdf in pdfs:
        with pdfplumber.open(pdf) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators='\n', length_function=len)
    chunks = splitter.split_text(text)
    return chunks

def main():
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    st.header("Chat with multiple PDF's :books:")
    st.text_input("Ask a question about your documents:")
    
    st.write(user_template.replace("{{MSG}}", "Hello robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("Your Documents")
        pdfs = st.file_uploader(label="Upload your PDF's here and Click on 'Process'", type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = extract_from_pdf(pdfs)
                text_chunks = split_text(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore, model)
    
if __name__ == '__main__':
    main()
