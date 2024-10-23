import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import conversational_retrieval
import pdfplumber


def get_conversation_chain(vectorstore, model, request):
    context =  vectorstore.get_context(request)
    question = f"Answer the following question {request}. with respect to the context: {context}"
    response = model.generate_content(question)
    return response

def get_vectorstore(chunks):
    #define the embedding model as well as the vector store
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-large')
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def extract_from_pdf(pdfs):
    text = ""
    for  pdf in pdfs:
        with pdfplumber.open(pdf) as pdf:
            for page in  pdf.pages:
                text += page.extract_text()
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200, separators='\n', length_function = len)
    chunks = splitter.split_text(text)
    return chunks




def main(model):
    st.set_page_config(page_title="Chat with multiple PDF's", page_icon=":books:")
    st.header("Chat with multiple PDF's :books:")
    #st.title("Langchain App")

    request = st.text_input("Ask a question:")
    response = None
    # if request:
    # # If input is provided, process the request
    #     response = model.generate_content(request)
    # else:
    #     # If input is empty, show a message or warning
    #     st.warning("Please enter a question to proceed.")
    
    #adding a side bar
    with st.sidebar:
        st.subheader("Your Documents")
        pdfs = st.file_uploader(label="upload your PDF's here and Click on 'Process'", type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text =  extract_from_pdf(pdfs)
                
                text_chunks =  split_text(raw_text)
                
                vectorstore  = get_vectorstore(text_chunks)
                
                conversation = get_conversation_chain(vectorstore, model, request)
                
                
        
    
    return conversation
    

if __name__=='__main__':
    api_key = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = main(model)
    print(response)
    if response:
        st.write("Answer:", response.text)
    else:
        st.write("No response")  # If no response is received, display a message
