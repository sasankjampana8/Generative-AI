import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai




def main(model):
    st.set_page_config(page_title="Chat with multiple PDF's", page_icon=":books:")
    st.header("Chat with multiple PDF's :books:")
    #st.title("Langchain App")

    request = st.text_input("Ask a question:")
    response = None
    if request:
    # If input is provided, process the request
        response = model.generate_content(request)
    else:
        # If input is empty, show a message or warning
        st.warning("Please enter a question to proceed.")
    
    #adding a side bar
    with st.sidebar:
        st.subheader("Your Documents")
        st.file_uploader(label="upload your PDF's here and Click on 'Process'", type=["pdf"])
        st.button("Process")
    
    return response
    

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
