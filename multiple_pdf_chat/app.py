import streamlit as st

st.title("Langchain App")

st.file_uploader(label="upload  file", type=["pdf", "docx", "txt"])
