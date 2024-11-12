import streamlit as st 
import os 
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chains import LLMChain


'''
System prompt: Research Paper Summarizer



'''

