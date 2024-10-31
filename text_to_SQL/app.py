import streamlit as st 
import os 
import google.generativeai as genai
from langchain_community.llms import Ollama
import psycopg2
from dotenv import load_dotenv 

def connect_to_db():
    conn = psycopg2.connect(database=os.getenv('db_name'), user=os.getenv('db_user'), password=os.getenv('db_password'), host=os.getenv('db_host'), port=os.getenv('db_port'))
    return conn

def get_db_data(conn, sql):
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    conn.close() 
    return results

def create_sql(query):
    model = Ollama(model='llama3')    
    system_prompt = '''You are an expert at writing PostgreSQL queries. Write a PostgreSQL query for the Customer database, which contains a single table called customer_master with columns: cif_id, customer_name, age, city, occupation, and risk_profile. Only return the SQL query and nothing else.No explanation requried. just the final sql query.
'''
    result = model.invoke(query)
    print(result)
    return result

    

def main():
    load_dotenv()
    st.set_page_config(page_title="Text to SQL", page_icon=":balloon:")
    conn =  connect_to_db()
    user_input = st.text_input(label="Enter your query:")
    if user_input:
        sql = create_sql(user_input)
        results = get_db_data(conn=conn, sql=sql)
        st.write(results)    

if __name__=="__main__":
    main()