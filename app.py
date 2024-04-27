import os
import requests
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


@st.cache_resource 
def create_embeddings():
    pdf_url = st.secrets["PDF_URL"]   
    # Downloading and extracting PDF from URL
    with requests.get(pdf_url, verify=False) as response:
        with open("temp_pdf.pdf", "wb") as pdf_file:
            pdf_file.write(response.content)
    
    # Reading PDF extracted
    with open("temp_pdf.pdf", "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
        )        
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

def main():
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    st.set_page_config(page_title='Chatea con la Ley 769!')
    st.header('Preguntale a la ley 769')
    
    knowledge_base = create_embeddings()
    # Getting user input
    user_question = st.text_input('Preguntale a la ley 769:', key='question')

    if user_question:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        docs = knowledge_base.similarity_search(user_question, 3)
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm, chain_type="stuff")
        cost = ""
        with get_openai_callback() as cb:
            respuesta = chain.run(input_documents=docs, question=user_question)
            cost = cb     
        st.write(respuesta) 
        st.write("---------------")    
        st.write("cost info:",cost)    


        
if __name__ == '__main__':
    main()