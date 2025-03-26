import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOllama
from langchain_community.callbacks.manager import get_openai_callback

def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase

def main():
    st.title("📄PDF Summarizer")
    st.write("Created by Abul Al Arabi")
    st.divider()

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # Text variable will store the pdf text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Create the knowledge base object
        knowledgeBase = process_text(text)

        query = "Summarize the content of the uploaded PDF file in approximately 3-5 sentences for a research work background study. Focus on capturing the main ideas and key points discussed in the document and ensure clarity and coherence in the summary. Present numerical data from the paper if available."

        if query:
            docs = knowledgeBase.similarity_search(query)
            llm = ChatOllama(
                model="llama3.1:latest",
                temperature=0.1,
                keep_alive="1h",
                max_tokens=40000,
            )
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)

            st.subheader('Summary Results:')
            st.write(response)

if __name__ == '__main__':
    main()