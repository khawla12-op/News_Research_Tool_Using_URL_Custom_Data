import streamlit as st
import os
import pickle
import time
import langchain
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit app layout
st.title("News Research Tool")
st.sidebar.title("News Article URLs")

# Get URLs from user input
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# Button to start processing
process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# Define faiss_directory outside the condition
faiss_directory = "faiss_index"
embeddings = OpenAIEmbeddings()
llm = OpenAI(temperature=0.9,max_tokens=500)
# Process URLs
if process_url_clicked:
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading started")
    data = loader.load()

    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Data Splitting started")
    docs = text_splitter.split_documents(data)

    # # Create embeddings
    # embeddings = OpenAIEmbeddings()

    # Check if FAISS vector index exists, if not create and save it
    if not os.path.exists(faiss_directory):
        vectorindex_openai = FAISS.from_documents(docs, embeddings)
        vectorindex_openai.save_local(faiss_directory)  # Save the index to 'faiss_index' folder

    # Load the saved FAISS index
    vectorindex_openai = FAISS.load_local(faiss_directory, embeddings, allow_dangerous_deserialization=True)
    main_placeholder.text("Embedding vector Started")

# Input field for the user’s query
query = main_placeholder.text_input("Question:")

# Process query if provided
if query:
    # Check if FAISS index directory exists and load the vector index
    if os.path.exists(faiss_directory):
        vectorindex_openai = FAISS.load_local(faiss_directory, embeddings, allow_dangerous_deserialization=True)

        # Use the vector index as a retriever
        retriever = vectorindex_openai.as_retriever()

        # Define the LLM (you need to make sure `llm` is initialized somewhere in your code)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        # Run the query and get the result
        result = chain({"question": query}, return_only_outputs=True)

        # Display the result
        st.header("Answer")
        st.subheader(result["answer"])
        #Dispalying sources:
        sources=result.get("sources","")
        if sources:
            st.subheader("Sources:")
            sources_list=sources.split("\n")
            for source in sources_list:
                st.write(source)
