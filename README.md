# Q&A Chatbot for Financial News

This project is a user-friendly news research tool built with Streamlit. The chatbot allows users to input article URLs and ask questions related to stock market and financial topics. The app retrieves relevant insights from the provided articles and uses custom data to answer questions.

## Project Architecture

The chatbot leverages the following components:

1. **Document Loader**: Loads articles from URLs provided by users.
2. **Text Splitting**: Splits long articles into manageable chunks for efficient processing.
3. **Vector Database**: Converts text chunks into vector representations for semantic search.
4. **Retrieval**: Retrieves the most relevant chunks based on user questions.
5. **Prompting**: Generates responses by combining the retrieved information and answering user queries.

## Key Features

- Input article URLs related to the stock market and finance.
- Ask questions about the articles and get accurate answers based on the content.
- Focus on the custom data provided.

