# Document Question Answering System

This project is a document question answering system that allows users to load PDF documents, preprocess them, and search for answers to specific questions based on the document content. The system utilizes Natural Language Processing (NLP) techniques such as text splitting, embedding generation, and vector similarity search to efficiently retrieve relevant information from the documents.

## Features

- Load PDF documents from a specified folder path
- Preprocess documents by splitting them into chunks and generating embeddings
- Create a vector database using FAISS for efficient similarity search
- Perform question answering by retrieving relevant text passages and generating natural language answers
- Evaluate the system's performance using provided test questions and expected answers

## Installation

1. Clone the repository: git clone https://github.com/your-username/document-qa-system.git
cd document-qa-system
2. Install the required Python packages:pip install -r requirements.txt
3. ## Usage

1. Mount your Google Drive in the Colab environment:

```python
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
2. Update the clients dictionary in main.py with the appropriate file paths and client details.
3. Run the main.py script: python main.py


This will load the specified PDF documents, preprocess them, set up the vector database, and run the evaluation process for each configured client.

Project Structure
main.py: The entry point of the application, responsible for orchestrating the document loading, preprocessing, and evaluation processes.
document_loader.py: Contains the DocumentLoader class for loading PDF documents from various sources.
text_splitter.py: Defines the TextSplitter class for splitting documents into chunks.
embedding_generator.py: Implements the EmbeddingGenerator class for generating embeddings from text.
document_preprocessor.py: Provides the DocumentPreprocessor class for preprocessing documents by splitting and generating embeddings.
faiss_vectordb.py: Contains the FAISSVectorDB class for creating and searching a vector database using FAISS.
llama_reader.py: Implements the LlamaReader class for generating natural language answers based on the retrieved information.
evaluation_component.py: Defines the EvaluationComponent class for evaluating the system's performance against a set of test questions.
logger.py: Contains a singleton Logger class for consistent logging throughout the application.
Configuration
The application can be configured by modifying the clients dictionary in main.py. Each client should have the following keys:

input_path: The path to the folder containing the PDF documents for the client.


## Hardware Requirements

This project utilizes the LLaMA  LLaMA is a computationally intensive model, and running it effectively may require a GPU (Graphics Processing Unit) with sufficient memory and computing power.





