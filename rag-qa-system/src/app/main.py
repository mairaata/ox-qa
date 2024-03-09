
import os
from google.colab import drive
from document_loader import DocumentLoader
from text_splitter import TextSplitter
from embedding_generator import EmbeddingGenerator
from document_preprocessor import DocumentPreprocessor
from faiss_vectordb import FAISSVectorDB
from llama_reader import LlamaReader
from evaluation_component import EvaluationComponent

def mount_google_drive():
    drive.mount('/content/gdrive', force_remount=True)

def load_documents(folder_path):
    doc_loader = DocumentLoader()
    documents = doc_loader.load_pdf_langchain(folder_path)
    print(f'{len(documents)} documents loaded')
    return documents

def setup_pipeline(documents):
    txt_splitter = TextSplitter()
    embeddings = EmbeddingGenerator()
    doc_prsr = DocumentPreprocessor(txt_splitter, embeddings)
    fs_db = FAISSVectorDB(doc_prsr)
    fs_db.add(documents)
    return fs_db

def main():
    mount_google_drive()

    clients = {
        'Advent': {
            'input_path': '/content/gdrive/MyDrive/fundfact/DDQ/AdventPolicy',
            'question_path': '/content/gdrive/MyDrive/fundfact/test_advent3.xlsx',
            'FUND': 'Advent International GPE X-A SCSp',
            'MANAGER_OR_ADVISOR': 'Advent International Fund Manager Sàrl',
            'client_name': 'advent'
        },
        # 'RoboCap': {...},
        # 'Chorus': {...},
        # 'Legalist': {...}
    }

    output_path = '/content/gdrive/MyDrive/fundfact/DDQ/'

    for client_name, client_details in clients.items():
        print(f'Processing {client_name}')
        print(client_details)

        documents_path = client_details['input_path']
        question_path = client_details['question_path']

        documents = load_documents(documents_path)
        vectordb = setup_pipeline(documents)
        reader = LlamaReader(vectordb)

        ev_obj = EvaluationComponent(documents_path, question_path, output_path, client_details)
        ev_obj.run_evaluation()

if __name__ == "__main__":
    main()