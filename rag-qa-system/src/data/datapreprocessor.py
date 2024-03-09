from abc import ABC, abstractmethod
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from typing import List, Dict
from typing import Tuple
import time
from utils.Logger import Logger
logger = Logger.get_logger()

class TextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 20):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split(self, documents: List[str]) -> List[str]:
        """
        Splits the given documents into smaller chunks.

        Args:
            documents (List[str]): A list of document texts.

        Returns:
            List[str]: A list of split document chunks.
        """
        return self.splitter.split_documents(documents)

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device: str = 'cpu'):
        """
        Initializes the EmbeddingGenerator with the specified model and device.

        Args:
            model_name (str, optional): The name of the HuggingFace model for generating embeddings. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
            device (str, optional): The device to use for generating embeddings ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        model_kwargs = {"device": device}
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    def generate(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for the given texts.

        Args:
            texts (List[str]): A list of text strings.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of float values.
        """
        return self.embeddings.embed_documents(texts)

class DocumentPreprocessor:
    def __init__(self, splitter: TextSplitter, embedding_generator: EmbeddingGenerator):
        self.splitter = splitter
        self.embedding_generator = embedding_generator

    def process(self, documents: List[str]) -> Tuple[List[str], List[List[float]]]:
        """
        Processes the given documents by splitting them into chunks and generating embeddings.

        Args:
            documents (List[str]): A list of document texts.

        Returns:
            Tuple[List[str], List[List[float]]]: A tuple containing the list of split document chunks and their corresponding embeddings.
        """
        texts = self.splitter.split(documents)
        embeddings = self.embedding_generator.generate(texts)
        return texts, embeddings

class VectorDB(ABC):
    @abstractmethod
    def add(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """
        Adds the given texts and their corresponding embeddings to the vector database.

        Args:
            texts (List[str]): A list of text strings.
            embeddings (List[List[float]]): A list of embeddings, where each embedding is a list of float values.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 10) -> List[str]:
        """
        Searches the vector database for the most relevant texts based on the given query.

        Args:
            query (str): The query text.
            k (int, optional): The number of top results to return. Defaults to 10.

        Returns:
            List[str]: A list of the most relevant text strings from the vector database.
        """
        pass

class FAISSVectorDB(VectorDB):
    def __init__(self, preprocessor: DocumentPreprocessor):
        self.preprocessor = preprocessor
        self.vectordb = None

    def add(self, documents: List[str]) -> None:
        """
        Adds the given documents to the FAISS vector database.

        Args:
            documents (List[str]): A list of document texts.
        """
        start_time = time.time()
        texts, embeddings = self.preprocessor.process(documents)
        end_time = time.time()
        logger.info(f"Processed {len(documents)} documents in {end_time - start_time:.2f} seconds")

        start_time = time.time()
        self.vectordb = FAISS.from_documents(texts, embeddings)
        end_time = time.time()
        logger.info(f"Vector DB creation for {len(documents)} documents in {end_time - start_time:.2f} seconds")

    def search(self, query: str, k: int = 10) -> List[str]:
        """
        Searches the FAISS vector database for the most relevant texts based on the given query.

        Args:
            query (str): The query text.
            k (int, optional): The number of top results to return. Defaults to 10.

        Returns:
            List[str]: A list of the most relevant text strings from the vector database.
        """
        if self.vectordb is None:
            logger.warning("Vector database is not initialized. Call `add` method first.")
            return []

        query_embedding = self.preprocessor.embedding_generator.generate([query])[0]
        top_results = self.vectordb.similarity_search_with_score(query_embedding, k=k)
        return [self.vectordb.docstore.search_by_id(doc_id)[0] for doc_id, _ in top_results]