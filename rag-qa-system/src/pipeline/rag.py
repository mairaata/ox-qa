from abc import ABC, abstractmethod
from data.datapreprocessor import VectorDB 

class Retriever(ABC):
    """Abstract retriever interface"""

    @abstractmethod
    def retrieve(self, query):
        pass


class Reader(ABC):
    """Abstract reader interface"""

    @abstractmethod
    def read(self, query):
        pass



class RAG:
    """Orchestrates the RAG components"""

    def __init__(self,
                 reader: Reader,
                 indexer: VectorDB):

        self.indexer = indexer
        self.retriever = self.indexer
        self.reader = reader

    def index(self, corpus):
        self.indexer.index(corpus)

    def answer(self, question):
        docs = self.retriever.retrieve(question)
        return self.reader.read(question, docs)