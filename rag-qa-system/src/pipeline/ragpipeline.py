from data.datapreprocessor import FAISSVectorDB
from data.datapreprocessor import DocumentPreprocessor
from data.datapreprocessor import EmbeddingGenerator
from models.lamapipeline import LlamaReader
from data.dataloader import DocumentLoader
from data.datapreprocessor import TextSplitter

from pipeline.rag import RAG

class LlamaChatRAG(RAG):

    def __init__(self,folder_path):
      #folder_path ='/content/gdrive/MyDrive/fundfact/DDQ/'
      self.doc_loader = DocumentLoader()
      self.documents = self.doc_loader.load_pdf_langchain(folder_path)

      print('{0} documents loaded'.format(len(self.documents)))


      self.txt_splitter = TextSplitter()
      self.embeddings = EmbeddingGenerator()
      self.doc_prsr = DocumentPreprocessor(self.txt_splitter,self.embeddings)

      self.indexer = FAISSVectorDB(self.doc_prsr)
      self.indexer.add(self.documents)
      self.reader = LlamaReader(self.indexer)

      super().__init__( self.reader, self.indexer)



    def answer(self, question):
        return self.reader.read(question)
