from google.colab import drive
from langchain.document_loaders import PyPDFDirectoryLoader
import time
import fitz
from utils.Logger import Logger
from typing import List, Dict

logger = Logger.get_logger()

class DocumentLoader:
    def load_pdf_from_google_drive(self, folder_path: str, mode: str = "doc") -> List[Dict[str, str]]:
        """
        Loads PDF documents from Google Drive.

        Args:
            folder_path (str): The path to the folder containing PDF files in Google Drive.
            mode (str, optional): The mode for loading PDFs. Defaults to "doc".
                - "doc": Load the entire content of each PDF as a single document.
                - "page": Load each page of the PDF as a separate document.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the file name and content (or pages) of each PDF.
        """
        start_time = time.time()
        docs = []

        try:
            file_paths = os.listdir(folder_path)
        except OSError as e:
            logger.error(f"Error listing files in the folder: {e}")
            return docs

        for f in file_paths:
            f_path = os.path.join(folder_path, f)
            try:
                if mode == 'doc':
                    pages = self.extract_data_from_pdf(f_path)
                    doc = {'name': f, 'content': pages}
                elif mode == 'page':
                    pages = self.extract_pages_from_pdf(f_path)
                    doc = {'name': f, 'pages': pages}
                else:
                    logger.warning(f"Invalid mode '{mode}' for loading PDFs. Skipping file: {f}")
                    continue
                docs.append(doc)
            except Exception as e:
                logger.error(f"Error loading file {f}: {e}")

        end_time = time.time()
        logger.info(f"Loaded {len(docs)} docs in {end_time - start_time:.2f} seconds")
        return docs

    def load_pdf_langchain(self, pdf_folder_path: str) -> List[str]:
        """
        Loads PDF documents from a folder using the LangChain library.

        Args:
            pdf_folder_path (str): The path to the folder containing PDF files.

        Returns:
            List[str]: A list of strings representing the content of each PDF.
        """
        start_time = time.time()
        try:
            loader = PyPDFDirectoryLoader(pdf_folder_path)
            docs = loader.load()
        except Exception as e:
            logger.error(f"Error loading PDFs from {pdf_folder_path}: {e}")
            docs = []

        end_time = time.time()
        logger.info(f"Loaded {len(docs)} docs in {end_time - start_time:.2f} seconds")
        return docs

    def extract_pages_from_pdf(self, pdf_path: str) -> List[str]:
        """
        Extracts the text content from each page of a PDF file.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            List[str]: A list of strings representing the text content of each page.
        """
        start_time = time.time()
        pages = []

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Error opening PDF file {pdf_path}: {e}")
            return pages

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            if self.should_include_page(text):
                pages.append(text)

        doc.close()
        end_time = time.time()
        logger.info(f"Extracted {len(pages)} pages from {pdf_path} in {end_time - start_time:.2f} seconds")
        return pages

    def should_include_page(self, page_text: str) -> bool:
        """
        Filters pages based on rules.

        Args:
            page_text (str): The text content of the page.

        Returns:
            bool: True if the page should be included, False otherwise.
        """
        if len(page_text) < 250:
            return False
        if any(keyword.lower() in page_text.lower() for keyword in ["table of contents", "contents", "pages"]):
            return False
        return True

    def extract_data_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts the entire text content from a PDF file.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            str: The text content of the PDF.
        """
        start_time = time.time()
        doc_text = ""

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Error opening PDF file {pdf_path}: {e}")
            return doc_text

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            if self.should_include_page(text):
                doc_text += text

        doc.close()
        end_time = time.time()
        logger.info(f"Extracted data from {pdf_path} in {end_time - start_time:.2f} seconds")
        return doc_text

    def extract_paragraphs_from_pdf(self, pdf_path: str) -> List[str]:
        """
        Extracts paragraphs from a PDF file.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            List[str]: A list of strings representing the paragraphs in the PDF.
        """
        start_time = time.time()
        paragraphs = []

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Error opening PDF file {pdf_path}: {e}")
            return paragraphs

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            current_paragraph = []
            for line in text.split('\n'):
                line = line.strip()
                if line:
                    current_paragraph.append(line)
                elif current_paragraph:
                    paragraphs.append('\n'.join(current_paragraph))
                    current_paragraph = []

            if current_paragraph:
                paragraphs.append('\n'.join(current_paragraph))

        doc.close()
        end_time = time.time()
        logger.info(f"Extracted {len(paragraphs)} paragraphs from {pdf_path} in {end_time - start_time:.2f} seconds")
        return paragraphs