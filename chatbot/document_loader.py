from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from bs4 import BeautifulSoup

class DocumentProcessor:
    def __init__(self, api_key=None):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    def load_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        return loader.load()

    def load_html(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            text = soup.get_text()
        return [{"page_content": text, "metadata": {"source": file_path}}]

    def process_files(self, files):
        all_docs = []
        for file in files:
            if file.endswith('.pdf'):
                docs = self.load_pdf(file)
                all_docs.extend(docs)
            elif file.endswith('.html'):
                docs = self.load_html(file)
                all_docs.extend(docs)
        return self.text_splitter.split_documents(all_docs) if all_docs else []