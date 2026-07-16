from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# Extract data from PDF File
def load_pdf_file(data):
    loader = DirectoryLoader(data,
                            glob = '*.pdf',
                            loader_cls = PyPDFLoader)
    documents=loader.load()
    return documents


# Split the Data into Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Download the embeddings from Hugging Face
def download_hugging_face_embeddings(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings