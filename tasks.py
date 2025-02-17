from celery import Celery
from celery.schedules import crontab
import logging
from pathlib import Path
from config import (
    WEAVIATE_REST_URL,
    WEAVIATE_CLIENT_NAME,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    REDIS_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
import weaviate
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import docx
from pypdf import PdfReader
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery('tasks', broker=REDIS_URL, backend=REDIS_URL)

class DocumentProcessor:
    def __init__(self):
        self.client = weaviate.Client(
            url=WEAVIATE_REST_URL,
            additional_headers={
                "X-Weaviate-Client-Name": WEAVIATE_CLIENT_NAME
            },
            trust_env=True
        )
        self.embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )
    
    def read_file_content(self, file_path):
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        
        elif file_extension == 'json':
            with open(file_path, 'r') as file:
                return json.dumps(json.load(file))
        
        elif file_extension == 'pdf':
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        
        elif file_extension == 'docx':
            doc = docx.Document(file_path)
            return ' '.join([paragraph.text for paragraph in doc.paragraphs])
    
    def process_text(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        return text_splitter.split_text(text)
    
    def process_document(self, file_path):
        try:
            start_time = time.time()
            logger.info(f"Started processing document: {file_path}")
            
            # Read content
            read_start = time.time()
            content = self.read_file_content(file_path)
            logger.info(f"Reading took: {time.time() - read_start:.2f} seconds")
            
            # Split into chunks
            chunk_start = time.time()
            chunks = self.process_text(content)
            logger.info(f"Chunking took: {time.time() - chunk_start:.2f} seconds")
            logger.info(f"Number of chunks: {len(chunks)}")
            
            # Process chunks in batches
            embed_start = time.time()
            batch_size = 50
            with self.client.batch as batch:
                batch.batch_size = batch_size
                for i, chunk in enumerate(chunks):
                    embedding = self.embeddings.embed_query(chunk)
                    batch.add_data_object(
                        data_object={
                            "content": chunk,
                            "source": Path(file_path).name,
                            "chunk_index": i,
                            "status": "processed"
                        },
                        class_name="Document",
                        vector=embedding
                    )
            
            logger.info(f"Embedding and storage took: {time.time() - embed_start:.2f} seconds")
            logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return False

# Celery task for document processing
@celery_app.task(name='tasks.process_document')
def process_document(file_path):
    processor = DocumentProcessor()
    return processor.process_document(file_path)

# Celery task for checking and processing pending documents
@celery_app.task(name='tasks.check_pending_documents')
def check_pending_documents():
    client = weaviate.Client(
        url=WEAVIATE_REST_URL,
        additional_headers={
            "X-Weaviate-Client-Name": WEAVIATE_CLIENT_NAME
        }
    )
    
    # Query for documents with pending status
    response = client.query.get(
        "Document",
        ["source"]
    ).with_where({
        "path": ["status"],
        "operator": "Equal",
        "valueString": "pending"
    }).do()
    
    pending_docs = response['data']['Get']['Document']
    for doc in pending_docs:
        process_document.delay(doc['source'])

# Configure periodic tasks
celery_app.conf.beat_schedule = {
    'check-pending-documents': {
        'task': 'tasks.check_pending_documents',
        'schedule': crontab(minute='*/5'),  # Run every 5 minutes
    },
} 