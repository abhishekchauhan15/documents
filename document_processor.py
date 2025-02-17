from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
import time
import threading
from queue import Queue
import logging
from config import UPLOAD_FOLDER
from langchain.text_splitter import RecursiveCharacterTextSplitter
import weaviate
from config import (
    WEAVIATE_REST_URL,
    WEAVIATE_CLIENT_NAME,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL
)
from langchain_ollama import OllamaEmbeddings
import json
import docx
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global task queue
processing_queue = Queue()

class DocumentProcessor:
    def __init__(self):
        self.client = weaviate.Client(
            url=WEAVIATE_REST_URL,
            additional_headers={
                "X-Weaviate-Client-Name": WEAVIATE_CLIENT_NAME
            }
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
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        
        elif file_extension == 'docx':
            doc = docx.Document(file_path)
            return ' '.join([paragraph.text for paragraph in doc.paragraphs])
    
    def process_text(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return text_splitter.split_text(text)
    
    def process_document(self, file_path):
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Read content
            content = self.read_file_content(file_path)
            
            # Split into chunks
            chunks = self.process_text(content)
            
            # Process chunks in batches
            batch_size = 50
            with self.client.batch as batch:
                batch.batch_size = batch_size
                
                for i, chunk in enumerate(chunks):
                    # Generate embedding
                    embedding = self.embeddings.embed_query(chunk)
                    
                    # Store in Weaviate
                    batch.add_data_object(
                        data_object={
                            "content": chunk,
                            "source": Path(file_path).name,
                            "chunk_index": i
                        },
                        class_name="Document",
                        vector=embedding
                    )
            
            logger.info(f"Successfully processed document: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return False

class FileHandler(FileSystemEventHandler):
    def __init__(self):
        self.processor = DocumentProcessor()
        
    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            if any(file_path.endswith(ext) for ext in ['.pdf', '.docx', '.json', '.txt']):
                processing_queue.put(file_path)
                logger.info(f"Added {file_path} to processing queue")

def process_queue():
    processor = DocumentProcessor()
    while True:
        try:
            file_path = processing_queue.get()
            processor.process_document(file_path)
            processing_queue.task_done()
        except Exception as e:
            logger.error(f"Error in queue processing: {str(e)}")
        time.sleep(1)  # Prevent CPU overload

def start_file_monitoring():
    # Create upload directory if it doesn't exist
    Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    
    # Start the file watcher
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, UPLOAD_FOLDER, recursive=False)
    observer.start()
    
    # Start the processing queue worker
    worker_thread = threading.Thread(target=process_queue, daemon=True)
    worker_thread.start()
    
    logger.info(f"Started monitoring {UPLOAD_FOLDER} for new documents")
    return observer, worker_thread 