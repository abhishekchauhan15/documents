import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import json
import docx
from pypdf import PdfReader
from werkzeug.utils import secure_filename
from weaviate import Client
from langchain_ollama import OllamaEmbeddings
import redis
from statistics import mean
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import datetime
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    REDIS_URL
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileHandler:
    CHUNK_SIZE = 4096  # Optimal chunk size for file reading
    
    @staticmethod
    def allowed_file(filename: str, allowed_extensions: set) -> bool:
        """Check if the file extension is allowed."""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

    @staticmethod
    def secure_file_save(file: Any, upload_folder: str) -> Optional[str]:
        """Securely save a file and return the file path."""
        try:
            filename = secure_filename(file.filename)
            logger.info(f"Saving file: {filename} to {upload_folder}")
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)
            logger.info(f"File saved successfully: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return None

class WeaviateHelper:
    _client = None
    
    @classmethod
    def get_client(cls):
        """Get or create Weaviate client."""
        if cls._client is None:
            try:
                cls._client = Client(
                    url="http://localhost:8080"
                )
                # Create schema if it doesn't exist
                if not cls._client.schema.exists("Document"):
                    cls._client.schema.create_class({
                        "class": "Document",
                        "properties": [
                            {
                                "name": "content",
                                "dataType": ["text"],
                                "description": "The content of the document chunk"
                            },
                            {
                                "name": "source",
                                "dataType": ["text"],
                                "description": "The source filename"
                            },
                            {
                                "name": "chunk_index",
                                "dataType": ["int"],
                                "description": "Index of the chunk in the document"
                            },
                            {
                                "name": "status",
                                "dataType": ["text"],
                                "description": "Processing status of the chunk"
                            }
                        ],
                        "vectorizer": "none"  # We'll provide vectors ourselves
                    })
                logger.info("Successfully connected to Weaviate")
            except Exception as e:
                logger.error(f"Failed to connect to Weaviate: {str(e)}")
                return None
        return cls._client

    @classmethod
    def close_client(cls):
        """Close Weaviate client connection."""
        if cls._client:
            cls._client = None

class IndexingHelper:
    def __init__(self, redis_url: str):
        """Initialize the IndexingHelper with Redis and document processing components."""
        self.redis_client = redis.from_url(redis_url)
        self._vector_store = None
        self.performance = PerformanceMonitor()
        
        # Initialize document loaders for each file type
        self.loaders = {
            'txt': TextLoader,
            'pdf': PyPDFLoader,
            'docx': Docx2txtLoader,
            'pptx': UnstructuredPowerPointLoader,
            'csv': CSVLoader,
            'xlsx': UnstructuredExcelLoader
        }
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )
    
    def initialize_vector_store(self, weaviate_client: Any):
        """Initialize the Weaviate vector store."""
        try:
            self._vector_store = Weaviate(
                client=weaviate_client,
                index_name="Document",
                text_key="content",
                embedding=self.embeddings,
                by_text=False
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def process_document(self, file_path: str, weaviate_client: Any, embedding_model: OllamaEmbeddings) -> bool:
        """Process and index a document."""
        try:
            # Initialize vector store if not already initialized
            if not self._vector_store:
                self.initialize_vector_store(weaviate_client)
            
            # Get appropriate loader based on file extension
            file_extension = Path(file_path).suffix.lower()[1:]
            loader_class = self.loaders.get(file_extension)
            
            if not loader_class:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Load and split document
            loader = loader_class(file_path)
            documents = loader.load()
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata to chunks
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'source': Path(file_path).name,
                    'chunk_index': i,
                    'file_type': file_extension,
                    'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'status': 'processed'
                })
            
            # Add documents to vector store
            self._vector_store.add_documents(chunks)
            
            logger.info(f"Successfully processed and indexed {len(chunks)} chunks from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return False

    def query_document(self, query: str, document_name: str, limit: int = 3) -> List[Dict]:
        """Query documents from the vector store."""
        try:
            if not self._vector_store:
                logger.error("Vector store not initialized")
                return []

            # Search in vector store with metadata filter
            results = self._vector_store.similarity_search_with_score(
                query,
                k=limit,
                filter={"source": document_name}
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                result = {
                    "content": doc.page_content,
                    "source": doc.metadata.get('source'),
                    "chunk_index": doc.metadata.get('chunk_index'),
                    "status": doc.metadata.get('status', 'retrieved'),
                    "score": score
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results for query in document {document_name}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying document: {str(e)}")
            return []

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def record_metric(self, name: str, value: float):
        self.metrics[name].append(value)
    
    def get_average(self, name: str):
        values = self.metrics.get(name, [])
        return mean(values) if values else 0
    
    def reset(self):
        self.metrics.clear()