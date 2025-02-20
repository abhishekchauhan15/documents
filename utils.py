import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import weaviate
from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings
from werkzeug.utils import secure_filename
import redis
from statistics import mean
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import datetime
import uuid
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    WEAVIATE_URL
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileHandler:
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
                cls._client = weaviate.Client(
                    url=WEAVIATE_URL,
                    startup_period=15  # Wait up to 15 seconds for Weaviate to be ready
                )
                # Test the connection
                cls._client.schema.get()
                logger.info("Successfully connected to Weaviate")
            except Exception as e:
                logger.error(f"Failed to connect to Weaviate: {str(e)}")
                cls._client = None
        return cls._client

    @classmethod
    def close_client(cls):
        """Close Weaviate client connection."""
        if cls._client:
            cls._client = None

class IndexingHelper:
    def __init__(self, redis_url: str):
        """Initialize the IndexingHelper with Redis and document processing components."""
        try:
            self.redis_client = redis.from_url(redis_url)
            # Test Redis connection
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
        
        self._vector_store = None
        self.performance = PerformanceMonitor()
        
        # Initialize document loaders for each file type
        self.loaders = {
            'txt': TextLoader,
            'pdf': PyPDFLoader,
            'docx': Docx2txtLoader,
            'json': TextLoader  # JSON files are treated as text
        }
        
        # Initialize text splitter with configurable parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata to ensure property names are valid for Weaviate."""
        cleaned = {}
        for key, value in metadata.items():
            # Replace dots and other invalid characters with underscores
            clean_key = key.replace('.', '_').replace('-', '_')
            # Ensure key starts with a letter or underscore
            if not clean_key[0].isalpha() and clean_key[0] != '_':
                clean_key = f'_{clean_key}'
            cleaned[clean_key] = value
        return cleaned

    def initialize_vector_store(self, weaviate_client: Any):
        """Initialize the Weaviate vector store."""
        try:
            # First, ensure the schema exists with proper configuration
            if not weaviate_client.schema.exists("Document"):
                logger.info("Creating Document schema in Weaviate")
                schema = {
                    "class": "Document",
                    "description": "A document chunk with its embeddings",
                    "vectorizer": "none",  # We'll provide vectors ourselves
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "The text content of the chunk",
                            "indexInverted": True
                        },
                        {
                            "name": "documentId",
                            "dataType": ["text"],
                            "description": "Unique identifier for the document",
                            "indexInverted": True
                        },
                        {
                            "name": "fileName",
                            "dataType": ["text"],
                            "description": "Original file name",
                            "indexInverted": True
                        },
                        {
                            "name": "chunkIndex",
                            "dataType": ["int"],
                            "description": "Index of this chunk in the document",
                            "indexInverted": True
                        },
                        {
                            "name": "processedAt",
                            "dataType": ["text"],
                            "description": "Timestamp when this chunk was processed",
                            "indexInverted": True
                        }
                    ]
                }
                weaviate_client.schema.create_class(schema)
                logger.info("Document schema created successfully")
            
            # Initialize the vector store with our embeddings
            self._vector_store = Weaviate(
                client=weaviate_client,
                index_name="Document",
                text_key="content",
                embedding=self.embeddings,
                by_text=False,
                attributes=["documentId", "fileName", "chunkIndex", "processedAt"]
            )
            logger.info("Vector store initialized successfully")
            
            # Verify the vector store was created
            if not hasattr(self._vector_store, 'similarity_search'):
                raise Exception("Vector store initialization failed - missing required methods")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def query_document(self, query: str, document_id: str, limit: int = 3):
        """Query documents from the vector store."""
        try:
            # Verify document exists
            doc_metadata = self.redis_client.hgetall(f"document:{document_id}")
            if not doc_metadata:
                raise ValueError(f"Document with ID {document_id} not found")

            # Initialize vector store if needed
            if self._vector_store is None or not hasattr(self._vector_store, 'similarity_search'):
                logger.info("Initializing vector store for query...")
                client = WeaviateHelper.get_client()
                if not client:
                    raise Exception("Failed to initialize Weaviate client")
                self.initialize_vector_store(client)

            if self._vector_store is None or not hasattr(self._vector_store, 'similarity_search'):
                raise Exception("Vector store initialization failed")

            # Query Weaviate with document_id filter
            results = self._vector_store.similarity_search(
                query,
                filter={"path": ["documentId"], "operator": "Equal", "valueString": document_id},
                k=limit
            )
            
            return {
                'document_id': document_id,
                'document_name': doc_metadata.get(b'document_name', b'').decode('utf-8'),
                'results': [
                    {
                        'content': doc.page_content,
                        'metadata': {
                            'documentId': doc.metadata.get('documentId'),
                            'fileName': doc.metadata.get('fileName'),
                            'chunkIndex': doc.metadata.get('chunkIndex'),
                            'processedAt': doc.metadata.get('processedAt')
                        }
                    } for doc in results
                ]
            }
            
        except Exception as e:
            logger.error(f"Error querying document: {str(e)}")
            raise

    def process_document(self, file_path: str, weaviate_client: Any, embedding_model: OllamaEmbeddings = None):
        """Process and index a document."""
        try:
            file_extension = Path(file_path).suffix.lower()[1:]
            document_id = str(uuid.uuid4())  # Generate a unique document ID
            document_name = Path(file_path).name
            
            # Store document metadata in Redis
            metadata = {
                'document_name': document_name,
                'document_id': document_id,
                'file_path': file_path,
                'timestamp': datetime.datetime.now().isoformat()
            }
            self.redis_client.hset(f"document:{document_id}", mapping=metadata)
            
            # Get appropriate loader
            if file_extension not in self.loaders:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            loader_class = self.loaders[file_extension]
            loader = loader_class(file_path)
            
            # Load and split the document
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            
            # Add document_id to metadata of each chunk and clean metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'documentId': document_id,
                    'fileName': document_name,
                    'chunkIndex': i,
                    'processedAt': datetime.datetime.now().isoformat()
                })
            
            # Initialize vector store if needed
            if self._vector_store is None or not hasattr(self._vector_store, 'similarity_search'):
                self.initialize_vector_store(weaviate_client)

            if self._vector_store is None or not hasattr(self._vector_store, 'similarity_search'):
                raise Exception("Vector store initialization failed")
            
            # Store document chunks in Weaviate using class embeddings
            self._vector_store.add_documents(chunks)
            
            logger.info(f"Successfully processed document: {document_name} with ID: {document_id}")
            return {'document_id': document_id, 'status': 'success'}
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

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