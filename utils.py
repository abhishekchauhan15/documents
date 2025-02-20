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
    REDIS_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
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

    def initialize_vector_store(self, weaviate_client: Any):
        """Initialize the Weaviate vector store."""
        try:
            if self._vector_store is None:
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
                                "description": "The text content of the chunk"
                            },
                            {
                                "name": "source",
                                "dataType": ["text"],
                                "description": "Source document name"
                            },
                            {
                                "name": "chunkIndex",
                                "dataType": ["int"],
                                "description": "Index of this chunk in the document"
                            },
                            {
                                "name": "processedAt",
                                "dataType": ["text"],
                                "description": "Timestamp when this chunk was processed"
                            }
                        ]
                    }
                    weaviate_client.schema.create_class(schema)
                    logger.info("Document schema created successfully")
                
                self._vector_store = Weaviate(
                    client=weaviate_client,
                    index_name="Document",
                    text_key="content",
                    embedding=self.embeddings,
                    by_text=False
                )
                logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def process_document(self, file_path: str, weaviate_client: Any, embedding_model: OllamaEmbeddings) -> bool:
        """Process and index a document."""
        try:
            # Record start time
            start_time = datetime.datetime.now()
            
            # Get file extension and name
            file_path = Path(file_path)
            file_ext = file_path.suffix[1:].lower()
            file_name = file_path.name
            
            logger.info(f"Step 1: Processing document: {file_name} (type: {file_ext})")
            
            if file_ext not in self.loaders:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Load document
            try:
                logger.info(f"Step 2: Loading document using {self.loaders[file_ext].__name__}")
                loader_class = self.loaders[file_ext]
                loader = loader_class(str(file_path))
                documents = loader.load()
                logger.info(f"Step 2 Complete: Loaded {len(documents)} document segments")
            except Exception as e:
                logger.error(f"Failed to load document: {str(e)}")
                raise
            
            # Clean and split text into chunks
            try:
                logger.info("Step 3: Splitting document into chunks")
                chunks = self.text_splitter.split_documents(documents)
                # Clean chunk metadata to only include allowed properties
                for i, chunk in enumerate(chunks):
                    chunk.metadata = {
                        "source": file_name,
                        "chunkIndex": i,
                        "processedAt": datetime.datetime.now().isoformat()
                    }
                logger.info(f"Step 3 Complete: Split into {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to split document: {str(e)}")
                raise
            
            # Initialize vector store
            try:
                logger.info("Step 4: Initializing vector store")
                self.initialize_vector_store(weaviate_client)
                logger.info("Step 4 Complete: Vector store initialized")
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {str(e)}")
                raise
            
            # Add to vector store
            try:
                logger.info("Step 5: Adding chunks to vector store")
                self._vector_store.add_documents(chunks)
                logger.info(f"Step 5 Complete: Added {len(chunks)} chunks to vector store")
            except Exception as e:
                logger.error(f"Failed to add documents to vector store: {str(e)}")
                raise
            
            # Save to Redis
            try:
                logger.info("Step 6: Saving metadata to Redis")
                redis_key = f"document:{file_name}"
                
                # Test Redis connection before saving
                self.redis_client.ping()
                logger.info("Redis connection verified")
                
                # Save metadata as strings
                metadata_mapping = {
                    "status": "processed",
                    "chunk_count": str(len(chunks)),
                    "processing_time": str((datetime.datetime.now() - start_time).total_seconds()),
                    "processed_at": datetime.datetime.now().isoformat()
                }
                
                # Save to Redis
                result = self.redis_client.hset(
                    redis_key,
                    mapping=metadata_mapping
                )
                logger.info(f"Redis hset result: {result}")
                
                # Verify the data was saved
                saved_data = self.redis_client.hgetall(redis_key)
                if not saved_data:
                    raise Exception("Failed to verify Redis save - no data found")
                logger.info(f"Step 6 Complete: Verified Redis data: {saved_data}")
                
            except Exception as e:
                logger.error(f"Failed to save to Redis: {str(e)}")
                raise
            
            logger.info(f"Successfully completed all steps for document: {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            try:
                # Attempt to save error status to Redis
                self.redis_client.hset(
                    f"document:{file_name}",
                    mapping={
                        "status": "error",
                        "error": str(e),
                        "error_time": datetime.datetime.now().isoformat()
                    }
                )
            except Exception as redis_error:
                logger.error(f"Additionally failed to save error status to Redis: {str(redis_error)}")
            return False

    def query_document(self, query: str, document_name: str, limit: int = 3) -> List[Dict]:
        """Query documents from the vector store."""
        try:
            # Record start time
            start_time = datetime.datetime.now()
            
            # Verify document exists
            doc_key = f"document:{document_name}"
            logger.info(f"Checking for document with key: {doc_key}")
            
            # List all keys in Redis for debugging
            all_keys = self.redis_client.keys("document:*")
            logger.info(f"Available document keys in Redis: {all_keys}")
            
            if not self.redis_client.exists(doc_key):
                logger.error(f"Document not found in Redis: {doc_key}")
                raise ValueError(f"Document not found: {document_name}")
            
            # Get document status
            doc_status = self.redis_client.hget(doc_key, "status")
            logger.info(f"Document status from Redis: {doc_status}")
            
            if doc_status != b"processed":
                logger.error(f"Document not ready - status: {doc_status}")
                raise ValueError(f"Document not ready: {document_name}")
            
            # Perform similarity search
            logger.info(f"Performing similarity search for query: {query}")
            results = self._vector_store.similarity_search_with_score(
                query,
                k=limit,
                filter={"source": document_name}
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score)
                })
            
            logger.info(f"Found {len(formatted_results)} results")
            
            # Record performance metrics
            end_time = datetime.datetime.now()
            query_time = (end_time - start_time).total_seconds()
            self.performance.record_metric("query_time", query_time)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying document: {str(e)}")
            raise

    def delete_document(self, document_name: str) -> bool:
        """Delete a document and its chunks from Weaviate."""
        try:
            # Delete from vector store
            self._vector_store.delete({"source": document_name})
            
            # Delete from Redis
            self.redis_client.delete(f"document:{document_name}")
            
            logger.info(f"Successfully deleted document: {document_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    def list_documents(self) -> List[str]:
        """List all available processed documents."""
        try:
            # Get all document keys from Redis
            all_keys = self.redis_client.keys("document:*")
            documents = []
            
            for key in all_keys:
                # Convert bytes to string and remove 'document:' prefix
                doc_name = key.decode('utf-8').replace('document:', '')
                # Check if document is processed
                status = self.redis_client.hget(key, "status")
                if status == b"processed":
                    documents.append(doc_name)
            
            logger.info(f"Found {len(documents)} processed documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
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