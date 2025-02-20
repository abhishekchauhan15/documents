import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import json
import docx
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
                cls._client = weaviate.Client(
                    url="http://localhost:8080"
                )
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
                
                self._vector_store = Weaviate(
                    client=weaviate_client,
                    index_name="Document",
                    text_key="content",
                    embedding=self.embeddings,
                    by_text=False,
                    attributes=["documentId", "fileName", "chunkIndex", "processedAt"]
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
            
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            
            # Get file extension and name
            file_path = Path(file_path)
            file_ext = file_path.suffix[1:].lower()
            file_name = file_path.name
            
            logger.info(f"Step 1: Processing document: {file_name} (ID: {document_id})")
            
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
                        "documentId": document_id,
                        "fileName": file_name,
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
                redis_key = f"document:{document_id}"
                
                # Test Redis connection before saving
                self.redis_client.ping()
                logger.info("Redis connection verified")
                
                # Save metadata as strings
                metadata_mapping = {
                    "status": "processed",
                    "documentId": document_id,
                    "fileName": file_name,
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
                
                # Save filename to document ID mapping for easy lookup
                self.redis_client.set(f"filename:{file_name}", document_id)
                
                # Verify the data was saved
                saved_data = self.redis_client.hgetall(redis_key)
                if not saved_data:
                    raise Exception("Failed to verify Redis save - no data found")
                logger.info(f"Step 6 Complete: Verified Redis data: {saved_data}")
                
            except Exception as e:
                logger.error(f"Failed to save to Redis: {str(e)}")
                raise
            
            logger.info(f"Successfully completed all steps for document: {file_name} (ID: {document_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            try:
                # Attempt to save error status to Redis
                self.redis_client.hset(
                    f"document:{document_id}",
                    mapping={
                        "status": "error",
                        "documentId": document_id,
                        "fileName": file_name,
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
            
            # First check if document exists in Redis
            available_docs = self.list_documents()
            matching_doc = None
            for doc in available_docs:
                if doc['fileName'] == document_name:
                    matching_doc = doc
                    break
                    
            if not matching_doc:
                raise ValueError(f"Document not found: {document_name}")
                
            document_id = matching_doc['documentId']
            
            # Initialize vector store if needed
            if self._vector_store is None:
                client = WeaviateHelper.get_client()
                if not client:
                    raise Exception("Failed to initialize Weaviate client")
                self.initialize_vector_store(client)
            
            # Perform similarity search with document ID filter
            logger.info(f"Performing similarity search for query: {query} in document: {document_name} (ID: {document_id})")
            
            # Construct where filter for Weaviate
            where_filter = {
                "operator": "Equal",
                "path": ["documentId"],
                "valueString": document_id
            }
            
            results = self._vector_store.similarity_search_with_score(
                query,
                k=limit,
                filter=where_filter
            )
            
            if not results:
                logger.warning(f"No results found for document: {document_name}")
                return []
                
            # Format results
            formatted_results = []
            for doc, score in results:
                # Verify this result belongs to our document
                if doc.metadata.get('documentId') != document_id:
                    logger.warning(f"Filtered out result from wrong document: {doc.metadata.get('documentId')}")
                    continue
                    
                formatted_results.append({
                    "content": doc.page_content,
                    "chunk_index": doc.metadata.get('chunkIndex', 0),
                    "relevance_score": float(score)
                })
            
            logger.info(f"Found {len(formatted_results)} results for document: {document_name}")
            
            # Record performance metrics
            end_time = datetime.datetime.now()
            query_time = (end_time - start_time).total_seconds()
            self.performance.record_metric("query_time", query_time)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying document: {str(e)}")
            raise

    def list_documents(self) -> List[Dict[str, str]]:
        """List all available processed documents."""
        try:
            # Get all document keys from Redis
            all_keys = self.redis_client.keys("document:*")
            documents = []
            
            for key in all_keys:
                try:
                    # Get document data
                    doc_data = self.redis_client.hgetall(key)
                    if not doc_data:
                        logger.warning(f"No data found for key: {key}")
                        continue

                    # Check if it's processed
                    status = doc_data.get(b"status", b"").decode('utf-8')
                    if status != "processed":
                        logger.debug(f"Skipping document with status: {status}")
                        continue

                    # Handle both old and new format
                    doc_id = doc_data.get(b"documentId", b"").decode('utf-8')
                    if not doc_id:
                        # Old format - key is document:filename
                        doc_id = key.decode('utf-8').replace('document:', '')
                        
                    file_name = doc_data.get(b"fileName", b"").decode('utf-8')
                    if not file_name:
                        # Old format - use the key as filename
                        file_name = key.decode('utf-8').replace('document:', '')

                    processed_at = doc_data.get(b"processed_at", b"unknown").decode('utf-8')
                    chunk_count = doc_data.get(b"chunk_count", b"0").decode('utf-8')

                    documents.append({
                        "documentId": doc_id,
                        "fileName": file_name,
                        "processed_at": processed_at,
                        "chunk_count": chunk_count
                    })
                    logger.info(f"Found document: {file_name} (ID: {doc_id})")
                except Exception as e:
                    logger.error(f"Error processing key {key}: {str(e)}")
                    continue
            
            logger.info(f"Found {len(documents)} processed documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
        
    def delete_document(self, document_name: str) -> bool:
        """Delete a document and its chunks from Weaviate."""
        try:
            # Get document ID
            document_id = self.redis_client.get(f"filename:{document_name}")
            if not document_id:
                logger.error(f"No document ID found for filename: {document_name}")
                return False
            
            document_id = document_id.decode('utf-8')
            
            # Delete from vector store using document ID
            self._vector_store.delete({"documentId": document_id})
            
            # Delete from Redis
            self.redis_client.delete(f"document:{document_id}")
            self.redis_client.delete(f"filename:{document_name}")
            
            logger.info(f"Successfully deleted document: {document_name} (ID: {document_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

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