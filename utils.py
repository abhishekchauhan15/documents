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
import json
from langchain.schema import Document

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

class JsonLoader:
    """Custom loader for JSON files."""
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self) -> List[Any]:
        """Load JSON content and convert it to documents."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                json_data = json.loads(file.read())
                
            # Convert JSON to string representation
            if isinstance(json_data, (dict, list)):
                # Pretty print JSON for better readability
                content = json.dumps(json_data, indent=2)
            else:
                content = str(json_data)
                
            # Create a document with the JSON content
            return [Document(page_content=content, metadata={"source": self.file_path})]
        except Exception as e:
            logger.error(f"Error loading JSON file: {str(e)}")
            raise

class IndexingHelper:
    def __init__(self, redis_url: str):
        """Initialize the IndexingHelper with Redis and document processing components."""
        try:
            self.redis_client = redis.from_url(redis_url)
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
            'json': JsonLoader
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
        """Clean metadata to ensure property names are valid for Weaviate GraphQL."""
        cleaned = {
            'documentId': metadata.get('documentId', ''),
            'fileName': metadata.get('fileName', ''),
            'chunkIndex': metadata.get('chunkIndex', 0),
            'processedAt': metadata.get('processedAt', datetime.datetime.now(datetime.UTC).isoformat())
        }
        
        # Store all other metadata as a JSON string in metaData field
        other_metadata = {}
        for key, value in metadata.items():
            if key not in cleaned:
                # Clean the key name
                clean_key = ''.join(c if c.isalnum() else '_' for c in str(key))
                clean_key = clean_key.strip('_')
                if not clean_key[0].isalpha():
                    clean_key = 'f_' + clean_key
                other_metadata[clean_key] = str(value)
        
        if other_metadata:
            cleaned['metaData'] = json.dumps(other_metadata)
        else:
            cleaned['metaData'] = '{}'
            
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
                    "vectorizer": "none",
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
                            "dataType": ["date"],
                            "description": "Timestamp when this chunk was processed",
                            "indexInverted": True
                        },
                        {
                            "name": "metaData",
                            "dataType": ["text"],
                            "description": "JSON string of additional metadata",
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
                attributes=["documentId", "fileName", "chunkIndex", "processedAt", "metaData"]
            )
            logger.info("Vector store initialized successfully")
            
            # Verify the vector store was created
            if not hasattr(self._vector_store, 'similarity_search'):
                raise Exception("Vector store initialization failed - missing required methods")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def process_document(self, file_path: str, weaviate_client: Any, embedding_model: OllamaEmbeddings = None):
        """Process and index a document."""
        try:
            file_extension = Path(file_path).suffix.lower()[1:]
            document_id = str(uuid.uuid4()) 
            document_name = Path(file_path).name
            
            logger.info(f"Processing document: {document_name} with ID: {document_id}")
            
            # Initialize vector store first if needed
            if self._vector_store is None or not hasattr(self._vector_store, 'similarity_search'):
                self.initialize_vector_store(weaviate_client)

            if self._vector_store is None or not hasattr(self._vector_store, 'similarity_search'):
                raise Exception("Vector store initialization failed")
            
            # Now check for existing document with same name
            existing_docs = self._vector_store.similarity_search(
                document_name,
                filter={
                    "operator": "Equal",
                    "path": ["fileName"],
                    "valueString": document_name
                },
                k=1
            )
            if existing_docs:
                old_doc_id = existing_docs[0].metadata.get('documentId')
                if old_doc_id:
                    logger.info(f"Found existing document with name {document_name}, deleting it first...")
                    self._vector_store.delete(
                        where_filter={
                            "path": ["documentId"],
                            "operator": "Equal",
                            "valueString": old_doc_id
                        }
                    )
            
            # Store document metadata in Redis
            metadata = {
                'document_name': document_name,
                'document_id': document_id,
                'file_path': file_path,
                'timestamp': datetime.datetime.now(datetime.UTC).isoformat()
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
            
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Add document_id to metadata of each chunk and clean metadata
            for i, chunk in enumerate(chunks):
                # First clean any existing metadata
                chunk.metadata.update({
                    'documentId': document_id,
                    'fileName': document_name,
                    'chunkIndex': i,
                    'processedAt': datetime.datetime.now(datetime.UTC).isoformat()
                })
                chunk.metadata = self._clean_metadata(chunk.metadata)
            
            # Store document chunks in Weaviate using class embeddings
            self._vector_store.add_documents(chunks)
            
            # Verify documents were added by doing a test query
            test_results = self._vector_store.similarity_search(
                "test",
                filter={
                    "operator": "Equal",
                    "path": ["documentId"],
                    "valueString": document_id
                },
                k=1
            )
            
            if not test_results:
                raise Exception("Failed to verify document indexing - no results found")
            
            logger.info(f"Successfully processed document: {document_name} with ID: {document_id}")
            return {'document_id': document_id, 'status': 'success'}
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
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

            logger.info(f"Querying document with ID: {document_id}")
            
            # First verify the document exists in Weaviate
            verify_results = self._vector_store.similarity_search(
                "test",  # Simple query to check existence
                filter={
                    "operator": "Equal",
                    "path": ["documentId"],
                    "valueString": document_id
                },
                k=1
            )
            
            if not verify_results:
                logger.warning(f"Document {document_id} not found in vector store. It may have been deleted.")
                raise ValueError(f"Document {document_id} not found in vector store")
            
            # Query Weaviate with document_id filter
            results = self._vector_store.similarity_search_with_score(
                query,
                k=limit,
                filter={
                    "operator": "Equal",
                    "path": ["documentId"],
                    "valueString": document_id
                }
            )
            
            # Log the results for debugging
            logger.info(f"Found {len(results)} results")
            for doc, score in results:
                logger.info(f"Score: {score}, Content: {doc.page_content[:100]}...")
            
            # Format the response
            response = {
                'document_id': document_id,
                'document_name': doc_metadata.get(b'document_name', b'').decode('utf-8'),
                'results': []
            }
            
            for doc, score in results:
                # Extract the metadata we want to return
                metadata = doc.metadata
                result = {
                    'content': doc.page_content,
                    'metadata': {
                        'document_id': metadata.get('documentId'),
                        'file_name': metadata.get('fileName'),
                        'chunk_index': metadata.get('chunkIndex'),
                        'processed_at': metadata.get('processedAt'),
                        'score': float(score)
                    }
                }
                
                # Add any additional metadata if present
                if 'metaData' in metadata:
                    try:
                        extra_metadata = json.loads(metadata['metaData'])
                        result['metadata']['extra'] = extra_metadata
                    except json.JSONDecodeError:
                        pass
                
                response['results'].append(result)
            
            return response
            
        except Exception as e:
            logger.error(f"Error querying document: {str(e)}")
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