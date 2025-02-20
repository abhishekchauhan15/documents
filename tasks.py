from celery import Celery
from celery.schedules import crontab
from pathlib import Path
import datetime
from config import (
    WEAVIATE_REST_URL,
    WEAVIATE_GRPC_URL,
    WEAVIATE_CLIENT_NAME,
    WEAVIATE_ADMIN_API_KEY,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    REDIS_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import FileHandler, WeaviateHelper, TimeTracker, logger, IndexingHelper
from typing import Any, Optional, List

# Initialize Celery
celery_app = Celery('tasks', broker=REDIS_URL, backend=REDIS_URL)
client = WeaviateHelper.get_client()

if client is None:
    raise Exception("Failed to initialize Weaviate client")

# Celery Configuration
celery_app.conf.update(
    broker_connection_retry_on_startup=True,
    worker_prefetch_multiplier=1,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    beat_schedule={
        'check-pending-documents': {
            'task': 'tasks.check_pending_documents',
            'schedule': crontab(minute='*/1'),  # Run every 1 minute
        },
    },
    timezone='UTC',
    enable_utc=True  # Only specify once
)

def get_utc_now():
    """Get current UTC time in a timezone-aware manner."""
    return datetime.datetime.now(datetime.timezone.utc)

class DocumentProcessor:
    def __init__(self):
        self.client = WeaviateHelper.get_client()
        if self.client is None:
            raise Exception("Failed to initialize Weaviate client")
            
        self.embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        self.indexer = IndexingHelper(REDIS_URL)
    
    @TimeTracker.track_time
    def process_document(self, file_path):
        logger.info(f"Processing document: {file_path}")
        return self.indexer.process_document(
            file_path=file_path,
            weaviate_client=self.client,
            embedding_model=self.embeddings,
            chunk_size=CHUNK_SIZE
        )

    @staticmethod
    def store_document_chunk(chunk: str, embedding: List[float], source: str, chunk_index: int) -> None:
        """Store a document chunk in Weaviate."""
        weaviate_client = WeaviateHelper.get_client()
        
        # Ensure the embedding is a flat list of floats
        if isinstance(embedding, list) and all(isinstance(i, float) for i in embedding):
            weaviate_client.data.create(
                data={
                    "content": chunk,
                    "source": source,
                    "chunk_index": chunk_index,
                    "status": "processed"
                },
                class_name="Document",
                vector=embedding  # Ensure this is a flat list of floats
            )
        else:
            logger.error("Invalid embedding format: must be a flat list of floats.")

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL
)

@celery_app.task(name='tasks.process_document')
def process_document_task(file_path: str):
    # Create an instance of IndexingHelper
    indexing_helper = IndexingHelper(redis_url=REDIS_URL)
    
    try:
        # Call process_document with the correct parameters
        result = indexing_helper.process_document(file_path, weaviate_client=client, embedding_model=embeddings)
        return result
    finally:
        indexing_helper.close()  # Ensure the client is closed

# Celery task for checking and processing pending documents
@celery_app.task(name='tasks.check_pending_documents')
def check_pending_documents():
    logger.info("Checking for pending documents...")
    client = WeaviateHelper.initialize_client(
        rest_url=WEAVIATE_REST_URL,
        grpc_url=WEAVIATE_GRPC_URL,
        client_name=WEAVIATE_CLIENT_NAME,
        api_key=WEAVIATE_ADMIN_API_KEY
    )
    if client is None:
        logger.error("Failed to initialize Weaviate client")
        return
    
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
        logger.info(f"Processing pending document: {doc['source']}")
        process_document_task.delay(doc['source']) 