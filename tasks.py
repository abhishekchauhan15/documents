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

# Initialize Celery
celery_app = Celery('tasks', broker=REDIS_URL, backend=REDIS_URL)

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
    return datetime.datetime.now(datetime.UTC)

class DocumentProcessor:
    def __init__(self):
        self.client = WeaviateHelper.initialize_client(
            rest_url=WEAVIATE_REST_URL,
            grpc_url=WEAVIATE_GRPC_URL,
            client_name=WEAVIATE_CLIENT_NAME,
            api_key=WEAVIATE_ADMIN_API_KEY
        )
        if self.client is None:
            raise Exception("Failed to initialize Weaviate client")
            
        self.embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        self.indexer = IndexingHelper(REDIS_URL)
    
    @TimeTracker.track_time
    def process_document(self, file_path):
        return self.indexer.process_document(
            file_path=file_path,
            weaviate_client=self.client,
            embeddings=self.embeddings,
            chunk_size=CHUNK_SIZE
        )

@celery_app.task(name='tasks.process_document')
def process_document(file_path):
    start_time = get_utc_now()
    logger.info(f"Starting document processing at {start_time}")
    processor = DocumentProcessor()
    result = processor.process_document(file_path)
    end_time = get_utc_now()
    logger.info(f"Finished document processing at {end_time}")
    return result

# Celery task for checking and processing pending documents
@celery_app.task(name='tasks.check_pending_documents')
def check_pending_documents():
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
        process_document.delay(doc['source']) 