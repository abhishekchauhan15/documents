from celery import Celery
from config import REDIS_URL, OLLAMA_MODEL, OLLAMA_BASE_URL
from utils import IndexingHelper, WeaviateHelper
from langchain_ollama import OllamaEmbeddings
import logging
import platform
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery with appropriate pool based on OS
if platform.system() == 'Windows':
    # Use solo pool for Windows
    celery = Celery('tasks', 
                    broker=REDIS_URL, 
                    backend=REDIS_URL,
                    pool='solo')
else:
    # Use default pool for other OS
    celery = Celery('tasks', 
                    broker=REDIS_URL, 
                    backend=REDIS_URL)

# Configure Celery
celery.conf.update(
    broker_connection_retry_on_startup=True,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True
)

@celery.task(bind=True)
def process_document_task(self, file_path: str) -> bool:
    """Process and index a document."""
    try:
        logger.info(f"Starting to process document: {file_path}")
        
        # Initialize components
        logger.info("Initializing Weaviate client...")
        client = WeaviateHelper.get_client()
        if not client:
            logger.error("Failed to initialize Weaviate client")
            raise Exception("Failed to initialize Weaviate client")
            
        logger.info("Initializing Ollama embeddings...")
        embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        logger.info("Initializing IndexingHelper...")
        indexing_helper = IndexingHelper(REDIS_URL)
        
        # Process the document
        logger.info("Starting document processing...")
        success = indexing_helper.process_document(
            file_path=file_path,
            weaviate_client=client,
            embedding_model=embeddings
        )
        
        if success:
            logger.info(f"Successfully processed document: {file_path}")
            # Verify Redis entry was created
            doc_name = Path(file_path).name
            doc_key = f"document:{doc_name}"
            status = indexing_helper.redis_client.hget(doc_key, "status")
            logger.info(f"Verified Redis entry - Key: {doc_key}, Status: {status}")
            return True
        else:
            logger.error(f"Failed to process document: {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {str(e)}")
        raise