from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Weaviate Configuration
WEAVIATE_URL = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY', None)

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1:latest')

# Redis Configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Document Processing Configurationś
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
MAX_DOCUMENT_SIZE = int(os.getenv('MAX_DOCUMENT_SIZE', '10485760'))  # 10MB default
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '768'))

# Flask Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {
    'pdf',
    'docx',
    'json',
    'txt'
}

# Performance Configuration
MAX_CONCURRENT_TASKS = int(os.getenv('MAX_CONCURRENT_TASKS', '5'))
QUERY_RESULT_LIMIT = int(os.getenv('QUERY_RESULT_LIMIT', '3'))
CACHE_TTL = int(os.getenv('CACHE_TTL', '3600'))  # 1 hour default

# Schema Configuration
DOCUMENT_SCHEMA = {
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
            "name": "metadata",
            "dataType": ["object"],
            "description": "Additional metadata about the document"
        }
    ],
    "vectorizer": "none"
}