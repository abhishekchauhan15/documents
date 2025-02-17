from dotenv import load_dotenv
import os
import weaviate
import certifi

# Load environment variables
load_dotenv()

# Weaviate Configuration
WEAVIATE_REST_URL = os.getenv('WEAVIATE_REST_URL')
WEAVIATE_GRPC_URL = os.getenv('WEAVIATE_GRPC_URL')
WEAVIATE_CLIENT_NAME = os.getenv('WEAVIATE_CLIENT_NAME')
WEAVIATE_GRPC_PORT = 50051
WEAVIATE_ADMIN_API_KEY = os.getenv('WEAVIATE_ADMIN_API_KEY')

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')

# Redis Configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Document Processing Configuration
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))

# Flask Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'json', 'txt'}