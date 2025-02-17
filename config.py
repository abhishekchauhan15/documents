from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Weaviate Configuration
WEAVIATE_REST_URL = os.getenv('WEAVIATE_REST_URL')
WEAVIATE_GRPC_URL = os.getenv('WEAVIATE_GRPC_URL')
WEAVIATE_CLIENT_NAME = os.getenv('WEAVIATE_CLIENT_NAME')

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')

# Flask Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'json', 'txt'} 