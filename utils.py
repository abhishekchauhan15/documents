import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import docx
from pypdf import PdfReader
from werkzeug.utils import secure_filename
from weaviate.connect import ConnectionParams
from weaviate.auth import AuthApiKey
import weaviate

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
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)
            return file_path
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return None

    @staticmethod
    def read_file_content(file_path: str) -> Optional[str]:
        """Read content from different file types."""
        try:
            file_extension = Path(file_path).suffix.lower()[1:]
            
            if file_extension == 'txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            
            elif file_extension == 'json':
                with open(file_path, 'r') as file:
                    return json.dumps(json.load(file))
            
            elif file_extension == 'pdf':
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                return text
            
            elif file_extension == 'docx':
                doc = docx.Document(file_path)
                return ' '.join([paragraph.text for paragraph in doc.paragraphs])
            
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None

class WeaviateHelper:
    @staticmethod
    def check_connection(client: Any) -> bool:
        """Check if Weaviate connection is working."""
        try:
            client.is_ready()
            logger.info("Successfully connected to Weaviate!")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {str(e)}")
            return False

    @staticmethod
    def store_document_chunk(batch: Any, chunk: str, embedding: List[float], 
                           source: str, chunk_index: int) -> None:
        """Store a document chunk in Weaviate."""
        batch.add_data_object(
            data_object={
                "content": chunk,
                "source": source,
                "chunk_index": chunk_index,
                "status": "processed"
            },
            class_name="Document",
            vector=embedding
        )

    @staticmethod
    def initialize_client(
        rest_url: str,
        grpc_url: str,
        client_name: str,
        api_key: str,
        http_port: int = 8080,
        grpc_port: int = 50051,
        secure: bool = True
    ) -> Any:
        """Initialize and return a Weaviate client with standard configuration."""
        try:
            connection_params = ConnectionParams.from_params(
                http_host=rest_url,
                http_port=http_port,
                http_secure=secure,
                grpc_host=grpc_url,
                grpc_port=grpc_port,
                grpc_secure=secure
            )

            auth_config = AuthApiKey(api_key=api_key)

            client = weaviate.WeaviateClient(
                connection_params=connection_params,
                auth_client_secret=auth_config,
                additional_headers={
                    "X-Weaviate-Client-Name": client_name
                }
            )
            
            # Test connection
            if WeaviateHelper.check_connection(client):
                return client
            return None
            
        except Exception as e:
            logger.error(f"Error initializing Weaviate client: {str(e)}")
            return None

class TimeTracker:
    @staticmethod
    def track_time(func):
        """Decorator to track execution time of functions."""
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
            return result
        return wrapper 