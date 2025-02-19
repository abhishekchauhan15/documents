import weaviate
from weaviate.connect import ConnectionParams
from weaviate.auth import AuthApiKey
from config import (
    WEAVIATE_REST_URL,
    WEAVIATE_GRPC_URL,
    WEAVIATE_CLIENT_NAME,
    WEAVIATE_ADMIN_API_KEY
)
from utils import WeaviateHelper

# Initialize Weaviate client
client = WeaviateHelper.initialize_client(
    rest_url=WEAVIATE_REST_URL,
    grpc_url=WEAVIATE_GRPC_URL,
    client_name=WEAVIATE_CLIENT_NAME,
    api_key=WEAVIATE_ADMIN_API_KEY
)

if client is None:
    raise Exception("Failed to initialize Weaviate client")

class_obj = {
    "class": "Document",
    "vectorizer": "none",
    "properties": [
        {
            "name": "content",
            "dataType": ["text"]
        },
        {
            "name": "source",
            "dataType": ["string"]
        },
        {
            "name": "chunk_index",
            "dataType": ["int"]
        },
        {
            "name": "page_number",
            "dataType": ["int"]
        },
        {
            "name": "timestamp",
            "dataType": ["date"]
        },
        {
            "name": "file_type",
            "dataType": ["string"]
        },
        {
            "name": "metadata",
            "dataType": ["object"]
        }
    ]
}

# Add the schema
client.schema.create_class(class_obj) 