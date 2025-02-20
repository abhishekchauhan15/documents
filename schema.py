import weaviate
from weaviate.connect import ConnectionParams
from weaviate.auth import AuthApiKey
from config import (
    WEAVIATE_REST_URL,
    WEAVIATE_GRPC_URL,
    WEAVIATE_CLIENT_NAME,
    WEAVIATE_ADMIN_API_KEY
)
from utils import FileHandler, WeaviateHelper, logger, IndexingHelper

# Get Weaviate client
client = WeaviateHelper.get_client()

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