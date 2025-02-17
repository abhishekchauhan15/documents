import weaviate
from config import WEAVIATE_REST_URL, WEAVIATE_CLIENT_NAME

client = weaviate.Client(
    url=WEAVIATE_REST_URL,
    additional_headers={
        "X-Weaviate-Client-Name": WEAVIATE_CLIENT_NAME
    }
)

class_obj = {
    "class": "Document",
    "vectorizer": "none",  # We'll provide our own vectors
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
        }
    ]
}

# Add the schema
client.schema.create_class(class_obj) 