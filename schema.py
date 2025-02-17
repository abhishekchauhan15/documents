import weaviate

client = weaviate.Client(
    url="http://localhost:8080"  # Update with your Weaviate instance URL
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