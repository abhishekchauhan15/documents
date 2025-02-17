from flask import Flask, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import json
from langchain_ollama import OllamaEmbeddings
from weaviate.connect import ConnectionParams
from weaviate.auth import AuthApiKey  
from config import (
    WEAVIATE_REST_URL,
    WEAVIATE_GRPC_URL,
    WEAVIATE_CLIENT_NAME,
    WEAVIATE_ADMIN_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS
)
from tasks import process_document
import logging
from utils import FileHandler, WeaviateHelper, logger
from pathlib import Path

app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Weaviate client
client = WeaviateHelper.initialize_client(
    rest_url=WEAVIATE_REST_URL,
    grpc_url=WEAVIATE_GRPC_URL,
    client_name=WEAVIATE_CLIENT_NAME,
    api_key=WEAVIATE_ADMIN_API_KEY
)

if client is None:
    raise Exception("Failed to initialize Weaviate client")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_weaviate_connection():
    return WeaviateHelper.check_connection(client)

def initialize_app():
    if not check_weaviate_connection():
        raise Exception("Could not connect to Weaviate")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

initialize_app()


@app.before_request
def before_request():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/ingest', methods=['POST'])
def ingest_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and FileHandler.allowed_file(file.filename, ALLOWED_EXTENSIONS):
        file_path = FileHandler.secure_file_save(file, app.config['UPLOAD_FOLDER'])
        if not file_path:
            return jsonify({'error': 'Error saving file'}), 500
        
        # Queue the document for processing
        task = process_document.delay(file_path)
        
        return jsonify({
            'message': 'Document uploaded successfully and queued for processing',
            'filename': Path(file_path).name,
            'task_id': task.id
        }), 200
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task = process_document.AsyncResult(task_id)
    response = {
        'task_id': task_id,
        'status': task.status,
    }
    if task.status == 'FAILURE':
        response['error'] = str(task.result)
    return jsonify(response)

@app.route('/query', methods=['POST'])
def query_document():
    data = request.json
    if not data or 'query' not in data or 'document_name' not in data:
        return jsonify({'error': 'Missing query or document name'}), 400
    
    query = data['query']
    document_name = data['document_name']
    
    try:
        # Initialize embeddings for query
        embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        # Generate embedding for the query
        query_embedding = embeddings.embed_query(query)
        
        # Search in Weaviate
        response = client.query.get(
            "Document",
            ["content", "source", "chunk_index"]
        ).with_where({
            "path": ["source"],
            "operator": "Equal",
            "valueString": document_name
        }).with_near_vector({
            "vector": query_embedding
        }).with_limit(3).do()
        
        results = response['data']['Get']['Document']
        
        return jsonify({
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    # Create upload folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Check if we can write to the upload folder
    can_write = os.access(UPLOAD_FOLDER, os.W_OK)
    
    return jsonify({
        'status': 'healthy'
    }), 200

if __name__ == '__main__':
    port = 5000
    host = '0.0.0.0'
    
    # Create upload folder at startup
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    logger.info(f"Starting server...")
    logger.info(f"Server URL: http://localhost:{port}")
    
    # Run the server without SSL for development
    app.run(
        host=host, 
        port=port, 
        debug=True,
        ssl_context=None
    ) 