from flask import Flask, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import json
from langchain_ollama import OllamaEmbeddings
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
from utils import FileHandler, WeaviateHelper, logger, IndexingHelper
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

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
        return jsonify({'error': 'Missing query or document_name'}), 400
    
    query = data['query']
    document_name = data['document_name']
    limit = data.get('limit', 3)
    
    try:
        # Initialize components
        embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        indexer = IndexingHelper(REDIS_URL)
        indexer.initialize_components(embeddings, client)
        
        # Query using LlamaIndex
        results = indexer.query_document(
            query=query,
            document_name=document_name,
            limit=limit
        )
        
        return jsonify({
            'query': query,
            'document': document_name,
            'model': OLLAMA_MODEL,
            'results': results,
            'total_chunks': len(results)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query/json', methods=['POST'])
def query_json_document():
    data = request.json
    if not data or 'document_name' not in data:
        return jsonify({'error': 'Missing document_name'}), 400
    
    document_name = data['document_name']
    
    try:
        # Initialize components
        embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        indexer = IndexingHelper(REDIS_URL)
        indexer.initialize_components(embeddings, client)
        
        # Query using JSON helper
        results = indexer.query_json_document(
            query=data,
            document_name=document_name
        )
        
        return jsonify(results), 200
        
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

@app.route('/update/<document_name>', methods=['POST'])
def update_document(document_name):
    """Update is equivalent to re-uploading with changes"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and FileHandler.allowed_file(file.filename, ALLOWED_EXTENSIONS):
        # First, delete existing document chunks
        try:
            client.batch.delete_objects(
                class_name="Document",
                where={
                    "path": ["source"],
                    "operator": "Equal",
                    "valueString": document_name
                }
            )
            
            # Then process new document
            file_path = FileHandler.secure_file_save(file, app.config['UPLOAD_FOLDER'])
            if not file_path:
                return jsonify({'error': 'Error saving file'}), 500
            
            task = process_document.delay(file_path)
            
            return jsonify({
                'message': 'Document update queued',
                'filename': Path(file_path).name,
                'task_id': task.id
            }), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

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