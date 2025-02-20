from flask import Flask, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
from langchain_ollama import OllamaEmbeddings
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    REDIS_URL
)
from tasks import process_document_task
import logging
from utils import FileHandler, WeaviateHelper, logger, IndexingHelper
from pathlib import Path
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Get Weaviate client
client = WeaviateHelper.get_client()

if client is None:
    raise Exception("Failed to initialize Weaviate client")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.before_request
def before_request():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/ingest', methods=['POST'])
def ingest_document():
    logger.info("Ingesting document...")
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and FileHandler.allowed_file(file.filename, ALLOWED_EXTENSIONS):
        file_path = FileHandler.secure_file_save(file, app.config['UPLOAD_FOLDER'])
        if not file_path:
            logger.error("Error saving file")
            return jsonify({'error': 'Error saving file'}), 500
        
        logger.info(f"Document uploaded successfully: {file_path}")
        task = process_document_task.delay(file_path)
        
        return jsonify({
            'message': 'Document uploaded successfully and queued for processing',
            'filename': Path(file_path).name,
            'task_id': task.id
        }), 200
    
    logger.warning("Invalid file type")
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    logger.info(f"Checking status for task_id: {task_id}")
    task = process_document_task.AsyncResult(task_id)
    response = {
        'task_id': task_id,
        'status': task.status,
    }
    if task.status == 'FAILURE':
        response['error'] = str(task.result)
        logger.error(f"Task {task_id} failed: {response['error']}")
    return jsonify(response)

@app.route('/query', methods=['POST'])
def query_document():
    logger.info("Querying document...")
    data = request.json
    if not data or 'query' not in data or 'document_name' not in data:
        logger.warning("Missing query or document_name")
        return jsonify({'error': 'Missing query or document_name'}), 400
    
    query = data['query']
    document_name = data['document_name']
    limit = data.get('limit', 3)
    
    try:
        # Initialize components directly
        embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        indexer = IndexingHelper(REDIS_URL)  # Create an instance of IndexingHelper

        print("indexer", indexer)

        indexer.initialize_vector_store(client)
        # Query using the updated method
        results = indexer.query_document(
            query=query,
            document_name=document_name,
            limit=limit,
            # embeddings=embeddings  # Pass embeddings if needed
        )
        
        logger.info(f"Query results for {document_name}: {results}")
        return jsonify({
            'query': query,
            'document': document_name,
            'model': OLLAMA_MODEL,
            'results': results,
            'total_chunks': len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
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
        indexer.initialize_components( client, embeddings)
        
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
    logger.info(f"Updating document: {document_name}")
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
            
            task = process_document_task.delay(file_path)
            
            logger.info(f"Document update queued for: {document_name}")
            return jsonify({
                'message': 'Document update queued',
                'filename': Path(file_path).name,
                'task_id': task.id
            }), 200
            
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(
    model=OLLAMA_MODEL,  # Ensure this is set to "llama3.1:latest"
    base_url=OLLAMA_BASE_URL  # Ensure this points to your local Ollama server
)

# Configure global settings
Settings.embed_model = embeddings
Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=50)

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