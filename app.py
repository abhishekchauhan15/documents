from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from langchain_ollama import OllamaEmbeddings
from config import (
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    REDIS_URL
)
from tasks import process_document_task
import logging
from utils import FileHandler, WeaviateHelper, logger, IndexingHelper
from pathlib import Path

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
        # Initialize components
        indexing_helper = IndexingHelper(REDIS_URL)
        indexing_helper.initialize_vector_store(client)
        
        # Perform query
        results = indexing_helper.query_document(query, document_name, limit)
        
        return jsonify({
            'query': query,
            'document': document_name,
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Error querying document: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        client = WeaviateHelper.get_client()
        if client is None:
            return jsonify({'status': 'error', 'message': 'Failed to connect to Weaviate'}), 500
        return jsonify({'status': 'healthy'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)