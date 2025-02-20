from flask import Flask, request, jsonify
import os
from config import (
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    REDIS_URL,
    QUERY_RESULT_LIMIT,
    MAX_DOCUMENT_SIZE
)
from tasks import process_document_task
import logging
from utils import FileHandler, WeaviateHelper, logger, IndexingHelper
from pathlib import Path

app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_DOCUMENT_SIZE

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
    """
    Ingest a new document for processing.
    Accepts multipart/form-data with a 'file' field.
    """
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

@app.route('/update/<document_name>', methods=['POST'])
def update_document(document_name):
    """
    Update an existing document.
    Accepts multipart/form-data with a 'file' field.
    """
    logger.info(f"Updating document: {document_name}")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and FileHandler.allowed_file(file.filename, ALLOWED_EXTENSIONS):
        try:
            # Delete existing document from Weaviate
            indexing_helper = IndexingHelper(REDIS_URL)
            indexing_helper.delete_document(document_name)
            
            # Process new document
            file_path = FileHandler.secure_file_save(file, app.config['UPLOAD_FOLDER'])
            if not file_path:
                return jsonify({'error': 'Error saving file'}), 500
            
            task = process_document_task.delay(file_path)
            
            return jsonify({
                'message': 'Document update queued',
                'filename': Path(file_path).name,
                'task_id': task.id
            }), 200
            
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/query', methods=['POST'])
def query_document():
    """Query a document with a search query."""
    try:
        data = request.get_json()
        if not data or 'query' not in data or 'document_name' not in data:
            return jsonify({'error': 'Missing query or document_name in request'}), 400
            
        query = data['query']
        document_name = data['document_name']
        limit = data.get('limit', 3)
        
        # Initialize helpers
        indexing_helper = IndexingHelper(REDIS_URL)
        
        # Get available documents
        available_docs = indexing_helper.list_documents()
        logger.info(f"Available documents: {available_docs}")
        
        # Find the exact document by name
        matching_doc = None
        for doc in available_docs:
            if doc['fileName'] == document_name:
                matching_doc = doc
                break
                
        if not matching_doc:
            logger.error(f"Document not found. Available documents: {available_docs}")
            return jsonify({
                'error': 'Document not found',
                'available_documents': available_docs
            }), 404
            
        # Query using the document name
        results = indexing_helper.query_document(
            query=query,
            document_name=document_name,
            limit=limit
        )
        
        return jsonify({
            'results': results,
            'document_info': matching_doc
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get the status of a document processing task."""
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

@app.route('/health', methods=['GET'])
def health_check():
    """Check the health of the application and its dependencies."""
    try:
        client = WeaviateHelper.get_client()
        if client is None:
            return jsonify({'status': 'error', 'message': 'Failed to connect to Weaviate'}), 500
        return jsonify({'status': 'healthy'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)