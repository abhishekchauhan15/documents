from flask import Flask, request, jsonify
import os
from config import (
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    REDIS_URL,
    QUERY_RESULT_LIMIT,
    MAX_DOCUMENT_SIZE,
)
from tasks import process_document_task
import logging
from utils import FileHandler, WeaviateHelper, logger, IndexingHelper

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
    
    if not FileHandler.allowed_file(file.filename, ALLOWED_EXTENSIONS):
        logger.warning(f"File type not allowed: {file.filename}")
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Save file
        file_path = FileHandler.secure_file_save(file, UPLOAD_FOLDER)
        if not file_path:
            return jsonify({'error': 'Failed to save file'}), 500

        # Queue the document processing task
        task = process_document_task.delay(file_path)
        
        return jsonify({
            'status': 'success',
            'message': 'Document uploaded and queued for processing',
            'task_id': task.id
        }), 200

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query_document():
    """Query a document with a search query."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        query = data.get('query')
        document_id = data.get('document_id')
        limit = data.get('limit', QUERY_RESULT_LIMIT)

        if not query:
            return jsonify({'error': 'No query provided'}), 400
        if not document_id:
            return jsonify({'error': 'No document_id provided'}), 400

        indexing_helper = IndexingHelper(REDIS_URL)
        results = indexing_helper.query_document(query, document_id, limit)
        
        return jsonify(results), 200

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 404
    except Exception as e:
        logger.error(f"Error querying document: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get the status of a document processing task."""
    try:
        task = process_document_task.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            return jsonify({'status': 'processing'})
            
        elif task.state == 'SUCCESS':
            result = task.result
            if isinstance(result, dict) and result.get('status') == 'success':
                return jsonify({
                    'status': 'completed',
                    'document_id': result.get('document_id'),
                    'message': 'Document processing completed successfully'
                })
            else:
                return jsonify({
                    'status': 'failed',
                    'error': 'Document processing failed - invalid result format'
                }), 500
                
        elif task.state == 'FAILURE':
            return jsonify({
                'status': 'failed',
                'error': str(task.result)
            }), 500
            
        else:  # STARTED, RETRY, or other states
            return jsonify({'status': 'processing'})
            
    except Exception as e:
        logger.error(f"Error checking task status: {str(e)}")
        return jsonify({'error': str(e)}), 500

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