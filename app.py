from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import json
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import weaviate
import docx
import PyPDF2
from config import (
    WEAVIATE_REST_URL,
    WEAVIATE_CLIENT_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS
)

app = Flask(__name__)

# Configure upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Weaviate client
client = weaviate.Client(
    url=WEAVIATE_REST_URL,
    additional_headers={
        "X-Weaviate-Client-Name": WEAVIATE_CLIENT_NAME
    }
)

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_text(text):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def read_file_content(file_path):
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    elif file_extension == 'json':
        with open(file_path, 'r') as file:
            return json.dumps(json.load(file))
    
    elif file_extension == 'pdf':
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    elif file_extension == 'docx':
        doc = docx.Document(file_path)
        return ' '.join([paragraph.text for paragraph in doc.paragraphs])

def check_weaviate_connection():
    try:
        client.schema.get()
        return True
    except Exception as e:
        print(f"Failed to connect to Weaviate: {str(e)}")
        return False

@app.before_first_request
def initialize():
    if not check_weaviate_connection():
        raise Exception("Could not connect to Weaviate")

@app.route('/ingest', methods=['POST'])
def ingest_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Read and process the document
        try:
            content = read_file_content(file_path)
            chunks = process_text(content)
            
            # Generate embeddings and store in Weaviate
            for i, chunk in enumerate(chunks):
                embedding = embeddings.embed_query(chunk)
                
                # Store in Weaviate
                client.batch.add_data_object(
                    data_object={
                        "content": chunk,
                        "source": filename,
                        "chunk_index": i
                    },
                    class_name="Document",
                    vector=embedding
                )
            
            return jsonify({
                'message': 'Document processed successfully',
                'filename': filename
            }), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True) 