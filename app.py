from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import json
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import weaviate
import docx
import PyPDF2

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'json', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Weaviate client
client = weaviate.Client(
    url="http://localhost:8080"  # Update with your Weaviate instance URL
)

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model="llama2")

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

@app.route('/query', methods=['POST'])
def query_document():
    data = request.json
    if not data or 'query' not in data or 'document_name' not in data:
        return jsonify({'error': 'Missing query or document name'}), 400
    
    query = data['query']
    document_name = data['document_name']
    
    try:
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

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True) 