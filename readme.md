# Document RAG System

A robust Retrieval-Augmented Generation (RAG) system for document processing and querying using LlamaIndex, Weaviate, and Ollama.

## System Architecture

### Components
- **Flask API Server**: Handles HTTP requests and document uploads
- **Celery Workers**: Manages asynchronous document processing
- **Redis**: Message broker and result backend for Celery
- **Weaviate**: Vector database for storing document embeddings
- **Ollama**: Local LLM for generating embeddings
- **LlamaIndex**: Document processing and retrieval framework

### Workflow
1. Document Upload → Flask API
2. Async Processing → Celery Worker
3. Embedding Generation → Ollama
4. Vector Storage → Weaviate
5. Query Processing → LlamaIndex + Weaviate
6. Result Delivery → Flask API

## Setup Instructions

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Weaviate Cloud Account (or self-hosted instance)
- 8GB+ RAM recommended

### Environment Setup
1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd document-rag-system
   ```
2. Create a `.env` file with the following configuration:
   ```ini
   # Weaviate Configuration
   WEAVIATE_REST_URL=your-weaviate-url
   WEAVIATE_GRPC_URL=your-weaviate-grpc-url
   WEAVIATE_CLIENT_NAME=your-client-name
   WEAVIATE_ADMIN_API_KEY=your-api-key
   
   # Ollama Configuration
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.1:latest
   
   # Redis Configuration
   REDIS_URL=redis://redis:6379/0
   
   # Document Processing
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   ```
3. Start services:
   ```sh
   docker-compose up --build
   ```

## API Documentation

### 1. Document Ingestion
**Endpoint:**
   ```http
   POST /ingest
   ```
**Headers:**
   ```http
   Content-Type: multipart/form-data
   ```
**Body Parameters:**
   ```
   file: <document>
   ```
Supported formats:
- PDF (.pdf)
- Word (.docx)
- Text (.txt)
- JSON (.json)

**Response:**
   ```json
   {
      "message": "Document uploaded successfully and queued for processing",
      "filename": "document.pdf",
      "task_id": "123e4567-e89b-12d3-a456-426614174000"
   }
   ```

### 2. Query Document
**Endpoint:**
   ```http
   POST /query
   ```
**Headers:**
   ```http
   Content-Type: application/json
   ```
**Body:**
   ```json
   {
      "query": "What is the main topic?",
      "document_name": "document.pdf",
      "limit": 3
   }
   ```
**Response:**
   ```json
   {
      "query": "What is the main topic?",
      "document": "document.pdf",
      "model": "llama3.1:latest",
      "results": [
         {
            "content": "...",
            "source": "document.pdf",
            "document_id": "...",
            "relevance_score": 0.92,
            "chunk_index": 1,
            "page_number": 1,
            "timestamp": "2024-01-01T12:00:00Z"
         }
      ],
      "total_chunks": 1
   }
   ```

### 3. JSON Query (Advanced)
**Endpoint:**
   ```http
   POST /query/json
   ```
**Headers:**
   ```http
   Content-Type: application/json
   ```
**Body:**
   ```json
   {
      "document_name": "data.json",
      "text": "find sales data",
      "aggregate": {
         "field": "amount",
         "operation": "all"
      },
      "group_by": {
         "field": "region",
         "aggregate_field": "amount",
         "operation": "sum"
      }
   }
   ```

## Design Choices

### Performance Optimizations
1. **Document Processing**
   - Parallel PDF page processing
   - Chunked file reading for large documents
   - Optimized document splitting strategies

2. **Query Processing**
   - Embedding caching in Redis
   - Parallel result processing
   - Streaming for large results

3. **Storage**
   - Efficient vector indexing
   - Metadata optimization
   - Batch processing for vectors

### Security Measures
- Secure file handling
- Input validation
- Non-root Docker containers
- Environment-based configuration
- API key authentication

## Development

### Local Development Setup
1. Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run services individually:
   ```sh
   # Terminal 1: Redis
   docker run -p 6379:6379 redis:alpine
   
   # Terminal 2: Ollama
   docker run -p 11434:11434 ollama/ollama
   
   # Terminal 3: Flask API
   python app.py
   
   # Terminal 4: Celery Worker
   celery -A tasks worker --pool=solo -l info
   ```
4. Check API health:
   ```sh
   curl http://localhost:5000/health
   ```

5. **Celery Tasks**
   - Check task status:
     ```sh
     curl http://localhost:5000/status/<task_id>
     ```

6. **Memory Issues**
   - Increase Docker memory limit
   - Adjust chunk size in `.env`
   - Monitor Redis memory usage

### Performance Tuning

1. **Document Processing**
   ```ini
   CHUNK_SIZE=1000  # Adjust based on document size
   CHUNK_OVERLAP=200  # Adjust for context preservation
   ```
2. **Query Performance**
   - Use `limit` parameter in queries
   - Enable result streaming for large documents
   - Implement proper indexing in Weaviate

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Create a Pull Request

## License
This project is licensed under the MIT License.

