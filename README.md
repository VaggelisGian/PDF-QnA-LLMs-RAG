# PDF-QnA-LLMs-RAG

## Introduction

Upload your documents (PDF or TXT), and the system processes them into a queryable knowledge base stored in a Neo4j graph database. You can then ask questions in natural language and get concise answers synthesized directly from the source material using Retrieval-Augmented Generation (RAG).


*   **Frontend:** An interactive web interface powered by **Streamlit**, making it easy to upload files and chat with your data.
*   **Backend:** A robust API built with **FastAPI** handles all the heavy lifting asynchronously.
*   **Document Handling:** We use **PyPDF2/pypdf** for reliable text extraction and **Langchain** (specifically `RecursiveCharacterTextSplitter`) to break down the text into manageable chunks.
*   **Understanding Content:** **Sentence Transformers** (via Langchain) create vector embeddings to capture semantic meaning. For deeper insights, we optionally use **spaCy/scispaCy** (with the `en_core_sci_sm` model) for Named Entity Recognition (NER), identifying key entities like names, organizations, and concepts.
*   **Knowledge Storage:** All this information – documents, text chunks, embeddings, and optionally entities/relationships – is stored in a **Neo4j graph database**. This allows for rich, interconnected data representation and efficient vector searches.
*   **Answering Questions (RAG):** We implement two Retrieval-Augmented Generation (RAG) methods:
    1.  **Standard RAG:** Finds relevant text chunks based on vector similarity to your question.
    2.  **Graph RAG:** Uses a Large Language Model (LLM) to generate a **Cypher** query, retrieving highly relevant, interconnected context from the Neo4j graph before synthesizing an answer.
*   **Easy Setup:** The entire system is containerized using **Docker** and orchestrated with **Docker Compose**, making setup and deployment straightforward.

Essentially, you upload documents, the system processes and stores them intelligently in Neo4j, and then you can query this knowledge base using either standard semantic search or the more advanced Graph RAG approach through the Streamlit UI.

*This setup was validated using Docker with LM Studio running locally, serving the `mistralai-mistral-7b-instruct-v0.2-smashed` model (Just-in-Time Model Loading OFF, context length ~8000).*

### Batch CLI Functionality

The Batch CLI functionality allows users to query multiple questions at once, either by uploading and processing a document or by querying a pre-processed document. This feature is useful for testing or retrieving answers for a set of predefined questions stored in a JSON file.

---

### How to Use the Batch CLI Feature

#### 1. **Prepare Your Questions**
   - Create a JSON file containing the questions you want to query. Use the following structure:
     ```json
     {
         "questions": [
             {
                 "question": "What are the essential components of the initial pre-hospital assessment in trauma care?",
                 "use_graph": false
             },
             {
                 "question": "What is the protocol for controlling external hemorrhage?",
                 "use_graph": true
             }
         ]
     }
     ```
   - Save this file (e.g., `test/batch_request.json`) in a location accessible to your scripts.

---

#### 2. **Option 1: Upload and Process a Document**
   Use the `run_upload_and_batch_test.ps1` script to upload a document, process it, and query your questions in one step.

   **Steps:**
   1. Open a terminal or PowerShell.
   2. Run the script with the following command:
      ```powershell
      .\test\run_upload_and_batch_test.ps1 -PdfFilePath ".\data\pdf_files\TRAUMA CARE PRE-HOSPITAL MANUAL.pdf"
      ```
   3. The script will:
      - Upload the specified PDF document.
      - Process the document into a queryable knowledge base.
      - Query the questions from the JSON file.
      - Save the results in a JSON file (e.g., `batch_results.json`).

---

#### 3. **Option 2: Query a Pre-Processed Document**
   If the document has already been uploaded and processed, you can skip the upload step and directly query the questions using the `run_upload_and_batch_test.ps1` script.

   **Steps:**
   1. Open a terminal or PowerShell.
   2. Run the script with the following command:
      ```powershell
      .\test\run_upload_and_batch_test.ps1 -DocumentTitleToQuery "TRAUMA CARE PRE-HOSPITAL MANUAL.pdf"
      ```
   3. The script will:
      - Query the questions from the JSON file against the pre-processed document.
      - Save the results in a JSON file (e.g., `batch_results.json`).

---

#### 4. **Understanding the Results**
   - The output JSON file will have the following structure:
     ```json
     {
         "results": [
             {
                 "question": "What are the essential components of the initial pre-hospital assessment in trauma care?",
                 "use_graph": false,
                 "answer": "The essential components include ensuring scene safety, assessing the mechanism of injury, performing a structured primary survey, and initiating life-saving interventions.",
             },
             {
                 "question": "What is the protocol for controlling external hemorrhage?",
                 "use_graph": true,
                 "answer": "Control of external hemorrhage includes applying direct pressure, using a tourniquet, and employing hemostatic dressings.",
             }
         ]
     }
     ```

---

### Notes:
- **Graph vs. Non-Graph Queries**:  
  - Set `"use_graph": true` to use the GraphRAG approach, which leverages the Neo4j graph database for interconnected context.
  - Set `"use_graph": false` to use the standard RAG approach, which retrieves answers based on vector similarity.

- **Customizing Output**:  
  - Modify the scripts to adjust file paths, output locations, or logging preferences as needed.

---

This functionality streamlines the process of querying multiple questions, making it ideal for batch testing or retrieving insights from large documents.

## Key Features

*   **Interactive Web UI:** Built with Streamlit for easy interaction.
*   **Flexible Document Upload:** Accepts both PDF and TXT files.
*   **Configurable Ingestion:** Option to bypass detailed graph building (NER, topics, relationships) for faster processing when only standard RAG is needed.
*   **Asynchronous Processing:** Ingests documents in the background with real-time progress updates in the UI.
*   **Robust Text Extraction:** Leverages PyPDF2/pypdf for reliable text extraction from PDFs.
*   **Advanced Text Processing:** Employs Langchain's `RecursiveCharacterTextSplitter` for effective text cleaning and chunking.
*   **Vector Embeddings:** Generates embeddings for text chunks using Sentence Transformers via Langchain.
*   **Graph-Based Storage:** Stores documents, chunks, embeddings, and optionally entities, topics, and relationships in Neo4j for rich contextual understanding. Includes Named Entity Recognition (NER) using spaCy/scispaCy (`en_core_sci_sm` by default).
*   **Efficient Vector Search:** Utilizes Neo4j vector indexes for fast similarity searches.
*   **Dual RAG Modes:**
    *   **Standard RAG:** Answers questions via semantic similarity search over text chunks.
    *   **Graph RAG:** Generates answers by first creating a Cypher query (using an LLM) to fetch relevant graph context, then synthesizing a response.
*   **Conversational Interface:** Engage in interactive chat sessions to query document content using the selected RAG mode.
*   **LLM Customization:** Adjust LLM temperature and max tokens for chat responses directly from the UI (compatible with OpenAI-like APIs, e.g., LM Studio).
*   **Modular API Backend:** FastAPI provides well-defined endpoints for uploads, chat, progress tracking, and health checks.
*   **Centralized Configuration:** Easily manage settings via environment variables using a dedicated `Config` class (`src/utils/config.py`).
*   **Standardized Logging:** Configurable logging (`src/utils/logger.py`) for both console and file output.

## System Architecture

The application uses Docker Compose to orchestrate the following services:

1.  **`frontend` (Streamlit - `src/frontend/app.py`):**
    *   Provides the user interface.
    *   Built using `src/frontend/Dockerfile`.
    *   Communicates with the `backend` service (`http://backend:8000`).
    *   Depends on `streamlit` and `requests` (`src/frontend/requirements.txt`).
    *   Polls the backend for ingestion progress.
2.  **`backend` (FastAPI - `src/backend/`):**
    *   Handles API requests, document ingestion pipeline (including NER with spaCy), and RAG logic.
    *   Built using the root `Dockerfile`. Installs PyTorch, spaCy model (`en_core_sci_sm`), and dependencies from `requirements.txt`.
    *   Connects to `redis` and `neo4j` services.
    *   Interfaces with external LLM/embedding services (e.g., host machine via `host.docker.internal`).
    *   Exposes API endpoints for uploads, chat, progress, and health.
    *   Core dependencies listed in the root `requirements.txt`.
3.  **`neo4j` (Neo4j Database):**
    *   Uses the official `neo4j` image with the APOC plugin.
    *   Stores the knowledge graph.
    *   **Persistence Note:** By default, graph data uses `tmpfs` and is **not persistent** across container restarts. For persistence, modify `docker-compose.yml` to use a named volume for the `/data` directory.
4.  **`redis` (Redis Cache):**
    *   Uses the official `redis` image.
    *   Stores job progress information.
    *   Data is persisted using a local volume (`./redis-data`).

**Core Utilities (`src/utils/`):**

*   **Configuration (`config.py`):** Loads settings from environment variables via a `Config` class.
*   **Logging (`logger.py`):** Provides a `setup_logger` function for consistent application logs.

## Setup & Installation (Docker Compose - Recommended)

**Prerequisites:**

*   Docker & Docker Compose installed.
*   An OpenAI-compatible API provider (like LM Studio) running locally or accessible via URL.
    *   *Tested Model:* `mistralai-mistral-7b-instruct-v0.2-smashed`.
    *   *Recommendation:* Ensure the model is fully loaded (disable Just-in-Time loading) and the server context length is adequate (e.g., ~8000 tokens).
**Steps:**

1.  **Clone the Repository:**
    ````bash
    git clone https://github.com/VaggelisGian/PDF-QnA-LLMs-RAG
    cd PDF-QnA-LLMs-RAG
    ````
2.  **Configure Environment Variables:**
    *   Create a `.env` (if needed) file in the project's root directory (`PDF-QnA-LLMs-RAG/.env`).
    *   Populate it with your settings. **Essential variables for Docker Compose:**
        *   `NEO4J_PASSWORD`: Set your desired password for the Neo4j database.
        *   `LM_STUDIO_API_BASE`:
            *   If using LM Studio on your **host machine** with Docker Desktop, the default `http://host.docker.internal:1234/v1` should work.
            *   Adjust `host.docker.internal` to your host's IP if needed (e.g., on Linux without Docker Desktop).
            *   Use the appropriate URL for cloud services or other setups.
    *   Review and customize other variables (ports, model names, etc.) as needed (refer to `src/utils/config.py` for details).
3.  **Build and Run Services:**
    ````bash
    docker-compose up --build -d
    ````
    This command builds the necessary Docker images and starts all services (`frontend`, `backend`, `neo4j`, `redis`) in the background.

## Usage Guide

1.  **Access UI:** Open your browser to the Streamlit UI (default: `http://localhost:8501`, or the host port mapped in `docker-compose.yml`). The UI will indicate when backend services are ready.
2.  **Upload Document:** Use the file uploader to select a PDF or TXT file.
3.  **Configure Ingestion (Optional):** Check "Skip Graph Build" for faster processing if you only need standard vector RAG.
4.  **Process:** Click "Process Document". Monitor the real-time progress updates.
5.  **Chat:** Once processing completes:
    *   Select the desired RAG mode (**Standard** or **Graph RAG**) in the sidebar. *Note: Graph RAG requires the graph build step.*
    *   Adjust LLM **Temperature** and **Max Tokens** in the sidebar if needed.
    *   Enter your question in the chat input at the bottom and press Enter.
    *   View the generated answer. Standard RAG mode will also display source chunks.

## Project Structure

```
PDF-QnA-LLMs-RAG/
├── data/
│   └── pdf_files/
│       ├── Pre-hospital management of the baby born at extreme preterm ge...
│       ├── Pediatric Advanced Life Support Study Guide (Pals) Fourth Edition.pdf
│       └── TRAUMA CARE PRE-HOSPITAL MANUAL.pdf
├── neo4j/
├── redis-data/
├── src/
│   ├── backend/
│   │   ├── __pycache__/
│   │   ├── api/
│   │   │   ├── __pycache__/
│   │   │   ├── endpoints.py
│   │   │   ├── models.py
│   │   │   ├── progress.py
│   │   │   └── websocket.py
│   │   ├── assistant/
│   │   │   ├── __pycache__/
│   │   │   ├── graph_rag.py
│   │   │   └── rag.py
│   │   ├── database/
│   │   │   ├── __pycache__/
│   │   │   ├── neo4j_client.py
│   │   │   └── redis_client.py
│   │   └── document_processing/
│   │       ├── __pycache__/
│   │       ├── add_embeddings.py
│   │       ├── pdf_loader.py
│   │       ├── process_documents.py
│   │       └── text_processor.py
│   ├── frontend/
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── main.py
├── test/
├── venv/
├── .env
├── docker-compose.yml
├── Dockerfile
├── image.png
├── README.md
└── requirements.txt
```