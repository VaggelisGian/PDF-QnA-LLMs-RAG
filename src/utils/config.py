import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv("API_KEY", "your_api_key_here")
    API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_CONNECT_TIMEOUT = int(os.getenv('REDIS_CONNECT_TIMEOUT', '5'))
    REDIS_SOCKET_TIMEOUT = int(os.getenv('REDIS_SOCKET_TIMEOUT', '5'))
    REDIS_JOB_EXPIRY_SECONDS = int(os.getenv('REDIS_JOB_EXPIRY_SECONDS', '86400'))

    PDF_DIRECTORY = os.getenv("UPLOAD_DIR", "data/pdf_files")
    PROCESSED_DIRECTORY = os.getenv("PROCESSED_DIRECTORY", "data/processed")

    SPACY_MODEL_NAME = os.getenv("SPACY_MODEL_NAME", "en_core_sci_sm")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

    TEXT_CHUNK_SIZE = int(os.getenv("TEXT_CHUNK_SIZE", "1000"))
    TEXT_CHUNK_OVERLAP = int(os.getenv("TEXT_CHUNK_OVERLAP", "200"))

    RAG_RETRIEVER_K = int(os.getenv("RAG_RETRIEVER_K", "6"))

    DEFAULT_LLM_TEMP = float(os.getenv("DEFAULT_LLM_TEMP", "0.1"))
    DEFAULT_LLM_MAX_TOKENS = int(os.getenv("DEFAULT_LLM_MAX_TOKENS", "512"))
    TOPIC_LLM_MAX_TOKENS = int(os.getenv("TOPIC_LLM_MAX_TOKENS", "100"))

    LM_STUDIO_API_BASE = os.getenv("LM_STUDIO_API_BASE", "http://localhost:1234/v1")
    LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

    INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "20"))
    TOPIC_EXTRACTION_CHUNK_SIZE = int(os.getenv("TOPIC_EXTRACTION_CHUNK_SIZE", "3500"))
    TOPIC_EXTRACTION_CONCURRENCY = int(os.getenv("TOPIC_EXTRACTION_CONCURRENCY", "10"))
    MAX_TOPICS_PER_DOC = int(os.getenv("MAX_TOPICS_PER_DOC", "32"))
