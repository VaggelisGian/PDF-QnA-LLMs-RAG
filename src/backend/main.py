import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import traceback
import os
import redis

from .api import endpoints as api_endpoints
from .api import progress as progress_api
from .assistant.rag import RAGAssistant
from .assistant.graph_rag import GraphRAGAssistant
from .database.neo4j_client import Neo4jClient

rag_assistant: RAGAssistant = None
graph_rag_assistant: GraphRAGAssistant = None

app = FastAPI(title="Intelligent PDF Retriever Backend")

origins = [
    "http://localhost",
    "http://localhost:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')
    try:
        redis_health_check_client = redis.Redis.from_url(
            redis_url, socket_connect_timeout=5, socket_timeout=5, decode_responses=True
        )
        redis_health_check_client.ping()
        print(f"Health Check using Redis at: {redis_url}")
        redis_health_check_client.close()
    except Exception as e:
        print(f"WARNING: Health check Redis connection failed at {redis_url}: {e}")

    print("Ensuring Neo4j vector index exists...")
    neo4j_index_client = None
    try:
        neo4j_index_client = Neo4jClient()
        embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))
        neo4j_index_client.ensure_vector_index(
            index_name="pdf_chunk_embeddings",
            node_label="Chunk",
            property_name="embedding",
            dimensions=embedding_dimension
        )
        print("Neo4j vector index check/creation initiated.")
    except Exception as idx_e:
        print(f"ERROR: Failed to ensure Neo4j vector index exists: {idx_e}")
        traceback.print_exc()
    finally:
        if neo4j_index_client:
            neo4j_index_client.close()

    print("Initializing Assistants on startup...")
    global rag_assistant, graph_rag_assistant
    app.state.rag_assistant_instance = None
    app.state.graph_rag_assistant_instance = None
    try:
        app.state.rag_assistant_instance = RAGAssistant()
        app.state.graph_rag_assistant_instance = GraphRAGAssistant()
        print("Assistants initialized.")
    except Exception as e:
        print(f"FATAL: Failed to initialize assistants: {e}")
        traceback.print_exc()

    print("Performing Neo4j connection test on startup...")
    neo4j_conn_test_client = None
    try:
        neo4j_conn_test_client = Neo4jClient()
        neo4j_conn_test_client.run_query("RETURN 1")
        print("Neo4j connection verified.")
    except Exception as neo_e:
        print(f"WARNING: Neo4j connection test failed on startup: {neo_e}")
    finally:
        if neo4j_conn_test_client:
            neo4j_conn_test_client.close()

app.include_router(api_endpoints.router, prefix="/api", tags=["api"])
app.include_router(progress_api.router, prefix="/api/progress", tags=["progress"])

@app.get("/api/health", tags=["health"])
async def health_check(request: Request):
    redis_status = "Redis connection failed"
    try:
        redis_client = progress_api.redis_client
        if not redis_client:
             redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')
             redis_client = redis.Redis.from_url(redis_url, socket_connect_timeout=1, socket_timeout=1)
        redis_client.ping()
        redis_status = "Redis connection OK"
    except Exception as e:
        redis_status = f"Redis connection failed: {e}"

    neo4j_status = "Neo4j connection failed"
    neo4j_client = None
    try:
        neo4j_client = Neo4jClient()
        neo4j_client.run_query("RETURN 1")
        neo4j_status = "Neo4j connection OK"
    except Exception as e:
        neo4j_status = f"Neo4j connection failed: {e}"
    finally:
        if neo4j_client:
            neo4j_client.close()

    rag_status = "RAG Assistant Initialized" if request.app.state.rag_assistant_instance else "RAG Assistant NOT Initialized"
    graph_rag_status = "GraphRAG Assistant Initialized" if request.app.state.graph_rag_assistant_instance else "GraphRAG Assistant NOT Initialized"

    if "failed" in redis_status or "failed" in neo4j_status or not request.app.state.rag_assistant_instance or not request.app.state.graph_rag_assistant_instance:
         raise HTTPException(status_code=503, detail={
             "redis": redis_status,
             "neo4j": neo4j_status,
             "rag_assistant": rag_status,
             "graph_rag_assistant": graph_rag_status
         })

    return {
        "status": "ok",
        "redis": redis_status,
        "neo4j": neo4j_status,
        "rag_assistant": rag_status,
        "graph_rag_assistant": graph_rag_status
    }

if __name__ == "__main__":
    uvicorn.run("src.backend.main:app", host="0.0.0.0", port=8000, reload=True)