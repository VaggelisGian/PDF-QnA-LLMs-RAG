from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Request , Form
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import shutil
import spacy
import chardet
import uuid
import json
import redis
import asyncio
import traceback
import requests
import datetime
from dotenv import load_dotenv
from src.backend.api.models import ChatRequest ,ChatResponse,SourceDocument,BatchChatResponse,BatchChatRequest,BatchChatRequestItem,BatchChatResponseItem
from src.backend.api.progress import redis_client as shared_redis_client
from src.backend.api.progress import router as progress_router
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.backend.api.progress import create_job, update_job_progress, update_job_status, progress_complete_job, complete_job_sync
from src.backend.document_processing.pdf_loader import PDFLoader
from src.backend.document_processing.text_processor import TextProcessor
from src.backend.database.neo4j_client import Neo4jClient
from src.backend.assistant.rag import RAGAssistant
from src.backend.assistant.graph_rag import GraphRAGAssistant
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

NLP_MODEL_NAME = os.getenv("SPACY_MODEL_NAME", "en_core_sci_sm")
try:
    nlp = spacy.load(NLP_MODEL_NAME)
    print(f"spaCy NER model '{NLP_MODEL_NAME}' loaded successfully.")
except OSError:
    print(f"ERROR: spaCy model '{NLP_MODEL_NAME}' not found.")
    print("Please ensure it's installed via pip, e.g.:")
    print("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz")
    nlp = None
except ImportError as import_err:
     print(f"ERROR: Failed to import spaCy or scispaCy: {import_err}")
     print("Please ensure 'spacy' and 'scispacy' are installed: pip install spacy scispacy")
     nlp = None

SPACY_LABEL_TO_ENTITY_TYPE = {
    "PERSON": "Person", "NORP": "Group", "FAC": "Facility", "ORG": "Organization",
    "GPE": "Location", "LOC": "Location", "PRODUCT": "Product", "EVENT": "Event",
    "WORK_OF_ART": "WorkOfArt", "LAW": "Law", "LANGUAGE": "Language", "DATE": "Date",
    "TIME": "Time", "PERCENT": "Percent", "MONEY": "Money", "QUANTITY": "Quantity",
    "ORDINAL": "Ordinal", "CARDINAL": "Cardinal", "ENTITY": "MedicalEntity",
}

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'vaggpinel')
UPLOAD_DIR = os.getenv('UPLOAD_DIR', 'data/pdf_files')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
EMBEDDING_DEVICE = os.getenv('EMBEDDING_DEVICE', 'cpu')
LM_STUDIO_API_BASE = os.getenv('LM_STUDIO_API_BASE', 'http://localhost:1234/v1')
LM_STUDIO_API_KEY = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')
DEFAULT_LLM_TEMP = float(os.getenv('DEFAULT_LLM_TEMP', '0.1'))
DEFAULT_LLM_MAX_TOKENS = int(os.getenv('DEFAULT_LLM_MAX_TOKENS', '512'))
TOPIC_LLM_MAX_TOKENS = int(os.getenv('TOPIC_LLM_MAX_TOKENS', '100'))
INGEST_BATCH_SIZE = int(os.getenv('INGEST_BATCH_SIZE', '20'))
TOPIC_EXTRACTION_CHUNK_SIZE = int(os.getenv('TOPIC_EXTRACTION_CHUNK_SIZE', '3500'))
TOPIC_EXTRACTION_CONCURRENCY = int(os.getenv('TOPIC_EXTRACTION_CONCURRENCY', '10'))
MAX_TOPICS_PER_DOC = int(os.getenv('MAX_TOPICS_PER_DOC', '32'))

try:
    redis_health_client = redis.Redis.from_url(
        REDIS_URL, socket_connect_timeout=2, socket_timeout=2, decode_responses=True
    )
    redis_health_client.ping()
    print(f"Health Check using Redis at: {REDIS_URL}")
except Exception as e:
    print(f"WARNING: Failed to initialize Redis client for health check: {e}")
    redis_health_client = None

router = APIRouter()
router.include_router(progress_router, prefix="/progress", tags=["progress"])
rag_assistant = None
graph_rag_assistant = None

async def extract_topics_with_llm(text: str, llm: ChatOpenAI, max_topics: int = MAX_TOPICS_PER_DOC) -> List[str]:
    print(f"Extracting topics from full text ({len(text)} chars) by chunking...")
    chunk_size = TOPIC_EXTRACTION_CHUNK_SIZE
    concurrency_limit = TOPIC_EXTRACTION_CONCURRENCY
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an expert topic extractor. Identify up to 5 main topics discussed *only* in the following text snippet. List each topic concisely on a new line, using title case (e.g., 'Patient Documentation'). **CRITICAL: DO NOT add any preamble, numbering (like 1., 2.), or bullet points (like -, *).** Focus only on the provided snippet."),
        ("user", "{input_text}")
    ])
    parser = StrOutputParser()
    chain = prompt | llm | parser
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"Divided text into {len(text_chunks)} chunks for topic extraction (chunk size: {chunk_size} chars).")
    tasks = []
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def process_chunk(text_chunk, chunk_index):
        async with semaphore:
            try:
                if not text_chunk.strip():
                    print(f"--- Skipping empty chunk {chunk_index + 1}/{len(text_chunks)}")
                    return []
                result = await chain.ainvoke({"input_text": text_chunk})
                topics_from_chunk = [topic.strip() for topic in result.split('\n') if topic.strip()]
                print(f"--- Topics found in chunk {chunk_index + 1}/{len(text_chunks)}: {topics_from_chunk}")
                return topics_from_chunk
            except Exception as e:
                error_str = str(e)
                if "context length" in error_str or "token" in error_str:
                     print(f"ERROR during topic extraction for chunk {chunk_index + 1} (Context Length Exceeded?): {e}")
                else:
                     print(f"ERROR during topic extraction for chunk {chunk_index + 1}: {e}")
                return []

    for i, chunk in enumerate(text_chunks):
        tasks.append(process_chunk(chunk, i))
    results = await asyncio.gather(*tasks)
    topic_counts = {}
    for topic_list in results:
        for topic in topic_list:
            normalized_topic = topic.lower().strip()
            if normalized_topic and len(normalized_topic) > 1 and not normalized_topic.startswith(('-', '*', '.')):
                topic_counts[normalized_topic] = topic_counts.get(normalized_topic, 0) + 1
    sorted_topics = sorted(topic_counts.items(), key=lambda item: item[1], reverse=True)
    final_topics = [topic for topic, count in sorted_topics[:max_topics]]
    final_topics_title_case = [topic.title() for topic in final_topics]
    print(f"Aggregated and deduplicated topics (Top {len(final_topics_title_case)} requested): {final_topics_title_case}")
    return final_topics_title_case

@router.post("/batch_chat", response_model=BatchChatResponse)
async def batch_chat_with_assistant(request: Request, batch_request: BatchChatRequest):
    """
    Processes a batch of questions using either RAG or GraphRAG concurrently,
    optionally scoped to a document_title.
    """
    rag_assistant_instance = request.app.state.rag_assistant_instance
    graph_rag_assistant_instance = request.app.state.graph_rag_assistant_instance
    doc_title_for_query = batch_request.document_title

    if rag_assistant_instance is None or graph_rag_assistant_instance is None:
        raise HTTPException(status_code=503, detail="Assistants not initialized")

    if not batch_request.questions:
        return BatchChatResponse(results=[])

    print(f"Received batch chat request with {len(batch_request.questions)} questions. Document context: '{doc_title_for_query if doc_title_for_query else 'All Documents'}'")

    try:
        llm_instance = ChatOpenAI(
            openai_api_key=LM_STUDIO_API_KEY,
            openai_api_base=LM_STUDIO_API_BASE,
            temperature=DEFAULT_LLM_TEMP,
            max_tokens=DEFAULT_LLM_MAX_TOKENS
        )
        print(f"Instantiated single LLM for batch chat with default params.")
        rag_assistant_instance.update_llm(llm_instance)
        graph_rag_assistant_instance.update_llm(llm_instance)
        print("Updated RAG and GraphRAG assistants with the batch LLM instance.")
    except Exception as e:
        print(f"ERROR: Failed to initialize or update LLM for batch chat: {e}")
        traceback.print_exc()
        error_results = [
            BatchChatResponseItem(
                question=item.question,
                use_graph=item.use_graph,
                error=f"Failed to initialize LLM for batch processing: {type(e).__name__}"
            ) for item in batch_request.questions
        ]
        return BatchChatResponse(results=error_results)

    async def process_single_question(item: BatchChatRequestItem) -> BatchChatResponseItem:
        question = item.question
        use_graph = item.use_graph
        response_item = BatchChatResponseItem(question=question, use_graph=use_graph)

        try:
            if use_graph:
                print(f"  Processing (GraphRAG for '{doc_title_for_query}'): {question[:50]}...")
                result = graph_rag_assistant_instance.query(
                    question,
                    current_doc_title=doc_title_for_query
                )
                if "error" in result:
                     response_item.error = result.get("error", "GraphRAG query failed.")
                     print(f"  > Error (GraphRAG): {response_item.error} - Details: {result.get('details')}")
                else:
                     response_item.answer = result.get("result", "Graph RAG failed to find an answer.")
                     response_item.sources = []
                     print(f"  > Answer (GraphRAG): {response_item.answer[:50]}...")
            else:
                print(f"  Processing (RAG): {question[:50]}...")
                result = rag_assistant_instance.query(question)
                response_item.answer = result.get("answer", "RAG failed to find an answer.")
                raw_sources = result.get("sources", [])
                response_item.sources = [SourceDocument(**src) for src in raw_sources if isinstance(src, dict)]
                print(f"  > Answer (RAG): {response_item.answer[:50]}... ({len(response_item.sources)} sources)")

        except Exception as e:
            error_msg = f"Failed processing question '{question[:50]}...': {type(e).__name__}"
            print(f"  > ERROR: {error_msg} - {e}")
            traceback.print_exc()
            response_item.error = error_msg
        return response_item

    tasks = [process_single_question(item) for item in batch_request.questions]
    results = await asyncio.gather(*tasks)

    print(f"Finished processing batch of {len(batch_request.questions)} questions for document '{doc_title_for_query}'.")
    return BatchChatResponse(results=results)

@router.get("/health")
async def health_check():
    neo4j_status = "down"
    try:
        neo4j_client_health = Neo4jClient(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        neo4j_client_health.run_query("RETURN 1 as test")
        neo4j_client_health.close()
        neo4j_status = "up"
    except Exception as neo4j_e:
        print(f"Health check Neo4j error: {neo4j_e}")
        neo4j_status = f"down: {type(neo4j_e).__name__}"

    redis_status = "down: client init failed"
    if redis_health_client:
        try:
            redis_health_client.ping()
            redis_status = "up"
        except redis.exceptions.ConnectionError as e: redis_status = f"down: ConnectionError: {str(e)}"
        except redis.exceptions.TimeoutError as e: redis_status = f"down: TimeoutError: {str(e)}"
        except Exception as e: redis_status = f"down: {type(e).__name__}: {str(e)}"

    backend_status = "up"
    overall_status = "healthy" if neo4j_status == "up" and redis_status == "up" else "unhealthy"

    if overall_status == "healthy":
         return {"status": overall_status, "services": {"neo4j": neo4j_status, "redis": redis_status, "backend": backend_status}}
    else:
         raise HTTPException(status_code=503, detail={"status": overall_status, "services": {"neo4j": neo4j_status, "redis": redis_status, "backend": backend_status}})

@router.post("/upload", status_code=200)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    skip_graph_build: bool = Form(False)
):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        job_id = str(uuid.uuid4())
        safe_filename = os.path.basename(file.filename or f"upload_{job_id}.tmp")
        if file.content_type == 'application/pdf' and not safe_filename.lower().endswith('.pdf'):
            safe_filename += '.pdf'
        elif file.content_type == 'text/plain' and not safe_filename.lower().endswith('.txt'):
             safe_filename += '.txt'

        file_path = os.path.join(UPLOAD_DIR, safe_filename)

        try:
            with open(file_path, "wb") as buffer:
                 shutil.copyfileobj(file.file, buffer)
        except Exception as save_e:
             print(f"ERROR saving uploaded file {safe_filename}: {save_e}")
             raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {save_e}")
        finally:
             if hasattr(file, 'file') and hasattr(file.file, 'close') and callable(file.file.close):
                 file.file.close()

        print(f"DEBUG Upload Endpoint: Received skip_graph_build = {skip_graph_build}")
        background_tasks.add_task(process_document, file_path, safe_filename, job_id, skip_graph_build)

        print(f"Upload successful for {safe_filename}, starting background job {job_id}")
        return {
            "message": "File uploaded and processing started",
            "filename": safe_filename,
            "job_id": job_id
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"ERROR during file upload: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during upload: {type(e).__name__}")

async def process_document(file_path, filename, job_id, skip_graph_build: bool):
    print(f"DEBUG process_document: Started job {job_id} with skip_graph_build = {skip_graph_build}")
    neo4j_client = None
    doc_id = filename.rsplit('.', 1)[0]
    text = ""
    total_items = 0
    is_pdf = filename.lower().endswith('.pdf')

    if skip_graph_build:
        print(f"Job {job_id}: Graph build skipped. Adjusting progress phases.")
        EXTRACTION_START_PERC = 0
        EXTRACTION_END_PERC = 50
        CHUNKING_START_PERC = EXTRACTION_END_PERC
        CHUNKING_END_PERC = 65
        BATCH_PROCESSING_START_PERC = CHUNKING_END_PERC
        BATCH_PROCESSING_END_PERC = 100
    else:
        EXTRACTION_START_PERC = 0
        EXTRACTION_END_PERC = 50
        TOPIC_EXTRACTION_START_PERC = EXTRACTION_END_PERC
        TOPIC_EXTRACTION_END_PERC = 55
        CHUNKING_START_PERC = TOPIC_EXTRACTION_END_PERC
        CHUNKING_END_PERC = 65
        BATCH_PROCESSING_START_PERC = CHUNKING_END_PERC
        BATCH_PROCESSING_END_PERC = 100

    embeddings_client = None
    try:
        print(f"Job {job_id}: Initializing embeddings model {EMBEDDING_MODEL} for ingestion...")
        embeddings_client = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': EMBEDDING_DEVICE}
        )
        print(f"Job {job_id}: Embeddings model initialized.")
    except Exception as emb_e:
        error_msg = f"Failed to initialize embedding model {EMBEDDING_MODEL}: {emb_e}"
        print(f"ERROR: {error_msg}")
        complete_job_sync(job_id, error_msg, final_status="failed")
        return

    topic_llm = None
    if not skip_graph_build:
        try:
            topic_llm = ChatOpenAI(
                openai_api_key=LM_STUDIO_API_KEY,
                openai_api_base=LM_STUDIO_API_BASE,
                temperature=DEFAULT_LLM_TEMP,
                max_tokens=TOPIC_LLM_MAX_TOKENS
            )
            print(f"Job {job_id}: Topic extraction LLM initialized.")
        except Exception as llm_e:
            error_msg = f"Failed to initialize topic extraction LLM: {llm_e}"
            print(f"ERROR: {error_msg}")
            topic_llm = None
    else:
        print(f"Job {job_id}: Skipping Topic LLM initialization as graph build is skipped.")

    if nlp is None:
        print(f"CRITICAL ERROR Job {job_id}: spaCy model '{NLP_MODEL_NAME}' failed to load. NER step cannot proceed.")
        complete_job_sync(job_id, f"spaCy model {NLP_MODEL_NAME} failed to load", final_status="failed")
        return

    try:
        print(f"Background task started for job {job_id}, file {filename}")
        await update_job_status(job_id, status="starting", message="Processing started...", percent_complete=EXTRACTION_START_PERC)

        if is_pdf:
            pdf_loader = PDFLoader(pdf_directory=None)
            await update_job_status(job_id, status="extracting", message="Extracting text from PDF...", percent_complete=EXTRACTION_START_PERC)
            text = await pdf_loader.extract_text_from_pdf(file_path, job_id)
            if not text:
                 print(f"Job {job_id}: PDF text extraction returned empty or failed.")
                 complete_job_sync(job_id, "PDF text extraction failed or returned empty.", final_status="failed")
                 return
            await update_job_status(job_id, status="extracted", message="PDF text extraction complete.", percent_complete=EXTRACTION_END_PERC)
            print(f"Job {job_id}: PDF extraction complete. Text length: {len(text)}")
        else:
            await update_job_status(job_id, status="extracting", message="Reading text file...", percent_complete=EXTRACTION_START_PERC)
            try:
                with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'rb') as f_raw: raw_data = f_raw.read()
                    detected_encoding = chardet.detect(raw_data)['encoding']
                    if detected_encoding: text = raw_data.decode(detected_encoding)
                    else: raise ValueError("Could not detect encoding for text file.")
                except Exception as enc_e:
                    error_msg = f"Error reading text file {filename} after encoding detection: {enc_e}"
                    print(f"ERROR: {error_msg}")
                    complete_job_sync(job_id, error_msg, final_status="failed")
                    return
            except Exception as txt_e:
                error_msg = f"Error reading text file {filename}: {txt_e}"
                print(f"ERROR: {error_msg}")
                complete_job_sync(job_id, error_msg, final_status="failed")
                return
            await update_job_status(job_id, status="extracted", message="Text file reading complete.", percent_complete=EXTRACTION_END_PERC)
            print(f"Job {job_id}: Text file reading complete. Text length: {len(text)}")

        extracted_topics = []
        if not skip_graph_build:
            await update_job_status(job_id, status="topic_extraction", message="Extracting topics...", percent_complete=TOPIC_EXTRACTION_START_PERC)
            if topic_llm and text:
                extracted_topics = await extract_topics_with_llm(text, topic_llm, max_topics=MAX_TOPICS_PER_DOC)
                print(f"Job {job_id}: extract_topics_with_llm returned {len(extracted_topics)} topics (limit was {MAX_TOPICS_PER_DOC}).")
            else:
                print(f"Job {job_id}: Skipping topic extraction (LLM not available or no text).")
            await update_job_status(job_id, status="topic_extracted", message=f"Topic extraction complete ({len(extracted_topics)} topics found).", percent_complete=TOPIC_EXTRACTION_END_PERC)
            await asyncio.sleep(0)
        else:
            print(f"Job {job_id}: Skipping Topic Extraction phase.")
            await update_job_status(job_id, status="chunking", message="Skipped topic extraction. Starting chunking...", percent_complete=CHUNKING_START_PERC)

        await update_job_status(job_id, status="chunking", message="Chunking document text (recursive)...", percent_complete=CHUNKING_START_PERC)
        try:
            text_processor = TextProcessor()
        except ValueError as val_err:
            error_msg = f"Failed to initialize TextProcessor: {val_err}"
            print(f"ERROR Job {job_id}: {error_msg}")
            complete_job_sync(job_id, error_msg, final_status="failed")
            return
        chunks = text_processor.process_text(text, verbose=True)
        total_items = len(chunks)
        print(f"Job {job_id}: Processed text into {total_items} recursive chunks.")
        if not chunks:
            await progress_complete_job(job_id, message="No text chunks generated after recursive processing.", final_status="completed_empty")
            print(f"Job {job_id}: No recursive chunks generated. Finishing.")
            return
        await update_job_status(job_id, status="chunked", message="Recursive text chunking complete.", percent_complete=CHUNKING_END_PERC)

        neo4j_client = Neo4jClient(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        neo4j_client.ensure_vector_index()

        neo4j_client.run_query(
             "MERGE (d:Document {id: $doc_id}) ON CREATE SET d.title = $title",
             {'doc_id': doc_id, 'title': filename}
        )
        try:
            shared_redis_client.set("current_doc_title", filename)
            print(f"Job {job_id}: Set 'current_doc_title' in Redis to '{filename}'")
        except Exception as redis_err:
            print(f"WARNING Job {job_id}: Failed to set 'current_doc_title' in Redis: {redis_err}")

        if extracted_topics:
            print(f"Job {job_id}: Storing {len(extracted_topics)} topics in Neo4j...")
            neo4j_client.add_document_topics(doc_id, extracted_topics)
            print(f"Job {job_id}: Topic storage call complete.")
        else:
            print(f"Job {job_id}: No topics to store in Neo4j (either none found or skipped).")
        await asyncio.sleep(0)

        await update_job_status(job_id, status="embedding_ner_neo4j", message="Starting embedding, NER, and storage...", percent_complete=BATCH_PROCESSING_START_PERC)
        chunks_added_count = 0
        batch_size = INGEST_BATCH_SIZE
        num_batches = (total_items + batch_size - 1) // batch_size
        batch_phase_range = BATCH_PROCESSING_END_PERC - BATCH_PROCESSING_START_PERC

        for i in range(0, total_items, batch_size):
            batch_texts = chunks[i : i + batch_size]
            if not batch_texts: continue

            current_batch_num = i // batch_size + 1
            start_batch_progress_fraction = (i / total_items) if total_items > 0 else 0
            start_batch_overall_percent = BATCH_PROCESSING_START_PERC + int(start_batch_progress_fraction * batch_phase_range)
            start_batch_overall_percent = min(start_batch_overall_percent, BATCH_PROCESSING_END_PERC)

            await update_job_status(
                job_id, status="embedding_ner_neo4j",
                message=f"Processing batch {current_batch_num}/{num_batches} (Embedding, NER)...",
                percent_complete=start_batch_overall_percent
            )

            try:
                batch_embeddings = embeddings_client.embed_documents(batch_texts)
            except Exception as emb_e:
                error_msg = f"Error generating embeddings for batch {current_batch_num}: {emb_e}"
                print(f"ERROR: {error_msg}")
                await update_job_status(job_id, status="error_embedding", message=error_msg)
                raise
            if len(batch_embeddings) != len(batch_texts):
                print(f"WARNING: Job {job_id}: Embedding count mismatch for batch {current_batch_num}. Skipping batch.")
                await update_job_status(job_id, status="warning_skip", message=f"Skipped batch {current_batch_num} due to embedding mismatch.")
                continue

            items_to_store = []
            entities_to_store_by_chunk = {}

            for j, text_chunk in enumerate(batch_texts):
                chunk_seq_id = i + j
                chunk_id = f"{doc_id}_{chunk_seq_id}"
                items_to_store.append({
                    "text": text_chunk, "embedding": batch_embeddings[j],
                    "doc_id": doc_id, "chunk_seq_id": chunk_seq_id
                })

                chunk_entities = []
                try:
                    spacy_doc = nlp(text_chunk)
                    for ent in spacy_doc.ents:
                        entity_type = SPACY_LABEL_TO_ENTITY_TYPE.get(ent.label_, "UnknownEntity")
                        if len(ent.text) > 1 and not ent.text.isdigit():
                            label_description = spacy.explain(ent.label_) or ""
                            chunk_entities.append({
                                "name": ent.text.strip(), "type": entity_type,
                                "description": label_description, "synonyms": []
                            })
                except Exception as ner_e:
                     print(f"WARNING Job {job_id}: NER failed for chunk {chunk_id}: {ner_e}")

                if chunk_entities and not skip_graph_build:
                    entities_to_store_by_chunk[chunk_id] = chunk_entities

            if items_to_store:
                try:
                    neo4j_client.add_document_chunks(items_to_store)
                    chunks_added_count += len(items_to_store)
                except Exception as neo_e:
                    error_msg = f"Error storing chunk batch {current_batch_num} in Neo4j: {neo_e}"
                    print(f"ERROR: {error_msg}")
                    await update_job_status(job_id, status="error_neo4j", message=error_msg)
                    raise

            if entities_to_store_by_chunk:
                 print(f"Job {job_id}: Storing entities/mentions for batch {current_batch_num}...")
                 try:
                    for chunk_id_key, entities_list in entities_to_store_by_chunk.items():
                         neo4j_client.add_entities_and_mentions(chunk_id_key, entities_list)
                 except Exception as ent_neo_e:
                      print(f"ERROR Job {job_id}: Storing entities/mentions failed for batch {current_batch_num}: {ent_neo_e}")
                      await update_job_status(job_id, status="error_neo4j_entities", message=f"Error storing entities batch {current_batch_num}")
            elif not skip_graph_build:
                 print(f"Job {job_id}: No entities found in batch {current_batch_num} to store relationships for.")

            end_batch_progress_fraction = (chunks_added_count / total_items) if total_items > 0 else 0
            end_batch_overall_percent = BATCH_PROCESSING_START_PERC + int(end_batch_progress_fraction * batch_phase_range)
            end_batch_overall_percent = min(end_batch_overall_percent, BATCH_PROCESSING_END_PERC)

            await update_job_status(
                job_id, status="embedding_ner_neo4j",
                message=f"Stored batch {current_batch_num}/{num_batches}, total stored: {chunks_added_count}",
                percent_complete=end_batch_overall_percent
            )
            await asyncio.sleep(0.01)

        await progress_complete_job(job_id, message="Processing complete - ready for querying")
        print(f"Background task finished successfully for job: {job_id}")

    except Exception as e:
        error_msg = f"Error processing document {filename} for job {job_id}: {e}\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}")
        try:
            complete_job_sync(job_id, f"Failed: {e}", final_status="failed")
        except Exception as status_e:
            print(f"ERROR Job {job_id}: Could not update final failed status: {status_e}")
    finally:
        if neo4j_client:
            neo4j_client.close()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(request: Request, chat_request: ChatRequest):
    rag_assistant_instance = request.app.state.rag_assistant_instance
    graph_rag_assistant_instance = request.app.state.graph_rag_assistant_instance

    if rag_assistant_instance is None or graph_rag_assistant_instance is None:
         raise HTTPException(status_code=503, detail="Assistants not initialized")

    if chat_request.use_graph:
         print("DEBUG Chat: Request received with use_graph=True.")

    try:
        llm_params = {"temperature": DEFAULT_LLM_TEMP, "max_tokens": DEFAULT_LLM_MAX_TOKENS}
        if chat_request.temperature is not None: llm_params["temperature"] = chat_request.temperature
        if chat_request.max_tokens is not None: llm_params["max_tokens"] = chat_request.max_tokens

        llm_instance = ChatOpenAI(
            openai_api_key=LM_STUDIO_API_KEY,
            openai_api_base=LM_STUDIO_API_BASE,
            temperature=llm_params["temperature"],
            max_tokens=llm_params["max_tokens"]
        )
        print(f"Using LLM for chat with params: {llm_params}")

        rag_assistant_instance.update_llm(llm_instance)
        graph_rag_assistant_instance.update_llm(llm_instance)

        if chat_request.use_graph:
            print("Using Graph RAG Assistant")
            current_doc_title = None
            try:
                current_doc_title = shared_redis_client.get("current_doc_title")
                if current_doc_title: print(f"Retrieved 'current_doc_title' from Redis: '{current_doc_title}'")
                else: print("No 'current_doc_title' found in Redis.")
            except Exception as redis_err:
                print(f"WARNING: Failed to retrieve 'current_doc_title' from Redis: {redis_err}")

            result = graph_rag_assistant_instance.query(
                chat_request.question,
                current_doc_title=current_doc_title
            )
            answer = result.get("result", "Graph RAG failed to find an answer.")
            sources = []
        else:
            print("Using Standard RAG Assistant")
            result = rag_assistant_instance.query(chat_request.question)
            answer = result.get("answer", "RAG failed to find an answer.")
            sources = result.get("sources", [])

        response_payload = ChatResponse(answer=answer, sources=sources)
        return response_payload

    except Exception as e:
        print(f"ERROR during chat: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during chat: {type(e).__name__}")
