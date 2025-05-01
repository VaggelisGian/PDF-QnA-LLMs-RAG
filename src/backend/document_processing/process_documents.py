from src.backend.document_processing.text_processor import TextProcessor
from src.backend.database.neo4j_client import Neo4jClient
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import time
import argparse
from tqdm import tqdm
from PyPDF2 import PdfReader
import chardet

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description='Process PDF/TXT documents, generate embeddings, and build a knowledge graph')
    parser.add_argument('--max-pages', type=int, default=None, help='Maximum pages to process per PDF (default: all)')
    parser.add_argument('--batch-size', type=int, default=int(os.getenv("INGEST_BATCH_SIZE", "50")), help='Batch size for Neo4j operations')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    lm_studio_api_base = os.getenv("LM_STUDIO_API_BASE", "http://localhost:1234/v1")
    lm_studio_api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
    chunk_size = int(os.getenv("TEXT_CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("TEXT_CHUNK_OVERLAP", "200"))
    embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "384")) # Default for all-MiniLM-L6-v2 is 384
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    data_dir = os.getenv("UPLOAD_DIR", "data/pdf_files")

    if not neo4j_password:
        print("ERROR: NEO4J_PASSWORD environment variable not set.")
        return

    text_processor = TextProcessor()
    neo4j_client = Neo4jClient(neo4j_uri, neo4j_user, neo4j_password)
    embeddings_client = OpenAIEmbeddings(
        openai_api_key=lm_studio_api_key,
        openai_api_base=lm_studio_api_base,
        model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    print("Components initialized.")

    files_to_process = [f for f in os.listdir(data_dir) if f.lower().endswith(('.pdf', '.txt'))]

    if not files_to_process:
        print(f'No PDF or TXT files found in {data_dir}')
        neo4j_client.close()
        return

    print(f'Processing {len(files_to_process)} files from {data_dir}')
    start_time = time.time()
    total_chunks_added = 0

    for filename in files_to_process:
        file_path = os.path.join(data_dir, filename)
        file_start = time.time()
        print(f'\nProcessing {filename}...')

        text = ""
        try:
            if filename.lower().endswith(".pdf"):
                with open(file_path, 'rb') as f:
                    reader = PdfReader(f)
                    num_pages = len(reader.pages)
                    pages_to_read = min(num_pages, args.max_pages) if args.max_pages else num_pages
                    print(f'  Reading {pages_to_read} pages from PDF...')
                    for i in range(pages_to_read):
                        page = reader.pages[i]
                        page_text = page.extract_text()
                        if page_text: text += page_text + "\n"
            elif filename.lower().endswith(".txt"):
                print(f'  Reading TXT file...')
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    detected_encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
                with open(file_path, 'r', encoding=detected_encoding) as f:
                    text = f.read()
            print(f'  Read content (Length: {len(text)} characters)')
        except Exception as read_e:
            print(f"  Error reading file {filename}: {read_e}")
            continue

        if not text.strip():
            print("  No text content found. Skipping.")
            continue

        doc_id = filename.rsplit('.', 1)[0]
        neo4j_client.run_query(
            "MERGE (d:Document {id: $doc_id}) ON CREATE SET d.title = $title",
            {'doc_id': doc_id, 'title': filename}
        )

        print("  Chunking text...")
        chunks = text_processor.process_text(text, verbose=args.verbose)
        print(f'  Generated {len(chunks)} chunks')

        if not chunks:
            print("  No chunks generated. Skipping Neo4j/Embedding steps.")
            continue

        batch_size = args.batch_size
        num_batches = (len(chunks) + batch_size - 1) // batch_size
        print(f'  Creating graph nodes, relationships, and embeddings in {num_batches} batches...')
        chunks_added_this_file = 0

        for i in tqdm(range(0, len(chunks), batch_size), desc="  Processing batches"):
            batch_texts = chunks[i : i + batch_size]
            if not batch_texts: continue

            try:
                batch_embeddings = embeddings_client.embed_documents(batch_texts)
            except Exception as e:
                print(f"\nError generating embeddings for batch starting at index {i}: {e}")
                continue

            if len(batch_embeddings) != len(batch_texts):
                 print(f"\nWarning: Embedding count mismatch for batch {i}. Expected {len(batch_texts)}, got {len(batch_embeddings)}. Skipping batch.")
                 continue

            batch_params = []
            for j, chunk_text in enumerate(batch_texts):
                chunk_index = i + j
                chunk_id = f'{doc_id}_c{chunk_index}'
                batch_params.append({
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'content': chunk_text,
                    'embedding': batch_embeddings[j]
                })

            if batch_params:
                try:
                    neo4j_client.run_query(
                        """
                        UNWIND $batch as row
                        MATCH (d:Document {id: row.doc_id})
                        MERGE (c:Chunk {id: row.chunk_id})
                        ON CREATE SET c.content = row.content, c.embedding = row.embedding
                        ON MATCH SET c.embedding = row.embedding
                        MERGE (d)-[:CONTAINS]->(c)
                        """,
                        {'batch': batch_params}
                    )
                    chunks_added_this_file += len(batch_params)
                except Exception as e:
                    print(f"\nError executing Neo4j batch query for batch {i}: {e}")
                    continue

        total_chunks_added += chunks_added_this_file
        print(f'  Added/Updated {chunks_added_this_file} chunks from {filename} to the graph')
        print(f'  Completed in {time.time() - file_start:.2f} seconds')

    print("\nCreating Neo4j vector index 'chunk_embeddings' (if it doesn't exist)...")
    try:
        index_name = "chunk_embeddings"
        neo4j_client.run_query(
            f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {embedding_dimension},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
        )
        print(f"Vector index '{index_name}' created or already exists.")
    except Exception as e:
        print(f"Error creating Neo4j vector index: {e}")
        print("Please ensure your Neo4j version supports vector indexes (5.11+ recommended).")


    neo4j_client.close()
    total_time = time.time() - start_time
    print(f'\nProcessing finished in {total_time:.2f} seconds total!')
    print(f'Total chunks added/updated: {total_chunks_added}')

if __name__ == '__main__':
    main()