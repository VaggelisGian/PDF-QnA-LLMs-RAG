from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import uuid

load_dotenv()

class Neo4jClient:
    def __init__(self, uri=None, user=None, password=None):
        uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        user = user or os.getenv('NEO4J_USERNAME', 'neo4j')
        password = password or os.getenv('NEO4J_PASSWORD')

        if not password:
             raise ValueError("Neo4j password not provided via argument or NEO4J_PASSWORD environment variable.")

        try:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            print(f"Neo4j driver initialized for URI: {uri}")
            self._driver.verify_connectivity()
            print("Neo4j connection verified.")
        except Exception as e:
            print(f"ERROR: Failed to create Neo4j driver or verify connection: {e}")
            self._driver = None
            raise

    def close(self):
        if self._driver is not None:
            try:
                self._driver.close()
                print("Neo4j driver closed.")
            except Exception as e:
                 print(f"ERROR closing Neo4j driver: {e}")
            finally:
                 self._driver = None

    def run_query(self, query, parameters=None):
        if self._driver is None:
            print("ERROR: Neo4j driver not initialized or closed. Cannot run query.")
            raise ConnectionError("Neo4j driver is not available.")

        try:
            with self._driver.session() as session:
                result = session.run(query, parameters)
                summary = result.consume()
                return summary
        except Exception as e:
            print(f"ERROR running Neo4j query: {e}")
            print(f"Query: {query}")
            print(f"Parameters: {parameters}")
            raise

    def add_document_chunks(self, items_to_store: list):
        if not items_to_store:
            print("No items provided to add_document_chunks.")
            return

        query = """
        UNWIND $items as item
        MERGE (d:Document {id: item.doc_id})
        MERGE (c:Chunk {id: item.doc_id + '_' + toString(item.chunk_seq_id)})
        ON CREATE SET
            c.text = item.text,
            c.embedding = item.embedding,
            c.doc_id = item.doc_id,
            c.chunk_seq_id = item.chunk_seq_id,
            c.created_at = timestamp()
        ON MATCH SET
            c.text = item.text,
            c.embedding = item.embedding,
            c.updated_at = timestamp()
        MERGE (d)-[:HAS_CHUNK]->(c)

        WITH c, item
        WHERE item.chunk_seq_id > 0
        MATCH (prev_c:Chunk {id: item.doc_id + '_' + toString(item.chunk_seq_id - 1)})
        MERGE (prev_c)-[:NEXT_CHUNK]->(c)

        RETURN count(c) as chunks_processed
        """
        try:
            summary = self.run_query(query, parameters={'items': items_to_store})
            if summary and hasattr(summary, 'counters'):
                 nodes_created = summary.counters.nodes_created
                 rels_created = summary.counters.relationships_created
                 nodes_set = summary.counters.properties_set
                 print(f"Neo4j batch stored (Chunks): {nodes_created} nodes created, {rels_created} relationships created, {nodes_set} properties set.")
            else:
                 print("Neo4j: add_document_chunks query executed, but summary or counters were not available.")

        except Exception as e:
            print(f"ERROR during batch storage in add_document_chunks: {e}")
            raise

    def add_entities_and_mentions(self, chunk_id: str, entities: List[Dict[str, Any]]):
        if not chunk_id or not entities:
            return

        query = """
        MATCH (c:Chunk {id: $chunk_id})
        WITH c
        UNWIND $entities as entity_data
        MERGE (e:Entity {name: entity_data.name, type: entity_data.type})
        ON CREATE SET
             e.description = entity_data.description,
             e.synonyms = entity_data.synonyms,
             e.id = randomUUID()
        MERGE (c)-[:MENTIONS]->(e)
        RETURN count(e) as entities_linked
        """
        try:
            summary = self.run_query(query, parameters={'chunk_id': chunk_id, 'entities': entities})

        except Exception as e:
            print(f"ERROR storing entities/mentions for chunk_id {chunk_id}: {e}")


    def ensure_vector_index(self, index_name="pdf_chunk_embeddings", node_label="Chunk", property_name="embedding", dimensions=384):
         index_query = f"""
         CREATE VECTOR INDEX {index_name} IF NOT EXISTS
         FOR (c:{node_label}) ON (c.{property_name})
         OPTIONS {{indexConfig: {{
             `vector.dimensions`: {dimensions},
             `vector.similarity_function`: 'cosine'
         }}}}
         """
         try:
             print(f"Ensuring vector index '{index_name}' on :{node_label}({property_name}) with dimensions {dimensions} exists...")
             self.run_query(index_query)
             print(f"Vector index '{index_name}' check/creation command executed.")
         except Exception as e:
             print(f"ERROR potentially occurred while ensuring vector index '{index_name}': {e}")


    def add_document_topics(self, doc_id: str, topics: List[str]):
        if not topics:
            print(f"No topics provided for doc_id {doc_id}.")
            return

        query = """
        MATCH (d:Document {id: $doc_id})
        WITH d
        UNWIND $topics as topic_name
        MERGE (t:Topic {name: topic_name})
        MERGE (d)-[:HAS_TOPIC]->(t)
        RETURN count(t) as topics_linked
        """
        try:
            summary = self.run_query(query, parameters={'doc_id': doc_id, 'topics': topics})
            if summary and hasattr(summary, 'counters'):
                 rels_created = summary.counters.relationships_created
                 nodes_created = summary.counters.nodes_created
                 print(f"Neo4j topics stored for {doc_id}: {nodes_created} Topic nodes created, {rels_created} HAS_TOPIC relationships created.")
            else:
                 print(f"Neo4j: add_document_topics query executed for {doc_id}, but summary/counters unavailable.")
        except Exception as e:
            print(f"ERROR storing topics for doc_id {doc_id}: {e}")
