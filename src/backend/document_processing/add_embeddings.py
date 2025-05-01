from src.backend.database.neo4j_client import Neo4jClient
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm
import numpy as np
import requests
import json
from datetime import datetime
from typing import Optional
import re

load_dotenv()

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def estimate_tokens(text):
    return len(text) // 4 + 1

def create_token_aware_batches(sentences, max_tokens=7500):
    batches = []
    current_batch = []
    current_token_count = 0

    for sentence_id, sentence_text in sentences:
        tokens = estimate_tokens(sentence_text)

        if current_token_count + tokens > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = [(sentence_id, sentence_text)]
            current_token_count = tokens
        else:
            current_batch.append((sentence_id, sentence_text))
            current_token_count += tokens

    if current_batch:
        batches.append(current_batch)

    return batches

def main():
    api_key = os.getenv("AZURE_AI_API_KEY")
    if not api_key:
        print("ERROR: AZURE_AI_API_KEY environment variable not set.")
        return

    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')

    if not neo4j_password:
        print("ERROR: NEO4J_PASSWORD environment variable not set.")
        return

    neo4j_client = None
    try:
        neo4j_client = Neo4jClient(neo4j_uri, neo4j_user, neo4j_password)

        print("Fetching sentences without embeddings...")
        results = neo4j_client.run_query("""
            MATCH (s:Sentence)
            WHERE s.embedding IS NULL
            RETURN s.id AS id, s.content AS content
            LIMIT 500
        """)

        sentences = [(record["id"], record["content"]) for record in results]
        total_sentences = len(sentences)
        print(f"Found {total_sentences} sentences without embeddings")

        if total_sentences == 0:
            print("No sentences need embeddings. Exiting.")
            return

        batches = create_token_aware_batches(sentences, max_tokens=7500)
        print(f"Created {len(batches)} batches respecting the 8000 token input limit")

        successful = 0
        request_times = []

        for i, batch in enumerate(batches):
            batch_ids = [s[0] for s in batch]
            batch_texts = [s[1] for s in batch]

            current_time = time.time()
            request_times = [t for t in request_times if current_time - t < 60]

            while len(request_times) >= 10:
                sleep_time = max(0, 61 - (current_time - request_times[0]))
                if sleep_time > 0:
                    print(f"Rate limit approaching, waiting {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)

                current_time = time.time()
                request_times = [t for t in request_times if current_time - t < 60]

            try:
                if i > 0:
                    time.sleep(5)

                request_times.append(time.time())

                print(f"Processing batch {i+1}/{len(batches)} with {len(batch_texts)} sentences...")

                response = requests.post(
                    "https://models.inference.ai.azure.com/embeddings",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "text-embedding-3-small",
                        "input": batch_texts
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    embeddings = [item["embedding"] for item in result["data"]]

                    for j, embedding in enumerate(embeddings):
                        neo4j_client.run_query(
                            "MATCH (s:Sentence {id: $id}) SET s.embedding = $embedding",
                            {"id": batch_ids[j], "embedding": embedding}
                        )

                    successful += len(embeddings)
                    print(f"Progress: {successful}/{total_sentences} sentences processed")

                elif response.status_code == 429:
                    error_data = response.json()
                    wait_time = 60

                    error_msg = error_data.get('error', {}).get('message', '')
                    print(f"Rate limit error: {error_msg}")

                    wait_match = re.search(r'Please wait (\d+) seconds', error_msg)
                    if wait_match:
                        parsed_wait = int(wait_match.group(1))
                        wait_time = min(parsed_wait, 300)

                    print(f"Rate limit exceeded, waiting {wait_time} seconds...")
                    time.sleep(wait_time)

                    print(f"Skipping batch {i+1} due to rate limit, will retry on next run.")
                    continue

                else:
                    print(f"Error: {response.status_code} - {response.text}")
                    time.sleep(30)

            except Exception as e:
                print(f"Error processing batch: {e}")
                time.sleep(30)

        print(f"Finished adding embeddings to {successful}/{total_sentences} sentences")

    except Exception as e:
         print(f"An error occurred: {e}")
    finally:
        if neo4j_client:
            neo4j_client.close()

if __name__ == "__main__":
    main()