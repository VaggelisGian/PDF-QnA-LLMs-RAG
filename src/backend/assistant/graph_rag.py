import os
import traceback
import logging
import re 
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import OutputParserException 
from neo4j.exceptions import ( 
    Neo4jError,
    ServiceUnavailable,
    AuthError as Neo4jAuthError,
    CypherSyntaxError,
    ConstraintError,
    ClientError as Neo4jClientError
)
from openai import AuthenticationError as OpenAIAuthenticationError, APIConnectionError as OpenAIAPIConnectionError 

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger(__name__)


load_dotenv()


CYPHER_GENERATION_TEMPLATE = """Task: Generate the WHERE, RETURN, and LIMIT clauses for a Cypher query to find relevant text chunks based on a user question. Assume the relevant Chunk node variable is 'c'.

Schema Information (for context only, mainly use Chunk.text):
Node properties:
Document {{title: STRING}}
Chunk {{text: STRING}}
Relationships:
(:Document)-[:HAS_CHUNK]->(:Chunk)

Instructions:
1.  Analyze the user's question: `{question}`.
2.  Identify the core subject. **If the core subject is an exact phrase (e.g., "principles of care", "comfort-focused care", "thermal care"), prioritize using this exact phrase as a keyword.**
3.  Extract other important single keywords related to the core subject and any key descriptive terms (e.g., "four", "key", "paramedics", "essential components", "BAPM").
4.  **Crucially, include specific keywords highly likely to appear *within the detailed answer itself*.** Examples:
    - If asking *how* something works (like Tranexamic Acid), include terms related to its mechanism (e.g., 'fibrinolysis', 'clot', 'bleeding').
    - If asking for an *approach* or *method* (like Spinal Injury), include terms for specific actions or equipment (e.g., 'immobilization', 'collar', 'backboard', 'log-roll', 'splint').
    - If asking for *components* (like Thermal Care), include specific items (e.g., 'bag', 'hat', 'mattress', 'swaddle', 'polythene', 'towel', 'blanket', 'warm', 'temperature').
    - If asking about *causes* or *mechanisms* (like Blunt Chest Trauma), include potential causes AND resulting conditions (e.g., 'collision', 'fall', 'assault', 'fracture', 'contusion', 'tamponade').
5.  Create a `WHERE` clause. **Start the clause with the word `WHERE` followed by a single space.**
6.  For each keyword/phrase identified (prioritizing exact phrases from step 2), add a condition using the format: `toLower(c.text) CONTAINS toLower('keyword or phrase')`.
7.  **CRITICAL SYNTAX:**
    - By default, enclose the keyword/phrase in single quotes.
    - **If a keyword/phrase contains a single quote (apostrophe), you MUST enclose the entire keyword/phrase in **double quotes** instead of single quotes.** Example: `toLower(c.text) CONTAINS toLower("patient's condition")`.
    - **Use the keyword `CONTAINS` exactly. Do NOT use typos like `CONTACS`.**
8.  **Combine ALL conditions using `OR`. Ensure there is a single space before and after each `OR`.** This aims for broad recall but gives higher weight to chunks containing the exact phrase.
9.  Return the chunk text using the exact format: `RETURN c.text AS result`
10. Limit results using the exact format: `LIMIT {limit}` (Use a reasonable limit like 15 for OR queries).
11. **Output ONLY the Cypher clauses starting from WHERE.** Ensure correct spacing everywhere. Do not include MATCH. Do not add any introductory text, explanations, or backticks.
12. **Ensure the entire generated Cypher text (WHERE, RETURN, LIMIT clauses) is a single block without invalid line continuation characters like backslashes (`\`) at the end of lines.**

Example for question "What is the patient's AVPU score?":
WHERE toLower(c.text) CONTAINS toLower("patient's") OR toLower(c.text) CONTAINS toLower('avpu') OR toLower(c.text) CONTAINS toLower('score') OR toLower(c.text) CONTAINS toLower('consciousness')
RETURN c.text AS result
LIMIT 15

User question: {question}
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question", "limit"], template=CYPHER_GENERATION_TEMPLATE
)



QA_PROMPT_TEMPLATE = """You are a highly precise and literal extraction assistant. Your ONLY task is to find the specific answer to the Question within the provided Context and extract it **verbatim** or synthesize it **strictly from the context** if verbatim is not possible. You follow instructions exactly.
**DO NOT** use external knowledge, summarize broadly, infer beyond the text, or add any commentary unless explicitly instructed.

Question: "{question}"

Context:
---------------------
{context}
---------------------

Instructions for Extraction/Synthesis:
1.  Read the Question: "{question}". Identify the **exact subject** (e.g., "thermal care components", "four core principles").
2.  Scan the Context *only* for text that directly lists or defines this **exact subject**. Look for introductory phrases like "The core principles are:", "Thermal care involves:", or similar, followed immediately by a list.
3.  **If you find a list or definition that directly answers the Question about the exact subject verbatim:**
    -   Extract **ONLY** the text describing the components/items directly related to the **exact subject**.
    -   **CRITICAL - EXCLUSION RULE:** You MUST NOT include items or details related to other procedures or general care, even if mentioned nearby in the context, unless they are explicitly part of the definition of the exact subject requested. Failure to exclude unrelated items is incorrect.
        -   **Example 1 (Thermal Care):** If the Question asks for "thermal care components", look for steps like 'place in polythene bag', 'apply hat', 'use heated mattress', 'ensure warm environment'. You MUST NOT include "cord clamping", "ventilation", "oxygen saturation", or "airway management". Extract ONLY the thermal care steps.
        -   **Example 2 (Core Principles):** If the Question asks for "core principles", extract ONLY high-level concepts like 'assessment', 'management', 'destination', 'communication'. DO NOT extract specific actions like 'summon help', 'prioritise mother', 'be aware of complications'.
    -   Extract the relevant text **exactly** as it appears in the Context, including a relevant introductory sentence if present.
    -   **Formatting Rule:** Use a numbered list (e.g., `1. Text`, `2. Text`) ONLY if the question asks for a specific number (e.g., 'four principles'). Otherwise, use bullet points (`• Text`). **CRITICAL: Do NOT mix numbered (e.g., `1.`) and bulleted (`•`) lists within the same answer.** Ensure correct spacing: `1. Text` or `• Text`. Ensure each list item starts on a new line.
    -   Ensure you extract the **MOST COMPLETE** definition available in the context for ALL relevant items in the list.
    -   STOP extracting immediately after the end of the definition for the LAST relevant item in the list. Do not include any subsequent unrelated sentences or paragraphs.
    -   Provide **ONLY** the extracted relevant list/definition. **ABSOLUTELY NO** other phrases like "The context does not contain..." or "The provided context does not contain...". **DO NOT** add prefixes like "Answer:". **REITERATION: NO EXTRA COMMENTARY AT ALL, NOT EVEN ABOUT THE CONTEXT ITSELF.**
4.  **If the Context does NOT contain the exact list or definition requested verbatim:**
    a. Check if the context *does* contain scattered facts that directly answer the core of the question.
    b. **If yes:** Synthesize a concise answer **strictly using only the facts found within the provided Context**. Do not add outside knowledge. Start the answer directly. Adhere to the Formatting Rule (bullets unless number specified, **NO MIXING**). **NO EXTRA COMMENTARY.**
    c. **If no:** Respond *only* with the exact phrase: `The context does not contain sufficient information to answer this question.` (Slightly shortened)
5.  **(Removed - Covered by new step 4c)**
6.  **Final Check:** Ensure your entire response adheres strictly to these rules, especially the EXCLUSION RULE, FORMATTING RULE (NO MIXING LISTS), verbatim extraction or context-only synthesis, completeness, and **ABSOLUTELY NO EXTRA COMMENTARY (unless stating context is insufficient as per rule 4c)**.

Answer based **ONLY** on the Context:"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=QA_PROMPT_TEMPLATE
)


class GraphRAGAssistant:
    def __init__(self, llm: Optional[BaseChatModel] = None):
        lm_studio_api_base = os.getenv('LM_STUDIO_API_BASE', 'http://localhost:1234/v1')
        lm_studio_api_key = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')
        neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_user = os.getenv('NEO4J_USERNAME', 'neo4j')
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'vaggpinel')
        
        self.retriever_limit = int(os.getenv('GRAPH_RAG_RETRIEVER_LIMIT', 15)) 
        
        self.qa_temp = 0.0

        logger.info("Connecting to Neo4j graph...")
        try:
            self._neo4j_uri = neo4j_uri
            self._neo4j_user = neo4j_user
            self._neo4j_password = neo4j_password
            self.graph = Neo4jGraph(
                url=self._neo4j_uri,
                username=self._neo4j_user,
                password=self._neo4j_password
            )
            
            self.graph.query("RETURN 1")
            logger.info("Neo4j graph object created and connection verified.")
        except Neo4jAuthError as e:
            logger.critical(f"Neo4j authentication failed: {e}", exc_info=True)
            raise ConnectionRefusedError(f"Neo4j authentication failed. Check credentials.") from e
        except ServiceUnavailable as e:
            logger.critical(f"Neo4j service unavailable at {self._neo4j_uri}: {e}", exc_info=True)
            raise ConnectionRefusedError(f"Neo4j service unavailable at {self._neo4j_uri}.") from e
        except Exception as e: 
            logger.critical(f"Failed during Neo4j graph initialization or connection test: {e}", exc_info=True)
            raise ConnectionRefusedError(f"Failed to connect to Neo4j: {e}") from e


        logger.info(f"Initializing LLM via LM Studio at: {lm_studio_api_base}")
        if llm:
            self.cypher_llm = llm
            self.qa_llm = llm
            logger.info("GraphRAG Assistant initialized with provided LLM for both Cypher and QA.")
            if hasattr(self.cypher_llm, 'temperature'):
                self.cypher_llm.temperature = 0.0 
            if hasattr(self.qa_llm, 'temperature'):
                 self.qa_llm.temperature = self.qa_temp 
        else:
            logger.info(f"Initializing default LLMs via LM Studio at: {lm_studio_api_base}")
            try:
                cypher_temp = 0.0 
                qa_temp = self.qa_temp 

                self.cypher_llm = ChatOpenAI(
                    openai_api_key=lm_studio_api_key,
                    openai_api_base=lm_studio_api_base,
                    temperature=cypher_temp
                )
                self.qa_llm = ChatOpenAI(
                    openai_api_key=lm_studio_api_key,
                    openai_api_base=lm_studio_api_base,
                    temperature=qa_temp
                )
                
                self.qa_llm.invoke("Test connection")
                logger.info(f"Default LLMs initialized and connection tested (Cypher Temp: {cypher_temp}, QA Temp: {qa_temp}).")
            except OpenAIAuthenticationError as e:
                 logger.critical(f"OpenAI authentication error: {e}", exc_info=True)
                 raise ConnectionRefusedError("AI model authentication failed. Check API key or service.") from e
            except OpenAIAPIConnectionError as e:
                 logger.critical(f"OpenAI API connection error: {e}", exc_info=True)
                 raise ConnectionRefusedError(f"Failed to connect to AI model service at {lm_studio_api_base}.") from e
            except Exception as e: 
                 logger.critical(f"Failed to initialize or test LLM from LM Studio: {e}", exc_info=True)
                 raise ConnectionRefusedError(f"Failed to initialize AI model: {e}") from e

        self._create_chains()
        
        logger.info(f"Graph RAG Assistant ready (Retrieval Limit: {self.retriever_limit}).")


    def _create_chains(self):
        logger.info("Creating Cypher generation and QA chains...")
        try:
            self.cypher_generation_chain = (
                CYPHER_GENERATION_PROMPT 
                | self.cypher_llm
                | StrOutputParser()
            )
            logger.info("Cypher generation chain created.")

            def format_context(context_list: list) -> str:
                if not context_list:
                    return "No context found."
                
                formatted_ctx = "\n\n".join([str(item.get('result', '')) for item in context_list if item.get('result')])
                
                logger.debug(f"Context being passed to QA LLM ({len(context_list)} chunks):\n---\n{formatted_ctx}\n---")
                return formatted_ctx

            self.qa_chain = (
                RunnablePassthrough.assign(
                    context=RunnableLambda(lambda x: format_context(x["context"]))
                )
                | QA_PROMPT 
                | self.qa_llm
                | StrOutputParser()
            )
            logger.info("QA chain created.")

        except Exception as e:
            logger.critical(f"Failed to create Langchain chains: {e}", exc_info=True)
            
            raise RuntimeError(f"Failed to create Langchain chains: {e}") from e

    def update_llm(self, llm: BaseChatModel):
        logger.info("Updating GraphRAG Assistant LLMs...")
        self.cypher_llm = llm
        self.qa_llm = llm
        if hasattr(self.cypher_llm, 'temperature'):
            self.cypher_llm.temperature = 0.0 
        if hasattr(self.qa_llm, 'temperature'):
             self.qa_llm.temperature = self.qa_temp 
        try:
            self._create_chains() 
            logger.info("GraphRAG Assistant LLMs updated and chains recreated.")
        except Exception as e:
             logger.error(f"Failed to recreate chains after LLM update: {e}", exc_info=True)
             


    
    def _post_process_result(self, result: str, question: str, context_data: list) -> str:
        """Applies sequential post-processing steps to the raw LLM result."""

        
        v7_no_context_phrase = "The context does not contain sufficient information to answer this question."
        
        
        llm_indicated_no_context = v7_no_context_phrase.lower() in result.strip().lower()


        raw_llm_result = result 
        final_result = result 

        
        if not context_data:
             override_result = "No relevant information found in the specified document context based on the query keywords."
             
             if llm_indicated_no_context:
                 logger.warning(f"Empty retrieval and LLM correctly indicated insufficient context. Overriding LLM response. Final result: {override_result}")
             else:
                 logger.warning(f"Empty retrieval, but LLM response was unexpected: '{result}'. Overriding. Final result: {override_result}")
             return override_result 

        
        processed_result = raw_llm_result 

        
        if llm_indicated_no_context:
            
            logger.warning(f"LLM indicated insufficient context (Raw: '{result.strip()}'). Standardizing response.")
            
            logger.info("Applying Commentary Removal (for insufficient context case)...")
            commentary_input = result.strip() 

            
            commentary_removed_result = re.sub(r'\s*\([^)]*\)\s*$', '', commentary_input, flags=re.IGNORECASE).strip()
            
            inline_removed_result = re.sub(r'\s*\(\b[\w\s.-]*?\b\)\s*', ' ', commentary_removed_result).strip()
            inline_removed_result = re.sub(r'\s{2,}', ' ', inline_removed_result).strip() 

            
            if inline_removed_result != commentary_input:
                logger.info(f"Removed commentary from insufficient context message. Cleaned: '{inline_removed_result}'")
            else:
                 logger.info("Commentary Removal: No commentary found or removed from insufficient context message.")

            
            final_result = v7_no_context_phrase
            logger.info(f"Returning standardized insufficient context message: '{final_result}'")
            return final_result 

        
        else:
            logger.info("LLM did not indicate insufficient context. Proceeding with full post-processing for answer.")
            

            
            logger.info("Applying Simplified List Formatting Cleanup...")
            list_cleaned_input = processed_result 
            formatted_result = re.sub(r'(\n\s*(?:•|\d+\.)\s*)\n(\S)', r'\1 \2', list_cleaned_input)
            if formatted_result != list_cleaned_input:
                processed_result = formatted_result.strip()
                logger.info(f"Applied simplified list formatting cleanup. Result:\n{processed_result}")
            else:
                processed_result = list_cleaned_input.strip() 
                logger.info("Simplified List Formatting Cleanup: No changes made.")
            

            
            logger.info("Applying Mixed List Normalization...")
            mixed_list_input = processed_result
            has_bullets = '•' in mixed_list_input
            has_numbers = re.search(r'^\s*\d+\.', mixed_list_input, re.MULTILINE) is not None
            if has_bullets and has_numbers:
                logger.warning("Detected mixed bullet and numbered list. Normalizing to bullets.")
                normalized_result = re.sub(r'^\s*\d+\.\s+', '• ', mixed_list_input, flags=re.MULTILINE)
                processed_result = normalized_result.strip()
                logger.info(f"Normalized mixed list to bullets. Result:\n{processed_result}")
            else:
                logger.info("Mixed List Normalization: No mixed lists detected or no normalization needed.")
                processed_result = mixed_list_input.strip() 
            


            
            
            logger.info("Skipping Targeted Filtering (relying on QA prompt exclusion rules).")
            


            
            logger.info("Applying Prefix Cleaning...")
            current_prefix_input = processed_result 
            prefix_cleaned_result = current_prefix_input
            
            prefixes_to_remove = [
                r'^\s*Answer:\s*',
                r'^\s*Here is the information based on the context:\s*', 
            ]
            prefix_removed = False
            for prefix_pattern in prefixes_to_remove:
                 new_result = re.sub(prefix_pattern, '', prefix_cleaned_result, count=1, flags=re.IGNORECASE | re.DOTALL).strip()
                 if new_result != prefix_cleaned_result.strip():
                     prefix_cleaned_result = new_result
                     prefix_removed = True
            if prefix_removed:
                processed_result = prefix_cleaned_result 
                logger.info(f"Removed prefix(es). Result after prefix cleaning:\n{processed_result}")
            else:
                logger.info("Prefix Cleaning: No prefixes found or removed.")
            

            
            logger.info("Applying Commentary Removal (for answer)...")
            commentary_input = processed_result 

            
            commentary_removed_result = re.sub(r'\s*\([^)]*\)\s*$', '', commentary_input, flags=re.IGNORECASE).strip()
            
            inline_removed_result = re.sub(r'\s*\(\b[\w\s.-]*?\b\)\s*', ' ', commentary_removed_result).strip()
            inline_removed_result = re.sub(r'\s{2,}', ' ', inline_removed_result).strip() 

            if inline_removed_result != commentary_input.strip(): 
                processed_result = inline_removed_result
                logger.info(f"Removed trailing and/or inline commentary. Result:\n{processed_result}")
            else:
                logger.info("Commentary Removal: No commentary found or removed.")
                processed_result = commentary_input.strip() 
            


            
            logger.info("Applying Simplified Truncation Check...")
            current_truncation_input = processed_result 
            is_list = '•' in current_truncation_input or re.search(r'^\s*\d+\.', current_truncation_input, re.MULTILINE)
            truncated_result = current_truncation_input

            if is_list:
                matches = list(re.finditer(r'(?:^|\n)(\s*(?:•|\d+\.)\s+.*(?:\n(?!\s*(?:•|\d+\.)).*)*)', current_truncation_input, re.MULTILINE))
                if matches:
                    last_match = matches[-1]
                    last_item_end_pos = last_match.end(0)
                    trailing_content = current_truncation_input[last_item_end_pos:].strip()
                    if trailing_content:
                        logger.warning(f"Truncation: Potential over-extraction detected (non-whitespace text after list: '{trailing_content[:50]}...'). Attempting truncation.")
                        truncated_result = current_truncation_input[:last_item_end_pos].strip()
                    else:
                        truncated_result = current_truncation_input[:last_item_end_pos].strip()
                        logger.info("Truncation: Found only whitespace after list. Stripping trailing whitespace.")

                if truncated_result != current_truncation_input.strip():
                     processed_result = truncated_result
                     logger.info(f"Applied truncation. Result after truncation:\n{processed_result}")
                else:
                     logger.info("Truncation: No over-extraction detected or truncation applied.")
            else:
                logger.info("Truncation: Not a list or no list items found. Skipping.")
            


            
            final_result = processed_result 

            if final_result != raw_llm_result:
                logger.info(f"Final result (before final markdown): '{final_result}'")
            else:
                logger.info(f"Final result (no post-processing applied or changes made before final markdown): '{final_result}'")


            
            logger.info("Skipping Bullet-to-Number Conversion (relying on QA prompt/normalization).")


            
            logger.info("Applying final Markdown list formatting (double newlines)...")
            markdown_input = final_result
            markdown_formatted_result = re.sub(r'\n(\s*(?:•|\d+\.)\s+)', r'\n\n\1', markdown_input)
            if markdown_formatted_result != markdown_input:
                logger.info("Applied final Markdown list formatting (double newlines).")
                final_result = markdown_formatted_result
            else:
                logger.info("Final Markdown list formatting: No changes needed.")
                final_result = markdown_input 
            

        
        return final_result

    
    def query(self, question: str, current_doc_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes a user query using Graph RAG, handling potential errors gracefully.
        """
        logger.info(f"GraphRAG Query: '{question}'")
        if current_doc_title:
            logger.info(f"GraphRAG Context Title: '{current_doc_title}'")
        else:
            logger.warning("GraphRAG Context Title: None provided (querying all documents)")

        final_result = ""
        user_error_message = "An error occurred while processing your query." 
        error_details = ""
        generated_clauses = "" 
        try:
            
            cypher_input = {"question": question, "limit": self.retriever_limit}
            generated_clauses = self.cypher_generation_chain.invoke(cypher_input)
            logger.debug(f"Raw Generated Cypher Clauses:\n{generated_clauses}")

            if not generated_clauses or not generated_clauses.strip().upper().startswith("WHERE"):
                 
                 raise ValueError(f"LLM did not generate valid Cypher clauses starting with WHERE. Output: '{generated_clauses}'")

            
            fixed_clauses = generated_clauses.replace("WHEREtoLower", "WHERE toLower")\
                                             .replace("ORtoLower", "OR toLower") \
                                             .replace("ANDtoLower", "AND toLower") \
                                             .replace(" CONTACS ", " CONTAINS ") \
                                             .replace(" CONTAITS ", " CONTAINS ") \
                                             .replace(' \\\n', ' ') 

            if fixed_clauses != generated_clauses:
                logger.warning(f"Applied Auto-Fix for Spacing/Typo/Linebreak. Original:\n{generated_clauses}\nFixed:\n{fixed_clauses}")
                generated_clauses = fixed_clauses
            else:
                logger.debug("No spacing/typo/linebreak auto-fix applied.")

            
            params = {}
            if current_doc_title:
                match_clause = "MATCH (d:Document {title: $doc_title})-[:HAS_CHUNK]->(c:Chunk)"
                params["doc_title"] = current_doc_title
            else:
                match_clause = "MATCH (c:Chunk)"

            full_cypher_query = f"{match_clause}\n{generated_clauses}"
            logger.info(f"Constructed Full Cypher Query:\n{full_cypher_query}")
            logger.info(f"Query Parameters: {params}")

            context_data = self.graph.query(full_cypher_query, params=params)
            context_count = len(context_data) if context_data else 0
            logger.info(f"Retrieved Context Data Count: {context_count}")

            
            qa_input = {"question": question, "context": context_data if context_data else []}
            raw_llm_result = self.qa_chain.invoke(qa_input)
            logger.info(f"Generated Final Answer (raw): {raw_llm_result}")

            
            
            final_result = self._post_process_result(raw_llm_result, question, context_data)

            
            logger.info(f"Returning final processed result: '{final_result}'")
            return {
                "query": question,
                "result": final_result,
                "document_title": current_doc_title if current_doc_title else "All Documents"
            }

        
        except ValueError as e: 
            error_type = type(e).__name__
            error_details = str(e)
            logger.error(f"LLM Generation Error: {error_details}", exc_info=True)
            user_error_message = "The AI failed to generate a valid database query based on your question."

        except CypherSyntaxError as e: 
            error_type = type(e).__name__
            error_details = str(e)
            logger.error(f"Neo4j Cypher Syntax Error: {error_details}", exc_info=True)
            
            user_error_message = "Failed to query the graph due to a syntax error in the generated query. This might involve issues with keywords, quotes, or structure."
            
            logger.error(f"Cypher query causing syntax error: {generated_clauses}")


        except ConstraintError as e: 
             error_type = type(e).__name__
             error_details = str(e)
             logger.warning(f"Neo4j Constraint Validation Failed: {error_details}", exc_info=True) 
             user_error_message = "Warning: Query might have failed due to graph constraint validation issues."

        except Neo4jClientError as e: 
             error_type = type(e).__name__
             error_details = str(e)
             
             if "ParameterMissing" in error_details:
                 logger.error(f"Neo4j Parameter Missing Error: {error_details}", exc_info=True)
                 user_error_message = "Failed to query the graph due to a missing parameter."
             
             elif "UnknownLabelWarning" in error_details or \
                  "UnknownRelationshipTypeWarning" in error_details or \
                  "UnknownPropertyKeyWarning" in error_details:
                 logger.warning(f"Neo4j Schema Mismatch Warning: {error_details}", exc_info=True) 
                 user_error_message = "Warning: Query might have returned incomplete results due to mismatch with graph structure."
             else:
                 logger.error(f"Neo4j Client Error: {error_details}", exc_info=True)
                 user_error_message = "An error occurred while querying the knowledge graph."

        except (ServiceUnavailable, Neo4jAuthError, ConnectionRefusedError) as e: 
            error_type = type(e).__name__
            error_details = str(e)
            logger.critical(f"Neo4j Connection/Auth Error: {error_details}", exc_info=True)
            user_error_message = "Error: Could not connect to the knowledge graph database."

        except (OpenAIAuthenticationError, OpenAIAPIConnectionError) as e: 
            error_type = type(e).__name__
            error_details = str(e)
            logger.critical(f"LLM Connection/Auth Error: {error_details}", exc_info=True)
            user_error_message = "Error: Could not connect to the AI model service."

        except OutputParserException as e: 
            error_type = type(e).__name__
            error_details = str(e)
            logger.error(f"QA Chain Output Parsing Error: {error_details}", exc_info=True)
            user_error_message = "The AI failed to format the final answer correctly."

        except Exception as e: 
            error_type = type(e).__name__
            error_details = str(e)
            logger.error(f"Unexpected Error during GraphRAG query: {error_details}", exc_info=True)
            
            
            user_error_message = "An unexpected error occurred while processing your query."

        
        
        error_type = error_type if 'error_type' in locals() else "UnknownError"
        error_details = error_details if 'error_details' in locals() else "No details available"

        error_response = {
            "query": question,
            "error": user_error_message,
            "details": f"{error_type}: {error_details}", 
            "document_title": current_doc_title if current_doc_title else "N/A"
        }
        logger.info(f"Returning error response: {error_response}")
        return error_response