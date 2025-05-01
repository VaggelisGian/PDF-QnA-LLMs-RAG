import os
import re
import logging
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_openai import ChatOpenAI

load_dotenv()


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


RAG_QA_PROMPT_TEMPLATE = """You are an assistant tasked with extracting specific information from the provided context ONLY.
Base your answer strictly on the text provided below. Do not add any information not present in the context.
Focus precisely on the user's question and the specific type of information requested.

**If the question asks for steps, actions, how to perform a procedure, OR indications (e.g., "What are the steps for X?", "How to do Y?", "Indications for Z"):**
1. Identify EVERY action, step, OR specific indication mentioned in the context that directly relates to the specific procedure, condition, AND timeframe mentioned in the question.
2. For **indications**: Include specific triggers or criteria mentioned (e.g., 'haemodynamic instability (SBP < 90)', 'failed needle decompression', 'signs of life ≤ 5 min'). Be precise about these criteria.
3. For **procedural steps**:
    a. Include initial assessment steps ONLY IF they are mentioned as part of the procedure in the context.
    b. Include actions taken based on assessments (e.g., 'if breathing is poor, provide ventilation').
    c. Include actions that involve *not* doing something if mentioned (e.g., "not dried", "do not attempt chest compressions if...").
    d. Include specific techniques or anatomical locations mentioned (e.g., 'apply pelvic binder AT THE GREATER TROCHANTERS', 'use JAW-THRUST', 'maintain MANUAL IN-LINE STABILIZATION (MILS)').
4. List ONLY these specific actions/steps/indications clearly and concisely as a numbered list.
5. **CRITICAL for Management/Steps:** Prioritize immediate life-saving interventions mentioned first (e.g., hemorrhage control before fluids). Do NOT include secondary management steps (like IV fluid administration for hemorrhage control, routine splinting) unless the context explicitly states they are part of the *immediate* first actions or directly addresses the primary problem (e.g., binder for pelvic hemorrhage).
6. Do NOT include general principles unless they are presented as specific steps/indications in the context.

**If the question asks for equipment, items, tools, or things to prepare (e.g., "What equipment is needed?", "List the items for Z"):**
1. Identify EVERY specific piece of equipment, item, or tool mentioned in the context that directly relates to the subject of the question.
2. Include details about *when* preparation should occur (e.g., "before birth") IF mentioned in the context.
3. Include any "backup" or alternative items mentioned (e.g., "warm towels if bag unavailable") IF mentioned in the context.
4. List ONLY these specific items clearly and concisely as a bulleted list (using '*' or '-').
5. Include brief descriptions or purposes if mentioned directly alongside the item in the context (e.g., "heated mattress for thermal stability").
6. Do NOT include procedural steps or actions, unless the action is intrinsic to identifying the item (e.g., "open your maternity pack").

**--- EXAMPLES ---**

**Example 1: Filtering irrelevant information & Specificity (Pelvic Binder)**
Context: "Apply a pelvic binder at the greater trochanters for suspected unstable pelvic fractures with hypotension (SBP < 90). If a thoracostomy was done, press the aorta. Stabilize other fractures. Rapid transport is key."
Question: "What are the indications for using a pelvic binder pre-hospital?"
Correct Answer:
1. Suspected unstable pelvic fracture WITH hypotension (SBP < 90).

**Example 2: Distinguishing procedures & Indications (Thoracic)**
Context: "Indications for thoracostomy include tension pneumothorax after failed needle decompression, and massive haemothorax. Indications for thoracotomy include penetrating injury to the cardiac box with recent signs of life (≤ 5 min)."
Question: "What are the indications for thoracostomy?"
Correct Answer:
1. Tension pneumothorax after failed needle decompression.
2. Massive haemothorax.

**Example 3: Prioritizing Immediate Actions (Hemorrhage)**
Context: "Control catastrophic external hemorrhage immediately using direct pressure and tourniquets high and tight. Consider TXA early. Once bleeding is controlled, gain IV access and administer fluids cautiously, aiming for permissive hypotension."
Question: "How should catastrophic external hemorrhage be managed pre-hospital?"
Correct Answer:
1. Apply direct pressure.
2. Apply tourniquet high and tight.
3. Consider TXA early.

**Example 4: Specific Techniques (Airway C-Spine)**
Context: "For suspected C-spine injury, maintain manual in-line stabilization (MILS) throughout airway management. Use a jaw-thrust maneuver. Secure the airway via RSI if GCS ≤ 8, maintaining MILS."
Question: "What is the recommended pre-hospital airway management strategy in a trauma patient with suspected cervical spine injury?"
Correct Answer:
1. Maintain Manual In-Line Stabilization (MILS).
2. Use jaw-thrust maneuver.
3. Secure airway (e.g., RSI if GCS ≤ 8) while maintaining MILS.

**--- END EXAMPLES ---**

**For ALL questions, do NOT include:**
    - General principles or statements of fact not directly answering the question.
    - Actions/items related to significantly different procedures or timeframes not covered by the question.
    - Secondary management steps (like routine IV fluids, general splinting) unless explicitly stated as immediate first actions or core indications in the context.
    - Actions related to communication, transport logistics (unless a key principle like 'rapid transport for definitive haemostasis'), or other types of care not directly requested.
    - Figure references or citations (e.g., "(Figure 2)", "[7]").
    - Introductory phrases (e.g., "Based on the context...", "The required equipment includes...") or concluding summaries.

**If the context does not contain the answer for the specific question, state only:** "The context does not provide the answer."

Context:
{context}

Question:
{question}

Answer based ONLY on the context:"""



RAG_QA_PROMPT = PromptTemplate(
    template=RAG_QA_PROMPT_TEMPLATE, input_variables=["context", "question"]
)


class RAGAssistant:


    DEFAULT_RETRIEVER_K = int(os.getenv("RAG_RETRIEVER_K", "6"))
    DEFAULT_LLM_TEMP = float(os.getenv("DEFAULT_LLM_TEMP", "0.1"))

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,

        retriever_k: Optional[int] = None,
    ):

        self.retriever_k = retriever_k if retriever_k is not None else self.DEFAULT_RETRIEVER_K
        logger.info(f"Initializing RAGAssistant (k={self.retriever_k})...")


        embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.neo4j_url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "vaggpinel")



        self.vector_store: Optional[Neo4jVector] = None
        self.retriever = None
        self.chain: Optional[RetrievalQA] = None
        self.llm: Optional[BaseChatModel] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None

        if not self._initialize_embeddings(embedding_model_name):
            logger.warning("Embeddings initialization failed. RAG Assistant may not function.")
            return

        if not self._initialize_vector_store():
            logger.warning("Vector store initialization failed. RAG Assistant may not function.")
            return

        if not self._initialize_llm(llm):
            logger.warning("LLM initialization failed. RAG Assistant may not function.")
            return


        self._update_chain()

        if self.chain:
            logger.info("RAG Assistant ready.")
        else:
            logger.warning("RAG Assistant initialized, but chain creation failed.")

    def _initialize_embeddings(self, model_name: str) -> bool:
        logger.info(f"Initializing local embeddings model: {model_name}")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name, model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")}
            )
            logger.info("Embeddings model initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embeddings model: {e}", exc_info=True)
            self.embeddings = None
            return False

    def _initialize_vector_store(self) -> bool:
        if not self.embeddings:
            logger.error("Embeddings not available, cannot initialize vector store.")
            return False

        logger.info("Connecting to Neo4j vector store (using Chunk nodes)...")
        try:
            self.vector_store = Neo4jVector.from_existing_index(
                embedding=self.embeddings,
                url=self.neo4j_url,
                username=self.neo4j_user,
                password=self.neo4j_password,
                index_name="pdf_chunk_embeddings",
                node_label="Chunk",
                text_node_property="text",
                embedding_node_property="embedding",


            )

            if not isinstance(self.vector_store, Neo4jVector):
                 logger.error("Neo4jVector.from_existing_index did not return a Neo4jVector instance.")
                 raise RuntimeError("Neo4jVector initialization failed type check")

            logger.info("Neo4j vector store connected successfully.")
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.retriever_k}
            )
            logger.info(
                f"Retriever created with k={self.retriever.search_kwargs.get('k', 'N/A')}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error connecting to Neo4j vector store or creating retriever: {e}",
                exc_info=True,
            )
            self.vector_store = None
            self.retriever = None
            return False

    def _initialize_llm(self, llm: Optional[BaseChatModel] = None) -> bool:
        lm_studio_api_base = os.getenv("LM_STUDIO_API_BASE", "http://localhost:1234/v1")
        lm_studio_api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
        default_llm_temp = self.DEFAULT_LLM_TEMP
        default_max_tokens = int(os.getenv("DEFAULT_LLM_MAX_TOKENS", "512"))

        if llm:

            self.llm = llm
            logger.info("RAG Assistant initialized with provided LLM.")
            return True
        else:
            logger.info(f"Initializing default LLM via LM Studio at: {lm_studio_api_base}")
            if not lm_studio_api_base:
                 logger.error("LM_STUDIO_API_BASE environment variable not set.")
                 return False
            try:

                self.llm = ChatOpenAI(
                    openai_api_key=lm_studio_api_key,
                    openai_api_base=lm_studio_api_base,
                    temperature=default_llm_temp,
                    max_tokens=default_max_tokens

                )
                logger.info("Default LLM initialized successfully.")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize LLM from LM Studio: {e}", exc_info=True)
                self.llm = None
                return False

    def _update_chain(self):
        if not self.retriever or not self.llm:
            logger.error("Cannot create/update chain. Retriever or LLM not initialized.")
            self.chain = None
            return

        try:

            chain_type_kwargs = {"prompt": RAG_QA_PROMPT}
            self.chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=True,
            )

            logger.info("RetrievalQA chain created/updated with V4.5 custom QA prompt.")

        except Exception as e:
            logger.error(f"Failed to create/update RetrievalQA chain: {e}", exc_info=True)
            self.chain = None

    def update_llm(self, llm: BaseChatModel):
        logger.info("Updating RAG Assistant LLM...")


        if self._initialize_llm(llm):
            self._update_chain()
            if self.chain:
                logger.info("RAG Assistant LLM updated and chain recreated.")
            else:
                logger.error("RAG Assistant LLM updated, but chain recreation failed.")
        else:
             logger.error("Failed to initialize the new LLM during update.")


    def query(self, question: str) -> Dict[str, Any]:
        if not self.chain or not self.retriever:
            error_msg = "RAG Assistant is not properly initialized (chain or retriever missing)."
            logger.error(error_msg)
            return {"answer": f"Error: {error_msg}", "sources": []}

        logger.info(f"RAG Query: {question}")
        try:


            result = self.chain.invoke({"query": question})
            logger.debug(f"RAG Result (raw): {result}")

            answer = result.get("result", "No answer found.")


            logger.debug(f"Answer before post-processing: {answer}")

            answer = re.sub(r"\s*[\(\[]\s*\d+\s*[\)\]]", "", answer)
            answer = answer.strip()
            logger.info(f"Answer after post-processing: {answer}")



            sources_data: List[Dict[str, Any]] = []
            source_documents = result.get("source_documents")
            if isinstance(source_documents, list):
                for doc in source_documents:
                    content = getattr(doc, "page_content", None)
                    metadata = getattr(doc, "metadata", {})
                    if content:
                        source_info = {
                            "content": content,
                            "metadata": metadata,
                        }
                        sources_data.append(source_info)
                logger.info(
                    f"RAG Sources (structured): {len(sources_data)} chunks retrieved with metadata."
                )
            else:
                logger.warning("No 'source_documents' list found in RAG result.")


            return {"answer": answer, "sources": sources_data}

        except Exception as e:
            error_msg = f"Error during RAG query execution: {type(e).__name__}"
            logger.error(f"{error_msg}: {e}", exc_info=True)
            return {
                "answer": f"An error occurred while processing the query. Please check the system logs for details.",
                "sources": [],
            }
