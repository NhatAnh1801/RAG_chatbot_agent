import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_chroma import Chroma

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from dotenv import load_dotenv

from src.models.embeddings.gte_multi_base import GTE
from src.prompt import*

import os
import requests
import time
import re       

load_dotenv()   

# DECLARE VARIABLES
PARENT_CHUNK_SIZE = 1000
CHILD_CHUNK_SIZE = 200
CLOUDFLARE_URL= os.getenv("CLOUDFLARE_URL")

class RagController:
    def __init__(self):
        '''
            Init vector DB and LLM here
        '''
        # INIT MODELS
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
            
        self.embedding_model = GTE()
        
        google_model = "google_genai:gemini-flash-lite-latest"
        self.llm_model = init_chat_model(
            model=google_model,
            api_key=GEMINI_API_KEY
        )
        
        # Init Chroma as vector database
        self.vector_db = Chroma(
            embedding_function=self.embedding_model,
            persist_directory='./data/chromadb'    
        )
        
        # INIT SMALL2BIG
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size = PARENT_CHUNK_SIZE,
            length_function = len,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size = CHILD_CHUNK_SIZE,
            length_function = len,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        fs = LocalFileStore("./data/docstore") 
        
        self.small2big_retriever = ParentDocumentRetriever(
            vectorstore=self.vector_db,
            docstore=create_kv_docstore(fs),
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": 2}
        )
        
    def ingest_docs(self, docs: list, batch_size = 200):
        """
        Chunk documents using a hierarchical "small-to-big" approach for optimized retrieval.
        Note:
        - The maximum batch size supported by ChromaDB is 5461 documents.
        - The batch_size parameter is set to 50 to prevent overloading the vector database,
        - A pdf page full of text is about 3000 characters. -> ~4 parents, each parents have ~6 child => 24 child chunks each page
            -> The maximum capacity is 227 pages
            -> Set batch_size to 150-200 is the sweet spot
        This ensures stable ingestion and efficient use of system resources.
        """
        if not docs:
            return
        
        # Check duplicate documents in the vector database
        existing = self.vector_db.get()
        ingested_keys = set(
            (m["source"], m["page"]) for m in existing["metadatas"]
        )
        new_docs = [
            d for d in docs
            if (d.metadata.get("source"), d.metadata.get("page")) not in ingested_keys
        ]
        
        skipped = len(docs) - len(new_docs)
        if skipped:
            print(f"-> [ingest_docs]: Skipping {skipped} already ingested pages")

        if not new_docs:
            print(f"-> [ingest_docs]: All documents already ingested, skipping...")
            return
        
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i : i + batch_size]
            self.small2big_retriever.add_documents(batch)
            
    def ingest_legal_docs(self):
        try:
            response = requests.get(
                f"{CLOUDFLARE_URL}/ingest",
                timeout=180
            )
            
            response.raise_for_status()
            data = response.json()
            
            if data["status"] != "ok":
                raise RuntimeError(data.get("message", "Unknown error"))
            
            documents = data["documents"]
            print(f"-> [ingest_legal_docs]: Received {len(documents)} documents")
            
            all_docs = []
            for document in data["documents"]:
                jurisdiction_meta = document["jurisdiction"]
                domain_meta = document["domain"]
                source = document["source"]
                
                print(f"Ingesting: {jurisdiction_meta} -> {domain_meta}")
                
                # Wrap pages into LangChain Documents
                docs = [
                    Document(
                        page_content=page_text,
                        metadata={
                            "jurisdiction": jurisdiction_meta,
                            "domain": domain_meta,
                            "source": source,
                            "page": i
                        }
                    )
                    for i, page_text in enumerate(document["pages"])
                    if page_text.strip() 
                ]
                all_docs.extend(docs)
                
            # Send to ChromaDB
            t0 = time.perf_counter()
            self.ingest_docs(all_docs)
            t1 = time.perf_counter()
            print(f"-> [ingest_legal_docs]: all docs are ingested in {t1 - t0: .4f}")
                
        except requests.exceptions.ConnectionError:
            print("ERROR: Cannot reach Colab. Is the tunnel still running?")
            raise
        except requests.exceptions.Timeout:
            print("-> [ingest_legal_docs]: Request timed out after 1 hour")
            raise
        except Exception as e:
            print(f"Ingestion failed: {e}")
            raise

    def get_retrieved_docs(self, jurisdiction: str, domain: str):
        # Verify if the jurisdiction and domain are valid
        if jurisdiction not in set(m["jurisdiction"] for m in self.vector_db.get()["metadatas"]):
            raise ValueError(f"Invalid jurisdiction: {jurisdiction}")
        if domain not in set(m["domain"] for m in self.vector_db.get()["metadatas"]):
            raise ValueError(f"Invalid domain: {domain}")
        
        @tool(response_format="content_and_artifact")
        def retrieve_doc(query:str) -> tuple:
            """
                Query documents from the user's question.
                Args:
                    query: The user's question or search query.
                    jurisdiction: The jurisdiction to filter by.
                    domain: The domain to filter by.
                Returns:
                    A tuple of (serialized text, list of Document objects).
            """
            try:
                self.small2big_retriever.vectorstore.search_kwargs = {
                    "filter": {
                        "$and": [
                            {"jurisdiction": {"$eq": jurisdiction}},
                            {"domain": {"$eq": domain}}
                        ]
                    }
                }
                
                retrieved_docs = self.small2big_retriever.invoke(query)
                serialized = "\n\n".join(
                    f"Source: {doc.metadata}\nContent: {doc.page_content}"
                    for doc in retrieved_docs
                )
                return serialized, retrieved_docs
            except Exception as e:
                print(f"Error during document retrieval: {e}")
                return "An error occurred while retrieving documents.", []
            
        return retrieve_doc
    
    def get_system_prompt(self, jurisdiction, domain):
        return system_prompt.format(jurisdiction=jurisdiction, domain=domain)
    
    def parse_response(self, response):
        try:
            clean = re.sub(r"```json|```", "", response).strip()
            if not clean.endswith("}"):
                clean += "}"
            return json.loads(clean)
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None
        
    def build_legal_agent(self, jurisdiction, domain):
        # build the RAG agent with the LLM, retrieval tools, and system prompt
        retrieve_doc = self.get_retrieved_docs(jurisdiction, domain)
        return create_agent(
            model=self.llm_model,
            tools=[retrieve_doc],
            system_prompt=self.get_system_prompt(jurisdiction, domain)
        )
     
    def _extract_source_info(self, messages):
        for msg in messages:
            if msg.type == "tool" and msg.name == "retrieve_doc":
                if hasattr(msg, 'artifact') and msg.artifact:
                    doc = msg.artifact[0]
                    return {
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page")
                    }
        return None
    
    def _extract_contexts(self, messages) -> list[str]:
        """
            Extract the contexts from agent response, used for evaluation.
        """
        contexts = []
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "tool" and hasattr(msg, "artifact"):
                if msg.artifact:
                    for doc in msg.artifact:
                        contexts.append(doc.page_content)
        return contexts
    
    def ask(self, agent, question, history=None, max_turns=5):
        messages = []
        if history:
            max_messages = max_turns * 2
            recent_history = history[-max_messages:]
            for msg in recent_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        messages.append({"role": "user", "content": question})
        inputs = {"messages": messages}
        
        response = None
        for chunk in agent.stream(input=inputs, stream_mode="values"):
            response = chunk    
            # print(f"-> [ask]: Received chunk: {chunk}")

        messages = response["messages"]
        final_ans = messages[-1].content
        if isinstance(final_ans, list):
            final_ans = " ".join(block["text"] for block in final_ans if block.get("type") == "text")
        
        # print(f"-> [ask]: messages from agent:\n{messages}")
        # print(f"-> [ask]: Final answer content:\n{final_ans}")
        
        # Debug token usage
        for msg in messages:
            if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                print(f"[{msg.type}] Tokens - input: {msg.usage_metadata.get('input_tokens')}, "
                    f"output: {msg.usage_metadata.get('output_tokens')}, "
                    f"total: {msg.usage_metadata.get('total_tokens')}")
        
        source_info = self._extract_source_info(messages)
        contexts = self._extract_contexts(messages)
        
        return {
            "type": "document_based" if source_info else "general",
            "source": source_info.get("source") if source_info else None,
            "page": source_info.get("page") if source_info else None,
            "answer": final_ans,
            "contexts": contexts
        }
            
    def _test_process_pdf(self):
        self.ingest_legal_docs()
    