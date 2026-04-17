from langchain_core.documents import Document
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from dotenv import load_dotenv

from src.models.embeddings.gte_multi_base import GTE
from src.prompt import*

import os
import requests
import hashlib
import time
import pickle
import json  

load_dotenv()   

# DECLARE VARIABLES
BM25_PATH = "./data/bm25/bm25_retriever.pkl"
BM25_HASH_PATH = "./data/bm25/bm25_content.hash"
CLOUDFLARE_URL= os.getenv("CLOUDFLARE_URL")

class RagController:
    def __init__(self):
        # Validate API key
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        # Models
        self.embedding_model = GTE()
        google_model = "google_genai:gemini-flash-lite-latest"
        self.llm_model = init_chat_model(
            model=google_model,
            api_key=GEMINI_API_KEY
        )
        
        # Vector database
        self.vector_db = Chroma(
            embedding_function=self.embedding_model,
            persist_directory='./data/chromadb'    
        )
        self.docstore = create_kv_docstore(LocalFileStore("./data/docstore"))
    
    def ingest_legal_docs(self):
        try:
            # response = requests.get(
            #     f"{CLOUDFLARE_URL}/ingest",
            #     timeout=180
            # )
            
            # response.raise_for_status()
            # data = response.json()
            
            # if data["status"] != "ok":
            #     raise RuntimeError(data.get("message", "Unknown error"))
            from pathlib import Path
            mock_response_path = "./data/json_files/Vietnam"
            data = {
                "status": "ok",
                "documents": [
                    json.load(open(f,encoding="utf-8")) 
                    for f in Path(mock_response_path).rglob("*.json")
                ]
            }

            documents = data["documents"]
            for document in documents:
                print(document.keys())
            print(f"-> [ingest_legal_docs]: Received {len(documents)} documents")
            
            parent_docs = {}    
            child_docs = []
            
            for document in documents:
                jurisdiction = document["jurisdiction"]
                domain = document["domain"]
                source = document["source"]
                
                print("Jurisdiction:", jurisdiction, "| Domain:", domain, "| Source:", source)
                
                for chunk in document["chunks"]:
                    parent_title = chunk["parent"]
                    parent_content = chunk.get("parent_content", "")
                    
                    parent_id = hashlib.md5(f"{source}__{parent_title})".encode()).hexdigest()
                    
                    parent_docs[parent_id] = Document(
                        page_content=parent_content,
                        metadata={
                            "parent_id": parent_id,
                            "parent_title": parent_title,
                            "jurisdiction": jurisdiction,
                            "domain": domain,
                            "source": source
                        }
                    )
                    
                    for child in chunk["children"]:
                        child_id = hashlib.md5(f"{source}__{child['title']}".encode()).hexdigest()
                        
                        child_docs.append(Document(
                            page_content=child["content"],
                            metadata={
                                "child_id": child_id,
                                "parent_id": parent_id,  
                                "parent_title": parent_title,
                                "child_title": child["title"],
                                "jurisdiction": jurisdiction,
                                "domain": domain,
                                "source": source,
                            }
                        ))
                
            # Send to ChromaDB
            t0 = time.perf_counter()
            
            self._add_child_and_parent_documents(child_docs=child_docs, parent_docs=parent_docs)
            
            t1 = time.perf_counter()
            print(f"[ingest_legal_docs]: all docs are ingested in {t1 - t0: .4f}")
                
        except requests.exceptions.ConnectionError:
            print("ERROR: Cannot reach Colab. Is the tunnel still running?")
            raise
        except requests.exceptions.Timeout:
            print("[ingest_legal_docs]: Request timed out after 1 hour")
            raise
        except Exception as e:
            print(f"Ingestion failed: {e}")
            raise
    
    def build_legal_agent(self, jurisdiction, domain):
        retrieve_doc = self._get_retrieved_docs(jurisdiction, domain)
        return create_agent(
            model=self.llm_model,
            tools=[retrieve_doc],
            system_prompt=self._get_system_prompt(jurisdiction, domain)
        )
        
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
    
    def _add_child_and_parent_documents(self, child_docs: list, parent_docs: dict, batch_size: int=200):
        """
            ChromaDB max batch: 5461 docs. 
            PDF page: ~24 child chunks. 
            Recommended batch_size: 150-200.
        """
        if not child_docs or not parent_docs:
            return

        ids = [d.metadata["child_id"] for d in child_docs]
        
        for i in range(0, len(child_docs), batch_size):
            child_batch = child_docs[i : i + batch_size]
            id_batch = ids[i : i + batch_size]
            self.vector_db.add_documents(child_batch, ids=id_batch)
            
        self.docstore.mset(list(parent_docs.items()))
        
        self._build_bm25_retriever(child_docs)

    def _compute_docs_hash(self, docs) -> str:
        content = "".join(sorted(d.metadata["child_id"] for d in docs))
        return hashlib.md5(content.encode()).hexdigest()

    def _build_bm25_retriever(self, child_docs, bm25_top_k=5):
        os.makedirs(os.path.dirname(BM25_PATH), exist_ok=True)
        current_hash = self._compute_docs_hash(child_docs)

        if os.path.exists(BM25_PATH) and os.path.exists(BM25_HASH_PATH):
            with open(BM25_HASH_PATH, "r") as f:
                saved_hash = f.read().strip()
            if saved_hash == current_hash:
                print("-> BM25 cache valid, loading from disk")
                with open(BM25_PATH, "rb") as f:
                    return pickle.load(f)

        print("-> Building BM25 index")
        bm25_retriever = BM25Retriever.from_documents(child_docs)
        bm25_retriever.k = bm25_top_k

        with open(BM25_PATH, "wb") as f:
            pickle.dump(bm25_retriever, f)
        with open(BM25_HASH_PATH, "w") as f:
            f.write(current_hash)

        return bm25_retriever
    
    def _load_bm25_retriever(self):
        """loads BM25 from disk."""
        if os.path.exists(BM25_PATH):
            with open(BM25_PATH, "rb") as f:
                return pickle.load(f)
        raise FileNotFoundError("BM25 index not found. Run ingestion first.")
              
    def _hybrid_retriever(self):
        vector_retriever = self.vector_db.as_retriever()
        bm25_retriever = self._load_bm25_retriever()
    
        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.6, 0.4]
        )
        
    def _get_retrieved_docs(self, jurisdiction: str, domain: str):
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
                retriever = self._hybrid_retriever()
                
                retrieved_child_chunks = retriever.invoke(query)
                
                # Filter based on jurisdiction and domain
                retrieved_child_chunks = [
                    d for d in retrieved_child_chunks
                    if d.metadata.get("jurisdiction") == jurisdiction
                    and d.metadata.get("domain") == domain
                ]

                # 3. Fetch parents
                parent_ids = list({r.metadata["parent_id"] for r in retrieved_child_chunks})
                parents = self.doc_store.mget(parent_ids)
                parents = [p for p in parents if p is not None]
                
                serialized = "\n\n".join(
                    f"Source: {doc.metadata}\nContent: {doc.page_content}"
                    for doc in parents
                )
                return serialized, parents
            except Exception as e:
                print(f"Error during document retrieval: {e}")
                return "An error occurred while retrieving documents.", [] 
            
        return retrieve_doc
    
    def _get_system_prompt(self, jurisdiction, domain):
        return system_prompt.format(jurisdiction=jurisdiction, domain=domain)
     
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

if __name__ == "__main__":
    controller = RagController()
    controller.ingest_legal_docs()