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

import os
import requests
import time

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
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
            
        self.embedding_model = GTE()
        
        self.model = init_chat_model(
            "google_genai:gemini-flash-lite-latest",
            api_key=self.GEMINI_API_KEY
        )
        
        # INIT DATABASE
        self.vector_db = Chroma(
            embedding_function=self.embedding_model,
            persist_directory='./data/chromadb'    
        )
        
        # INIT SMALL2BIG
        self.small2big_retriever = None
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
            parent_splitter=parent_splitter
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
            
            print(f"-> response: {response}")
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
        
    def ask(self, question, jurisdiction, domain, history=None, max_turns=5):
        @tool(response_format="content_and_artifact")
        def retrieve_doc(query:str) -> tuple:
            """
            Query documents from the user's question.
            Args:
                query: The user's question or search query.
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
        
        system_prompt = f"""You are an expert Legal AI Assistant specializing in {jurisdiction} law, specifically within the {domain} domain.
        ## Context
        - You will receive a conversation history followed by the user's current legal question.
        - You MUST answer the question accurately based ONLY on the provided legal documents.
        - You have access to specialized tool_call actions, which should be leveraged to retrieve relevant documents and evidence necessary for answering user questions.
        
        ## Message Format
        You will receive messages in this order:
        - Previous conversation turns (for context)
        - The current user question (LAST message) -> this is what you need to answer    
        
        ## Constraint
        - Whenever you need to reference or search legal information in order to answer the user's question, you MUST use the retrieve_doc tool to retrieve the relevant documents first.
        - If the retrieved documents do not contain relevant information to answer the question, clearly state: "I couldn't find this information in the {jurisdiction}: {domain} documents."
        - If no documents are retrieved, clearly respond: "Sorry, I don't have information regarding legal matters in the {jurisdiction}: {domain} documents."
        - Do not fabricate laws, precedents, or rely on external knowledge outside the provided documents.
        
        ## Output format
        Strictly follow this output format:
        ```
        From my legal database, my answer is:
        [INSERT YOUR ANSWER HERE]
        ```
        """
        
        tools = [retrieve_doc]
        
        agent = create_agent(
            model=self.model,
            tools=tools,
            system_prompt=system_prompt
        )
        
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

        messages = response["messages"]

        # Extract key messages
        human_msg   = messages[0]
        tool_call   = messages[1]  # AIMessage with function_call
        tool_result = messages[2]  # ToolMessage with retrieved docs
        final_ans   = messages[-1] # Final AIMessage

        # ─── Query ────────────────────────────────────────────
        print(f"-> [ask]: Query: {human_msg.content}")

        # ─── Retrieval ────────────────────────────────────────
        print(f"-> [ask]: Retrieved {len(tool_result.artifact)} chunks | "
            f"Sources: {set(d.metadata['source'] for d in tool_result.artifact)} | "
            f"Pages: {[d.metadata['page'] for d in tool_result.artifact]}")

        # ─── Token Usage ──────────────────────────────────────
        t1 = tool_call.usage_metadata
        t2 = final_ans.usage_metadata
        print(f"-> [ask]: Tokens | "
            f"Call 1: {t1['total_tokens']} | "
            f"Call 2: {t2['total_tokens']} | "
            f"Total: {t1['total_tokens'] + t2['total_tokens']}")
        
        if response and "messages" in response:
            return response["messages"][-1].content
        return "No response"
        
    def _test(self):
        print("\n" + "="*50)
        print("🚀 STARTING RAG CONTROLLER TEST")
        print("="*50)
        
        # 2. Run the ingestion pipeline
        print("\n⚙️  STEP 1: Ingesting Documents...")
        try:
            self.ingest_legal_docs()
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            return

        # 3. Test the Agent Query
        print("\n🤖 STEP 2: Testing the LLM Agent...")
        
        # Test parameters that match the expected folder/file structure
        test_jurisdiction = "VietNam"
        test_domain = "AI Law"
        test_question = "Khi nào hệ thống trí tuệ nhân tạo bị coi là rủi ro cao ?"
        
        print(f"   Jurisdiction: {test_jurisdiction}")
        print(f"   Domain:       {test_domain}")
        print(f"   Question:     {test_question}")
        print("\nThinking...")

        try:
            response = self.ask(
                question=test_question,
                jurisdiction=test_jurisdiction,
                domain=test_domain
            )
            
            print("\n" + "="*50)
            print("🎯 AGENT RESPONSE:")
            print("="*50)
            print(response)
            
        except Exception as e:
            print(f"\n❌ Query failed: {e}")
            
    def _test_process_pdf(self):
        self.ingest_legal_docs()
        
def check_connection():
    response = requests.get(
        f"{CLOUDFLARE_URL}/health",
        timeout=60
    )
    print(response.text)
        
# if __name__ == "__main__":
#     rag = RagController()
#     rag._test_process_pdf()
