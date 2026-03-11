from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_docling.loader import DoclingLoader
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorDevice, EasyOcrOptions, AcceleratorOptions
from docling.document_converter import PdfFormatOption
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyMuPDFLoader

from langchain_classic.storage import InMemoryStore
from langchain_chroma import Chroma

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from models.embeddings.gte_multi_base import GTE
from dotenv import load_dotenv

import os
import fitz
import time
import torch

load_dotenv()
class RagController:
    def __init__(self):
        '''
            Init vector DB and LLM here
        '''
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
            
        self.embedding_model = GTE()
        
        self.vector_db = Chroma(
            embedding_function=self.embedding_model,
            persist_directory='./data/chromadb'    
        )
        self.small2big_retriever = None
        
        self.store = InMemoryStore()    # Store in RAM (will be replaced in the future)
        
        self.model = init_chat_model(
            "google_genai:gemini-flash-lite-latest",
            api_key=self.GEMINI_API_KEY
        )
        
    def is_text_extractable(self, docs, threshold=200):
            '''
                Checks if the extracted documents have enough text per page.
            '''
            if not docs:
                return False
                
            try:
                total_chars = sum(len(d.page_content) for d in docs)
                avg_chars = total_chars / len(docs)
                return avg_chars > threshold
            
            except Exception as e:
                print(f"Error checking text length: {e}")
                return False
        
    def load_and_process_pdf(self, file_path):
        '''
            Load the pdf file content and process it
            Args:
                file_path: Path to the pdf file
            Returns:
                Text content 
        '''
        print(f"--> [TIMING] Attempting PyMuPDF extraction on {file_path}...")
        start_pymupdf = time.perf_counter()
        
        try:
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
        except Exception as e:
            print(f"PyMuPDF failed completely: {e}")
            docs = []
        
        end_pymupdf = time.perf_counter()
            
        if self.is_text_extractable(docs):
            print(f"--> [SUCCESS] Native PDF detected. Extraction finished in {end_pymupdf - start_pymupdf:.4f} seconds.")
            return docs
        
        # Fallback: Using OCR for scanned PDF file
        use_gpu = torch.cuda.is_available()
        print(f"GPU Available for OCR: {use_gpu}")
        device = AcceleratorDevice.CUDA if use_gpu else AcceleratorDevice.CPU
        
        pipeline_options = PdfPipelineOptions(
            allow_external_plugins=True,
            do_ocr = True,
            accelerator_options = AcceleratorOptions(device=device)
            )
        
        pipeline_options.ocr_options = EasyOcrOptions(
            lang=["vi", "en"],
        )
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        loader = DoclingLoader(file_path, converter=converter)
        
        docs = []
        
        try:
            print(f"--> [TIMING] Scanned PDF detected. Starting Docling OCR on {file_path}. This may take a while...")
            start_ocr = time.perf_counter()
            
            doc_iter = loader.lazy_load()
            for doc in doc_iter:
                docs.append(doc)
                
            end_ocr = time.perf_counter()
            print(f"--> [TIMING] Docling OCR finished in {end_ocr - start_ocr:.4f} seconds.")
        except Exception as e:
            print(f"Error loading document(s) from {file_path}: {e}")
            
        return docs
    
    def ingest_docs(self, docs: list, parent_chunk_size: int = 1000, child_chunk_size: int=200, batch_size = 200):
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
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size = parent_chunk_size,
            length_function = len,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size = child_chunk_size,
            length_function = len,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.small2big_retriever = ParentDocumentRetriever(
            vectorstore=self.vector_db,
            docstore=self.store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )
        
        print(f"--> [TIMING] Starting embedding generation and ChromaDB ingestion for {len(docs)} documents...")
        
        start_embed = time.perf_counter()
        # Add data top retriever
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            self.small2big_retriever.add_documents(batch)
        
        end_embed = time.perf_counter()
        print(f"--> [TIMING] Embedding and ingestion finished in {end_embed - start_embed:.4f} seconds.")
          
    def index_data(self, file_path):
        file_content = self.load_and_process_pdf(file_path)
        clean_docs = filter_complex_metadata(file_content)
        for idx, doc in enumerate(clean_docs[:5], 1):
            print(f'Doc {idx} page content:\n{doc.page_content}\n{"-" * 50}')
        self.ingest_docs(clean_docs)
        
        return len(clean_docs)
    
    def ask(self, question, history=None, max_turns=5):
        @tool(response_format="content_and_artifact")
        def retrieve_doc(query:str) -> tuple:
            """
            Search the vector database for documents relevant to the user's question.
            Args:
                query: The user's question or search query.
            Returns:
                A tuple of (serialized text, list of Document objects).
            """
            retrieved_docs  = self.small2big_retriever.invoke(query)
            serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        
        doc_count = self.vector_db._collection.count()
        
        system_prompt = f"""You are a helpful assistant with access to a document retrieval tool
        ## Context
        - The user has uploaded documents to a knowledge base ({doc_count} chunks indexed).
        - You will receive a conversation history followed by the user's current question.
        
        ## Message Format
        You will receive messages in this order:
        - Previous conversation turns (for context)
        - The current user question (LAST message) — this is what you need to answer    
        
        ## Constraint
        - retrieve_doc tool to search the knowledge base FIRST, then answer based on what you find.
        - If the retrieved documents do not contain relevant information to answer the question, clearly state: "I couldn't find this information in the uploaded documents." Do not fabricate answers or rely on external knowledge outside the provided documents.
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

        if response and "messages" in response:
            return response["messages"][-1].content
        return "No response"
        
    def test(self):
        # Test the retrieve_doc function
        #test_file = r"D:\Pycharm projects\RAG_agent\src\congress.pdf"
        test_file = r"D:\Pycharm projects\RAG_agent\src\tri_tue_nhan_tao.pdf"
        query = "what is author name?"
        if os.path.exists(test_file):
            try:

                # Time the indexing step
                start_index = time.perf_counter()
                num_docs = self.index_data(test_file)
                end_index = time.perf_counter()
                print(f"Successfully indexed {num_docs} documents from {test_file} (Time taken: {end_index - start_index:.4f} seconds)")

                # # Test asking a query and time it
                # history = []
                # start_ask = time.perf_counter()
                # response = self.ask(query, history)
                # end_ask = time.perf_counter()
                # print(f'Query: {query}\nResponse: {response}')
                # print(f"ask() function time taken: {end_ask - start_ask:.4f} seconds")
            except Exception as e:
                print(f"Error indexing file: {e}")
        else:
            print(f"Test file '{test_file}' not found")
            
if __name__ == "__main__":
    rag = RagController()
    rag.test()