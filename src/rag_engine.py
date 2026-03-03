
from  langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from google import genai
from src.models.embeddings.gte_multi_base import GTE


from dotenv import load_dotenv
import os

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
        
        self.model = init_chat_model(
            "google_genai:gemini-flash-lite-latest",
            api_key=self.GEMINI_API_KEY
        )
        
    def load_and_process_pdf(self, file_path):
        '''
            Load the pdf file content and process it
            Args:
                file_path: Path to the pdf file
            Returns:
                Text content 
        '''
        loader = PyMuPDFLoader(file_path) 
        return loader.load()


    def split_doc(self, docs: list, chunk_size: int = 1000,  overlap: int = 200):
        '''
            - Chunk into simpler text 
        '''
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = overlap,
            length_function = len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        return chunks
    
    def index_data(self, file_path):
        '''
            - Get the file content uploaded
            - Chunk 
            - Save as metadata
        '''
        file_content = self.load_and_process_pdf(file_path)
        
        chunks = self.split_doc(file_content)
        if not chunks:
            raise ValueError(f"No content extracted from {file_path}")
    
        _= self.vector_db.add_documents(chunks)
        
        return len(chunks)
    
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
            retrieved_docs  = self.vector_db.similarity_search(query, k=3)
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
        print(f'Messages: {messages}')
        inputs = {"messages": messages}
        
        
        response = None
        for chunk in agent.stream(input=inputs, stream_mode="values"):
            response = chunk

        if response and "messages" in response:
            return response["messages"][-1].content
        return "No response"
        
    def test(self):
        pass
        # Test the retrieve_doc function
        # query = "What is this document about?"
        # serialized, retrieved_docs = self.retrieve_doc(query)
        # for i, r in enumerate(retrieved_docs):
        #     print(f'chunk: {r} at idx: {i}')
        #     print('*'*50)
        # test_file = "./src/[CV]Vuong Nhat Anh.pdf"
        # if os.path.exists(test_file):
        #     try:
        #         num_docs = self.index_data(test_file)
        #         print(f"Successfully indexed {num_docs} documents from {test_file}")
        #     except Exception as e:
        #         print(f"Error indexing file: {e}")
        # else:
        #     print(f"Test file '{test_file}' not found")
            
if __name__ == "__main__":
    rag = RagController()
    rag.test()