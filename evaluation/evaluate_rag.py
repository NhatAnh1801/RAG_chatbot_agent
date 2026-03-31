from langchain.chat_models import init_chat_model
from ragas import EvaluationDataset

from dotenv import load_dotenv
from src.rag_engine import RagController

import os

load_dotenv()  
 
class RagEvaluation:
    def __init__(self):
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        llm_model = "google_genai:gemini-flash-lite-latest"
        
        self.llm_judge = init_chat_model(
            model = llm_model,
            api_key=GEMINI_API_KEY
        )
        
        self.rag_controller = RagController()
        
    def func_testing(self):
        rag_controller = RagController()
        jurisdiction = "Vietnam"
        domain = "AI law"
        
        question = "Khi nào hệ thống trí tuệ nhân tạo bị coi là rủi ro cao ?"
        
        agent = rag_controller.build_legal_agent(jurisdiction, domain)
        response = rag_controller.ask(agent, question)
        
if __name__== "__main__":
    try:
        rag_judge = RagEvaluation()
        rag_judge.func_testing()
        print("success")
    except Exception as e:
        print(f"Error: {e}")
    