from langchain.chat_models import init_chat_model
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

from dotenv import load_dotenv
from src.rag_engine import RagController
import json

# Load test_data.json as a ragas EvaluationDataset
test_path = "evaluation/dataset/test_data.json"
test_data = EvaluationDataset.from_jsonl(test_path)

import os

load_dotenv()  
 
class RagEvaluation:
    def __init__(self):
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        model = "google_genai:gemini-flash-lite-latest"
        
        llm = init_chat_model(
            model = model,
            api_key=GEMINI_API_KEY
        )
        
        self.rag_controller = RagController()
        self.evaluator_llm = LangchainLLMWrapper(llm)
        
    def func_testing(self):
        rag_controller = RagController()
        jurisdiction = "Vietnam"
        domain = "AI law"
        
        question = "Khi nào hệ thống trí tuệ nhân tạo bị coi là rủi ro cao ?"
        
        agent = rag_controller.build_legal_agent(jurisdiction, domain)
        response = rag_controller.ask(agent, question)
    
    def run_evaluation(self, sample: list):
        result = evaluate(dataset=sample,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=self.evaluator_llm)
        print(result)
        
if __name__== "__main__":
    try:
        rag_judge = RagEvaluation()
        rag_judge.func_testing()
        print("success")
    except Exception as e:
        print(f"Error: {e}")
    