import traceback
from ragas import EvaluationDataset, evaluate
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from dotenv import load_dotenv
from google import genai
from src.rag_engine import RagController

import asyncio
import json
import os

load_dotenv()  

evaluation_dataset_path = "evaluation/dataset/evaluation_dataset.json"
validation_dataset_path = "evaluation/dataset/validation_dataset.json"
 
class RagEvaluator:
    def __init__(self):
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        os.getenv("GEMINI_API_KEY")
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        self.llm = llm_factory(
            "gemini-flash-lite-latest",
            client=client,
            provider="google",
            use_async=True
        )
        print(f"self.llm: {self.llm}")
        self.rag_controller = RagController()
        
        with open(validation_dataset_path, "r", encoding="utf-8") as f:
            self.test_data = json.load(f)    
            
    def generate_legal_evaluation_samples(self, evaluation_dataset: list[dict]) -> list:
        evaluation_samples = []
        unique_domains = set(item["domain"] for item in evaluation_dataset)
        for domain in unique_domains:
            agent = self.rag_controller.build_legal_agent("Vietnam", domain)
            
            domain_questions = [item for item in evaluation_dataset if item["domain"] == domain]
            
            for index, item in enumerate(domain_questions):
                print(f"Running in item number: {index} from {domain}")
                result = self.rag_controller.ask(agent, item["question"])
                print(f"result: {result}")
                evaluation_samples.append({
                    "user_input":          item["question"],
                    "response":            result["answer"],
                    "retrieved_contexts":  result["contexts"],
                    "reference":           item["ground_truth"]
                })  

        return evaluation_samples
    
    async def _run_legal_evaluation(self, evaluation_dataset: list) -> list:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        llm = llm_factory(
            "gemini-flash-lite-latest",
            client=client,
            provider="google",
            use_async=True
        )
        embeddings = embedding_factory("google", model="gemini-embedding-001")
        
        faith_metric     = Faithfulness(llm=llm)
        relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
        precision_metric = ContextPrecision(llm=llm)
        recall_metric    = ContextRecall(llm=llm)

        evaluation_samples = self.generate_legal_evaluation_samples(evaluation_dataset)
        
        evaluation_results = []
        try:
            for i, row in enumerate(evaluation_samples):
                print(f"\nScoring {i+1}/{len(evaluation_samples)} elements...")
                
                faith     =  await faith_metric.ascore(user_input=row["user_input"], response=row["response"], retrieved_contexts=row["retrieved_contexts"])
                relevancy =  await relevancy_metric.ascore(user_input=row["user_input"], response=row["response"])
                precision =  await precision_metric.ascore(user_input=row["user_input"], retrieved_contexts=row["retrieved_contexts"], reference=row["reference"])
                recall    =  await recall_metric.ascore(user_input=row["user_input"], retrieved_contexts=row["retrieved_contexts"], reference=row["reference"])

                print(f"reference: {row["reference"]}")
                evaluation_results.append({
                    "question":          row["user_input"],
                    "faithfulness":      faith,
                    "answer_relevancy":  relevancy,
                    "context_precision": precision,
                    "context_recall":    recall,
                })
        finally:
            await client.aio.aclose()
            
        return evaluation_results
    
    def run_legal_evaluation(self, evaluation_dataset: list):
        evaluation_samples = self.generate_legal_evaluation_samples(evaluation_dataset)

        test_data = EvaluationDataset.from_list(evaluation_samples)
        
        result = evaluate(
            dataset=test_data,
            metrics=[
                Faithfulness(llm=self.evaluator_llm),
                ContextRecall(llm=self.evaluator_llm),
                ContextPrecision(llm=self.evaluator_llm),
                AnswerRelevancy(llm=self.evaluator_llm, embeddings=self.evaluator_embeddings)
            ],
        )
        
        print(result)
        return result.to_pandas()
    
if __name__== "__main__":
    try:
        evaluator = RagEvaluator()
        # results = evaluator.run_legal_evaluation(evaluator.test_data)
        result = asyncio.run(evaluator._run_legal_evaluation(evaluator.test_data))
        print(result)
        print("success")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    