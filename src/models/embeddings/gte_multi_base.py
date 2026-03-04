import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from langchain_core.embeddings import Embeddings
from typing import List

class GTE(Embeddings):
    def __init__(self, batch_size: int=64):
        self.model_name_or_path = 'Alibaba-NLP/gte-multilingual-base'
        self.model = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True, dtype=torch.float16)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        self.batch_size = batch_size 
        
        self.model.eval()
        
    def _embedding(self, texts: List[str]) -> List[List[float]]:
        batch_dict = self.tokenizer(
            texts, 
            max_length=1024, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}   # Move the tensors to GPU 
        
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            
        embeddings = outputs.last_hidden_state[:, 0]    # CLS
        
        embeddings = F.normalize(embeddings, p=2, dim=1) # L2 Normalization
        
        # Clean up RAM
        if torch.cuda.is_available():
            del batch_dict, outputs
            torch.cuda.empty_cache()
        
        return embeddings.tolist()
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            embeddings = self._embedding(batch_texts)
            all_embeddings.extend(embeddings)
        return all_embeddings
        
    def embed_query(self, text: str) -> List[float]:
        return self._embedding([text])[0]
    
    def find_optimal_batch_size(self):
        """
            Stress test to identify the optimal batch_size on the GPU
        """
        sample_text = "test" * 1024 # max token length
        
        current_batch = 1
        optimal_batch = 1
        
        while True:
            try:
                dummy_test = [sample_text] * current_batch
                _ = self._embedding(dummy_test)
                
                optimal_batch = current_batch
                print(f"✅ Pass: batch_size = {current_batch}")
                current_batch *= 2
                
            except torch.cuda.OutOfMemoryError:
                print(f"❌ OOM Error: Out of memory at batch_size = {current_batch}.")
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"❌ OOM Error: Out of memory at batch_size = {current_batch}.")
                else:
                    raise e
                break
            finally:
                torch.cuda.empty_cache()

        print("-" * 30)
        if optimal_batch == 1 and current_batch == 1:
            print(f"⚠️ WARNING: The GPU cannot handle even batch_size = {current_batch}! Consider reducing max_seq_length.")
        else:
            print(f"🎯 RESULT: The highest safe optimal batch_size is: {optimal_batch}")
            



        