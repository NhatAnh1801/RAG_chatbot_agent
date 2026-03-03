import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from langchain_core.embeddings import Embeddings
from typing import List

class GTE(Embeddings):
    def __init__(self):
        self.model_name_or_path = 'Alibaba-NLP/gte-multilingual-base'
        self.model = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def _embedding(self, texts: List[str]) -> List[List[float]]:
        batch_dict = self.tokenizer(texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}   # Move the tensors to GPU 
        
        with torch.no_grad:
            outputs = self.model(**batch_dict) 
            
        embeddings = outputs.last_hidden_state[:, 0]    # CLS
        
        embeddings = F.normalize(embeddings, p=2, dim=1) # L2 Normalization
        
        return embeddings.tolist()
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embedding(texts)
        
    def embed_query(self, text: str) -> List[float]:
        return self._embedding([text])[0]

        