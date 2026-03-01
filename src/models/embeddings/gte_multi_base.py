import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class GTE:
    def __init__(self):
        self.model_name_or_path = 'Alibaba-NLP/gte-multilingual-base'
        self.model = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
    
    def encode(self, text: str, dimension: int=768):
        '''
            - The output dimension of the output embedding, should be in [128, 768]
        '''
        # Tokenize the input texts
        batch_dict = self.tokenizer(text, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.device) for k, v in self.batch_dict.items()}   # Move the tensors to GPU 
        
        outputs = self.model(**batch_dict)
        
        embeddings = outputs.last_hidden_state[:, 0][:dimension]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        
    def similarity_search():
        pass