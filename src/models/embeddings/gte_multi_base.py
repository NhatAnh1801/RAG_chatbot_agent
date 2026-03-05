import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from langchain_core.embeddings import Embeddings
from typing import List
from matplotlib import pyplot as plt
import time

class GTE(Embeddings):
    def __init__(self, batch_size: int=32):
        self.model_name_or_path = 'Alibaba-NLP/gte-multilingual-base'
        self.model = AutoModel.from_pretrained(
                self.model_name_or_path, 
                trust_remote_code=True, 
                dtype=torch.float16,
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        
        self.device = torch.device("cuda")
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
        
        with torch.inference_mode():
            outputs = self.model(**batch_dict)
            
        embeddings = outputs.last_hidden_state[:, 0]    # CLS
        
        embeddings = F.normalize(embeddings, p=2, dim=1) # L2 Normalization
        
        return embeddings.tolist()
      
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            embeddings = self._embedding(batch_texts)
            all_embeddings.extend(embeddings)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return all_embeddings
        
    def embed_query(self, text: str) -> List[float]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self._embedding([text])[0]
    
    def find_optimal_batch_size(self, max_test_batch: int=256):
            """
                Stress test to identify the maximum safe batch_size on the GPU
                and plot the throughput to find the optimal performance sweet spot.
            """

            print("Starting stress test and performance benchmark...")
            sample_text = "test " * 1024 
            
            current_batch = 1
            
            tested_batches = []
            throughputs = []
            
            TIMEOUT_SECONDS = 10.0  # Prevent GPU uses RAM when run out of VRAM 
            while current_batch <= max_test_batch:
                try:
                    dummy_test = [sample_text] * current_batch
                    
                    # 1. Warm-up: Run once to get the GPU engines running (crucial for accurate timing)
                    warmup_start = time.time()
                    _ = self._embedding(dummy_test)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize() # Force Python to wait until GPU finishes
                    
                    warmup_time = time.time() - warmup_start
                    
                    # Check if the program is using system RAM
                    if warmup_time > TIMEOUT_SECONDS:
                        print(f"Execution time is too long ({warmup_time:.2f}s). The GPU appears to be out of VRAM and is relying on system RAM, which will significantly degrade performance.")
                        break 
                    
                    # 2. Benchmark: Measure the time across a few runs to get a stable average
                    num_runs = 3
                    start_time = time.time()
                    
                    for _ in range(num_runs):
                        _ = self._embedding(dummy_test)
                        
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.time()
                    
                    # 3. Calculate metrics
                    avg_time = (end_time - start_time) / num_runs
                    throughput = current_batch / avg_time # How many chunks processed per second
                    
                    tested_batches.append(current_batch)
                    throughputs.append(throughput)
                    
                    print(f"✅ Pass: batch_size = {current_batch:<4} | Speed: {throughput:.2f} samples/sec")
                    
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
            
            if not tested_batches:
                print("⚠️ WARNING: The GPU cannot handle even batch_size = 1! Consider reducing max_seq_length.")
                return
                
            # Find the batch size that yielded the highest throughput
            best_throughput = max(throughputs)
            best_perf_batch = tested_batches[throughputs.index(best_throughput)]
            
            
            # Update the class attribute to use the absolute best batch size from now on
            self.batch_size = best_perf_batch
            
            # ================= PLOTTING THE RESULTS =================
            plt.figure(figsize=(10, 6))
            plt.plot(tested_batches, throughputs, marker='o', linestyle='-', color='b', linewidth=2)
            
            # Highlight the optimal batch size with a red star
            plt.plot(best_perf_batch, best_throughput, marker='*', color='r', markersize=15, label=f'Sweet Spot (BS={best_perf_batch})')
            
            plt.title('GPU Throughput vs Batch Size', fontsize=14)
            plt.xlabel('Batch Size (Log Scale)', fontsize=12)
            plt.ylabel('Throughput (Samples / Second)', fontsize=12)
            
            # Use log scale for X-axis since we multiplied by 2 each time
            plt.xscale('log', base=2)
            plt.xticks(tested_batches, labels=[str(b) for b in tested_batches]) 
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig('batch_size_benchmark.png')
            plt.show()
        



        