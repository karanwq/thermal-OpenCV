import os
os.environ["HF_HOME"] = "C:/hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import time
import torch
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions
from flashrank import Ranker, RerankRequest
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

class NativeEnterpriseRAGEngine:
    def __init__(self):
        print("[INFO] Initializing Native Local RAG Infrastructure...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"[LLM] Loading {model_id} natively onto GPU via CUDA...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=150,
            temperature=0.01,
            do_sample=False
        )
        
        self.chroma_client = chromadb.Client()
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
        self.vector_db = self.chroma_client.create_collection(
            name="local_enterprise_wiki", 
            embedding_function=self.emb_fn
        )
        
        self.reranker = Ranker()
        self.raw_documents = []
        self.bm25 = None

    def ingest_knowledge_base(self, documents: list):
        """Indexes raw text data into both the sparse keyword engine and vector matrix."""
        self.raw_documents = documents
        tokenized_corpus = [doc.lower().split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        string_ids = [str(i) for i in range(len(documents))]
        self.vector_db.add(documents=documents, ids=string_ids)
        print(f"[SUCCESS] Enterprise Index Synchronized. Nodes loaded: {len(documents)}")

    def run_hybrid_retrieval(self, query: str, top_k=4) -> list:
        """Executes parallel matches across semantic vectors and exact keywords."""
        vector_results = self.vector_db.query(query_texts=[query], n_results=top_k)
        dense_candidates = vector_results['documents'][0] if vector_results['documents'] else []
        
        tokenized_query = query.lower().split(" ")
        sparse_candidates = self.bm25.get_top_n(tokenized_query, self.raw_documents, n=top_k)        
        return list(set(dense_candidates + sparse_candidates))

    def run_reranking_pipeline(self, query: str, candidate_pool: list, target_count=2) -> list:
        """Filters out non-relevant context items using the cross-encoder."""
        formatted_passages = [{"id": idx, "text": text} for idx, text in enumerate(candidate_pool)]
        request = RerankRequest(query=query, passages=formatted_passages)
        rerank_results = self.reranker.rerank(request)
        return [result['text'] for result in rerank_results[:target_count]]

    def execute_secured_query(self, query: str) -> str:
        """Orchestrates hybrid retrieval, filters noise, formats system prompts, and fires inference."""
        start_time = time.time()
        
        candidates = self.run_hybrid_retrieval(query, top_k=4)
        filtered_context = self.run_reranking_pipeline(query, candidates, target_count=2)
        context_block = "\n---\n".join(filtered_context)
        
        messages = [
            {"role": "system", "content": (
                "You are an advanced data-grounded assistant. Answer the user query using ONLY "
                "the text provided below. If the answer cannot be found in the text, reply with exactly "
                "'ERROR: INS_CONTEXT_UNAVAILABLE'. Do not invent details or use background info."
            )},
            {"role": "user", "content": f"DOCUMENTATION CONTEXT:\n{context_block}\n\nQUERY: {query}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = self.llm_pipeline(prompt)
        generated_text = outputs[0]["generated_text"]
        
        final_answer = generated_text[len(prompt):].strip()
        
        print(f"\n[METRICS] Execution Latency: {time.time() - start_time:.2f}s")
        return final_answer

if __name__ == "__main__":
    mock_datacenter_logs = [
        "Telemetry cluster Gamma-6 reported Critical Error 901 indicating a cooling pump failure at 14:22.",
        "System firmware upgrade schedule states all worker nodes must be moved to v4.12 by Q3.",
        "The emergency contact number for site security engineering operations is 555-0199.",
        "Project Phoenix internal repository credentials must be renewed every 60 calendar days."
    ]
    
    rag_engine = NativeEnterpriseRAGEngine()
    rag_engine.ingest_knowledge_base(mock_datacenter_logs)
    
    q1 = "What does Critical Error 901 mean on Gamma-6?"
    print(f"\nQuery: {q1}\nAnswer: {rag_engine.execute_secured_query(q1)}")
    
    q2 = "What is the budget for Project Phoenix?"
    print(f"\nQuery: {q2}\nAnswer: {rag_engine.execute_secured_query(q2)}")