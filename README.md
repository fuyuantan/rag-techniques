1.Reformulate: Use **BART model** to reformulate.

```python RAG_Reformulate.py```

![1](https://github.com/user-attachments/assets/41e145b6-0979-4b71-a83b-fbfc299c0f0d)


2.Hybrid Retrieval：sparse (BM25) + dense (FAISS + Sentence Transformer Embeddings)

```pip install faiss-gpu```<br>
```python RAG_Hybrid_Retrieval.py```

![2](https://github.com/user-attachments/assets/571993af-ecf8-4828-ac2b-4f77dba5ad31)


3.Re-ranking
Use cross-encoder (ms-marco-MiniLM-L-6-v2) to count the scores of the pair data (query + initial retrieval results), and then rerank based on the scores.

```python RAG_Re-ranking.py```

![3](https://github.com/user-attachments/assets/8a06d5ba-be5d-488f-9af4-c578e7b050a8)
