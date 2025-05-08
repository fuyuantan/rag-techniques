from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import torch
import os
# Add your hf_token
os.environ["HF_TOKEN"] = "hf_your_token"

dense_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
dense_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def generate_embedding(text):
    """Generate dense vector using mean pooling"""
    inputs = dense_tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = dense_model(**inputs)

    attention_mask = inputs['attention_mask']
    embeddings = outputs.last_hidden_state

    expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
    sum_embeddings = torch.sum(embeddings * expanded_mask, axis=1)
    sum_mask = torch.clamp(expanded_mask.sum(axis=1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings.cpu().numpy()


# Sample document collection
documents = [
    "Transformers use self-attention mechanisms to process input sequences in "
    "parallel, making them efficient for long sequences.",
    "The attention mechanism in transformers allows the model to focus on different "
    "parts of the input sequence when generating each output element.",
    "Transformer models have a fixed context length determined by the positional "
    "encoding and self-attention mechanisms.",
    "To handle sequences longer than the context length, transformers can use "
    "techniques like sliding windows or hierarchical processing.",
    "Recurrent Neural Networks (RNNs) process sequences sequentially, which can be "
    "inefficient for long sequences.",
    "Long Short-Term Memory (LSTM) networks are a type of RNN designed to handle "
    "long-term dependencies in sequences.",
    "The Transformer architecture was introduced in the paper 'Attention Is All "
    "You Need' by Vaswani et al.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a "
    "transformer-based model designed for understanding the context of words.",
    "GPT (Generative Pre-trained Transformer) is a transformer-based model designed "
    "for natural language generation.",
    "Transformer-XL extends the context length of transformers by using a "
    "segment-level recurrence mechanism."
]

# Prepare for sparse retrieval (BM25)
tokenized_corpus = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# Prepare for dense retrieval (FAISS)
document_embeddings = generate_embedding(documents)
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)


def hybrid_retrieval(query, k=3, alpha=0.5):
    """Hybrid retrieval: Use both the BM25 and L2 index on FAISS"""
    # Sparse score of each document with BM25
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalize BM25 scores to [0,1] unless all elements are zero
    if max(bm25_scores) > 0:
        bm25_scores = bm25_scores / max(bm25_scores)

    # Sort all documents according to L2 distance to query
    query_embedding = generate_embedding(query)
    distances, indices = index.search(query_embedding, len(documents))

    # Dense score: 1/distance as similarity metric, then normalize to [0,1]
    eps = 1e-5  # a small value to prevent division by zero
    dense_scores = 1 / (eps + np.array(distances[0]))
    dense_scores = dense_scores / max(dense_scores)

    # Combine scores = affine combination of sparse and dense scores
    combined_scores = alpha * dense_scores + (1 - alpha) * bm25_scores

    # Get top-k documents
    top_indices = np.argsort(combined_scores)[::-1][:k]
    results = [(documents[idx], combined_scores[idx]) for idx in top_indices]
    return results


# Retrieve documents using hybrid retrieval
query = "How do transformers handle long sequences?"
results = hybrid_retrieval(query)
print(f"Query: {query}")
for i, (doc, score) in enumerate(results):
    print(f"Document {i + 1} (Score: {score:.4f}):")
    print(doc)
    print()