from transformers import BartForConditionalGeneration, BartTokenizer # Use BartForConditionalGeneration
import torch
import os
# Add your hf_token
os.environ["HF_TOKEN"] = "hf_your_token"

# Load BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large") # Use BartForConditionalGeneration

# Optional: Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def reformulate_query(query, n=2):
    inputs = tokenizer(query, return_tensors="pt").to(device)
    outputs = model.generate( # This line requires model to be an instance of a class with a generation head
        **inputs,
        max_length=64,
        num_beams=10,
        num_return_sequences=n,
        temperature=1.5,
        top_k=50,
        do_sample=True
    )
    reformulations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    all_queries = [query] + reformulations
    return all_queries

query = "How do transformer-based systems process natural language?"
reformulated_queries = reformulate_query(query)
print(f"Original Query: {query}")
print("Reformulated Queries:")
for i, q in enumerate(reformulated_queries[1:], 1):
    print(f"{i}. {q}")