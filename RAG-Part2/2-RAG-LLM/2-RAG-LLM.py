import os
import PyPDF2
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import torch

import generation_module
from qwen2 import generate_answer

# 设置你的 HuggingFace Token
os.environ["HF_TOKEN"] = "hf_XXX"
# 设置你的 PDF 文件路径
PDF_PATH = "test-pdf2.pdf"

# --- 全局设置 (模块内部) ---
DEVICE = None
DENSE_TOKENIZER = None
DENSE_MODEL = None


# --- 嵌入模型设置 ---
def setup_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    global DEVICE, DENSE_TOKENIZER, DENSE_MODEL
    if DENSE_MODEL is not None and DENSE_TOKENIZER is not None:
        return True
    try:
        DENSE_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        DENSE_MODEL = AutoModel.from_pretrained(model_name)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        DENSE_MODEL.to(DEVICE)
        print(f"稠密嵌入模型 '{model_name}' 已加载到 {DEVICE}")
        return True
    except Exception as e:
        print(f"加载稠密嵌入模型 '{model_name}' 时出错: {e}")
        return False


# --- 核心功能函数 ---
def generate_embedding_internal(text_list):
    if DENSE_TOKENIZER is None or DENSE_MODEL is None:
        if not setup_embedding_model():
            raise EnvironmentError("稠密模型未初始化且无法自动设置。")
    if not isinstance(text_list, list): text_list = [text_list]
    inputs = DENSE_TOKENIZER(text_list, padding=True, truncation=True, return_tensors="pt", max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = DENSE_MODEL(**inputs)
    attention_mask = inputs['attention_mask']
    embeddings = outputs.last_hidden_state
    expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
    sum_embeddings = torch.sum(embeddings * expanded_mask, axis=1)
    sum_mask = torch.clamp(expanded_mask.sum(axis=1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings.cpu().numpy()


def extract_text_from_pdf(pdf_path):
    text = ""
    if not os.path.exists(pdf_path):
        print(f"错误: PDF文件 '{pdf_path}' 不存在。")
        return text
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                extracted_page_text = page.extract_text()
                if extracted_page_text: text += extracted_page_text + "\n"
    except Exception as e:
        print(f"读取PDF {pdf_path} 时出错: {e}")
    return text


def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= text_len: break
        start += (chunk_size - overlap)
    return [chunk for chunk in chunks if chunk.strip()]


def initialize_retrievers(pdf_text_chunks):
    if not pdf_text_chunks:
        print("错误：文本块列表为空，无法初始化检索器。")
        return None, None
    tokenized_corpus = [doc.lower().split() for doc in pdf_text_chunks]
    bm25_retriever = BM25Okapi(tokenized_corpus)
    print("BM25 检索器已初始化。")
    if DENSE_MODEL is None:
        if not setup_embedding_model():
            print("错误：嵌入模型加载失败，无法初始化FAISS。")
            return bm25_retriever, None
    document_embeddings = generate_embedding_internal(pdf_text_chunks)
    dimension = document_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(document_embeddings)
    print(f"FAISS 索引已构建，包含 {faiss_index.ntotal} 个向量。")
    return bm25_retriever, faiss_index


def hybrid_retrieval(query_text, bm25_obj, faiss_idx, all_chunks, k_results=3, alpha=0.5):
    if not all_chunks: return []
    use_bm25 = bm25_obj is not None
    use_faiss = faiss_idx is not None and faiss_idx.ntotal > 0
    if not use_bm25 and not use_faiss: return []

    bm25_scores = np.array([0.0] * len(all_chunks))
    if use_bm25:
        tokenized_query = query_text.lower().split()
        try:
            bm25_scores_list = bm25_obj.get_scores(tokenized_query)
            if len(bm25_scores_list) == len(all_chunks): bm25_scores = np.array(bm25_scores_list)
        except Exception:
            pass

    faiss_scores_dict = {}
    if use_faiss:
        query_embedding = generate_embedding_internal([query_text])
        num_dense_candidates = min(max(k_results * 2, 10) if len(all_chunks) > 10 else len(all_chunks),
                                   faiss_idx.ntotal)
        if num_dense_candidates > 0:
            distances, faiss_indices = faiss_idx.search(query_embedding, k=num_dense_candidates)
            faiss_scores_dict = {idx: 1.0 / (1.0 + dist) for idx, dist in zip(faiss_indices[0], distances[0]) if
                                 idx != -1}

    def normalize_scores(scores_arr):
        if scores_arr.size == 0: return np.array([])
        min_s, max_s = np.min(scores_arr), np.max(scores_arr)
        return np.full_like(scores_arr, 0.5 if min_s == max_s else 0.0, dtype=float) if min_s == max_s else (
                                                                                                                        scores_arr - min_s) / (
                                                                                                                        max_s - min_s)

    norm_bm25 = normalize_scores(bm25_scores) if use_bm25 else np.array([0.0] * len(all_chunks))
    norm_faiss = np.array([0.0] * len(all_chunks))
    if use_faiss and faiss_scores_dict:
        temp_scores = np.array([faiss_scores_dict.get(i, 0.0) for i in range(len(all_chunks))])
        # 只归一化实际被FAISS检索到的（分数 > 0 的）块的分数
        relevant_indices = temp_scores > 0
        if np.any(relevant_indices):
            normalized_relevant_faiss_scores = normalize_scores(temp_scores[relevant_indices])
            norm_faiss[relevant_indices] = normalized_relevant_faiss_scores

    hybrid_scores_tuples = []
    for i in range(len(all_chunks)):
        final_score = 0.0
        s_bm25 = norm_bm25[i] if use_bm25 and i < len(norm_bm25) else 0.0
        s_faiss = norm_faiss[i] if use_faiss and i < len(norm_faiss) else 0.0

        if use_bm25 and use_faiss:
            final_score = (1 - alpha) * s_bm25 + alpha * s_faiss
        elif use_bm25:
            final_score = s_bm25
        elif use_faiss:
            final_score = s_faiss
        hybrid_scores_tuples.append((final_score, i))

    hybrid_scores_tuples.sort(key=lambda x: x[0], reverse=True)
    return [{"chunk_index": idx, "text": all_chunks[idx], "score": score} for score, idx in
            hybrid_scores_tuples[:k_results]]


# --- 主测试/演示函数 ---
def test_retrieval_and_generation_system():
    """
    测试PDF文本提取、分块、混合检索以及使用Gemini生成回答的功能。
    """
    print("--- 开始测试检索与生成系统 (使用现有PDF和Gemini) ---")

    # 1. 确保嵌入模型已加载
    if not setup_embedding_model():
        print("错误：嵌入模型加载失败，测试终止。")
        return

    # 2. 指定要加载的PDF文件路径
    pdf_path = PDF_PATH  # !!! 确保此文件存在 !!!
    print(f"尝试加载PDF文件: '{pdf_path}'")

    # 3. 提取文本和分块
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        print(f"未能从 '{pdf_path}' 提取文本或文件不存在。测试终止。")
        return

    text_chunks = chunk_text(raw_text, chunk_size=300, overlap=70)
    if not text_chunks:
        print("未能生成文本块。测试终止。")
        return
    print(f"\n从'{pdf_path}'生成了 {len(text_chunks)} 个文本块。")

    # 4. 初始化检索器
    bm25_retriever, faiss_retriever = initialize_retrievers(text_chunks)
    if bm25_retriever is None and (faiss_retriever is None or faiss_retriever.ntotal == 0):
        print("所有检索器都未能有效初始化或FAISS索引为空。测试终止。")
        return

    # 5. 执行示例检索
    sample_query = "AI Agent的自我完善功能是什么"
    print(f"\n对查询 \"{sample_query}\" 执行混合检索 (k=3, alpha=0.6):")

    retrieved_documents = hybrid_retrieval(
        query_text=sample_query,
        bm25_obj=bm25_retriever,
        faiss_idx=faiss_retriever,
        all_chunks=text_chunks,
        k_results=3,
        alpha=0.6
    )

    if retrieved_documents:
        print("\n检索到的文档块:")
        retrieved_texts_for_hf = []
        for doc in retrieved_documents:
            print(f"  块索引: {doc['chunk_index']}, 分数: {doc['score']:.4f}")
            retrieved_texts_for_hf.append(doc['text'])

        if retrieved_texts_for_hf:
            print("\n--- 使用HuggingFace Qwen 模型生成回答 ---")
            try:
                final_answer = generate_answer(
                    query=sample_query,
                    retrieved_docs=retrieved_texts_for_hf[0],  # 取检索到的第一个文本块
                    max_new_tokens=300
                )
                print("\nHuggingFace Qwen 模型生成的回答:")
                print(final_answer)
            except Exception as e:
                print(f"生成回答时出错: {e}")
        else:
            print("\n没有检索到相关文本块，跳过HuggingFace模型生成。")
    else:
        print("未能检索到任何文档。")

    print("\n--- 检索与生成系统测试结束 ---")


if __name__ == "__main__":
    test_retrieval_and_generation_system()