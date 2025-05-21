import os
import PyPDF2
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import torch
# 导入评估函数
from evaluate_retrieval import evaluate_retrieval

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
    """初始化稠密嵌入模型和分词器"""
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
            raise EnvironmentError("稠密模型未初始化且无法自动设置。请先调用 setup_embedding_model()")
    if not isinstance(text_list, list):
        text_list = [text_list]
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
                if extracted_page_text:
                    text += extracted_page_text + "\n"
    except Exception as e:
        print(f"读取PDF {pdf_path} 时出错: {e}")
    return text


def chunk_text(text, chunk_size=150, overlap=50):
    """分块文本并为每个切片分配索引"""
    chunks = []
    start = 0
    text_len = len(text)
    index = 0
    while start < text_len:
        end = start + chunk_size
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append({"index": index, "text": chunk_text})
            index += 1
        if end >= text_len:
            break
        start += (chunk_size - overlap)
    return chunks


def initialize_retrievers(pdf_text_chunks):
    if not pdf_text_chunks:
        print("错误：文本块列表为空，无法初始化检索器。")
        return None, None
    # 提取文本内容用于 BM25
    chunk_texts = [chunk["text"] for chunk in pdf_text_chunks]
    tokenized_corpus = [doc.lower().split() for doc in chunk_texts]
    bm25_retriever = BM25Okapi(tokenized_corpus)
    print("BM25 检索器已初始化。")
    if DENSE_MODEL is None:
        if not setup_embedding_model():
            print("错误：嵌入模型加载失败，无法初始化FAISS。")
            return bm25_retriever, None
    document_embeddings = generate_embedding_internal(chunk_texts)
    dimension = document_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(document_embeddings)
    print(f"FAISS 索引已构建，包含 {faiss_index.ntotal} 个向量。")
    return bm25_retriever, faiss_index


def hybrid_retrieval(query_text, bm25_obj, faiss_idx, all_chunks, k_results=3, alpha=0.5):
    if not all_chunks:
        print("警告: 文本块列表为空，无法进行检索。")
        return []

    use_bm25 = bm25_obj is not None
    use_faiss = faiss_idx is not None and faiss_idx.ntotal > 0

    if not use_bm25 and not use_faiss:
        print("警告: BM25 和 FAISS 检索器均不可用。")
        return []

    chunk_texts = [chunk["text"] for chunk in all_chunks]
    bm25_scores = np.array([0.0] * len(all_chunks))

    if use_bm25:
        tokenized_query = query_text.lower().split()
        try:
            bm25_scores_list = bm25_obj.get_scores(tokenized_query)
            if len(bm25_scores_list) == len(all_chunks):
                bm25_scores = np.array(bm25_scores_list)
            else:
                print(
                    f"警告: BM25返回的分数数量 ({len(bm25_scores_list)}) 与文本块数量 ({len(all_chunks)}) 不匹配。BM25分数可能不准确。")
        except Exception as e:
            print(f"BM25检索时出错: {e}。BM25分数将全为0。")

    faiss_scores_dict = {}
    if use_faiss:
        query_embedding = generate_embedding_internal([query_text])
        num_dense_candidates = max(k_results * 2, 10) if len(all_chunks) > 10 else len(all_chunks)
        num_dense_candidates = min(num_dense_candidates, faiss_idx.ntotal)
        if num_dense_candidates > 0:
            distances, faiss_indices = faiss_idx.search(query_embedding, k=num_dense_candidates)
            faiss_scores_dict = {idx: 1.0 / (1.0 + dist) for idx, dist in zip(faiss_indices[0], distances[0]) if
                                 idx != -1}
        else:
            print("警告: FAISS 候选数量为0或索引为空，跳过稠密检索。")

    def normalize_scores(scores_array_or_list):
        scores_array = np.array(scores_array_or_list)
        if scores_array.size == 0:
            return np.array([])
        min_score = np.min(scores_array)
        max_score = np.max(scores_array)
        if max_score == min_score:
            return np.full_like(scores_array, 0.5 if min_score == max_score else 0.0, dtype=float)
        return (scores_array - min_score) / (max_score - min_score)

    norm_bm25_scores = np.array([0.0] * len(all_chunks))

    if use_bm25:
        norm_bm25_scores = normalize_scores(bm25_scores)
    norm_faiss_scores = np.array([0.0] * len(all_chunks))

    if use_faiss and faiss_scores_dict:
        retrieved_faiss_values_for_norm = []
        indices_for_faiss_norm = []
        for i in range(len(all_chunks)):
            if i in faiss_scores_dict:
                retrieved_faiss_values_for_norm.append(faiss_scores_dict[i])
                indices_for_faiss_norm.append(i)
        if retrieved_faiss_values_for_norm:
            normalized_retrieved_faiss = normalize_scores(retrieved_faiss_values_for_norm)
            for original_idx, norm_score in zip(indices_for_faiss_norm, normalized_retrieved_faiss):
                norm_faiss_scores[original_idx] = norm_score

    hybrid_scores_tuples = []
    for i in range(len(all_chunks)):
        final_score = 0.0
        has_bm25_score = use_bm25 and norm_bm25_scores.size > 0 and i < len(norm_bm25_scores)
        has_faiss_score = use_faiss and norm_faiss_scores.size > 0 and i < len(norm_faiss_scores)
        if has_bm25_score and has_faiss_score:
            final_score = (1 - alpha) * norm_bm25_scores[i] + alpha * norm_faiss_scores[i]
        elif has_bm25_score:
            final_score = norm_bm25_scores[i]
        elif has_faiss_score:
            final_score = norm_faiss_scores[i]
        hybrid_scores_tuples.append((final_score, i))

    hybrid_scores_tuples.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, doc_idx in hybrid_scores_tuples[:k_results]:
        results.append(
            {"chunk_index": all_chunks[doc_idx]["index"], "text": all_chunks[doc_idx]["text"], "score": score})
    return results


def test_retrieval_system():
    """测试PDF文本提取、分块、混合检索和评估功能"""
    print("--- 开始测试检索系统 (使用现有PDF) ---")
    if not setup_embedding_model():
        print("错误：嵌入模型加载失败，测试终止。")
        return []
    pdf_path = PDF_PATH
    print(f"尝试加载PDF文件: '{pdf_path}'")
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        print(f"未能从 '{pdf_path}' 提取文本或文件不存在。测试终止。")
        return []
    text_chunks = chunk_text(raw_text, chunk_size=150, overlap=50)
    if not text_chunks:
        print("未能生成文本块。测试终止。")
        return []
    print(f"\n从'{pdf_path}'生成了 {len(text_chunks)} 个文本块。")

    output_txt_path = "retrieved_text_chunks.txt"
    try:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            for chunk in text_chunks:
                f.write(f"--- Chunk {chunk['index']} ---\n")
                f.write(chunk['text'])
                f.write("\n\n--------------------\n\n")
        print(f"所有文本块已按顺序保存到: {output_txt_path}")
    except Exception as e:
        print(f"保存文本块到文件时出错: {e}")

    if text_chunks:
        print("前3个文本块示例:")
        for chunk in text_chunks[:3]:
            print(f"  块 {chunk['index']}: \"{chunk['text'][:100].replace(chr(10), ' ')}...\"")
    bm25_retriever, faiss_retriever = initialize_retrievers(text_chunks)
    if bm25_retriever is None and (faiss_retriever is None or faiss_retriever.ntotal == 0):
        print("所有检索器都未能有效初始化或FAISS索引为空。测试终止。")
        return []

    sample_query = "什么是AI Agent？"
    print(f"\n对查询 \"{sample_query}\" 执行混合检索 (k=3, alpha=0.5):")
    retrieved_documents = hybrid_retrieval(
        query_text=sample_query,
        bm25_obj=bm25_retriever,
        faiss_idx=faiss_retriever,
        all_chunks=text_chunks,
        k_results=3,
        alpha=0.5
    )
    if retrieved_documents:
        print("\n检索到的文档块:")
        for doc in retrieved_documents:
            print(f"  块索引: {doc['chunk_index']}")
            print(f"  分数: {doc['score']:.4f}")
            print(f"  文本: \"{doc['text'][:150].replace(chr(10), ' ')}...\"")
            print("-" * 20)
    else:
        print("未能检索到任何文档。")

    # 评估检索结果
    retrieved_indices = [doc["chunk_index"] for doc in retrieved_documents]
    retrieved_texts = [doc["text"] for doc in retrieved_documents]
    eval_result = evaluate_retrieval(sample_query, retrieved_indices, retrieved_texts)

    print("\n--- 检索评估结果 ---")
    print(f"问题：{eval_result['query']}")
    print(f"命中率：{eval_result['hit_rate']:.4f}")
    print(f"召回率：{eval_result['recall']:.4f}")
    print(f"精确率：{eval_result['precision']:.4f}")
    print(f"是否命中：{eval_result['matched']}")

    print(f"关键词匹配率：{eval_result['keyword_match_rate']:.4f}")
    print("\n--- 检索系统测试结束 ---")
    return retrieved_documents


if __name__ == "__main__":
    test_retrieval_system()