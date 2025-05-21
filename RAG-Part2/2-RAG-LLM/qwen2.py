import torch
from transformers import pipeline

generator = pipeline("text-generation", model="Qwen/Qwen2-0.5B-Instruct", device=0 if torch.cuda.is_available() else -1)


def generate_answer(query, retrieved_docs, max_new_tokens=50):
    """
    使用 Qwen2-0.5B-Instruct 模型根据查询和检索到的文档生成完整、通顺的回答

    Args:
        query (str): 用户的查询
        retrieved_docs (list): 检索到的文档列表（字符串列表）
        max_new_tokens (int): 生成回答的最大长度（token 数）

    Returns:
        str: 生成的回答
    """
    # 拼接查询和检索到的文档
    context = "\n".join(retrieved_docs).replace('\n', '')  # 去除里面的所有换行
    # 提示词模板
    prompt = f"根据以下上下文回答问题，答案要简洁、准确：\n上下文：{context}\n问题：{query}\n答案："
    print("--- 拼接了 提示词模板、检索到的文本块 和 query 的 完整 prompt 如下 --- \n" + prompt)

    # 使用 Qwen 模型生成回答
    response = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        truncation=True,
        pad_token_id=generator.tokenizer.eos_token_id,
        temperature=0.7
    )

    # 提取生成的回答并清理
    answer = response[0]["generated_text"].replace(prompt, "").strip()
    return answer


# 示例调用（供测试，实际使用时由检索代码调用）
if __name__ == "__main__":
    query = "法国的首都是什么？"
    retrieved_docs = [
        "法国的首都是巴黎，这座城市以其文化和历史而闻名。",
        "埃菲尔铁塔是巴黎的著名地标，建于1889年。"
    ]
    answer = generate_answer(query, retrieved_docs)
    print(f"问题：{query}")
    print(f"答案：{answer}")