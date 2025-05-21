# evaluate_retrieval.py
def evaluate_retrieval(query, retrieved_chunk_indices, retrieved_texts):
    """
    评估检索效果，计算命中率、召回率和精确率

    Args:
        query (str): 用户的查询
        retrieved_chunk_indices (list): 检索到的 chunk 索引列表
        retrieved_texts (list): 检索到的 chunk 文本列表

    Returns:
        dict: 包含命中率、召回率、精确率的结果
    """
    # 定义真值数据
    ground_truth_data_zh = [
        {
            "query": "什么是AI Agent？",
            "golden_chunk_indices": [0],  # 示例真值索引
            "golden_answer_keywords": ["自主", "代理", "自动化"]
        },
        {
            "query": "什么是规划功能",
            "golden_chunk_indices": [5],
            "golden_answer_keywords": ["制定", "战略计划"]
        },
        {
            "query": "什么是自我完善功能",
            "golden_chunk_indices": [7],
            "golden_answer_keywords": ["自我改进", "自适应", "自我完善"]
        }
    ]

    # 初始化结果
    result = {
        "query": query,
        "hit_rate": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "matched": False
    }

    # 查找匹配的真值数据
    for ground_truth in ground_truth_data_zh:
        if ground_truth["query"] == query:
            # 1.使用 切片的数字索引 进行评估
            golden_indices = set(ground_truth["golden_chunk_indices"])
            retrieved_indices = set(retrieved_chunk_indices)

            # 计算交集
            intersection = golden_indices.intersection(retrieved_indices)

            # 命中率：如果有至少一个正确索引，则命中
            result["hit_rate"] = 1.0 if len(intersection) > 0 else 0.0

            # 召回率：检索到的真值索引数 / 总真值索引数
            result["recall"] = len(intersection) / len(golden_indices) if len(golden_indices) > 0 else 0.0

            # 精确率：检索到的真值索引数 / 总检索索引数
            result["precision"] = len(intersection) / len(retrieved_indices) if len(retrieved_indices) > 0 else 0.0

            # 是否匹配
            result["matched"] = len(intersection) > 0

            # 2.使用 黄金/真值数据集的关键词 进行评估
            # 计算 关键词 匹配率
            if retrieved_texts:
                golden_keywords = ground_truth["golden_answer_keywords"]
                matched_keywords = 0
                for keyword in golden_keywords:
                    if any(keyword in text for text in retrieved_texts):
                        matched_keywords += 1
                result["keyword_match_rate"] = matched_keywords / len(golden_keywords) if len(
                    golden_keywords) > 0 else 0.0

            break

    return result


# 示例调用（供测试，实际使用时由检索代码调用）
if __name__ == "__main__":
    query = "什么是AI Agent？"
    retrieved_chunk_indices = [1, 2, 3]
    retrieved_texts = [
        "AI Agent 是一种智能代理，能够自主执行任务。",
        "自我完善功能包括学习和优化。",
        "其他无关内容。"
    ]
    result = evaluate_retrieval(query, retrieved_chunk_indices, retrieved_texts)
    print(f"问题：{query}")
    print(f"检索到的索引：{retrieved_chunk_indices}")
    print(f"检索到的文本：{[text[:50] + '...' for text in retrieved_texts]}")
    print(f"评估结果：{result}")