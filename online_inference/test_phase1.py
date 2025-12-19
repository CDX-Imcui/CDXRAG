import json
from table_pipeline import TableRAGPipeline
import pandas as pd
import torch
import re
from utils.tool_utils import Embedder

def construct_table_text_pairs(pipeline, top_k: int = 3, score_threshold: float = 0.45) -> pd.DataFrame:
    """
    核心功能：构造表-文对 (Table-Text Alignment)
    逻辑：利用向量相似度矩阵，为每一行表格找到最匹配的 Top-K 文本片段。

    Args:
        pipeline: 已经执行过 build_index 的 TableRAGPipeline 实例
        top_k: 每一行保留几个最相关的文本块
        score_threshold: 相似度阈值，低于此分数的文本会被丢弃
    """
    print("\n=== 🔗 Phase 3: Constructing Table-Text Pairs ===")

    if pipeline.table_embeddings is None or pipeline.text_embeddings is None:
        raise ValueError("❌ Index not built! Please run pipeline.build_index() first.")

    # 1. 计算全局相似度矩阵 [Num_Rows, Num_Texts]
    # 这一步非常快，完全在 GPU 上并行
    with torch.no_grad():
        similarity_matrix = torch.matmul(
            pipeline.table_embeddings,
            pipeline.text_embeddings.t()
        )

    aligned_data = []

    # 2. 遍历每一行进行对齐
    for row_idx in range(len(pipeline.df)):
        # 获取当前行的相似度分数
        row_scores = similarity_matrix[row_idx]

        # 获取 Top-K 的索引和分数
        top_scores, top_indices = torch.topk(row_scores, k=min(top_k * 2, len(pipeline.raw_text_list)))

        # 获取当前行的实体名称（用于硬过滤）
        row_entity = pipeline.df.iloc[row_idx][pipeline.pk_col]
        entity_keywords = {w.lower() for w in re.split(r'\W+', str(row_entity)) if len(w) > 3}

        found_texts = []

        for score, text_idx in zip(top_scores.cpu().numpy(), top_indices.cpu().numpy()):
            if score < score_threshold:
                continue

            text_content = pipeline.raw_text_list[text_idx]

            # 3. 实体一致性检查 (Entity Consistency Check)
            # 只有当文本包含实体关键词时，才认为是有效的“成对”数据
            # 这能过滤掉虽然语义相似（都是浏览器）但讲的是别人的（讲Chrome的配到了Firefox行）情况
            if entity_keywords and not any(kw in text_content.lower() for kw in entity_keywords):
                continue

            found_texts.append({
                "text_id": text_idx,
                "text_content": text_content,
                "score": float(score)
            })

            if len(found_texts) >= top_k:
                break

        # 4. 构造数据记录
        if found_texts:
            for item in found_texts:
                aligned_data.append({
                    "row_id": row_idx,
                    "entity": row_entity,
                    "row_content": pipeline.documents[row_idx]['text'] if hasattr(pipeline, 'documents') else str(
                        pipeline.df.iloc[row_idx].to_dict()),
                    "matched_text": item['text_content'],
                    "similarity_score": round(item['score'], 4)
                })

    # 5. 转为 DataFrame 展示
    pair_df = pd.DataFrame(aligned_data)
    print(f"✅ Constructed {len(pair_df)} pairs from {len(pipeline.df)} rows.")
    return pair_df


def main():
    # 1. 读取表格
    df = pd.read_excel("data/dev_excel/Mobile_browser_0.xlsx")

    # 2. 读取 JSON 并转化为字符串列表 List[str]
    with open("data/dev_doc/Mobile_browser_0.json", 'r') as f:
        json_data = json.load(f)
    # 将字典的值提取出来，形成一个 List[str]
    text_list = list(json_data.values())

    llm_path = "./models/bge-m3"

    # 这里的 embedding_model_name 可以换成你本地 BGE 模型的路径，或者 HuggingFace Hub ID
    pipeline = TableRAGPipeline(
        df=df,
        external_text_list=text_list,
        llm_backbone="qwen2.5:7b",
        embedder=Embedder(llm_path)
    )

    pipeline.build_index()

    # 3. 在线问答 (Phase 2)
    # Case 1: 之前的 Android 版本问题
    q1 = "Of the free and open source software browsers, which is currently on stable version 10?"
    ans1 = pipeline.query(q1)
    print(f"\n📝 Final Answer 1: {ans1}")

    print("-" * 50)

    # Case 2: 测试列筛选能力
    q2 = "What engine does the Blackberry Browser use?"
    ans2 = pipeline.query(q2)
    print(f"\n📝 Final Answer 2: {ans2}")


if __name__ == "__main__":
    main()
