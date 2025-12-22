import json
from table_pipeline import TableRAGPipeline
import pandas as pd
from utils.tool_utils import Embedder


def main(table_id, questions):
    # 1. 读取表格
    df = pd.read_excel(f"data/dev_excel/{table_id}.xlsx")

    # 2. 读取 JSON 并转化为字符串列表 List[str]
    with open(f"data/dev_doc/{table_id}.json", 'r') as f:
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

    for i, q in enumerate(questions):
        answer = pipeline.query(q)
        print(f"\n📝 Final Answer {i}: {answer}")
        print("-" * 50)


if __name__ == "__main__":
    # question1 = "Of the free and open source software browsers, which is currently on stable version 10?"
    # question2 = "What engine does the Blackberry Browser use?"
    # table_id = "Mobile_browser_0"
    # questions = [question1,question2]

    question = "What is the middle name of the player with the second most National Football League career rushing yards ?"
    table_id = "List_of_National_Football_League_rushing_yards_leaders_0"
    #  "answer-text": "Jerry"
    questions = [question]

    main(table_id, questions)
