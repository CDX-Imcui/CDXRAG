import json
import argparse
from tqdm import tqdm
import sys
import re
from typing import Dict, List, Any
from online_inference.chat_utils import get_chat_result
from online_inference.prompt import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from online_inference.utils.utils import read_in_lines, read_in
from online_inference.config import *
import pandas as pd


def llm_eval(
        new_case: List = None,
        file_path: str = None,
        max_workers: int = 10,
        output_file: str = "evaluation.xlsx"
):
    """
    LLM based answer evaluation via qwen 72b.
    """
    if not new_case:
        new_case = read_in_lines(file_path=file_path)

    score_all = 0.0
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {executor.submit(single_llm_eval, case): case for case in new_case}
        for future in tqdm(as_completed(future_to_case), total=len(new_case), desc="Evaluating"):
            case = future_to_case[future]
            score = future.result()
            score_all += score
            results.append({
                "query": case['question'],
                "golden": case['answer-text'],
                "gen": case['tablerag_answer'],
                "table": case['table_id'],
                "score": score
            })

    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"LLM evaluation results saved to {output_file}")
    print("Final score", score_all / len(new_case))
    return


def single_llm_eval(case: Dict = None):
    pattern = r'\[\[(\d+)\]\]'
    golden = case['answer-text']
    gen = case['tablerag_answer']
    ques = case['question']

    eval_prompt = EVALUATION_PRONPT.format(question=ques, golden=golden, gen=gen)
    messages = [{"role": "user", "content": eval_prompt}]
    response = get_chat_result(messages=messages, llm_config=config_mapping[backbone])

    matches = re.findall(pattern, response.content)
    return float(matches[0]) if matches else 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="entry args")
    parser.add_argument('--backbone', type=str, default="qwen2.5:72b")
    parser.add_argument('--result_file_path', type=str,
                        default=r"../output.json", help="source file path")
    _args, _unparsed = parser.parse_known_args()

    backbone = _args.backbone

    data = read_in_lines(_args.result_file_path)
    questions = [d["question"] for d in data]

    llm_eval(new_case=data, file_path=_args.result_file_path)

# qwen2.5:32b Final score 0.3475409836065574         2:18:01   0.3540983606557377
# qwen2.5:72b Final score 0.5704918032786885         12:40:58  0.58、 0.5986842105263158

    # def read_in_lines(file_path: str) -> List[Dict[str, Any]]:
    #     """
    #     兼容三种常见场景：
    #     1) 完整的 JSON array: [ {...}, {...} ]
    #     2) ndjson / jsonlines: 每行一个 JSON 对象
    #     3) 多个 pretty-printed JSON 对象按 `}\n{` 或以空行分隔的情况
    #     """
    #     with open(file_path, "r", encoding="utf-8") as f:
    #         raw = f.read()
    #     if not raw.strip():
    #         return []
    #
    #     # 1) 如果是一个 JSON 数组，直接解析
    #     first_char = next((c for c in raw if not c.isspace()), "")
    #     if first_char == "[":
    #         try:
    #             return json.loads(raw)
    #         except Exception:
    #             # 如果解析失败，继续后续更宽松的尝试
    #             pass
    #
    #     # 2) 尝试按对象边界拆分：以 "}\n{" 或 "}\r\n{" 为分隔
    #     parts = re.split(r"}\s*\n\s*{", raw)
    #     if len(parts) > 1:
    #         objects = []
    #         for i, part in enumerate(parts):
    #             if i == 0:
    #                 s = part + "}"
    #             elif i == len(parts) - 1:
    #                 s = "{" + part
    #             else:
    #                 s = "{" + part + "}"
    #             try:
    #                 objects.append(json.loads(s))
    #             except Exception:
    #                 # 某块解析失败，回退到逐行累积策略
    #                 objects = None
    #                 break
    #         if objects is not None:
    #             return objects
    #
    #     # 3) 最后退回到逐行累积策略（对 pretty-printed 单个对象跨多行有效）
    #     objs = []
    #     buf_lines = []
    #     for line in raw.splitlines():
    #         stripped = line.strip()
    #         if not stripped and not buf_lines:
    #             continue  # 跳过多余空行
    #         buf_lines.append(line)
    #         # 简单启发式：如果这一行以 '}' 结尾，尝试把 buffer 作为一个 JSON 解析
    #         if stripped.endswith('}'):
    #             candidate = "\n".join(buf_lines)
    #             try:
    #                 objs.append(json.loads(candidate))
    #                 buf_lines = []
    #             except Exception:
    #                 # 还没到完整对象（或者字符串里有 '}'），继续累积
    #                 continue
    #     # 若最后还有残留 buffer，尝试解析一次
    #     if buf_lines:
    #         try:
    #             objs.append(json.loads("\n".join(buf_lines)))
    #         except Exception:
    #             pass
    #
    #     return objs