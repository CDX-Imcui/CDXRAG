import pandas as pd
import json
import os
from typing import Tuple, List


def load_data_for_pipeline(excel_path: str, json_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    加载 Excel 和 JSON，并做基础清洗，供 TableRAGPipeline 使用
    """
    # 1. 加载表格
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path)
    df = df.astype(str).replace('nan', '')  # 基础清洗

    # 2. 加载文本
    text_list = []
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 兼容 List 或 Dict 格式
        if isinstance(raw_data, list):
            text_list = [str(x) for x in raw_data]
        elif isinstance(raw_data, dict):
            text_list = list(raw_data.values())

    return df, text_list