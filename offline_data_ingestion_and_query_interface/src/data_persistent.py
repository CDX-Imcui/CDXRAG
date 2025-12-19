import os
import pandas as pd
import json
import random as randum
from tqdm import tqdm
from service import transfer_name
from dtype_mapping import (
    INTEGER_DTYPE_MAPPING,
    FLOAT_DTYPE_MAPPING,
    OTHER_DTYPE_MAPPING,
    SPECIAL_INTEGER_DTYPE_MAPPING
)
import warnings
from common_utils import transfer_name, SCHEMA_DIR, sql_alchemy_helper


def infer_and_convert(series):
    # 尝试转换为整数
    try:
        return pd.to_numeric(series, downcast='integer')
    except ValueError:
        pass

    # 尝试转换为浮点数
    try:
        return pd.to_numeric(series, downcast='float')
    except ValueError:
        pass

    # 尝试转换为日期时间
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # 忽略特定类型的警告
            return pd.to_datetime(series)
    except ValueError:
        pass

    # 如果都不行，返回原始数据
    return series


def pandas_to_mysql_dtype(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        if str(dtype) in SPECIAL_INTEGER_DTYPE_MAPPING:
            return SPECIAL_INTEGER_DTYPE_MAPPING[str(dtype)]
        return INTEGER_DTYPE_MAPPING.get(dtype, 'INT')

    elif pd.api.types.is_float_dtype(dtype):
        return FLOAT_DTYPE_MAPPING.get(dtype, 'FLOAT')

    elif pd.api.types.is_bool_dtype(dtype):
        return OTHER_DTYPE_MAPPING['boolean']

    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return OTHER_DTYPE_MAPPING['datetime']

    elif pd.api.types.is_timedelta64_dtype(dtype):
        return OTHER_DTYPE_MAPPING['timedelta']

    elif pd.api.types.is_string_dtype(dtype):
        return OTHER_DTYPE_MAPPING['string']

    elif pd.api.types.is_categorical_dtype(dtype):
        return OTHER_DTYPE_MAPPING['category']

    else:
        return OTHER_DTYPE_MAPPING['default']

def get_sample_values(series):
    valid_values = [str(x) for x in series.dropna().unique() if pd.notnull(x) and len(str(x)) < 64]
    sample_values = randum.sample(valid_values, min(3, len(valid_values)))
    return sample_values if sample_values else ['no sample values available']

def get_schema_and_data(df):
    column_list = []
    for col in df.columns:
        cur_column_list = []
        if isinstance(df[col], pd.DataFrame):
            print(f"Column {col} is a DataFrame, skipping...")
            raise ValueError(f"Column {col} is a DataFrame, which is not supported.")   
        cur_column_list.append(col)
        cur_column_list.append(pandas_to_mysql_dtype(df[col].dtype))
        cur_column_list.append('sample values:' + str(get_sample_values(df[col])))

        # 形成三元组
        column_list.append(cur_column_list)

    return column_list

def generate_schema_info(df: pd.DataFrame, file_name: str):
    try:
        column_list = get_schema_and_data(df)
    except:
        print(f"{file_name} 列存在问题")
        raise ValueError(f"Error processing file: {file_name}")

    table_name = transfer_name(file_name)

    schema_dict = {
        'table_name': table_name,
        'column_list': column_list           
    }

    return schema_dict, table_name


def transfer_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗 DataFrame 的列名：
    1. 使用 transfer_name_func 转换列名。
    2. 如果第一个列名为空或 NaN，设置为 'No'。
    3. 处理重复列名，重复列名加后缀 _1, _2 等。

    参数：
        df: 需要处理列名的 DataFrame。
        transfer_name_func: 一个函数，用于转换列名（如去除空格、统一格式等）。

    返回：
        列名处理后的 DataFrame。
    """
    df = df.copy()

    # 第一步：统一转换列名
    df.columns = [transfer_name(col) for col in df.columns]

    # 第二步：首列为空或 NaN 时命名为 'No'
    df.columns = [
        'No' if i == 0 and (not col or pd.isna(col)) else col
        for i, col in enumerate(df.columns)
    ]

    # 第三步：处理重复列名
    seen = {}
    new_columns = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_columns.append(col)
    df.columns = new_columns

    return df


def parse_excel_file_and_insert_to_db(excel_file_outer_dir: str):
    if not os.path.exists(excel_file_outer_dir):
        raise FileNotFoundError(f"File not found: {excel_file_outer_dir}")
    

    for file_name in tqdm(os.listdir(excel_file_outer_dir)):
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            full_path = os.path.join(excel_file_outer_dir, file_name)
            df = pd.read_excel(full_path)
            
            df_convert = df.apply(infer_and_convert)
            df_convert = transfer_df_columns(df_convert)

            schema_dict, table_name = generate_schema_info(df_convert, file_name)

            # 确保目录存在
            if not os.path.exists(SCHEMA_DIR):
                os.makedirs(SCHEMA_DIR)

            with open(f"{SCHEMA_DIR}/{table_name}.json", 'w', encoding='utf-8') as f:
                json.dump(schema_dict, f, ensure_ascii=False)
            
            sql_alchemy_helper.insert_dataframe_batch(df_convert, table_name)


if __name__ == "__main__":
    parse_excel_file_and_insert_to_db('../dataset/hybridqa/dev_excel/')

"""
这段脚本本质上是在自己撸一个“小型 ETL 管道”：用 pandas 把 Excel 解析成 DataFrame，再做一轮规则驱动的类型推断和列名清洗，最后一边生成 schema JSON，一边通过 SQLAlchemy 批量写进 MySQL。没有什么黑魔法或模型，都是显式规则和 pandas 自带能力。

入口是 `parse_excel_file_and_insert_to_db`。它遍历你指定的目录，把每个 `.xlsx/.xls` 文件用 `pd.read_excel` 读进来，这一步就是“解析 Excel”的核心手段——完全依赖 pandas 的 `read_excel`，由它处理单元格到 DataFrame 的转换。读完以后，先对整个 DataFrame 调用 `df.apply(infer_and_convert)`，也就是对每一列跑一遍你自己写的“类型自动识别”流程。

`infer_and_convert` 这个函数就是你用来“推断列类型”的方法链。对传入的一列数据 `series`，它先尝试用 `pd.to_numeric(..., downcast='integer')` 把整列转成整数，如果里面混了非整数字符就会抛 `ValueError`，你就放弃这一条路径；接着再试一次 `pd.to_numeric(..., downcast='float')`，看能不能当浮点数；如果还是不行，再用 `pd.to_datetime` 去尝试解析日期时间，中间用 warnings 屏蔽掉烦人的告警。三条路都失败，就保留原始类型。这一串就是“逐列多轮尝试”的启发式类型推断法：不是事先根据样本统计去判别类型，而是直接“能转就转，转不动就跳下一种类型”。

有了比较合理的 pandas dtype 之后，你又写了一个 `pandas_to_mysql_dtype`，把 pandas 的 dtype 映射成 MySQL 里的字段类型。这里用的是一套纯规则映射：通过 `pd.api.types.is_integer_dtype / is_float_dtype / is_datetime64_any_dtype` 这些类型检查，按类别再查对应的 `INTEGER_DTYPE_MAPPING / FLOAT_DTYPE_MAPPING / OTHER_DTYPE_MAPPING` 字典，决定最终在数据库里是 `INT / BIGINT / FLOAT / DATETIME / VARCHAR...` 之类。特殊的整型还走了 `SPECIAL_INTEGER_DTYPE_MAPPING`，算是对 pandas 那堆奇怪的 `int8/int16/int32` 做细粒度控制。

在写库之前，你还做了一层列名的“清洗和规范化”，这个工作完全由 `transfer_df_columns` 完成。它先对所有列名跑一遍 `transfer_name`（去掉空格、统一大小写之类应该都在那儿干了），如果第一列名是空或 NaN，就强制改成 `'No'`，避免出现匿名列。然后用一个 `seen` 字典给重复列名自动加 `_1、_2` 这种后缀，保证最终每个列名唯一，不会在建表和执行 SQL 时踩雷。

生成 schema 的部分则是 `generate_schema_info` + `get_schema_and_data` 这一对函数在干活。`get_schema_and_data` 逐列遍历 DataFrame，对每个列构造一个三元组 `[列名, MySQL 类型, 'sample values:[...样本...]']`。样本是 `get_sample_values` 从去重后的非空值里随机抽最多 3 个出来的，用来给你之后做 NL2SQL 或表格理解时当“列语义提示”。最后，`generate_schema_info` 把这些列信息打包成一个字典 `{table_name, column_list}`，根据文件名跑一遍 `transfer_name` 得到表名，然后写到 `SCHEMA_DIR` 下对应的 JSON 里，`ensure_ascii=False` 保证里面的中文是可读的。

整个流程的最后一步是通过 `sql_alchemy_helper.insert_dataframe_batch(df_convert, table_name)` 把 DataFrame 批量插入到数据库里。这里你没有展开内部实现，但从命名看，大概率是用 SQLAlchemy 根据 DataFrame 的列和你推断出的 dtype 去建表（如果不存在）或直接插入，再用批量插入优化性能。这一步完成了从“Excel 文件 → 清洗过的表格 → MySQL 数据表”的导入。

所以如果一句话概括：你当前解析导入数据用的方法是：依托 pandas 的 `read_excel` 读原始表，配上一套手写的列类型推断（多次尝试 `to_numeric` / `to_datetime`）、规则化列名和 dtype→MySQL 类型映射，再用 SQLAlchemy 封装的批量插入把 DataFrame 写进数据库，同时把简化的 schema 和列样本导出成 JSON，给后面的 NL2SQL / TableRAG 当“结构化表描述”。这属于很典型的“规则驱动 ETL + schema 推断”的范式，优点是可控、透明，缺点是对脏数据的鲁棒性和复杂类型（混合单位、复杂字符串）会比较有限。
"""