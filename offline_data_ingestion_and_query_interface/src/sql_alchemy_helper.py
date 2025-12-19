import pandas as pd
import pymysql
from sqlalchemy import create_engine, text
import math
import json

from decimal import Decimal
from datetime import date, datetime
import uuid

def default_serializer(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, bytes):
        return obj.decode(errors='replace')  # 或使用 base64 编码
    raise TypeError(f"Type {type(obj)} not serializable")

class SQL_Alchemy_Helper:
    def __init__(self, config):
        user = config["user"]
        password = config["password"]
        host = config["host"]
        port = config["port"]
        charset = config.get("charset", "utf8mb4")
        db = config.get("database", "mysql")  # 默认库

        self.engine = create_engine(
            f'mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset={charset}',
            pool_pre_ping=True,        # 防止 MySQL server has gone away
            pool_recycle=1800,         # 30分钟回收连接
            pool_size=10,
            max_overflow=20
        )

    def execute_sql(self, sql, args=None):
        """
        执行 insert/update/delete
        """
        with self.engine.begin() as conn:
            conn.execute(text(sql), args or {})

    def fetchall(self, sql, args=None):
        """
        执行 select 查询
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), args or {})
            rows = result.fetchall()

            json_result = [dict(row._mapping) for row in rows]

            json_result_str = json.dumps(json_result, ensure_ascii=False,  default=default_serializer)

            if len(json_result_str) > 1000:
                return json_result_str[:1000]
            else:
                return json_result_str

    def fetch_dataframe(self, sql, args=None):
        """
        查询结果转为 DataFrame
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=args)
            return df

    def insert_dataframe_batch(self, df, table_name, batch_size=1000):
        """
        DataFrame 批量插入
        """
        df.to_sql(table_name, self.engine, index=False, if_exists='replace', chunksize=batch_size, method='multi')