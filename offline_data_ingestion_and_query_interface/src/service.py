# -*- coding: utf-8 -*-
import re
import json
import time
from log_service import logger
from prompt import *
from handle_requests import get_llm_response
from common_utils import transfer_name, SCHEMA_DIR, sql_alchemy_helper


def extract_sql_statement(resp_content):  
    """
    从响应内容中提取SQL语句。
    
    Args:
        resp_content (str): 响应内容，包含SQL语句。
    
    Returns:
        str: 提取的SQL语句。
    """
    # 使用正则表达式匹配SQL语句
    match = re.search(r'```sql([\s\S]*?)```', resp_content, re.DOTALL)
    if match:
        matched_text = match.group(1).strip()
        sql_text = re.sub(r'\s+', ' ', matched_text)  # 压缩空白字符
        return sql_text
    else:
        logger.error(f"No SQL statement found in the response content. Response content: {resp_content}")
        return resp_content
    

def process_tablerag_request(table_name_list, query,backbone):
    """
    Process the request for TableRAG.
    
    Args:
        table_name_list (list): List of table names related to the query.
        query (str): The query string to be processed.
    
    Returns:
        str: A mock response for demonstration purposes.
    """
    # Here you would implement the actual logic to process the request
    # For demonstration, we will just return a mock response

    schema_list = []
    for table_name in table_name_list:
        table_name = transfer_name(table_name)
        schema_path = f"{SCHEMA_DIR}/{table_name}.json"
        schema_dict = json.load(open(schema_path, 'r', encoding='utf-8'))
        schema_list.append(schema_dict)
    
    nl2sql_prompt = NL2SQL_USER_PROMPT.format(
        schema_list=json.dumps(schema_list, ensure_ascii=False),
        user_query=query
    )

    nl2sql_start_time = time.time()
    resp_content = get_llm_response(
        system_prompt=NL2SQL_SYSTEM_PROMPT,
        user_prompt=nl2sql_prompt,
        model=backbone
    )
    nl2sql_end_time = time.time()
    nl2sql_time_cusumed = nl2sql_end_time - nl2sql_start_time

    sql_str = extract_sql_statement(resp_content)

    sql_excution_start_time = time.time()
    try:
        sql_excution_result = sql_alchemy_helper.fetchall(sql_str)
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        sql_excution_result = f"SQL execution failed: {str(e)}"
        
    sql_excution_end_time = time.time()
    sql_excution_time_cusumed = sql_excution_end_time - sql_excution_start_time

    time_consumed_str = f"NL2SQL time: {nl2sql_time_cusumed:.2f}s, SQL execution time: {sql_excution_time_cusumed:.2f}s"

    res_dict = {
        'query': query,
        'nl2sql_prompt': nl2sql_prompt,
        'nl2sql_response': resp_content,
        'sql_str': sql_str,
        'sql_execution_result': sql_excution_result,
        'time_consumed': time_consumed_str
    }
    return res_dict
    

        
