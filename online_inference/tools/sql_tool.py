import json
import requests
import logging
from functools import wraps
from typing import Any
import os
import time
import threading
import traceback
import sys

sys.path.append("../")
from online_inference.chat_utils import *
from online_inference.config import sql_service_url

function_lock = threading.Lock()
logger = init_logger('./logs/test.log', logging.INFO)

def with_retry(max_retries=3, backoff_factor=5) :
    def decorator(func) :
        @wraps(func)
        def wrapper(*args, kwargs) :
            retries = 0
            while retries < max_retries :
                try :
                    return func(*args, *kwargs)
                except Exception as e :
                    retries += 1
                    if retries > max_retries :
                        raise e
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    print(f"Retry {retries}/{max_retries} after {wait_time: .2f}s due to {e}")
        return wrapper
    return decorator

@with_retry(max_retries=3)
def get_excel_rag_response(table_name_list, query, repo_id) :
    """
    Run SQL generation and execute SQL in the database.

    Args:
        table_name_list[List]: num of tables
        repo_id: ID of the knowledge base
    """
    url = sql_service_url

    headers = {
        'Content-Type': 'application/json',
    }

    body = {
        'repo_id': repo_id,
        'table_name_list': table_name_list,
        'query': query
    }

    try :
        resp = requests.post(url=url, json=body, headers=headers, verify=False)
        answer = json.loads(resp.text)
        return answer
    except json.JSONDecodeError as json_decode_e :
        print(resp.text)
    except Exception as e :
        print(e)
        raise e


def get_excel_rag_response_plain(table_name_list: list = [], query: str = None) :
    """
    Call the SQL generation and execution service.

    Args: 
        table_name_list(list): List of table names
        query(str): input query
    
    Returns:
        answer(dict)
    """
    url = sql_service_url

    headers = {
        'Content-Type': 'application/json',
    }

    body = {
        'table_name_list': table_name_list,
        'query': query
    }
    
    try_times = 5
    while True :
        try :
            resp = requests.post(url=url, json=body, headers=headers, verify=False, timeout=60)
            answer = json.loads(resp.text)
            return answer
        except Exception as e :
            logger.error(f"SQL error, the model return is : {resp.text}")
            traceback.print_exc
            try_times -= 1
    return {}


if __name__ == '__main__' :
    res = get_excel_rag_response_plain(
        query="What is the middle name of the player with the second most National Football League career rushing yards ?",
        table_name_list=[
        "List_of_National_Football_League_rushing_yards_leaders_0"
        ]
    )
    print(res)