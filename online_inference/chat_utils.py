import json
import requests
from openai import OpenAI
import logging
from functools import wraps
from typing import Dict, Any, Optional
import os
import hashlib
import time
# from online_inference.config import sql_service_url ,config_mapping
import httpx


def init_logger(name='my_logger', level=logging.DEBUG, log_file='app.log'):
    """
    Initialize a logger with console and file handlers.

    Args:
        level(int): logging level(DEBUG, INFO, WARINING...)
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level=level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


def get_chat_result(
        messages: object,
        tools: object = None,
        tool_choice: object = None,
        llm_config: Dict = None
):
    """
    Get LLM generation result of different API backend, e.g. gpt-4o.
    """
    client = OpenAI(
        api_key=llm_config.get('api_key', ''),
        base_url=llm_config.get('url', '')
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=llm_config.get('model', 'qwen-flash'),
            tools=tools,
            temperature=0.1
        )
        return chat_completion.choices[0].message
    except:
        service_url = llm_config.get('url', '')
        payload = {
            "model": llm_config.get('model', ''),
            "messages": messages,
            "temperature": 0.1,
            "tools": tools
        }
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post(
            service_url,
            json=payload,
            headers=headers
        )
        return json.loads(response.text)['choices'][0]["message"]
