import logging
from logging.handlers import RotatingFileHandler

def setup_logger():
    logger = logging.getLogger("my_app")  # 使用唯一名称
    logger.setLevel(logging.INFO)
    
    # 文件日志
    file_handler = RotatingFileHandler("app.log", maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return logger

# 初始化全局 logger
logger = setup_logger()