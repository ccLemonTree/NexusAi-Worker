import os
import logging
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI
# 创建日志目录
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志格式
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def setup_logger(name: str, log_file: str, level = logging.INFO) -> logging.Logger:
    """创建并配置日志器"""
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if not logger.handlers:
        # 创建文件处理器（按天切割，保留7天）
        file_handler = TimedRotatingFileHandler(
            os.path.join(LOG_DIR, log_file),
            when="D",  # 按天切割
            interval=1,
            backupCount=7  # 保留7天的日志
        )
        file_handler.setFormatter(formatter)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # 添加处理器到日志器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# 创建主应用日志器
app_logger = setup_logger("app", "app.log")

# 苍穹智搜
CangQiong_Smart_Search_Logger = setup_logger("smart_search", "CangQiong_smart_search.log")
CangQiong_Smart_Insert_Logger = setup_logger("smart_insert", "CangQiong_smart_insert.log")


# 苍穹智眸
CangQiong_Smart_Vllm_logger = setup_logger("smart_vllm", "All_BigModels.log")
CangQiong_Smart_Model_logger = setup_logger("smart_model", "All_BigModels.log")


# 创建API请求日志器
request_logger = setup_logger("request", "request.log")

# 创建错误日志器（只记录ERROR级别以上的日志）
error_logger = setup_logger("error", "error.log", logging.ERROR)

# Kafka Worker 日志
Kafka_Consumer_logger = setup_logger("kafka_consumer", "kafka_consumer.log")
Kafka_Handler_logger  = setup_logger("kafka_handler",  "kafka_handler.log")
Kafka_Producer_logger = setup_logger("kafka_producer", "kafka_producer.log")


# FastAPI启动时配置
def configure_logging(app: FastAPI) -> None:
    @app.on_event("startup")
    def startup_event():
        app_logger.info("FastAPI应用启动")

    @app.on_event("shutdown")
    def shutdown_event():
        app_logger.info("FastAPI应用关闭")