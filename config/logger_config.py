from loguru import logger
import os

def configure_console_logger(name: str = __name__):
    """配置并返回用于控制台输出的日志记录器"""
    # 创建一个独立的日志记录器实例
    console_logger = logger.bind(name=name)
    # 移除默认的处理器（如果有）
    console_logger.remove()
    # 添加新的处理器，将日志输出到控制台
    console_logger.add(
        sink=lambda msg: print(msg, end=""),
        format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {module}:{line} - {message}",
        level="ERROR",
    )
    return console_logger

