from loguru import logger

def configure_logger(name: str = __name__):
    """返回预配置的日志对象"""
    # 移除默认的处理器
    logger.remove()
    
    # 添加新的处理器，使用与原来相似的格式
    logger.add(
        sink=lambda msg: print(msg, end=""),  # 使用标准输出
        format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {module}:{line} - {message}",
        level="INFO"
    )
    
    # loguru 不需要考虑名字空间，但如果你需要，可以用以下方式标记
    logger_with_context = logger.bind(name=name)
    
    return logger_with_context