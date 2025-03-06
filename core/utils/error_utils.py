from typing import Dict, Any
import re

ERROR_PATTERNS = {
    "timeout": r"time[ -]?out|timed? out",
    "model_not_found": r"model not found|找不到模型",
    "rate_limit": r"rate limit|请求过多",
    "auth_error": r"auth|authenticat|权限"
}

def classify_error(error: Exception) -> str:
    """智能错误分类器"""
    error_msg = str(error).lower()
    
    for err_type, pattern in ERROR_PATTERNS.items():
        if re.search(pattern, error_msg):
            return err_type
    return "unknown"

def generate_error_response(
    error_type: str, 
    context: Dict[str, Any]
) -> str:
    """动态生成错误响应"""
    error_templates = {
        "timeout": "请求超时（已尝试{retries}次），建议简化问题",
        "model_not_found": "模型服务异常，技术团队已收到通知",
        "rate_limit": "系统繁忙，请稍候重试",
        "default": "暂时无法处理请求，错误代码：{error_id}"
    }
    
    template = error_templates.get(error_type, error_templates["default"])
    return template.format(
        retries=context.get('retry_count', 0),
        error_id=context.get('error_id', 'UNKNOWN')
    )
