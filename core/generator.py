import time
from typing import List, Dict, Optional
from core.utils.context_utils import format_context, calculate_context_score
from core.utils.error_utils import classify_error, generate_error_response
from config import settings
from config.logger_config import configure_logger

logger = configure_logger(__name__)

# -------------------------
# LLM 生成模板配置
# -------------------------
RESPONSE_TEMPLATE = """基于以下上下文（按相关性排序），用中文专业、简明地回答。遵循格式：
1. 答案正文
2. 来源标注: [ID列表]

上下文:
{context}

问题: {query}
答案："""

ERROR_MESSAGES = {
    "timeout": "请求超时，请简化问题后重试",
    "model_not_found": "模型服务不可用",
    "default": "服务暂时不可用，请稍后再试"
}

# -------------------------
# 生成主函数
# -------------------------
def generate_llm_response(
    query: str,
    context: List[Dict],
    max_retries: int = 3,
    temperature: float = 0.3,
    max_tokens: int = 300,
    context_top_n: int = 3,  # 显式化关键参数
    **kwargs  # 保留扩展能力
) -> Optional[str]:
    """
    增强版生成器主要改进点：
    1. 显式参数与兼容参数分离
    2. 安全获取上下文配置
    3. 稳定类型标注
    """
    if calculate_context_score(context) < 0.5:
        return "当前知识库信息不足，建议补充问题细节"
    
    context_str = format_context(context, top_n=context_top_n)  # ✓ 正确引用
    prompt = RESPONSE_TEMPLATE.format(context=context_str, query=query)
    
    retry_count = 0
    backoff_factor = 1.5
    
    while retry_count < max_retries:
        try:
            response = settings.OLLAMA_CLIENT.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": 0.95
                },
                stream=False,
            )
            return postprocess_response(response, context)
        
        except Exception as e:
            error_type = classify_error(e)
            logger.warning(f"尝试 {retry_count+1}/{max_retries} 失败: [{error_type}] {str(e)}")
            
            if retry_count == max_retries - 1:
                return ERROR_MESSAGES.get(error_type, ERROR_MESSAGES["default"])
            
            sleep_time = min(5, backoff_factor ** retry_count)
            time.sleep(sleep_time)
            retry_count += 1
    return None

# # -------------------------
# # 工具函数
# # -------------------------
# def format_context(context: List[Dict], top_n: int = 3) -> str:
#     """格式化上下文信息"""
#     return "\n------\n".join(
#         f"[ID:{item['metadata']['q_id']}] {item.get('text', '')[:200]}..."  # 截断长文本
#         for item in context[:top_n]
#     )

# def _classify_error(error: Exception) -> str:
#     """错误类型分类器"""
#     if "timed out" in str(error).lower():
#         return "timeout"
#     elif "not found" in str(error).lower():
#         return "model_not_found"
#     return "default"

# def _postprocess_response(response: dict, context: List[Dict]) -> str:
#     """响应后处理"""
#     try:
#         answer = response["response"].strip()
#         source_ids = list({c["metadata"]["q_id"] for c in context[:3]})
        
#         # 检查是否已包含来源
#         if "来源" not in answer:
#             answer += f"\n来源标注: {source_ids}"
        
#         # 格式化编码处理
#         return answer.encode('utf-8', 'ignore').decode('utf-8')
#     except KeyError:
#         logger.error("响应格式异常: 缺少 'response' 字段")
#         return ERROR_MESSAGES["default"]
