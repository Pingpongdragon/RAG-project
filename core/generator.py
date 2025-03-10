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
CONTEXT_PROMPT_TEMPLATE = '''
你是一位专业顾问，请严格根据提供的参考资料回答问题。

参考资料片段：
{context}

用户问题：
{query}

回答要求：
1. 优先使用参考资料中的信息
2. 如果引用资料，需标注具体来源编号如 [来源1] 
3. 资料不足时可结合常识回答，但需明确说明
'''

NO_CONTEXT_PROMPT_TEMPLATE = '''
请基于你的专业知识回答以下问题：

{query}

回答要求：
1. 使用简洁中文，避免技术术语
2. 如果不确定答案，建议提供探索方向
3. 明确说明回答是否基于通用知识
'''

# -------------------------
# 生成主函数
# -------------------------
def generate_llm_response(
    query: str,
    context: List[Dict],
    max_retries: int = 3,
    temperature: float = 0.3,
    max_tokens: int = 300,
    context_top_n: int = 3,
    context_score_threshold: float = 0.5  # 可配置阈值
) -> Optional[str]:
    """支持无上下文生成的全功能版"""
    # =============== 动态判断上下文有效性 ================
    use_context = bool(context) and (calculate_context_score(context) >= context_score_threshold)
    
    # =============== 智能构造Prompt ================
    if use_context:
        context_str = format_context(context, top_n=context_top_n)
        prompt = CONTEXT_PROMPT_TEMPLATE.format(
            context=context_str, 
            query=query
        )
    else:
        prompt = NO_CONTEXT_PROMPT_TEMPLATE.format(query=query)
        logger.info("无有效上下文，启用通用回答模式")

    # =============== 生成逻辑 ================
    for retry in range(max_retries):
        try:
            response = settings.OLLAMA_CLIENT.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": 0.9
                },
                stream=False
            )
            cleaned_response = _sanitize_output(
                response.get("response", ""),
                has_context=use_context  # 传递上下文使用标志★
            )
            return cleaned_response
            
        except Exception as e:
            error_type = classify_error(e)
            logger.error(f"Generation attempt {retry+1} failed: {error_type}")
    
    return None

def _sanitize_output(raw_response: str, has_context: bool) -> str:
    """根据是否使用上下文智能后处理"""
    cleaned = (
        raw_response.strip()
        .replace('\r\n', '\n')
        .encode('utf-8', 'ignore').decode('utf-8')
    )
    
    # 仅当使用上下文时检查来源标注
    if has_context and not any(f"[来源{i}]" in cleaned for i in range(1,4)):
        cleaned += "\n[注意：此回答未引用提供的参考资料]"
    
    return cleaned[:3000]  # 确保长度可控


