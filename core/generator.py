import time
from typing import List, Dict, Optional
from core.utils.context_utils import format_context
from core.utils.error_utils import classify_error
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
1. 必须优先使用参考资料中的信息，并在回答中标注具体来源编号，例如：[来源1]、[来源2]。
2. 如果问题需要结合多个参考资料片段回答，请分别标注每个来源，例如：
   - 第一部分 [来源1]
   - 第二部分 [来源2]
3. 如果参考资料中没有相关信息，请明确说明“参考资料中未找到相关信息”。
4. 如果未标注来源，回答将被视为无效。请确保回答准确且完整。
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
    max_retries: int = settings.GENERATION_CONFIG["max_retries"],
    temperature: float = settings.GENERATION_CONFIG["temperature"],
    max_tokens: int = settings.GENERATION_CONFIG["max_tokens"],
    context_top_n: int = settings.GENERATION_CONFIG["context_top_n"],
) -> Optional[str]:
    """支持无上下文生成的全功能版"""
    
    # =============== 参数提取 ================
    top_p = settings.GENERATION_CONFIG["top_p"]
    stream = settings.GENERATION_CONFIG["stream"]
    
    # =============== 智能构造Prompt ================
    use_context = bool(context not in (None, [])) 
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
                model=settings.OLLAMA_MODEL,  # 使用配置中的模型名
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p
                },
                stream=stream
            )
            cleaned_response = _sanitize_output(
                response.get("response", ""),
                has_context=use_context
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


