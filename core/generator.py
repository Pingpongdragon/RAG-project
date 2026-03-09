import re
from typing import List, Dict, Optional,Tuple
from core.utils.context_utils import format_context
from config import settings
from config.logger_config import configure_console_logger
from openai import OpenAI
import os
try:
    from swift.llm import InferRequest, RequestConfig
    HAS_SWIFT = True
except ImportError:
    HAS_SWIFT = False

logger = configure_console_logger(__name__)


# -------------------------
# API 客户端初始化
# -------------------------
def _get_api_client():
    """根据配置返回对应的 OpenAI 兼容客户端"""
    if settings.ACTIVE_MODEL_TYPE == "qwen":
        return OpenAI(
            api_key=settings.QWEN_API_CONFIG["api_key"],
            base_url=settings.QWEN_API_CONFIG["base_url"]
        )
    elif settings.ACTIVE_MODEL_TYPE == "deepseek":
        return OpenAI(
            api_key=settings.DEEPSEEK_CONFIG["api_key"],
            base_url=settings.DEEPSEEK_CONFIG["base_url"]
        )
    else:
        return None  # 使用本地模型

def _get_model_name():
    """返回当前激活模型的名称"""
    if settings.ACTIVE_MODEL_TYPE == "qwen":
        return settings.QWEN_API_CONFIG["model"]
    elif settings.ACTIVE_MODEL_TYPE == "deepseek":
        return settings.DEEPSEEK_CONFIG["model"]
    else:
        return None


# -------------------------
# LLM 生成模板配置
# -------------------------
CONTEXT_PROMPT_TEMPLATE_ZH = '''
直接给出最终答案，不要包含思考过程：

格式要求：
1. 回答包括答案和参考文献内容，其余不需要
2. 参考文献需要内容而不是标号
3. 无相关证据时明确说明

[文本]
{context}

[问题]
{query}

[答案和参考文献]
'''

NO_CONTEXT_PROMPT_TEMPLATE_ZH = '''
用1-2句中文直接回答：
格式要求：
1. 说完答案即可
2. 不确定时建议研究方向

问题：
{query}

答案：
'''

CONTEXT_PROMPT_TEMPLATE_EN = '''
Provide the final answer directly (in English), without including the reasoning process:

Format requirements:
1. Begin with your answer and include the full content of the references at the end
2. References should contain content rather than just numbers
3. Clearly state if there is no relevant evidence

[contexts]
{context}

[Question]
{query}

[Answer And References]
'''


NO_CONTEXT_PROMPT_TEMPLATE_EN = '''
Answer directly in 1-2 sentences (in English):

Format requirements:
1. Give a concise answer to the question
2. Suggest research directions if uncertain

Question: {query}

[Answer]
'''


# -------------------------
# 生成主函数
# -------------------------
def generate_llm_response(
    query: str,
    context: List[Dict],
    language: str = "en",  
    max_retries: int = settings.MAX_RETRIES,
    context_top_n: int = settings.CONTEXT_TOP_N,
) -> Tuple[Optional[str], Optional[str]]:
    """支持中英文的全功能版，通过 language 参数动态切换"""
    
    # =============== 智能构造Prompt ================
    use_context = bool(context not in (None, [])) 
    context_str = ""
    
    if use_context:
        context_str = format_context(context, top_n=context_top_n, language=language)
        if language == "en":
            prompt = CONTEXT_PROMPT_TEMPLATE_EN.format(context=context_str, query=query)
        else:
            prompt = CONTEXT_PROMPT_TEMPLATE_ZH.format(context=context_str, query=query)
    else:
        if language == "en":
            prompt = NO_CONTEXT_PROMPT_TEMPLATE_EN.format(query=query)
            logger.info("No valid context found, enabling generic response mode.")
        else:
            prompt = NO_CONTEXT_PROMPT_TEMPLATE_ZH.format(query=query)
            logger.info("无有效上下文,启用通用回答模式")
    
    # =============== 生成逻辑 ================
    api_client = _get_api_client()
    
    for retry in range(max_retries):
        try:
            if api_client:  # 使用 API
                response = api_client.chat.completions.create(
                    model=_get_model_name(),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=settings.MAX_NEW_TOKEN,
                    temperature=settings.TEMPERATURE
                )
                response_text = response.choices[0].message.content
            else:  # 使用本地模型
                ENGINE = settings.model_manager.get_engine(
                    settings.MODEL_DIR, 
                    settings.ADAPTER_DIR, 
                    settings.MODEL_TYPE
                )
                infer_request = [InferRequest(messages=[{'role': 'user', 'content': prompt}])]
                request_config = RequestConfig(
                    max_tokens=settings.MAX_NEW_TOKEN,
                    temperature=settings.TEMPERATURE
                )
                resp_list = ENGINE.infer(infer_request, request_config)
                response_text = resp_list[0].choices[0].message.content
            
            return context_str, _post_process(response_text, use_context, language)
            
        except Exception as e:
            logger.error(f"Generation attempt {retry+1} failed: {e}")
    
    return context_str, "无法生成回答，请稍后再试。" if language == "zh" else "Unable to generate a response. Please try again later."


def generate_batch_llm_response(
    queries: List[str],
    contexts: List[List[Dict]],
    language: str = "en",
    max_batch_size: int = settings.MAX_BATCH_SIZE,
) -> List[str]:
    """Batch 生成函数，大幅提升实验速度"""
    
    # 1. 批量构造 Prompts
    prompts = []
    has_context_flags = []

    for query, context in zip(queries, contexts):
        use_context = bool(context not in (None, []))
        has_context_flags.append(use_context)
        
        context_str = ""
        if use_context:
            context_str = format_context(context, top_n=settings.CONTEXT_TOP_N, language=language)
            
        if language == "en":
            prompt = CONTEXT_PROMPT_TEMPLATE_EN.format(context=context_str, query=query)
        else:
            prompt = CONTEXT_PROMPT_TEMPLATE_ZH.format(context=context_str, query=query)
            
        if not use_context:
            if language == "en":
                prompt = NO_CONTEXT_PROMPT_TEMPLATE_EN.format(query=query)
            else:
                prompt = NO_CONTEXT_PROMPT_TEMPLATE_ZH.format(query=query)
        
        prompts.append(prompt)

    # 2. 获取 API 客户端或本地引擎
    api_client = _get_api_client()
    
    # 3. 执行批量推理
    try:
        if api_client:  # 使用 API (逐个调用，因为 OpenAI API 不支持真正的批量)
            resp_list = []
            for prompt in prompts:
                response = api_client.chat.completions.create(
                    model=_get_model_name(),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=settings.MAX_NEW_TOKEN,
                    temperature=settings.TEMPERATURE
                )
                resp_list.append(response.choices[0].message.content)
        else:  # 使用本地模型批量推理
            ENGINE = settings.model_manager.get_engine(
                settings.MODEL_DIR, 
                settings.ADAPTER_DIR, 
                settings.MODEL_TYPE
            )
            infer_requests = [
                InferRequest(messages=[{'role': 'user', 'content': p}]) 
                for p in prompts
            ]
            request_config = RequestConfig(
                max_tokens=settings.MAX_NEW_TOKEN, 
                temperature=settings.TEMPERATURE
            )
            responses = ENGINE.infer(infer_requests, request_config)
            resp_list = [resp.choices[0].message.content for resp in responses]
            
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        return ["" for _ in queries]

    # 4. 批量后处理
    final_responses = []
    for i, raw_text in enumerate(resp_list):
        processed = _post_process(raw_text, has_context_flags[i], language)
        final_responses.append(processed)
        
    return final_responses
    
# 关键逻辑：截断思维链内容 🚀
# =================== 改进版后处理流程 ===================
def _post_process(raw_response: str, has_context: bool, language: str = "zh") -> str:
    # 找到 '</think>' 后的内容
    index = raw_response.find('</think>')
    if index != -1:
        raw_response = raw_response[index + len('</think>'):]
    
    if language == "en":
        conclusion_pattern = r'\[Conclusion\]\s*[:：]?\s*(.*?)\s*(?:\[References\]|Answer|Response)'
    else:
        conclusion_pattern = r'结论\s*[:：]?\s*(.*?)\s*(?:答案|回答|原始答案)'
    match = re.search(conclusion_pattern, raw_response, re.DOTALL)
    processed_text = match.group(1).strip() if match else raw_response.strip()

    # 保持原始格式，不进行额外的换行或空格处理

    # 返回处理后的文本
    return processed_text



