import re
from typing import List, Dict, Optional,Tuple
from RAG_project.core.utils.context_utils import format_context
from RAG_project.config import settings
from RAG_project.config.logger_config import configure_console_logger
from swift.llm import InferRequest, RequestConfig
import os

logger = configure_console_logger(__name__)
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
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

CONTEXT_PROMPT_MMLU_TEMPLATE_EN = '''
You are answering a multiple-choice question from the MMLU (Massive Multitask Language Understanding) benchmark, 
give the answer (in English), without including the reasoning process:

Format requirements:
1. Give your answer like "The answer is X", where X is one of: A, B, C, or D, and include the full content of the chosen option
2. Begin with your answer and include the full content of the references at the end
3. References should contain content rather than just numbers
4. Clearly state if there is no relevant evidence

[contexts]
{context}

[Question]
{query}

[Answer And References]
'''

NO_CONTEXT_PROMPT_MMLU_TEMPLATE_EN = '''
Answer directly in 1-2 sentences (in English):

Format requirements:
1. Do not include any additional text after your answer to the question
2. Suggest research directions if uncertain

Question: {query}

[Answer]
'''

NO_CONTEXT_PROMPT_TEMPLATE_EN = '''
You are answering a multiple-choice question from the MMLU (Massive Multitask Language Understanding) benchmark, answer directly in 1-2 sentences (in English):

Format requirements:
1. Give your answer like "The answer is X", where X is one of: A, B, C, or D, and include the full content of the chosen option
2. Do not include any additional text after your answer to the question
3. Suggest research directions if uncertain

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
    context_str = ""  # 初始化 context_str，确保在所有情况下都有值
    
    if use_context:
        context_str = format_context(context, top_n=context_top_n, language=language)
        if language == "en":
            if settings.IS_MMLU:
                prompt = CONTEXT_PROMPT_MMLU_TEMPLATE_EN.format(context=context_str, query=query)
            else:
                prompt = CONTEXT_PROMPT_TEMPLATE_EN.format(context=context_str, query=query)
        else:
            prompt = CONTEXT_PROMPT_TEMPLATE_ZH.format(context=context_str, query=query)
    else:
        if language == "en":
            if settings.IS_MMLU:
                prompt = NO_CONTEXT_PROMPT_MMLU_TEMPLATE_EN.format(query=query)
                logger.info("No valid context found, enabling generic response mode for MMLU.")
            else:
                prompt = NO_CONTEXT_PROMPT_TEMPLATE_EN.format(query=query)
                logger.info("No valid context found, enabling generic response mode.")
        else:
            prompt = NO_CONTEXT_PROMPT_TEMPLATE_ZH.format(query=query)
            logger.info("无有效上下文，启用通用回答模式")
    
    # 加载推理引擎
    ENGINE = settings.model_manager.get_engine(settings.MODEL_DIR, settings.ADAPTER_DIR, settings.MODEL_TYPE)
    
    # =============== 生成逻辑 ================
    for retry in range(max_retries):
        try:
            infer_request = [InferRequest(messages=[{'role': 'user', 'content': prompt}])]
            request_config = RequestConfig(max_tokens=settings.MAX_NEW_TOKEN,temperature=settings.TEMPERATURE)
            resp_list = ENGINE.infer(infer_request, request_config)
            response = resp_list[0].choices[0].message.content
            return context_str, _post_process(response, use_context, language)
        except Exception as e:
            logger.error(f"Generation attempt {retry+1} failed: {e}")
    
    # 所有重试失败后返回提示信息
    return context_str, "无法生成回答，请稍后再试。" if language == "zh" else "Unable to generate a response. Please try again later."


def generate_batch_llm_response(
    queries: List[str],
    contexts: List[List[Dict]], # 注意：这是一个列表的列表，每个查询对应一组上下文
    language: str = "en",
    max_batch_size: int = settings.MAX_BATCH_SIZE,
) -> List[str]:
    """
    Batch 生成函数，大幅提升实验速度
    """
    
    # 1. 批量构造 Prompts
    prompts = []
    # 记录每个样本是否有上下文，用于后处理
    has_context_flags = [] 

    for query, context in zip(queries, contexts):
        use_context = bool(context not in (None, []))
        has_context_flags.append(use_context)
        
        context_str = ""
        if use_context:
            context_str = format_context(context, top_n=settings.CONTEXT_TOP_N, language=language)
            
        # 根据语言和任务选择模板
        if language == "en":
            if settings.IS_MMLU:
                prompt = CONTEXT_PROMPT_MMLU_TEMPLATE_EN.format(context=context_str, query=query)
            else:
                prompt = CONTEXT_PROMPT_TEMPLATE_EN.format(context=context_str, query=query)
        else:
            prompt = CONTEXT_PROMPT_TEMPLATE_ZH.format(context=context_str, query=query)
            
        # 无上下文时的 fallback 逻辑 (与单条保持一致)
        if not use_context:
            if language == "en":
                if settings.IS_MMLU:
                    prompt = NO_CONTEXT_PROMPT_MMLU_TEMPLATE_EN.format(query=query)
                else:
                    prompt = NO_CONTEXT_PROMPT_TEMPLATE_EN.format(query=query)
            else:
                prompt = NO_CONTEXT_PROMPT_TEMPLATE_ZH.format(query=query)
        
        prompts.append(prompt)

    # 2. 获取引擎 (单例)
    ENGINE = settings.model_manager.get_engine(settings.MODEL_DIR, settings.ADAPTER_DIR, settings.MODEL_TYPE)
    
    # 3. 构造批量请求
    # Swift/vLLM 的 infer 接口通常支持传入 list[InferRequest]
    infer_requests = [InferRequest(messages=[{'role': 'user', 'content': p}]) for p in prompts]
    request_config = RequestConfig(max_tokens=settings.MAX_NEW_TOKEN, temperature=settings.TEMPERATURE,
                                   )
    
    # 4. 执行批量推理 (最耗时的步骤，现在并行了)
    try:
        resp_list = ENGINE.infer(infer_requests, request_config)
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        # 如果批量失败，返回空字符串列表以避免程序崩溃
        return ["" for _ in queries]

    # 5. 批量后处理
    final_responses = []
    for i, resp in enumerate(resp_list):
        raw_text = resp.choices[0].message.content
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
    if language == "en":
        processed_text = f"{processed_text}"
    else:
        processed_text = f"{processed_text}"

    # 返回处理后的文本
    return processed_text



