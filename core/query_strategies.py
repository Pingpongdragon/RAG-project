"""
Advanced RAG Query Strategies

参考: https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies

实现的策略:
1. Query Expansion   - 用 LLM 扩展简短查询为详细版本
2. Multi-Query RAG   - 生成多个查询变体并行检索 + 去重 
3. Self-Reflective RAG - 检索后自评, 低分则修正查询重试
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)


def _get_llm_client() -> Tuple[Optional[OpenAI], Optional[str]]:
    """获取 LLM 客户端 (复用 generator 的配置)"""
    if settings.ACTIVE_MODEL_TYPE == "qwen":
        client = OpenAI(
            api_key=settings.QWEN_API_CONFIG["api_key"],
            base_url=settings.QWEN_API_CONFIG["base_url"],
        )
        model = settings.QWEN_API_CONFIG["model"]
    elif settings.ACTIVE_MODEL_TYPE == "deepseek":
        client = OpenAI(
            api_key=settings.DEEPSEEK_CONFIG["api_key"],
            base_url=settings.DEEPSEEK_CONFIG["base_url"],
        )
        model = settings.DEEPSEEK_CONFIG["model"]
    else:
        return None, None
    return client, model


# ============================================================
# Strategy 1: Query Expansion
# ============================================================

QUERY_EXPANSION_PROMPT = """You are a query expansion assistant. Take brief user queries and expand them
into more detailed, comprehensive versions that:
1. Add relevant context and clarifications
2. Include related terminology and concepts
3. Specify what aspects should be covered
4. Maintain the original intent
5. Keep it as a single, coherent question

Expand the query to be 2-3x more detailed while staying focused.
Output ONLY the expanded query, nothing else."""


def expand_query(query: str) -> str:
    """
    Strategy 1: Query Expansion
    
    用 LLM 将简短 query 扩展为详细版本, 提高检索精度。
    
    参考: ottomator-agents Strategy #5
    - Input:  "What is RAG?"
    - Output: "What is Retrieval-Augmented Generation (RAG), how does it combine 
               information retrieval with language generation, what are its key 
               components and architecture?"
    """
    client, model = _get_llm_client()
    if client is None:
        logger.warning("No LLM client available for query expansion, returning original query")
        return query
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QUERY_EXPANSION_PROMPT},
                {"role": "user", "content": f"Expand this query: {query}"},
            ],
            temperature=settings.QUERY_EXPANSION_TEMPERATURE,
            max_tokens=256,
        )
        expanded = response.choices[0].message.content.strip()
        logger.info(f"Query expanded: '{query[:50]}...' → '{expanded[:80]}...'")
        return expanded
    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        return query


# ============================================================
# Strategy 2: Multi-Query RAG
# ============================================================

MULTI_QUERY_PROMPT = """Generate {n} different variations/perspectives of the following search query.
Each variation should approach the topic from a different angle while maintaining the same intent.
Output ONLY the variations, one per line, numbered 1-{n}. No other text."""


def generate_query_variations(query: str, num_variations: int = 3) -> List[str]:
    """
    Strategy 2: Multi-Query RAG
    
    生成多个查询变体, 从不同角度检索, 然后去重合并。
    
    参考: ottomator-agents Strategy #6
    - Input:  "How does photosynthesis work?"
    - Output: ["How does photosynthesis work?",
               "What is the biological process of converting sunlight to energy in plants?",
               "Explain the light and dark reactions in plant metabolism",
               "What are the steps of carbon fixation in chloroplasts?"]
    """
    client, model = _get_llm_client()
    if client is None:
        logger.warning("No LLM client for multi-query, returning original query only")
        return [query]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": MULTI_QUERY_PROMPT.format(n=num_variations) + f"\n\nOriginal query: {query}"},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        text = response.choices[0].message.content.strip()
        
        # 解析编号列表
        variations = []
        for line in text.split("\n"):
            line = line.strip()
            # 去掉编号前缀如 "1. " 或 "1) "
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned and len(cleaned) > 5:
                variations.append(cleaned)
        
        # 始终包含原始查询
        all_queries = [query] + variations[:num_variations]
        logger.info(f"Multi-query: generated {len(all_queries)} variations")
        return all_queries
    except Exception as e:
        logger.error(f"Multi-query generation failed: {e}")
        return [query]


def deduplicate_results(results_lists: List[List[Dict]], key: str = "text", top_k: int = 10) -> List[Dict]:
    """
    对多组检索结果去重, 保留每个文档的最高分数。
    
    Args:
        results_lists: 多次检索的结果列表
        key: 用于去重的字段名
        top_k: 最终返回数量
    """
    seen = {}
    for results in results_lists:
        for item in results:
            doc_key = item.get(key, "")
            if not doc_key:
                continue
            # 取最优分数
            existing = seen.get(doc_key)
            if existing is None:
                seen[doc_key] = item
            else:
                # 比较 rerank_score 或 hybrid_score
                new_score = item.get("scores", {}).get("rerank", 0) or item.get("scores", {}).get("hybrid", 0)
                old_score = existing.get("scores", {}).get("rerank", 0) or existing.get("scores", {}).get("hybrid", 0)
                if new_score > old_score:
                    seen[doc_key] = item
    
    # 按最高分排序
    deduped = sorted(
        seen.values(),
        key=lambda x: x.get("scores", {}).get("rerank", 0) or x.get("scores", {}).get("hybrid", 0),
        reverse=True,
    )
    return deduped[:top_k]


# ============================================================
# Strategy 3: Self-Reflective RAG
# ============================================================

GRADE_PROMPT = """You are evaluating retrieval quality. Given a query and retrieved documents, 
grade how relevant and useful the retrieval results are on a scale of 1-5.

Query: {query}

Retrieved content (first 500 chars): {content}

Grade 1-5 where:
1 = Completely irrelevant
2 = Barely related  
3 = Somewhat relevant but missing key info
4 = Mostly relevant and useful
5 = Highly relevant and comprehensive

Respond with ONLY a single number (1-5)."""


REFINE_PROMPT = """The following search query returned low-relevance results. 
Suggest an improved version of the query that might retrieve better results.
Keep the same intent but rephrase it to be more specific and searchable.

Original query: {query}

Respond with ONLY the improved query, nothing else."""


def grade_retrieval(query: str, results: List[Dict]) -> int:
    """
    Strategy 3: Self-Reflective RAG - 评分阶段
    
    让 LLM 评估检索结果与 query 的相关性 (1-5 分)。
    
    参考: ottomator-agents Strategy #10
    """
    client, model = _get_llm_client()
    if client is None:
        return 5  # 无法评估则跳过
    
    # 拼接检索内容摘要
    content_preview = "\n".join(
        [r.get("text", "")[:200] for r in results[:3]]
    )[:500]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": GRADE_PROMPT.format(query=query, content=content_preview)},
            ],
            temperature=0,
            max_tokens=5,
        )
        text = response.choices[0].message.content.strip()
        # 提取数字
        match = re.search(r"[1-5]", text)
        grade = int(match.group()) if match else 3
        logger.info(f"Self-reflection grade: {grade}/5 for query '{query[:50]}...'")
        return grade
    except Exception as e:
        logger.error(f"Grading failed: {e}")
        return 3  # 默认中等分数


def refine_query(query: str) -> str:
    """
    Strategy 3: Self-Reflective RAG - 修正阶段
    
    当检索质量评分低时, 用 LLM 修正查询。
    """
    client, model = _get_llm_client()
    if client is None:
        return query
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": REFINE_PROMPT.format(query=query)},
            ],
            temperature=0.3,
            max_tokens=128,
        )
        refined = response.choices[0].message.content.strip()
        logger.info(f"Query refined: '{query[:50]}...' → '{refined[:80]}...'")
        return refined
    except Exception as e:
        logger.error(f"Query refinement failed: {e}")
        return query
