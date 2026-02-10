from typing import List, Dict

def format_context(
    context: List[Dict], 
    top_n: int = 3, 
    language: str = "zh"  # 默认为中文
) -> str:
    """标准化上下文格式化，支持中英文大语言模型
    
    Args:
        context: 上下文字典列表，需包含text和metadata字段
        top_n: 选择最相关的top_n条上下文（默认3条）
        language: 语言类型，"zh"表示中文，"en"表示英文
        
    Returns:
        格式化后的多段文本，每段结构：
        中文模式：[来源#序号 文档ID:XXX] 截断文本...
        英文模式：[Source#Index DocID:XXX] Truncated text...
        """
    if not context or not isinstance(context, list):
        return "[无相关上下文]" if language == "zh" else "[No relevant context]"
    
    formatted_lines = []
    for idx, item in enumerate(context[:top_n], start=1):
        # 安全获取正文内容（空值保护）
        raw_text = item.get('text', '') or ""
        
        # 多级元数据获取（防御性编程）
        metadata = item.get('metadata') or {}
        doc_id = str(metadata.get('doc_id', '未知ID')).strip()[:20]  # 防止过长ID
        
        # 文本预处理
        clean_text = raw_text.replace('\n', ' ').replace('\t', ' ')
        trunc_text = (clean_text[:300] + '...') if len(clean_text) > 300 else clean_text
        
        # 根据语言动态生成标签
        if language == "en":
            source_tag = f"[Source#{idx} DocID:{doc_id}]"
        else:
            source_tag = f"[来源#{idx} 文档ID:{doc_id}]"
        
        formatted_lines.append(f"{source_tag} {trunc_text}")
    
    # 增强版分隔符（更适合LLM输入的视觉分隔）
    separator = "\n\n" + "="*30 + "\n"  # 添加明显的视觉分割线
    if language == "zh":
        return separator.join(formatted_lines) if formatted_lines else "[无有效上下文]"
    else:
        return separator.join(formatted_lines) if formatted_lines else "[No relevant context]"


    

