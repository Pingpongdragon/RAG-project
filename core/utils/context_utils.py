from typing import List, Dict

def format_context(context: List[Dict], top_n: int = 3) -> str:
    """标准化上下文格式化"""
    if not context:
        return "[无相关上下文]"
    
    items = []
    for idx, item in enumerate(context[:top_n], 1):
        text = item.get('text', '')[:250].replace('\n', ' ')  # 防注入换行符
        source_id = item['metadata'].get('doc_id', '未知')      # 防御性访问
        
        items.append(
            f"[[来源{idx} ID:{source_id}]] {text}..." 
            if len(text) >= 250 else 
            f"[[来源{idx} ID:{source_id}]] {text}"
        )
    
    return "\n\n• ".join(items)

