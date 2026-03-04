import sys
import os
from openai import OpenAI

# 将项目根目录添加到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_direct_connection():
    """直接使用已知可用的配置进行测试"""
    print("正在测试本地 Qwen3 模型连接...")
    
    client = OpenAI(
        api_key="none",
        base_url="http://202.45.128.234:5788/v1/"
    )
    model_name = "/nfs/whlu/models/Qwen3-Coder-30B-A3B-Instruct"
    
    try:
        print(f"正在连接至模型: {model_name}...")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello, this is a connection test."}],
            max_tokens=10
        )
        print("✅ 访问成功！")
        print(f"响应内容: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ 访问失败: {e}")
        return False

def test_config_based_connection():
    """使用 settings.py 配置进行测试"""
    from core.generator import _get_api_client, _get_model_name
    from config import settings
    
    print(f"\n正在测试配置文件中的模型: {settings.ACTIVE_MODEL_TYPE}")
    client = _get_api_client()
    model = _get_model_name()
    
    if not client:
        print("错误: 未能获取 API 客户端，请检查 ACTIVE_MODEL_TYPE 配置。")
        return False

    try:
        print(f"正在连接至模型: {model}...")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello, this is a connection test."}],
            max_tokens=10
        )
        print("✅ 访问成功！")
        print(f"响应内容: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ 访问失败: {e}")
        return False

if __name__ == "__main__":
    # 先测试已知可用的配置
    direct_ok = test_direct_connection()
    
    # 再测试项目配置
    config_ok = test_config_based_connection()
    
    if direct_ok and not config_ok:
        print("\n⚠️ 直接连接成功但配置文件连接失败，请检查 config/settings.py 中的配置是否正确。")