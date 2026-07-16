import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# ================= 配置 =================
DATA_ROOT = Path("./data")
RAW_DATA_DIR = DATA_ROOT / "raw_data"
DATASET_NAME = "fiqa"
URL = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET_NAME}.zip"

def download_and_unzip():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DATA_DIR / f"{DATASET_NAME}.zip"
    extract_path = RAW_DATA_DIR / DATASET_NAME

    # 1. 下载
    if not zip_path.exists():
        print(f"🚀 正在下载 {DATASET_NAME} 数据集 (可能需要几分钟)...")
        response = requests.get(URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as file, tqdm(
            desc=DATASET_NAME,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    else:
        print(f"📦 压缩包已存在: {zip_path}")

    # 2. 解压
    if not extract_path.exists():
        print("📂 正在解压...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print(f"✅ 解压完成: {extract_path}")
    else:
        print(f"✅ 数据目录已准备好: {extract_path}")

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


MODEL_NAME = "BAAI/bge-reranker-base"

def download_model():
    print(f"\n🚀 开始下载模型: {MODEL_NAME}")
    print(f"   镜像源: {os.environ.get('HF_ENDPOINT')}")
    
    try:
        # 1. 使用 snapshot_download 下载所有文件到缓存
        path = snapshot_download(
            repo_id=MODEL_NAME,
            resume_download=True,
            local_files_only=False
        )
        print(f"✅ 模型文件下载完成，存储路径: {path}")
        
        # 2. 尝试加载一次，确保文件完整可用
        print("🔄 正在尝试加载模型以验证完整性...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        
        print(f"🎉 验证成功！模型已准备就绪。")
        
    except Exception as e:
        print(f"\n❌ 下载或加载失败: {str(e)}")
        print("建议检查网络连接或稍后重试。")



if __name__ == "__main__":
    download_model()