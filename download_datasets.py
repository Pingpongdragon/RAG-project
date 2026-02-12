import os
import json
import urllib.request
from pathlib import Path
from typing import Optional
import tarfile
import zipfile
from datasets import load_dataset

class DatasetDownloader:
    """下载 Wizard of Wikipedia 和 HotpotQA 数据集"""
    
    def __init__(self, output_dir: str = "./datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_wizard_of_wikipedia(self) -> None:
        """从 HuggingFace 下载 Wizard of Wikipedia 数据集"""
        print("正在从 HuggingFace 下载 Wizard of Wikipedia 数据集...")
        
        wizard_dir = self.output_dir / "wizard_of_wikipedia"
        wizard_dir.mkdir(exist_ok=True)
        
        try:
            # 从 HuggingFace 加载数据集
            print("  正在加载数据集...")
            dataset = load_dataset("chujiezheng/wizard_of_wikipedia")
            
            # 保存每个 split
            for split in dataset.keys():
                output_file = wizard_dir / f"{split}.json"
                
                if output_file.exists():
                    print(f"✓ {split}.json 已存在，跳过")
                    continue
                
                print(f"  保存 {split} split...")
                # 转换为 list of dicts
                data = [item for item in dataset[split]]
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                print(f"  ✓ {split}.json 保存完成 ({len(data)} 条记录)")
        
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            print("  💡 提示: 请确保已安装 datasets 库")
            print("     pip install datasets huggingface-hub")
    
    def download_hotpotqa(self) -> None:
        """从 HuggingFace 下载 HotpotQA 数据集"""
        print("正在从 HuggingFace 下载 HotpotQA 数据集...")
        
        hotpot_dir = self.output_dir / "hotpotqa"
        hotpot_dir.mkdir(exist_ok=True)
        
        try:
            # 从 HuggingFace 加载数据集
            print("  正在加载数据集...")
            dataset = load_dataset("hotpot_qa", "distractor")
            
            # 保存每个 split
            for split in dataset.keys():
                output_file = hotpot_dir / f"{split}_distractor.json"
                
                if output_file.exists():
                    print(f"✓ {split}_distractor.json 已存在，跳过")
                    continue
                
                print(f"  保存 {split} split (distractor 版本)...")
                data = [item for item in dataset[split]]
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                print(f"  ✓ {split}_distractor.json 保存完成 ({len(data)} 条记录)")
            
            # 加载 fullwiki 版本
            print("  正在加载 fullwiki 版本...")
            dataset_fullwiki = load_dataset("hotpot_qa", "fullwiki")
            
            for split in dataset_fullwiki.keys():
                output_file = hotpot_dir / f"{split}_fullwiki.json"
                
                if output_file.exists():
                    print(f"✓ {split}_fullwiki.json 已存在，跳过")
                    continue
                
                print(f"  保存 {split} split (fullwiki 版本)...")
                data = [item for item in dataset_fullwiki[split]]
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                print(f"  ✓ {split}_fullwiki.json 保存完成 ({len(data)} 条记录)")
        
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            print("  💡 提示: 请确保已安装 datasets 库")
            print("     pip install datasets huggingface-hub")
    
    def verify_datasets(self) -> None:
        """验证数据集完整性"""
        print("\n验证数据集完整性...")
        
        # 验证 Wizard of Wikipedia
        wizard_dir = self.output_dir / "wizard_of_wikipedia"
        if wizard_dir.exists():
            files = list(wizard_dir.glob("*.json"))
            print(f"✓ Wizard of Wikipedia: {len(files)} 个文件")
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name}: {size_mb:.2f} MB")
        else:
            print(f"✗ Wizard of Wikipedia 目录不存在")
        
        # 验证 HotpotQA
        hotpot_dir = self.output_dir / "hotpotqa"
        if hotpot_dir.exists():
            files = list(hotpot_dir.glob("*.json"))
            print(f"✓ HotpotQA: {len(files)} 个文件")
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name}: {size_mb:.2f} MB")
        else:
            print(f"✗ HotpotQA 目录不存在")
    
    def load_sample_data(self, dataset: str = "wizard") -> None:
        """加载并显示样本数据"""
        print(f"\n加载 {dataset} 样本数据...")
        
        if dataset == "wizard":
            file_path = self.output_dir / "wizard_of_wikipedia" / "train.json"
        else:  # hotpotqa
            file_path = self.output_dir / "hotpotqa" / "train_distractor.json"
        
        if not file_path.exists():
            print(f"文件不存在: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 显示第一条样本
            if isinstance(data, list) and len(data) > 0:
                sample = data[0]
                print(f"\n示例样本 (共 {len(data)} 条):")
                print(json.dumps(sample, indent=2, ensure_ascii=False)[:500] + "...")
            elif isinstance(data, dict):
                print(f"数据集包含 {len(data)} 条记录")
        
        except Exception as e:
            print(f"加载失败: {e}")


def main():
    """主函数"""
    downloader = DatasetDownloader(output_dir="./datasets")
    
    # 下载数据集
    downloader.download_wizard_of_wikipedia()
    downloader.download_hotpotqa()
    
    # 验证数据集
    downloader.verify_datasets()
    
    # 加载样本
    downloader.load_sample_data("wizard")
    downloader.load_sample_data("hotpotqa")
    
    print("\n✓ 所有数据集下载完成！")
    print(f"数据位置: {downloader.output_dir.absolute()}")


if __name__ == "__main__":
    main()