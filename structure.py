import json
from datasets import load_dataset
from pathlib import Path
from typing import Dict, Any
import os

# 使用镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class DatasetStructureViewer:
    """查看数据集结构"""
    
    def __init__(self):
        self.indent = "  "
    
    def print_structure(self, obj: Any, level: int = 0, max_str_len: int = 100):
        """递归打印数据结构"""
        prefix = self.indent * level
        
        if isinstance(obj, dict):
            print(f"{prefix}{{")
            for key, value in list(obj.items())[:10]:  # 只显示前10个键
                print(f"{prefix}  '{key}': ", end="")
                if isinstance(value, (dict, list)):
                    print()
                    self.print_structure(value, level + 2, max_str_len)
                else:
                    val_str = str(value)
                    if len(val_str) > max_str_len:
                        val_str = val_str[:max_str_len] + "..."
                    print(f"{repr(val_str)},")
            if len(obj) > 10:
                print(f"{prefix}  ... ({len(obj) - 10} more keys)")
            print(f"{prefix}}}")
        
        elif isinstance(obj, list):
            print(f"{prefix}[")
            for i, item in enumerate(obj[:3]):  # 只显示前3个元素
                self.print_structure(item, level + 1, max_str_len)
                if i < min(2, len(obj) - 1):
                    print(f"{prefix},")
            if len(obj) > 3:
                print(f"{prefix}  ... ({len(obj) - 3} more items)")
            print(f"{prefix}]")
        
        else:
            val_str = str(obj)
            if len(val_str) > max_str_len:
                val_str = val_str[:max_str_len] + "..."
            print(f"{prefix}{repr(val_str)}")
    
    def view_wizard_of_wikipedia(self):
        """查看 Wizard of Wikipedia 数据集结构"""
        print("=" * 80)
        print("📚 Wizard of Wikipedia 数据集结构")
        print("=" * 80)
        
        try:
            # 加载数据集（只加载一小部分）
            print("\n正在加载数据集...")
            dataset = load_dataset("chujiezheng/wizard_of_wikipedia", split="train[:10]")
            
            print(f"\n✅ 数据集信息:")
            print(f"  - Split: train (showing first 10 samples)")
            print(f"  - Total features: {len(dataset.features)}")
            print(f"  - Features: {list(dataset.features.keys())}")
            
            # 显示第一个样本的完整结构
            print(f"\n📋 第一个样本的完整结构:")
            sample = dataset[0]
            self.print_structure(sample)
            
            # 显示字段说明
            print(f"\n📝 字段说明:")
            print(f"  • chosen_topic: 选择的主题")
            print(f"  • persona: 角色设定")
            print(f"  • wizard_eval: 巫师评分")
            print(f"  • dialog: 对话内容列表")
            print(f"  • knowledge: 每轮对话的候选知识段落")
            print(f"  • topics: 可选主题列表")
            
            # 统计信息
            print(f"\n📊 统计信息:")
            topics_count = len(sample.get('topics', []))
            dialog_turns = len(sample.get('dialog', []))
            knowledge_turns = len(sample.get('knowledge', []))
            print(f"  • 可选主题数: {topics_count}")
            print(f"  • 对话轮数: {dialog_turns}")
            print(f"  • 知识段落组数: {knowledge_turns}")
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
    
    def view_hotpotqa_distractor(self):
        """查看 HotpotQA Distractor 数据集结构"""
        print("\n" + "=" * 80)
        print("🔥 HotpotQA Distractor 数据集结构")
        print("=" * 80)
        
        try:
            print("\n正在加载数据集...")
            dataset = load_dataset("hotpot_qa", "distractor", split="train[:10]")
            
            print(f"\n✅ 数据集信息:")
            print(f"  - Split: train (showing first 10 samples)")
            print(f"  - Total features: {len(dataset.features)}")
            print(f"  - Features: {list(dataset.features.keys())}")
            
            # 显示第一个样本
            print(f"\n📋 第一个样本的完整结构:")
            sample = dataset[0]
            self.print_structure(sample)
            
            # 显示字段说明
            print(f"\n📝 字段说明:")
            print(f"  • id: 样本唯一标识")
            print(f"  • question: 问题")
            print(f"  • answer: 答案")
            print(f"  • type: 问题类型 (comparison/bridge)")
            print(f"  • level: 难度级别 (easy/medium/hard)")
            print(f"  • supporting_facts: 支持事实 [[title, sent_id], ...]")
            print(f"  • context: 10篇文档 (2篇金标准 + 8篇干扰)")
            
            # 统计信息
            print(f"\n📊 统计信息:")
            print(f"  • 问题类型: {sample.get('type')}")
            print(f"  • 难度: {sample.get('level')}")
            print(f"  • 文档数量: {len(sample.get('context', []))}")
            print(f"  • 支持事实数量: {len(sample.get('supporting_facts', []))}")
            
            # 显示 context 结构
            if sample.get('context'):
                print(f"\n📄 Context 结构示例 (第一篇文档):")
                first_doc = sample['context'][0]
                print(f"  Title: {first_doc[0]}")
                print(f"  Sentences: {len(first_doc[1])} 句")
                print(f"  First sentence: {first_doc[1][0][:100]}...")
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
    
    def view_hotpotqa_fullwiki(self):
        """查看 HotpotQA FullWiki 数据集结构"""
        print("\n" + "=" * 80)
        print("🌐 HotpotQA FullWiki 数据集结构")
        print("=" * 80)
        
        try:
            print("\n正在加载数据集...")
            dataset = load_dataset("hotpot_qa", "fullwiki", split="train[:10]")
            
            print(f"\n✅ 数据集信息:")
            print(f"  - Split: train (showing first 10 samples)")
            print(f"  - Total features: {len(dataset.features)}")
            print(f"  - Features: {list(dataset.features.keys())}")
            
            # 显示第一个样本
            print(f"\n📋 第一个样本的完整结构:")
            sample = dataset[0]
            self.print_structure(sample)
            
            # 显示字段说明
            print(f"\n📝 字段说明:")
            print(f"  • id: 样本唯一标识")
            print(f"  • question: 问题")
            print(f"  • answer: 答案")
            print(f"  • type: 问题类型")
            print(f"  • level: 难度级别")
            print(f"  • supporting_facts: 支持事实")
            print(f"  ⚠️  注意: FullWiki 版本没有 context 字段")
            print(f"  ⚠️  需要从完整 Wikipedia 中检索文档")
            
            # 统计信息
            print(f"\n📊 统计信息:")
            print(f"  • 问题类型: {sample.get('type')}")
            print(f"  • 难度: {sample.get('level')}")
            print(f"  • 支持事实数量: {len(sample.get('supporting_facts', []))}")
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
    
    def compare_datasets(self):
        """对比两个数据集的差异"""
        print("\n" + "=" * 80)
        print("🔍 数据集对比")
        print("=" * 80)
        
        print("""
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ 特性                │ Wizard of Wikipedia  │ HotpotQA            │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ 任务类型            │ 知识对话生成         │ 多跳问答             │
│ 数据格式            │ 对话轮次             │ 问题-答案对          │
│ 知识来源            │ Wikipedia            │ Wikipedia           │
│ 推理跳数            │ 1跳                  │ 2-4跳                │
│ 平均对话轮数        │ 9轮                  │ N/A                 │
│ 候选文档数          │ 多个段落/轮          │ 10篇 (distractor)   │
│ 标注类型            │ 对话+知识选择        │ 问答+支持事实        │
│ 训练集大小          │ ~18k 对话            │ ~90k 问题            │
└─────────────────────┴──────────────────────┴──────────────────────┘

📌 使用建议:
  • Wizard of Wikipedia: 适合对话系统、知识增强生成
  • HotpotQA: 适合推理能力、多跳问答研究
        """)
    
    def save_samples_to_file(self, output_dir: str = "./dataset_samples"):
        """保存样本到文件以便详细查看"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 80)
        print("💾 保存样本到文件")
        print("=" * 80)
        
        datasets_to_save = [
            ("chujiezheng/wizard_of_wikipedia", None, "wizard_sample.json"),
            ("hotpot_qa", "distractor", "hotpotqa_distractor_sample.json"),
            ("hotpot_qa", "fullwiki", "hotpotqa_fullwiki_sample.json"),
        ]
        
        for dataset_name, config, filename in datasets_to_save:
            try:
                print(f"\n正在保存 {dataset_name} ({config or 'default'})...")
                
                if config:
                    dataset = load_dataset(dataset_name, config, split="train[:5]")
                else:
                    dataset = load_dataset(dataset_name, split="train[:5]")
                
                # 转换为 list of dicts
                samples = [dict(item) for item in dataset]
                
                output_file = output_path / filename
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(samples, f, ensure_ascii=False, indent=2)
                
                print(f"✅ 已保存 {len(samples)} 个样本到: {output_file}")
                
            except Exception as e:
                print(f"❌ 保存失败: {e}")


def main():
    viewer = DatasetStructureViewer()
    
    # 查看各个数据集结构
    viewer.view_wizard_of_wikipedia()
    viewer.view_hotpotqa_distractor()
    viewer.view_hotpotqa_fullwiki()
    
    # 对比数据集
    viewer.compare_datasets()
    
    # 保存样本到文件
    viewer.save_samples_to_file()
    
    print("\n" + "=" * 80)
    print("✅ 所有数据集结构查看完成！")
    print("💡 提示: 查看 ./dataset_samples/ 目录下的 JSON 文件获取更多细节")
    print("=" * 80)


if __name__ == "__main__":
    main()