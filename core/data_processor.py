from typing import List, Tuple
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from datasets import load_dataset
from RAG_project.models.embeddings import SafeHuggingFaceEmbedder
from RAG_project.config import settings
import pickle
from RAG_project.config.logger_config import configure_console_logger
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import CharacterTextSplitter
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from streamingqa import extraction

logger = configure_console_logger(__name__)  # 使用统一配置的logger

# 创建 LangChain 兼容的 Embeddings 对象
class CompatibleEmbeddings:
    """兼容 LangChain 的 Embeddings 包装器"""
    def __init__(self):
        self.client = SafeHuggingFaceEmbedder().embedder
        self.model_name = settings.EMBEDDING_MODEL
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        embeddings = self.client.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        embedding = self.client.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()
    
    def __call__(self, text: str) -> List[float]:
        """使对象可调用，用于向后兼容"""
        return self.embed_query(text)
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步嵌入文档（fallback到同步）"""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """异步嵌入查询（fallback到同步）"""
        return self.embed_query(text)

# 全局 embedding service
embedding_service = CompatibleEmbeddings()

# 创建 LangChain 兼容的 Embeddings 对象
# 全局 embedding service
embedding_service = CompatibleEmbeddings()

# -------------------------
# 数据集处理相关
# -------------------------

def _load_wmt_dataset(counter_size=30000) -> List[Document]:
    """加载WMT数据集"""
    logger.info(f"🚀 正在加载WMT数据集")
    wmt_docs = []
    wmt_dir = Path(settings.WMT_DIR)
    # 假设这里配置了WMT存档文件路径和去重排序键文件路径
    wmt_archive_file_paths = [
        str(wmt_dir / 'news-docs.2019.en.filtered.gz'),
    ]
    streamingqa_dir = Path(settings.STREAMINGQA_DIR)    
    deduplicated_sorting_keys_file_path = str(streamingqa_dir / 'wmt_sorting_key_ids.txt.gz')

    try:
        wmt_doc_objects = extraction.get_deduplicated_wmt_docs(
            wmt_archive_files=wmt_archive_file_paths,
            deduplicated_sorting_keys_file=deduplicated_sorting_keys_file_path
        )
         # 随机采样文档
        wmt_passage_objects = extraction.get_wmt_passages_from_docs(wmt_doc_objects)
        counter = 0
        for wmt_passage in wmt_passage_objects:
            if counter >= counter_size:
                break
            passage = Document(
                page_content=wmt_passage.text.decode(),
                metadata={
                    "doc_id": wmt_passage.id.split('_')[0], 
                    "source": "WMT"
                }
            )
            wmt_docs.append(passage)
            counter += 1
    except Exception as e:
        logger.error(f"🔥 加载WMT数据集失败: {str(e)}")

    logger.info(f"📊 加载完成: 共 {len(wmt_docs)} 条")
    return wmt_docs

def _load_hf_dataset(cfg=None) -> List[Document]:
    """加载数据集，针对MMLU只保留问题和正确答案
    
    Args:
        cfg: 数据集配置，为None时使用默认配置
        
    Returns:
        List[Document]: 加载的问答文档列表
    """
    if not cfg:
        cfg = settings.KNOWLEDGE_DATASET_CONFIG
    
    logger.info(f"🚀 正在加载数据集 {cfg['dataset_name']} from {cfg['dataset_source']}")
    
    if cfg['dataset_source'] == 'local':
        return _load_wmt_dataset()
    
    # 判断是否为MMLU数据集，以便特殊处理
    is_mmlu = "mmlu" in cfg['dataset_name'].lower()
    documents = []
    
    # 处理配置名称
    config_names = cfg['config_name']
    if isinstance(config_names, str):
        config_names = [config_names]
        
    for config_name in config_names:
        logger.info(f"加载配置: {config_name}")
        try:
            # 加载数据集
            cache_dir = Path(settings.DATA_CACHE_DIR) 
            raw_data = load_dataset(
                cfg['dataset_name'], 
                config_name,
                split=cfg['split'],
                cache_dir=str(cache_dir)
            )
            
            # MMLU特殊处理: 提取问题和正确答案
            if is_mmlu and "question" in raw_data.column_names and "answer" in raw_data.column_names:
                logger.info(f"检测到MMLU数据集，提取问题和正确答案...")
                
                valid_items = []
                for item in raw_data:
                    if "question" in item and "choices" in item and "answer" in item:
                        try:
                            # 映射答案字母到选项
                            answer_index = item['answer']
                            if 0 <= answer_index < len(item["choices"]):
                                valid_items.append({
                                    "id": item.get(cfg['id_column'], f"item_{len(valid_items)}"),
                                    "question": item["question"],
                                    "correct_answer": item["choices"][answer_index],
                                    "subject": config_name.replace("_", " ")
                                })
                        except Exception as e:
                            logger.error(f"处理MMLU问题时出错: {str(e)}")
                
                # 创建文档
                config_docs = []
                for item in valid_items:
                    # 格式化内容为问题和正确答案
                    content = f"Subject: {item['subject']}\nQuestion: {item['question']}\nAnswer: {item['correct_answer']}"
                    
                    doc = Document(
                        page_content=content,
                        metadata={ 
                            "doc_id": f"{config_name}_{item['id']}", 
                            "source": f"{cfg['dataset_name']}_{config_name}",
                            "config": config_name,
                            "subject": item['subject']
                        }
                    )
                    config_docs.append(doc)
                
                logger.info(f"从MMLU配置 {config_name} 提取了 {len(config_docs)} 个问答对")
                documents.extend(config_docs)
                
            # 标准处理流程
            else:
                required_columns = cfg['text_columns'] + [cfg['id_column']]
                valid_items = [
                    item for item in raw_data 
                    if all(k in item for k in required_columns)
                ]
                
                logger.info(f"配置 {config_name} 加载完成: 有效 {len(valid_items)}条")
                
                # 创建标准文档
                config_docs = [
                    Document(
                        page_content="\n".join(str(item[col]) for col in cfg['text_columns']),
                        metadata={ 
                            "doc_id": f"{config_name}_{item[cfg['id_column']]}", 
                            "source": f"{cfg['dataset_name']}_{config_name}",
                            "config": config_name
                        }
                    ) for item in valid_items
                ]
                
                documents.extend(config_docs)
                
        except Exception as e:
            logger.error(f"🔥 配置 {config_name} 加载失败: {str(e)}")
    
    logger.info(f"📊 所有配置加载完成: 共 {len(documents)} 条文档")
    return documents

def _build_hybrid_vector_index(docs: List[Document]) -> FAISS:
    """基于纯文本的混合索引构建"""
    
    cfg = settings.KNOWLEDGE_DATASET_CONFIG
    text_splitter = CharacterTextSplitter(
        chunk_size=cfg['chunk_size'],
        chunk_overlap=cfg['chunk_overlap']
    )
    
    processed = []
    for doc in docs:
        try:
            # 直接分块原始文本内容
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                processed.append(Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": doc.metadata["doc_id"],
                        "chunk_id": i,
                        "source": doc.metadata["source"]
                    }
                ))
        except KeyError as e:
            logger.error(f"元数据字段缺失: {e} 于文档 {doc.metadata}")
        except Exception as e:
            logger.error(f"处理异常: {str(e)}")
    
    logger.info(f"🎯 生成 {len(processed)} 个分块")
    return FAISS.from_documents(
        processed, 
        embedding_service,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )

# -------------------------
# 索引管理主函数
# -------------------------
def load_or_build_index() -> Tuple[FAISS, List[Document]]:
    """自动化索引全生命周期管理"""
    
    cfg = settings.KNOWLEDGE_DATASET_CONFIG
    index_path = Path(cfg['index_save_path'])
    docs_file = Path(settings.DATA_CACHE_DIR)  / "docs.pkl"
    
    # 当存在完整索引时加载
    if (index_path / "index.faiss").exists() and docs_file.exists():
        logger.info(f"🔄 加载现有索引: {index_path}")
        vector_db = FAISS.load_local(
            folder_path=str(index_path),
            embeddings=embedding_service,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
        )
        with open(docs_file, "rb") as f:
            docs = pickle.load(f)
        return vector_db, docs
    
    # 新建索引流程
    logger.warning("⚠️ 无可用索引，启动构建流程...")
    try:
        docs = _load_hf_dataset()
        if not docs:
            raise ValueError("数据集加载后为空，请检查配置")
            
        index = _build_hybrid_vector_index(docs)
        
        # 保证保存路径存在
        index_path.mkdir(parents=True, exist_ok=True)
        index.save_local(str(index_path))
        
        with open(docs_file, "wb") as f:
            pickle.dump(docs, f)
            
        logger.success(f"✅ 新索引保存至项目下目录 {index_path}")
        return index, docs
    except Exception as e:
        logger.critical(f"❌ 索引构建失败: {str(e)}", exc_info=True)
        raise

# -------------------------
# 增量更新相关功能（新增部分）
# -------------------------

import hashlib
import json
from datetime import datetime

class KnowledgeBaseUpdater:
    """知识库增量更新管理器"""
    
    def __init__(self, vector_db: FAISS, docs_list: List[Document]):
        self.vector_db = vector_db
        self.docs_list = docs_list
        self.metadata_file = Path(settings.DATA_CACHE_DIR) / "kb_metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """加载知识库元数据"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                # 调试：打印元数据结构
                
                # 自动修正：确保所有 documents 的 value 都是 dict
                if isinstance(self.metadata.get("documents"), dict):
                    fixed_docs = {}
                    for k, v in self.metadata.get("documents", {}).items():
                        if isinstance(v, dict):
                            fixed_docs[k] = v
                        else:
                            logger.warning(f"⚠️ 文档 {k} 的值不是字典，跳过: {v}")
                            fixed_docs[k] = {}
                    self.metadata["documents"] = fixed_docs
                else:
                    logger.warning(f"⚠️ documents字段类型异常: {type(self.metadata.get('documents'))}")
                    self.metadata["documents"] = {}
            else:
                logger.info("📝 元数据文件不存在，初始化为空")
                self.metadata = {
                    "documents": {},  # doc_id -> {hash, timestamp, chunks_count}
                    "last_updated": None,
                    "total_chunks": 0
                }
        except Exception as e:
            logger.error(f"❌ 加载元数据失败: {e}，初始化为空")
            self.metadata = {
                "documents": {},
                "last_updated": None,
                "total_chunks": 0
            }

    def save_metadata(self):
        """保存元数据到文件"""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def _calculate_content_hash(self, content: str) -> str:
        """计算内容的哈希值"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def add_documents(self, new_docs: List[Document], force_update=False) -> bool:
        """增量添加新文档到知识库"""
        logger.info(f"🚀 开始增量添加 {len(new_docs)} 个文档")
        
        cfg = settings.KNOWLEDGE_DATASET_CONFIG
        text_splitter = CharacterTextSplitter(
            chunk_size=cfg['chunk_size'],
            chunk_overlap=cfg['chunk_overlap']
        )
        
        new_chunks = []
        updated_docs = 0
        
        for doc in new_docs:
            doc_id = doc.metadata.get("doc_id")
            if not doc_id:
                logger.warning(f"文档缺少doc_id，跳过: {doc.page_content[:50]}...")
                continue
            
            content_hash = self._calculate_content_hash(doc.page_content)
            
            # 检查文档是否已存在且内容未变化
            if doc_id in self.metadata["documents"] and not force_update:
                existing_hash = self.metadata["documents"][doc_id].get("hash")
                if existing_hash == content_hash:
                    logger.info(f"文档 {doc_id} 内容无变化，跳过")
                    continue
                else:
                    # 内容已变化，先删除旧版本
                    self.remove_document(doc_id, save_metadata=False)
            
            # 处理新文档
            try:
                chunks = text_splitter.split_text(doc.page_content)
                chunk_count = 0
                
                for i, chunk in enumerate(chunks):
                    chunk_doc = Document(
                        page_content=chunk,
                        metadata={
                            "doc_id": doc_id,
                            "chunk_id": i,
                            "source": doc.metadata.get("source", "unknown"),
                            "timestamp": datetime.now().isoformat(),
                            **{k: v for k, v in doc.metadata.items() if k not in ["doc_id", "chunk_id"]}
                        }
                    )
                    new_chunks.append(chunk_doc)
                    chunk_count += 1
                
                # 更新元数据
                self.metadata["documents"][doc_id] = {
                    "hash": content_hash,
                    "timestamp": datetime.now().isoformat(),
                    "chunks_count": chunk_count,
                    "source": doc.metadata.get("source", "unknown")
                }
                updated_docs += 1
                
            except Exception as e:
                logger.error(f"处理文档 {doc_id} 时出错: {str(e)}")
                continue
        
        if new_chunks:
            # 添加新chunks到向量数据库
            logger.info(f"📝 添加 {len(new_chunks)} 个新块到向量数据库")
            self.vector_db.add_documents(new_chunks)
            
            # 更新文档列表
            self.docs_list.extend(new_chunks)
            
            # 保存更新后的索引和文档
            self._save_index_and_docs()
            
            # 更新总计数
            self.metadata["total_chunks"] = len(self.docs_list)
            self.save_metadata()
            
            logger.info(f"✅ 成功添加/更新 {updated_docs} 个文档，生成 {len(new_chunks)} 个块")
            return True
        
        logger.info("📋 没有新内容需要添加")
        return False
    
    def remove_document(self, doc_id: str, save_metadata: bool = True) -> bool:
        """删除指定文档的所有块"""
        logger.info(f"🗑️ 删除文档: {doc_id}")
        
        # 找到要删除的块索引
        indices_to_remove = []
        for i, doc in enumerate(self.docs_list):
            if doc.metadata.get("doc_id") == doc_id:
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            logger.warning(f"未找到文档 {doc_id}")
            return False
        
        # 从后往前删除，避免索引变化
        for i in reversed(indices_to_remove):
            del self.docs_list[i]
        
        # 重建向量索引（FAISS不支持直接删除，需要重建）
        if self.docs_list:
            self._rebuild_vector_index()
        else:
            # 如果没有文档了，创建空索引
            self.vector_db = FAISS.from_documents(
                [Document(page_content="dummy", metadata={"doc_id": "dummy"})], 
                embedding_service,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )
        
        # 更新元数据
        if doc_id in self.metadata["documents"]:
            del self.metadata["documents"][doc_id]
        
        self.metadata["total_chunks"] = len(self.docs_list)
        
        if save_metadata:
            self.save_metadata()
            self._save_index_and_docs()
        
        logger.info(f"✅ 已删除文档 {doc_id}，移除 {len(indices_to_remove)} 个块")
        return True
    
    def update_document(self, doc_id: str, new_content: str, new_metadata: dict = None) -> bool:
        """更新现有文档的内容"""
        logger.info(f"✏️ 更新文档: {doc_id}")
        
        # 先删除旧文档
        if not self.remove_document(doc_id, save_metadata=False):
            logger.warning(f"文档 {doc_id} 不存在，将作为新文档添加")
        
        # 创建新文档
        updated_metadata = new_metadata or {}
        updated_metadata["doc_id"] = doc_id
        
        new_doc = Document(
            page_content=new_content,
            metadata=updated_metadata
        )
        
        # 添加新版本
        success = self.add_documents([new_doc], force_update=True)
        
        if success:
            logger.info(f"✅ 文档 {doc_id} 更新完成")
        else:
            logger.error(f"❌ 文档 {doc_id} 更新失败")
        
        return success
    
    def _rebuild_vector_index(self):
        """重建向量索引"""
        logger.info("🔄 重建向量索引...")
        if self.docs_list:
            self.vector_db = FAISS.from_documents(
                self.docs_list, 
                embedding_service,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )
    
    def _save_index_and_docs(self):
        """保存索引和文档到文件"""
        cfg = settings.KNOWLEDGE_DATASET_CONFIG
        index_path = Path(cfg['index_save_path'])
        docs_file = Path(settings.DATA_CACHE_DIR) / "docs.pkl"
        
        # 保存索引
        index_path.mkdir(parents=True, exist_ok=True)
        self.vector_db.save_local(str(index_path))
        
        # 保存文档
        with open(docs_file, "wb") as f:
            pickle.dump(self.docs_list, f)
    
    def get_document_info(self, doc_id: str = None) -> dict:
        """获取文档信息"""
        if doc_id:
            return self.metadata["documents"].get(doc_id, {})
        else:
            return {
                "total_documents": len(self.metadata["documents"]),
                "total_chunks": self.metadata["total_chunks"],
                "last_updated": self.metadata["last_updated"],
                "documents": self.metadata["documents"]
            }
    
    def list_sources(self) -> List[str]:
        """列出所有数据源"""
        sources = set()
        for doc_info in self.metadata["documents"].values():
            if isinstance(doc_info, dict):
                sources.add(doc_info.get("source", "unknown"))
        return sorted(list(sources))

def get_knowledge_base_updater() -> KnowledgeBaseUpdater:
    """获取知识库更新器的便捷函数"""
    vector_db, docs_list = load_or_build_index()
    return KnowledgeBaseUpdater(vector_db, docs_list)

def demo_incremental_update():
    """演示增量更新功能"""
    logger.info("🔬 开始增量更新演示...")
    
    try:
        # 获取更新器
        updater = get_knowledge_base_updater()
        
        # 显示当前状态
        info = updater.get_document_info()
        logger.info(f"📊 当前知识库状态: {info['total_documents']} 个文档，{info['total_chunks']} 个块")
        
        # 添加新文档示例
        new_docs = [
            Document(
                page_content="这是一个新的测试文档，用于演示增量更新功能。它包含了关于机器学习的基础知识。",
                metadata={"doc_id": "test_doc_1", "source": "manual_test", "topic": "machine_learning"}
            ),
            Document(
                page_content="深度学习是机器学习的一个子集，它使用多层神经网络来学习数据的表示。",
                metadata={"doc_id": "test_doc_2", "source": "manual_test", "topic": "deep_learning"}
            )
        ]
        
        # 执行增量添加
        success = updater.add_documents(new_docs)
        if success:
            logger.info("✅ 新文档添加成功")
        
        # 更新文档示例
        updated_success = updater.update_document(
            "test_doc_1", 
            "这是一个更新后的测试文档，现在包含了更多关于自然语言处理的信息。",
            {"doc_id": "test_doc_1", "source": "manual_test", "topic": "nlp"}
        )
        
        if updated_success:
            logger.info("✅ 文档更新成功")
        
        # 显示更新后状态
        final_info = updater.get_document_info()
        logger.info(f"📊 更新后知识库状态: {final_info['total_documents']} 个文档，{final_info['total_chunks']} 个块")
        
        # 列出所有数据源
        try:
            sources = updater.list_sources()
            logger.info(f"📋 数据源列表: {sources}")
        except Exception as e:
            logger.error(f"❌ 调用 list_sources 失败: {e}")
            import traceback
            logger.error(f"❌ 详细错误:\n{traceback.format_exc()}")

        # 搜索测试
        if updater.vector_db:
            logger.info("🔍 测试向量搜索功能...")
            try:
                results = updater.vector_db.similarity_search("机器学习", k=3)
                logger.info(f"搜索'机器学习'找到 {len(results)} 个相关块:")
                for i, result in enumerate(results):
                    logger.info(f"  结果 {i+1}: {result.page_content[:80]}...")
            except Exception as e:
                logger.warning(f"向量搜索测试失败: {e}")
        
        logger.info("🎉 增量更新演示完成！")
        
    except Exception as e:
        logger.error(f"❌ 增量更新演示失败: {str(e)}")
        import traceback
        logger.error(f"❌ 详细错误信息:\n{traceback.format_exc()}")

if __name__ == "__main__":
    # 运行增量更新演示
    demo_incremental_update()