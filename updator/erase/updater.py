"""
ERASE 更新器: 三步流水线 (Retrieve → Update → Add)

论文: "Language Modeling with Editable External Knowledge" (Li et al., 2024)
链接: https://arxiv.org/abs/2406.11830
简称: ERASE = Enhancing Retrieval Augmentation with Self-consistent Editing

=== 核心流程 (Section 3) ===

当一篇新文档 d 到达时，ERASE 执行三步操作来更新知识库 K:

  Step 1  Retrieve(K, d):
          用文档 d 的 embedding 在 KB 中检索 top-k 个最相关的事实 f_j
          → 这些事实是"可能受到新文档影响"的候选集

  Step 2  Update(f_j, H_j, d):  ← 这一步最关键，包含两轮 LLM 调用
          [第一轮 - 分类] 对每个候选事实，让 LLM 判断:
            - "Reinforce":   新文档支持该事实 → 加一条 (timestamp, True) 到历史
            - "Make False":  新文档推翻该事实 → 加一条 (timestamp, False)
            - "No Change":   新文档与该事实无关 → 不改动

          [第二轮 - 改写] 对所有被判为 Make False 的事实:
            - 把事实 + 新文档 + 仍为真的事实列表 → LLM
            - LLM 尝试将假事实改写成新的真事实
            - 例: "Elizabeth II is Queen" → "Charles III is King"
            - 若无法改写则跳过

  Step 3  Add_facts(d):
          让 LLM 从文档 d 中提取全新的原子事实 (atomic facts)
          → embed 后加入 KB，初始状态为 True

=== 推理方式 (Appendix A.3) ===

回答问题时:
  1. 用问题 embedding 检索 top-k 相关事实 (包括历史中有 False 的)
  2. 将事实和它们的真假历史一起格式化给 LLM
  3. LLM 根据历史推理时间线，给出当前正确答案

=== Prompt 模板 (Appendix A) ===

论文附录给出了具体 prompt 格式:
  - A.1: Classify prompt ("Answer: Reinforce / Make False / No Change")
  - A.2: Rewrite prompt ("rewrite: ..." or "no rewrite possible")
  - A.3: Inference prompt (facts + history → answer)
  - Extract prompt: 从文档中提取原子事实 (论文中隐含，非正式附录)
"""

import re
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple

from updator.erase.knowledge_base import ERASEKnowledgeBase, FactEntry, RetrievalResult

logger = logging.getLogger(__name__)


# ============================================================
# Prompt 模板 — 严格对照论文 Appendix A 的格式
# ============================================================

# --- Appendix A.1: 分类 Prompt ---
# 第一轮 LLM 调用: 判断已有事实的真假是否因新文档而改变
# 输出格式: "Answer: Reinforce" / "Answer: Make False" / "Answer: No Change"
PROMPT_CLASSIFY_FACT = """[Input]
[Timestamp: {timestamp}]
{document}
[End Input]

The fact "{fact}" was previously true. In light of the input, is "{fact}" likely still true as of {timestamp}?
Begin by summarizing the changes we learned from the input, then reasoning briefly about them to give your final answer with
"Answer: Reinforce" (if the input makes the fact more likely) or
"Answer: Make False" (if the input makes the fact less likely) or
"Answer: No Change" (if the input doesn't affect the truth value of the fact).
"""

# --- Appendix A.2: 改写 Prompt ---
# 第二轮 LLM 调用: 将被判为 False 的事实改写成新的真事实
# 输入: 原事实 + 文档 + 当前仍为真的事实列表 (提供上下文)
# 输出格式: "rewrite: <新事实>" 或 "no rewrite possible"
PROMPT_REWRITE_FACT = """[Input]
[Timestamp: {timestamp}]
{document}
Other True Facts at {timestamp}: {still_true_facts}
[End Input]

The fact "{fact}" was previously true but no longer. Given the above input and true facts, can you rewrite it into one that is true as of {timestamp}?
Output your answer in form "rewrite: rewritten fact" or "no rewrite possible".
"""

# --- 事实提取 Prompt ---
# 从新文档中提取原子事实 (Step 3)
# 要求: simple, independent, standalone, decontextualized
# 指代消解: 将所有代词替换为完整名称
PROMPT_EXTRACT_FACTS = """Extract all facts from the input text, with each fact on a new line and without bullet points or numbered lists.
Facts should be simple, independent, standalone, and decontextualized.
Break up long facts into smaller facts.
Resolve all references (e.g. pronouns, definite articles, etc.) by copying full reference object everywhere it is referenced.
Only include facts referring to the current world state (what is true *now*), as opposed to facts true in the past.
If there are no facts, please output "No facts."

Input:
[Timestamp: {timestamp}]
{document}
"""

# --- Appendix A.3: 推理 Prompt (多选题版本) ---
# 检索事实 + 历史 → 组装 prompt → LLM 推理
PROMPT_INFERENCE = """Read the statements/passages below then answer the question below

*** BEGIN STATEMENTS ***
{facts_with_history}
*** END STATEMENTS ***

Given the above statements are true and any prior knowledge you have, answer the following question at timestep {timestamp}?:
{question}

Briefly reason then answer with one of the following options: {answer_choices}
"""

# --- Appendix A.3: 推理 Prompt (开放问答版本) ---
PROMPT_INFERENCE_FREE = """Read the statements/passages below then answer the question below

*** BEGIN STATEMENTS ***
{facts_with_history}
*** END STATEMENTS ***

Given the above statements are true and any prior knowledge you have, answer the following question at timestep {timestamp}?:
{question}

Briefly reason then give your final answer.
"""


# ============================================================
# ERASE 更新器
# ============================================================

class ERASEUpdater:
    """
    ERASE 完整流水线 — 将 ERASEKnowledgeBase 与 LLM 结合。

    主要接口:
      - ingest_document(document, timestamp):  处理新文档 → 三步更新 KB
      - answer_question(question, timestamp):  用 KB 中的事实回答问题

    与 ComRAG/QARC 的关键对比:
      - ERASE 是"文档驱动"的: 有新文档 → 更新 KB，不关心用户兴趣
      - ComRAG 是"查询驱动"的: 有新查询 → 更新 QA 记忆
      - QARC 是"兴趣驱动"的: 检测用户兴趣漂移 → 重组 KB 文档集合

    使用示例:
        kb = ERASEKnowledgeBase()
        updater = ERASEUpdater(kb, embed_fn=my_embed, llm_fn=my_llm)
        updater.ingest_document("Charles became King...", "2022-09-08")
        answer = updater.answer_question("Who is the UK monarch?", "2022-09-08")
    """

    def __init__(
        self,
        knowledge_base: ERASEKnowledgeBase,
        embed_fn: Callable,
        llm_fn: Callable,
        retrieve_top_k: int = 20,
        inference_top_k: int = 10,
        inference_threshold: float = 0.7,
    ):
        """
        参数:
            knowledge_base:       ERASE 知识库实例
            embed_fn:             嵌入函数 text → np.ndarray
            llm_fn:               LLM 调用函数 prompt → str
            retrieve_top_k:       Step 1 检索数量 (更新时)
            inference_top_k:      推理时检索数量
            inference_threshold:  推理时最低相似度阈值 (论文用 0.7)
        """
        self.kb = knowledge_base
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.retrieve_top_k = retrieve_top_k
        self.inference_top_k = inference_top_k
        self.inference_threshold = inference_threshold

        # 更新统计
        self._docs_processed = 0
        self._facts_reinforced = 0
        self._facts_made_false = 0
        self._facts_rewritten = 0
        self._facts_added = 0

    # ============================================================
    # 主入口: 处理新文档 (三步流水线)
    # ============================================================

    def ingest_document(
        self,
        document: str,
        timestamp: str,
    ) -> Dict[str, Any]:
        """
        处理一篇新文档 — 完整的 ERASE 三步流水线。

        流程:
          1. Retrieve: 用文档 embedding 在 KB 中找候选事实
          2. Update:   两轮 LLM 判断 + 改写
          3. Add:      从文档中提取新的原子事实加入 KB

        参数:
            document:   新文档文本
            timestamp:  文档时间戳 (用于历史记录)

        返回:
            包含 reinforced, made_false, rewritten, new_facts 等统计的字典
        """
        self._docs_processed += 1
        logger.info(f"[ERASE] 处理文档 #{self._docs_processed} @ {timestamp}")

        # ---- Step 1: Retrieve ----
        # 对应论文 Eq(2): Retrieve(K, d) = arg top-k sim(E(d), E(f_j))
        # 注意: 使用较低阈值 0.3，因为需要找到所有"可能受影响"的事实
        doc_embedding = self.embed_fn(document)
        retrieved = self.kb.retrieve_for_update(doc_embedding, top_k=self.retrieve_top_k)
        logger.info(f"[ERASE] Step 1 Retrieve: 找到 {len(retrieved)} 个候选事实")

        # ---- Step 2: Update ----
        # 两轮 LLM: 分类 → 改写
        update_result = self._update_facts(retrieved, document, timestamp)

        # ---- Step 3: Add ----
        # LLM 提取新的原子事实 → embed → 加入 KB
        new_facts = self._add_new_facts(document, timestamp)
        self._facts_added += len(new_facts)
        logger.info(f"[ERASE] Step 3 Add: 提取并添加了 {len(new_facts)} 个新事实")

        return {
            "docs_processed": self._docs_processed,
            "retrieved_count": len(retrieved),
            "reinforced": update_result["reinforced"],
            "made_false": update_result["made_false"],
            "rewritten": update_result["rewritten"],
            "new_facts_added": new_facts,
            "kb_stats": self.kb.get_statistics(),
        }

    def answer_question(
        self,
        question: str,
        timestamp: str,
        answer_choices: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        使用 ERASE KB 回答问题 — 对应论文 Appendix A.3 推理流程。

        流程:
          1. 用问题 embedding 检索 top-k 相关事实
          2. 格式化事实 + 历史信息 (包含已变假的事实，让 LLM 理解时间线)
          3. 组装 prompt → LLM 生成答案

        关键设计: 即使事实已变假，也把它的历史展示给 LLM，
        因为 "Elizabeth II was Queen (true at 2020, false at 2022-09)"
        这类信息能帮助 LLM 做时间推理。
        """
        query_embedding = self.embed_fn(question)
        retrieved = self.kb.retrieve(
            query_embedding,
            top_k=self.inference_top_k,
            threshold=self.inference_threshold,
            only_true=False,  # 包含已变假的事实 — 历史信息有助于推理
        )

        # 格式化: "fact (true at t1, false at t2)"
        facts_with_history = self.kb.format_facts_for_inference(retrieved)

        # 根据是否有选项使用不同 prompt 模板
        if answer_choices:
            choices_str = ", ".join(answer_choices)
            prompt = PROMPT_INFERENCE.format(
                facts_with_history=facts_with_history,
                timestamp=timestamp,
                question=question,
                answer_choices=choices_str,
            )
        else:
            prompt = PROMPT_INFERENCE_FREE.format(
                facts_with_history=facts_with_history,
                timestamp=timestamp,
                question=question,
            )

        answer = self.llm_fn(prompt)

        return {
            "question": question,
            "answer": answer,
            "timestamp": timestamp,
            "facts_used": [r.entry.fact for r in retrieved],
            "num_facts": len(retrieved),
        }

    # ============================================================
    # Step 2 实现: 两轮 LLM 更新
    # ============================================================

    def _update_facts(
        self,
        retrieved: List[RetrievalResult],
        document: str,
        timestamp: str,
    ) -> Dict[str, List]:
        """
        Step 2 完整实现 — 两轮 LLM 调用。

        第一轮 (Appendix A.1 - 分类):
          对每个候选事实 f_j:
            prompt = CLASSIFY_FACT(timestamp, document, fact)
            LLM → "Answer: Reinforce / Make False / No Change"
            → 相应更新 KB 中的历史记录 H_j

        第二轮 (Appendix A.2 - 改写):
          对所有被判为 Make False 的事实:
            prompt = REWRITE_FACT(timestamp, document, still_true_facts, fact)
            LLM → "rewrite: <新事实>" 或 "no rewrite possible"
            → 若成功改写则替换 KB 中的事实 (保留 fact_id 以便追踪)

        论文的设计意图: 第二轮给 LLM 提供"仍为真的事实"作为上下文，
        帮助 LLM 更准确地改写。例如，知道"Charles III became King"后，
        更容易将"Elizabeth II is Queen"改写为"Charles III is King"。
        """
        reinforced_ids = []
        false_ids = []
        no_change_ids = []

        # --- 第一轮: 分类 ---
        for r in retrieved:
            entry = r.entry
            label = self._classify_fact(entry.fact, document, timestamp)

            if label == "reinforce":
                # 标记为仍然为真 → 加 (timestamp, True) 到历史
                self.kb.reinforce_fact(entry.fact_id, timestamp)
                reinforced_ids.append(entry.fact_id)
                self._facts_reinforced += 1
            elif label == "make_false":
                # 标记为已变假 → 加 (timestamp, False) 到历史
                self.kb.make_fact_false(entry.fact_id, timestamp)
                false_ids.append(entry.fact_id)
                self._facts_made_false += 1
            else:  # no_change
                no_change_ids.append(entry.fact_id)

        logger.info(
            f"[ERASE] 第一轮分类完成: reinforce={len(reinforced_ids)}, "
            f"make_false={len(false_ids)}, no_change={len(no_change_ids)}"
        )

        # --- 第二轮: 尝试改写 Make False 的事实 ---
        # 收集"仍为真的事实"作为上下文 (最多取前10个，避免 prompt 过长)
        still_true_facts = [
            self.kb.entries[fid].fact
            for fid in reinforced_ids + no_change_ids
            if fid in self.kb.entries
        ]
        still_true_str = ", ".join(f'"{f}"' for f in still_true_facts[:10])

        rewritten = []
        for fid in false_ids:
            if fid not in self.kb.entries:
                continue
            entry = self.kb.entries[fid]
            rewrite_result = self._attempt_rewrite(
                entry.fact, document, timestamp, still_true_str
            )
            if rewrite_result:
                # 改写成功 → 用新文本和新 embedding 替换 KB 中的事实
                new_embedding = self.embed_fn(rewrite_result)
                self.kb.rewrite_fact(fid, rewrite_result, new_embedding, timestamp)
                rewritten.append({"fact_id": fid, "old": entry.fact, "new": rewrite_result})
                self._facts_rewritten += 1
                logger.info(f"[ERASE] 改写成功 [{fid}]: '{entry.fact[:40]}' → '{rewrite_result[:40]}'")

        return {
            "reinforced": reinforced_ids,
            "made_false": false_ids,
            "rewritten": rewritten,
        }

    def _classify_fact(
        self, fact: str, document: str, timestamp: str
    ) -> str:
        """
        第一轮 LLM 调用 — 分类事实的真假变化。

        Prompt 格式 (Appendix A.1):
          [Input] [Timestamp] document [End Input]
          The fact "..." was previously true. Is it still true?
          Answer: Reinforce / Make False / No Change

        返回: 'reinforce' | 'make_false' | 'no_change'
        """
        prompt = PROMPT_CLASSIFY_FACT.format(
            timestamp=timestamp, document=document, fact=fact
        )
        response = self.llm_fn(prompt)
        return self._parse_classification(response)

    def _attempt_rewrite(
        self,
        fact: str,
        document: str,
        timestamp: str,
        still_true_facts: str,
    ) -> Optional[str]:
        """
        第二轮 LLM 调用 — 尝试将假事实改写为真事实。

        Prompt 格式 (Appendix A.2):
          [Input] document + still_true_facts [End Input]
          The fact "..." was previously true but no longer.
          rewrite: <new fact> / no rewrite possible

        返回: 新事实文本，或 None (无法改写)
        """
        prompt = PROMPT_REWRITE_FACT.format(
            timestamp=timestamp,
            document=document,
            still_true_facts=still_true_facts,
            fact=fact,
        )
        response = self.llm_fn(prompt)
        return self._parse_rewrite(response)

    # ============================================================
    # Step 3 实现: 原子事实提取
    # ============================================================

    def _add_new_facts(
        self, document: str, timestamp: str
    ) -> List[FactEntry]:
        """
        Step 3: 从文档中提取新的原子事实。

        流程:
          1. LLM 提取: document → ["fact1", "fact2", ...]
          2. 每条事实 embed → 加入 KB

        提取要求 (参考 claim decomposition 文献):
          - Simple:    简单句，不含从句
          - Independent: 不依赖其他事实
          - Standalone: 独立可理解
          - Decontextualized: 所有代词替换为完整实体名

        注意: 只提取"当前为真"的事实，过去式的事实不提取。
        """
        prompt = PROMPT_EXTRACT_FACTS.format(
            timestamp=timestamp, document=document
        )
        response = self.llm_fn(prompt)
        facts = self._parse_extracted_facts(response)

        added = []
        for fact in facts:
            if not fact.strip():
                continue
            embedding = self.embed_fn(fact)
            entry = self.kb.add_fact(
                fact=fact,
                embedding=embedding,
                timestamp=timestamp,
                source=document[:100],
            )
            added.append(entry)

        return added

    # ============================================================
    # LLM 输出解析器
    # ============================================================

    @staticmethod
    def _parse_classification(response: str) -> str:
        """解析分类结果 — 从 LLM 输出中提取 Reinforce/Make False/No Change"""
        text = response.lower()
        # 优先匹配格式化的 "Answer: ..." 前缀
        if "answer: reinforce" in text or "answer:reinforce" in text:
            return "reinforce"
        elif "answer: make false" in text or "answer:make false" in text:
            return "make_false"
        elif "answer: no change" in text or "answer:no change" in text:
            return "no_change"
        # 回退: 关键词扫描
        if "reinforce" in text:
            return "reinforce"
        elif "make false" in text or "false" in text:
            return "make_false"
        return "no_change"

    @staticmethod
    def _parse_rewrite(response: str) -> Optional[str]:
        """解析改写结果 — 'rewrite: <新事实>' 或 'no rewrite possible'"""
        text = response.strip()
        # 匹配 "rewrite: ..." 格式
        m = re.search(r'rewrite\s*:\s*(.+)', text, re.IGNORECASE)
        if m:
            rewritten = m.group(1).strip().rstrip('."')
            if rewritten and "no rewrite" not in rewritten.lower():
                return rewritten
        if "no rewrite possible" in text.lower():
            return None
        return None

    @staticmethod
    def _parse_extracted_facts(response: str) -> List[str]:
        """解析原子事实提取结果 — 从 LLM 输出中逐行提取事实"""
        if "no facts" in response.lower():
            return []
        lines = response.strip().splitlines()
        facts = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 去掉 LLM 可能加的 bullet / 序号
            line = re.sub(r'^[\-\*\d\.]+\s*', '', line).strip()
            if line and len(line) > 5:
                facts.append(line)
        return facts

    # ============================================================
    # 统计
    # ============================================================

    def get_statistics(self) -> Dict:
        """返回更新统计信息"""
        return {
            "docs_processed": self._docs_processed,
            "facts_reinforced": self._facts_reinforced,
            "facts_made_false": self._facts_made_false,
            "facts_rewritten": self._facts_rewritten,
            "facts_added": self._facts_added,
            "kb": self.kb.get_statistics(),
        }
