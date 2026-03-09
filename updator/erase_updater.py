"""
ERASE Updater: Full three-step pipeline (Retrieve → Update → Add)

Paper: "Language Modeling with Editable External Knowledge" (Li et al., 2024)
https://arxiv.org/abs/2406.11830  (ERASE = Enhancing Retrieval Augmentation with Self-consistent Editing)

Three steps when a new document d arrives (Section 3):
  Step 1: Retrieve(K, d)        -> top-k facts similar to d
  Step 2: Update(f_j, H_j, d)  -> classify each retrieved fact as:
            reinforce / no change / make false  [First LLM pass]
            -> for "make false" facts: attempt Rewrite             [Second LLM pass]
  Step 3: Add_facts(d)          -> extract new atomic facts from d, add to K

Inference (Appendix A.3):
  retrieve true facts for the query, format with history, prompt LLM
"""

import re
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple

from updator.erase_knowledge_base import ERASEKnowledgeBase, FactEntry, RetrievalResult

logger = logging.getLogger(__name__)


# ============================================================
# Prompt Templates (Appendix A of the paper - exact formats)
# ============================================================

# First pass: classify fact as reinforce / make false / no change
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

# Second pass: rewrite a false fact into a true one
PROMPT_REWRITE_FACT = """[Input]
[Timestamp: {timestamp}]
{document}
Other True Facts at {timestamp}: {still_true_facts}
[End Input]

The fact "{fact}" was previously true but no longer. Given the above input and true facts, can you rewrite it into one that is true as of {timestamp}?
Output your answer in form "rewrite: rewritten fact" or "no rewrite possible".
"""

# Fact extraction: extract atomic facts from a document
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

# Inference: answer question using facts with history
PROMPT_INFERENCE = """Read the statements/passages below then answer the question below

*** BEGIN STATEMENTS ***
{facts_with_history}
*** END STATEMENTS ***

Given the above statements are true and any prior knowledge you have, answer the following question at timestep {timestamp}?:
{question}

Briefly reason then answer with one of the following options: {answer_choices}
"""

PROMPT_INFERENCE_FREE = """Read the statements/passages below then answer the question below

*** BEGIN STATEMENTS ***
{facts_with_history}
*** END STATEMENTS ***

Given the above statements are true and any prior knowledge you have, answer the following question at timestep {timestamp}?:
{question}

Briefly reason then give your final answer.
"""


# ============================================================
# ERASE Updater
# ============================================================

class ERASEUpdater:
    """
    ERASE pipeline: integrates ERASEKnowledgeBase with an LLM to process
    incoming documents and keep the KB consistent.

    Usage:
        kb = ERASEKnowledgeBase()
        updater = ERASEUpdater(kb, embed_fn=my_embed, llm_fn=my_llm)

        # Ingest a new document
        result = updater.ingest_document(
            document="Charles has become King Charles III...",
            timestamp="2022-09-08",
        )

        # Answer a question using the KB
        answer = updater.answer_question(
            question="Who is the head of state of Scotland?",
            timestamp="2022-09-08",
        )
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
        Args:
            knowledge_base:       ERASEKnowledgeBase instance
            embed_fn:             function(text: str) -> np.ndarray
            llm_fn:               function(prompt: str) -> str
            retrieve_top_k:       Number of facts to retrieve for update
            inference_top_k:      Number of facts to use at inference
            inference_threshold:  Similarity threshold for inference retrieval
        """
        self.kb = knowledge_base
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.retrieve_top_k = retrieve_top_k
        self.inference_top_k = inference_top_k
        self.inference_threshold = inference_threshold

        # Update statistics
        self._docs_processed = 0
        self._facts_reinforced = 0
        self._facts_made_false = 0
        self._facts_rewritten = 0
        self._facts_added = 0

    # ============================================================
    # Main entry: ingest a new document
    # ============================================================

    def ingest_document(
        self,
        document: str,
        timestamp: str,
    ) -> Dict[str, Any]:
        """
        Process a new document through the full ERASE pipeline.

        Args:
            document:  New document text
            timestamp: Timestamp of the document

        Returns:
            Summary dict with what was reinforced/made_false/rewritten/added
        """
        self._docs_processed += 1
        logger.info(f"[ERASE] Ingesting document #{self._docs_processed} @ {timestamp}")

        # Step 1: Retrieve relevant facts
        doc_embedding = self.embed_fn(document)
        retrieved = self.kb.retrieve_for_update(doc_embedding, top_k=self.retrieve_top_k)
        logger.info(f"[ERASE] Step 1: Retrieved {len(retrieved)} candidate facts")

        # Step 2: Update retrieved facts (two-pass LLM)
        update_result = self._update_facts(retrieved, document, timestamp)

        # Step 3: Add new facts extracted from the document
        new_facts = self._add_new_facts(document, timestamp)
        self._facts_added += len(new_facts)
        logger.info(f"[ERASE] Step 3: Added {len(new_facts)} new facts")

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
        Answer a question using the ERASE KB (Inference / Appendix A.3).

        Args:
            question:       The question to answer
            timestamp:      Current timestamp for context
            answer_choices: If given, use multiple-choice prompt

        Returns:
            {"answer": str, "facts_used": List[str], "prompt": str}
        """
        query_embedding = self.embed_fn(question)
        retrieved = self.kb.retrieve(
            query_embedding,
            top_k=self.inference_top_k,
            threshold=self.inference_threshold,
            only_true=False,  # include history of false facts for context
        )

        facts_with_history = self.kb.format_facts_for_inference(retrieved)

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
    # Step 1 helpers
    # ============================================================

    def _update_facts(
        self,
        retrieved: List[RetrievalResult],
        document: str,
        timestamp: str,
    ) -> Dict[str, List]:
        """
        Two-pass update (Appendix A.1):
        Pass 1: classify each fact as reinforce / make_false / no_change
        Pass 2: attempt rewrite for all make_false facts (conditioned on still-true facts)
        """
        reinforced_ids = []
        false_ids = []
        no_change_ids = []

        # --- Pass 1: Classify ---
        for r in retrieved:
            entry = r.entry
            label = self._classify_fact(entry.fact, document, timestamp)

            if label == "reinforce":
                self.kb.reinforce_fact(entry.fact_id, timestamp)
                reinforced_ids.append(entry.fact_id)
                self._facts_reinforced += 1
            elif label == "make_false":
                self.kb.make_fact_false(entry.fact_id, timestamp)
                false_ids.append(entry.fact_id)
                self._facts_made_false += 1
            else:  # no_change
                no_change_ids.append(entry.fact_id)

        logger.info(
            f"[ERASE] Pass 1: reinforce={len(reinforced_ids)}, "
            f"make_false={len(false_ids)}, no_change={len(no_change_ids)}"
        )

        # --- Pass 2: Attempt Rewrite for false facts ---
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
                new_embedding = self.embed_fn(rewrite_result)
                self.kb.rewrite_fact(fid, rewrite_result, new_embedding, timestamp)
                rewritten.append({"fact_id": fid, "old": entry.fact, "new": rewrite_result})
                self._facts_rewritten += 1
                logger.info(f"[ERASE] Rewrote [{fid}]: '{entry.fact[:40]}' -> '{rewrite_result[:40]}'")

        return {
            "reinforced": reinforced_ids,
            "made_false": false_ids,
            "rewritten": rewritten,
        }

    def _classify_fact(
        self, fact: str, document: str, timestamp: str
    ) -> str:
        """
        First LLM pass: classify fact as 'reinforce' | 'make_false' | 'no_change'.
        Returns one of those three strings.
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
        Second LLM pass: rewrite a false fact into a currently-true one.
        Returns the rewritten fact string, or None if no rewrite possible.
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
    # Step 3 helpers
    # ============================================================

    def _add_new_facts(
        self, document: str, timestamp: str
    ) -> List[FactEntry]:
        """
        Step 3: Extract atomic facts from the document and add them to KB.
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
    # LLM response parsers
    # ============================================================

    @staticmethod
    def _parse_classification(response: str) -> str:
        """Parse 'Answer: Reinforce / Make False / No Change' from LLM response."""
        text = response.lower()
        if "answer: reinforce" in text or "answer:reinforce" in text:
            return "reinforce"
        elif "answer: make false" in text or "answer:make false" in text:
            return "make_false"
        elif "answer: no change" in text or "answer:no change" in text:
            return "no_change"
        # Fallback: keyword scan
        if "reinforce" in text:
            return "reinforce"
        elif "make false" in text or "false" in text:
            return "make_false"
        return "no_change"

    @staticmethod
    def _parse_rewrite(response: str) -> Optional[str]:
        """Parse 'rewrite: <new fact>' or 'no rewrite possible' from LLM response."""
        text = response.strip()
        # Look for "rewrite: ..." pattern
        m = re.search(r'rewrite\s*:\s*(.+)', text, re.IGNORECASE)
        if m:
            rewritten = m.group(1).strip().rstrip('."')
            if rewritten and "no rewrite" not in rewritten.lower():
                return rewritten
        # No rewrite
        if "no rewrite possible" in text.lower():
            return None
        return None

    @staticmethod
    def _parse_extracted_facts(response: str) -> List[str]:
        """Parse a list of atomic facts from the LLM extraction response."""
        if "no facts" in response.lower():
            return []
        lines = response.strip().splitlines()
        facts = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Strip leading bullets/numbers
            line = re.sub(r'^[\-\*\d\.]+\s*', '', line).strip()
            if line and len(line) > 5:
                facts.append(line)
        return facts

    # ============================================================
    # Statistics
    # ============================================================

    def get_statistics(self) -> Dict:
        return {
            "docs_processed": self._docs_processed,
            "facts_reinforced": self._facts_reinforced,
            "facts_made_false": self._facts_made_false,
            "facts_rewritten": self._facts_rewritten,
            "facts_added": self._facts_added,
            "kb": self.kb.get_statistics(),
        }
