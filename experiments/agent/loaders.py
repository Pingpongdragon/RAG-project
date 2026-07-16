"""Active agent, session, and access-trace dataset loaders.

The retained suite has one role per dataset:

* MIND: natural timestamped multi-user object access with strong reuse.
* Wizard of Wikipedia: multi-turn, topic-labelled dialogue with gold knowledge
  selection and cross-dialogue evidence reuse.
* MultiDoc2Dial: multi-document goal-oriented dialogue with exact document and
  span references on every served agent turn.
* MTRAG is loaded separately by ``mtrag_loader.py`` because it has multi-document
  qrels and a dedicated exact-residency runner.

Mind2Web was removed from the active loader registry: its previous controlled
stream contained substantial exact-query repetition and did not add a cleaner
claim than Wizard of Wikipedia for the current paper.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import zipfile

from experiments.agent.config import PROJECT_DIR, log


NO_PASSAGE = "no_passages_used __knowledge__ no_passages_used"


def _knowledge_id(value: str) -> str:
    digest = hashlib.sha256(value.strip().encode("utf-8")).hexdigest()[:20]
    return f"wow-{digest}"


def _knowledge_document(value: str) -> dict:
    value = str(value).strip()
    if " __knowledge__ " in value:
        title, text = value.split(" __knowledge__ ", 1)
    else:
        title, text = "knowledge", value
    evidence_id = _knowledge_id(value)
    return {
        "doc_id": evidence_id,
        "title": evidence_id,
        "text": f"{title}. {text}".strip(),
        "source_title": title,
    }


def load_wizard_of_wikipedia(n_source=None, split="test"):
    """Load Wizard of Wikipedia as an ordered multi-session RAG trace.

    Each dialogue is one session and preserves its original turn order.  The
    selected knowledge sentence is the post-service evidence access.  Candidate
    knowledge is included in the cold corpus, but the selected label is never
    placed in the query text.  The dataset has no global timestamp, so a runner
    must use ``session_round_robin`` or an explicitly controlled domain schedule
    rather than treating file order as natural chronology.
    """

    path = PROJECT_DIR / "datasets" / "wizard_of_wikipedia" / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Wizard of Wikipedia split is missing: {path}")
    dialogues = json.loads(path.read_text())
    if n_source and n_source < len(dialogues):
        dialogues = dialogues[: int(n_source)]

    documents = {}
    queries = []
    for dialogue_index, dialogue in enumerate(dialogues):
        topic = str((dialogue.get("topics") or ["unknown"])[0])
        posts = list(dialogue.get("post") or [])
        responses = list(dialogue.get("response") or [])
        knowledge_turns = list(dialogue.get("knowledge") or [])
        labels = list(dialogue.get("labels") or [])
        turn_count = min(len(posts), len(knowledge_turns), len(labels))
        conversation_id = f"wow-{split}-{dialogue_index:06d}"
        history = []

        for turn_index in range(turn_count):
            candidates = []
            for value in knowledge_turns[turn_index]:
                value = str(value).strip()
                if not value or value == NO_PASSAGE:
                    continue
                document = _knowledge_document(value)
                documents.setdefault(document["title"], document)
                candidates.append(document["title"])

            label = labels[turn_index]
            selected = None
            if isinstance(label, int) and 0 <= label < len(knowledge_turns[turn_index]):
                selected_value = str(knowledge_turns[turn_index][label]).strip()
                if selected_value and selected_value != NO_PASSAGE:
                    selected = _knowledge_document(selected_value)
                    documents.setdefault(selected["title"], selected)

            current_post = str(posts[turn_index]).strip()
            history_view = " ".join(history[-4:]) or "no prior turns"
            if selected is not None:
                queries.append({
                    "qidx": len(queries),
                    "query_id": f"{conversation_id}-turn-{turn_index:03d}",
                    "question": (
                        f"Topic: {topic}. Dialogue history: {history_view}. "
                        f"User: {current_post}"
                    ),
                    "answer": (
                        str(responses[turn_index])
                        if turn_index < len(responses)
                        else ""
                    ),
                    "sf_titles": [selected["title"]],
                    "ctx_titles": list(dict.fromkeys(candidates)),
                    "access_title": selected["title"],
                    "qtype": "wizard_dialogue_turn",
                    "conversation_id": conversation_id,
                    "session_id": conversation_id,
                    "turn_idx": turn_index,
                    "domain": topic,
                    "topic": topic,
                    "global_time_order": "unobserved",
                    "evidence_visibility": "hidden",
                })

            history.append(f"User: {current_post}")
            if turn_index < len(responses):
                history.append(f"Wizard: {responses[turn_index]}")

    doc_pool = [documents[key] for key in sorted(documents)]
    title_to_idx = {doc["title"]: index for index, doc in enumerate(doc_pool)}
    log.info(
        "[wizard_of_wikipedia] split=%s dialogues=%s pool=%s turns=%s topics=%s",
        split,
        f"{len(dialogues):,}",
        f"{len(doc_pool):,}",
        f"{len(queries):,}",
        f"{len({query['topic'] for query in queries}):,}",
    )
    return doc_pool, queries, title_to_idx


def load_multidoc2dial(n_source=None, split="test"):
    """Load MultiDoc2Dial without inventing a global event chronology.

    A request is a user turn followed by an evidence-grounded agent response.
    The next agent turn's document references are post-service evidence keys;
    they are never included in ``question`` or a precomputed candidate list.
    Runners may form candidates causally from the question and cold corpus.
    """

    root = PROJECT_DIR / "datasets" / "multidoc2dial" / "multidoc2dial"
    document_path = root / "multidoc2dial_doc.json"
    dialogue_path = root / f"multidoc2dial_dial_{split}.json"
    if not document_path.exists() or not dialogue_path.exists():
        raise FileNotFoundError(
            "MultiDoc2Dial is missing. Expected the official archive under "
            f"{root}"
        )

    document_data = json.loads(document_path.read_text(encoding="utf-8"))[
        "doc_data"
    ]
    documents = {}
    for domain, domain_documents in document_data.items():
        for doc_id, source in domain_documents.items():
            documents[str(doc_id)] = {
                "doc_id": str(doc_id),
                "title": str(doc_id),
                "text": str(source.get("doc_text") or ""),
                "source_title": str(source.get("title") or doc_id),
                "domain": str(domain),
            }

    raw = json.loads(dialogue_path.read_text(encoding="utf-8"))["dial_data"]
    dialogue_rows = [
        (str(domain), dialogue)
        for domain, dialogues in raw.items()
        for dialogue in dialogues
    ]
    if n_source is not None:
        dialogue_rows = dialogue_rows[: int(n_source)]

    queries = []
    for domain, dialogue in dialogue_rows:
        conversation_id = str(dialogue["dial_id"])
        turns = list(dialogue.get("turns") or [])
        history = []
        request_index = 0
        for index, turn in enumerate(turns):
            role = str(turn.get("role") or "")
            utterance = str(turn.get("utterance") or "").strip()
            if role == "user" and index + 1 < len(turns):
                response = turns[index + 1]
                references = response.get("references") or []
                support_documents = list(dict.fromkeys(
                    str(reference["doc_id"])
                    for reference in references
                    if str(reference.get("doc_id") or "") in documents
                ))
                support_spans = list(dict.fromkeys(
                    f'{reference["doc_id"]}::{reference["id_sp"]}'
                    for reference in references
                    if str(reference.get("doc_id") or "") in documents
                ))
                if (
                    str(response.get("role") or "") == "agent"
                    and support_documents
                ):
                    history_view = " ".join(history[-6:]) or "no prior turns"
                    queries.append({
                        "qidx": len(queries),
                        "query_id": (
                            f"{conversation_id}-request-{request_index:03d}"
                        ),
                        "question": (
                            f"Dialogue history: {history_view}. User: {utterance}"
                        ),
                        "answer": str(response.get("utterance") or ""),
                        "sf_titles": support_documents,
                        "sf_spans": support_spans,
                        "qtype": "multidoc2dial_agent_turn",
                        "conversation_id": conversation_id,
                        "session_id": conversation_id,
                        "turn_idx": request_index,
                        "source_turn_id": int(turn.get("turn_id", index)),
                        "domain": domain,
                        "global_time_order": "unobserved",
                        "evidence_visibility": "hidden_until_service",
                    })
                    request_index += 1
            if utterance:
                history.append(f"{role}: {utterance}")

    doc_pool = [documents[key] for key in sorted(documents)]
    title_to_idx = {doc["title"]: index for index, doc in enumerate(doc_pool)}
    log.info(
        "[multidoc2dial] split=%s dialogues=%s pool=%s requests=%s domains=%s",
        split,
        f"{len(dialogue_rows):,}",
        f"{len(doc_pool):,}",
        f"{len(queries):,}",
        f"{len({query['domain'] for query in queries}):,}",
    )
    return doc_pool, queries, title_to_idx


def load_mind_news_context(n_source=None):
    """Map MIND-small to a history-only hidden next-access stream.

    Events remain in the official behavior timestamp order.  The clicked news
    identifier is exposed only as post-service access feedback; repeated clicks
    by different users create natural shared-cache reuse.
    """

    archive = PROJECT_DIR / "datasets" / "mind" / "MINDsmall_train.zip"
    if not archive.exists():
        raise FileNotFoundError(
            f"MIND-small archive not found: {archive}. See datasets/mind/README.md"
        )

    with zipfile.ZipFile(archive) as source:
        news_rows = {}
        with source.open("MINDsmall_train/news.tsv") as news_file:
            for raw_line in news_file:
                fields = raw_line.decode("utf-8").rstrip("\n").split("\t")
                if len(fields) < 5:
                    continue
                news_id, category, subcategory, title, abstract = fields[:5]
                news_rows[news_id] = {
                    "category": category,
                    "subcategory": subcategory,
                    "headline": title,
                    "abstract": abstract,
                }

        events = []
        with source.open("MINDsmall_train/behaviors.tsv") as behavior_file:
            for row_index, raw_line in enumerate(behavior_file):
                fields = raw_line.decode("utf-8").rstrip("\n").split("\t")
                if len(fields) != 5:
                    continue
                impression_id, user_id, raw_time, history, impressions = fields
                event_time = datetime.strptime(
                    raw_time, "%m/%d/%Y %I:%M:%S %p"
                ).replace(tzinfo=timezone.utc)
                timestamp = int(event_time.timestamp())
                candidate_ids = []
                positive_ids = []
                for item in impressions.split():
                    news_id, label = item.rsplit("-", 1)
                    if news_id not in news_rows:
                        continue
                    candidate_ids.append(news_id)
                    if label == "1":
                        positive_ids.append(news_id)
                for click_index, news_id in enumerate(positive_ids):
                    events.append((
                        timestamp,
                        row_index,
                        click_index,
                        impression_id,
                        user_id,
                        news_id,
                        tuple(candidate_ids),
                        history,
                    ))

    doc_pool = []
    title_to_idx = {}
    for news_id, news in news_rows.items():
        title_to_idx[news_id] = len(doc_pool)
        text = (
            f"Category {news['category']} / {news['subcategory']}. "
            f"{news['headline']}. {news['abstract']}"
        ).strip()
        doc_pool.append({
            "doc_id": f"mind-{news_id}",
            "title": news_id,
            "text": text,
            "category": news["category"],
            "subcategory": news["subcategory"],
        })

    events.sort(key=lambda event: (event[0], event[1], event[2]))
    queries = []
    for event in events:
        (
            timestamp,
            _,
            _,
            impression_id,
            user_id,
            news_id,
            candidate_ids,
            history,
        ) = event
        news = news_rows[news_id]
        history_ids = (
            [value for value in history.split() if value in news_rows]
            if history
            else []
        )
        target_title_in_history = any(
            news_rows[value]["headline"] == news["headline"]
            for value in history_ids
        )
        history_view = " | ".join(
            f"{news_rows[value]['category']}/"
            f"{news_rows[value]['subcategory']}: "
            f"{news_rows[value]['headline']}"
            for value in history_ids[-5:]
        ) or "no previous clicks"
        category_counts = Counter(
            news_rows[value]["category"]
            for value in candidate_ids
            if value in news_rows
        )
        slate_view = ", ".join(
            f"{category}:{count}"
            for category, count in sorted(category_counts.items())
        ) or "unknown"
        queries.append({
            "qidx": len(queries),
            "question": (
                f"Previously read: {history_view}. "
                f"Current candidate categories: {slate_view}."
            ),
            "answer": news["headline"],
            "sf_titles": [news_id],
            "ctx_titles": list(candidate_ids),
            "qtype": "observed_item_access",
            "access_title": news_id,
            "event_ts": int(timestamp),
            "impression_id": impression_id,
            "agent_id": user_id,
            "user_id": user_id,
            "category": news["category"],
            "subcategory": news["subcategory"],
            "history_size": len(history.split()) if history else 0,
            "history_titles": history_ids[-5:],
            "query_view": "history",
            "target_in_history": bool(news_id in history_ids),
            "target_title_in_history": bool(target_title_in_history),
            "evidence_visibility": "hidden",
            "preserve_order": True,
        })

    log.info(
        "[mind_news_context] pool=%s events=%s users=%s range=%s..%s",
        f"{len(doc_pool):,}",
        f"{len(queries):,}",
        f"{len({query['user_id'] for query in queries}):,}",
        queries[0]["event_ts"],
        queries[-1]["event_ts"],
    )
    return doc_pool, queries, title_to_idx


LOADERS = {
    "mind_news_context": load_mind_news_context,
    "multidoc2dial": load_multidoc2dial,
    "wizard_of_wikipedia": load_wizard_of_wikipedia,
}
