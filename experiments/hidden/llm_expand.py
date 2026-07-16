"""LLM-based query decomposition to improve multi-hop retrieval.

Decomposes each question into 2 sub-questions using a 30B LLM,
then returns augmented query embeddings (mean-pooled) and
augmented entity lists (union).
"""
import json, logging, hashlib, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import urllib.request, urllib.error

log = logging.getLogger(__name__)

LLM_URL   = "http://202.45.128.234:5788/v1/chat/completions"
LLM_MODEL = "/nfs/whlu/models/Qwen3-Coder-30B-A3B-Instruct"

_PROMPT = (
    "Decompose the following multi-hop question into exactly 2 simpler sub-questions, "
    "each targeting a single fact. Return ONLY a JSON array of 2 strings.\n\n"
    "Question: \"{q}\"\nOutput:"
)

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)


_STOP_ENTITY_HEADS = {
    "Do", "Does", "Did", "Were", "Was", "Are", "Is", "Which", "What",
    "Who", "Whose", "When", "Where", "Both", "The", "A", "An",
}


def _candidate_entities(question: str) -> list[str]:
    """Lightweight title/entity span extraction for local decomposition.

    This is a deterministic diagnostic fallback, not a dataset oracle: it uses
    only the question text and avoids support titles / labels.
    """
    # Prefer comparison spans introduced by common object nouns, then fall back
    # to generic capitalized spans. This catches film/book/song titles without
    # hard-coding 2Wiki support documents.
    spans = []
    intro = re.search(
        r"\b(?:films?|movies?|books?|novels?|songs?|albums?|works?|shows?|series)\s+(.+?)(?:\?| have | has | were | was | is | are | by )",
        question,
        flags=re.IGNORECASE,
    )
    if intro:
        piece = intro.group(1)
        if not re.match(r"^(?:has|have|is|are|was|were|with|whose|which|that)\b", piece, re.I):
            spans.extend(re.split(r"\s+(?:and|or)\s+|;\s*", piece))

    cap_pat = re.compile(
        r"(?:[A-Z][\w'&.-]*(?:\s+(?:of|the|and|&|in|to|for|from|The|A|An|"
        r"[A-Z][\w'&().:-]*|\d{4}))*?)"
        r"(?=\s+(?:and|or|have|has|was|were|is|are|directed|written|produced|composed)|[?,])"
    )
    spans.extend(m.group(0) for m in cap_pat.finditer(question))

    out, seen = [], set()
    for raw in spans:
        ent = raw.strip(" ,?;:'\"")
        ent = re.sub(r"\s+", " ", ent)
        if not ent:
            continue
        if ent.split()[0] in _STOP_ENTITY_HEADS and len(ent.split()) <= 2:
            continue
        if len(ent) < 3 or ent.lower() in seen:
            continue
        seen.add(ent.lower())
        out.append(ent)
        if len(out) >= 4:
            break
    return out


def heuristic_decompose(question: str) -> list[str]:
    """Question-text-only decomposition used when the LLM endpoint is absent."""
    ql = question.lower()
    relation_queries = []
    if "director" in ql or "directed" in ql:
        relation_queries.append("director of {entity}")
    if "producer" in ql or "produced" in ql:
        relation_queries.append("producer of {entity}")
    if "writer" in ql or "screenwriter" in ql or "written" in ql or "wrote" in ql:
        relation_queries.append("writer of {entity}")
    if "composer" in ql or "soundtrack" in ql or "music" in ql:
        relation_queries.append("composer of {entity}")
    if "actor" in ql or "actress" in ql or "starring" in ql or "starred" in ql:
        relation_queries.append("cast of {entity}")
    if "author" in ql or "written by" in ql:
        relation_queries.append("author of {entity}")
    if "country" in ql or "nationality" in ql:
        relation_queries.append("nationality associated with {entity}")
    if "born" in ql or "birth" in ql:
        relation_queries.append("birth information for {entity}")
    if "died" in ql or "death" in ql:
        relation_queries.append("death information for {entity}")
    if "release" in ql or "released" in ql:
        relation_queries.append("release date of {entity}")

    if not relation_queries:
        relation_queries = ["support evidence for {entity}"]

    entities = _candidate_entities(question)
    subqs = []
    for ent in entities[:3]:
        for tpl in relation_queries[:2]:
            subqs.append(tpl.format(entity=ent))
            if len(subqs) >= 4:
                return subqs
    return subqs or [question]


def _decompose_one(question: str, timeout: int = 30) -> list[str]:
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": _PROMPT.format(q=question)}],
        "max_tokens": 150, "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        LLM_URL, data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                text = json.loads(r.read())['choices'][0]['message']['content'].strip()
            s, e = text.find('['), text.rfind(']') + 1
            if s >= 0 and e > s:
                parsed = json.loads(text[s:e])
                if parsed and all(isinstance(x, str) for x in parsed):
                    return parsed[:3]
        except Exception as exc:
            if attempt == 2:
                log.warning(f"decompose failed for '{question[:60]}': {exc}")
            else:
                time.sleep(1)
    return [question]  # fallback: no decomposition


def batch_decompose(
    queries: list[dict],
    tag: str = '',
    max_workers: int = 40,
    mode: str | None = None,
) -> list[list[str]]:
    """Return list of sub-question lists (one per query)."""
    mode = (mode or os.environ.get('MO2_DECOMPOSE_MODE') or 'llm').lower()
    cache_prefix = 'heuristic_decompose' if mode == 'heuristic' else 'llm_decompose'
    cache_key = hashlib.md5((mode + tag + ''.join(q['question'] for q in queries)).encode()).hexdigest()[:12]
    cache_path = CACHE_DIR / f'{cache_prefix}_{cache_key}.json'
    if cache_path.exists():
        log.info(f'Loading cached {mode} decompositions from {cache_path.name}')
        with open(cache_path) as f:
            return json.load(f)

    if mode == 'heuristic':
        log.info(f'Decomposing {len(queries)} queries with local heuristic.')
        results = [heuristic_decompose(q['question']) for q in queries]
        with open(cache_path, 'w') as f:
            json.dump(results, f)
        log.info(f'Cached heuristic decompositions to {cache_path.name}')
        return results

    log.info(f'Decomposing {len(queries)} queries with LLM (workers={max_workers})...')
    results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_decompose_one, q['question']): i for i, q in enumerate(queries)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            if done % 200 == 0 or done == len(queries):
                log.info(f'  LLM decomposed {done}/{len(queries)}')

    with open(cache_path, 'w') as f:
        json.dump(results, f)
    log.info(f'Cached decompositions to {cache_path.name}')
    return results


def augment_with_llm(queries: list[dict], query_embs: np.ndarray,
                     query_ents: list[list[str]], nlp,
                     tag: str = '', max_workers: int = 40):
    """Return augmented (query_embs, query_ents) using LLM sub-question expansion.

    Args:
        queries: list of query dicts (must have 'question' field)
        query_embs: np.ndarray shape (N, D)
        query_ents: list of entity lists per query
        nlp: spaCy nlp object for entity extraction on sub-questions
        tag: cache tag string
        max_workers: LLM concurrency

    Returns:
        aug_embs: np.ndarray shape (N, D) — unit-normed mean of original + sub-q embeddings
        aug_ents: list of entity lists — union of original + sub-q entities
    """
    from sentence_transformers import SentenceTransformer
    from config import EMBED_MODEL, BGE_QUERY_PREFIX
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    sbert = SentenceTransformer(EMBED_MODEL, device=device)
    _qprefix = BGE_QUERY_PREFIX if 'bge' in EMBED_MODEL.lower() else ''

    sub_lists = batch_decompose(queries, tag=tag, max_workers=max_workers)

    # Encode all sub-questions flat then split back
    flat_subs = [sq for sqs in sub_lists for sq in sqs]
    log.info(f'Encoding {len(flat_subs)} sub-questions...')
    from tqdm import tqdm
    sub_all_embs = sbert.encode([_qprefix + q for q in flat_subs], batch_size=256, show_progress_bar=True,
                                normalize_embeddings=True)

    aug_embs = np.empty_like(query_embs)
    aug_ents = []
    ptr = 0
    for i, sqs in enumerate(sub_lists):
        n = len(sqs)
        sub_e = sub_all_embs[ptr:ptr + n]
        ptr += n
        stacked = np.vstack([query_embs[i:i+1], sub_e])  # (1+n, D)
        mean_e = stacked.mean(axis=0)
        norm = np.linalg.norm(mean_e)
        aug_embs[i] = mean_e / norm if norm > 1e-9 else mean_e

        # Entity augmentation: run spaCy NER on sub-questions
        sub_ents = set(query_ents[i]) if query_ents else set()
        for sq in sqs:
            doc = nlp(sq)
            sub_ents.update(ent.text for ent in doc.ents)
        aug_ents.append(list(sub_ents))

    log.info(f'LLM augmentation done. Avg sub-ents per query: '
             f'{np.mean([len(e) for e in aug_ents]):.1f}')
    return aug_embs, aug_ents
