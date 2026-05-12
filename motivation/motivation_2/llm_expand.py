"""LLM-based query decomposition to improve multi-hop retrieval.

Decomposes each question into 2 sub-questions using a 30B LLM,
then returns augmented query embeddings (mean-pooled) and
augmented entity lists (union).
"""
import json, logging, hashlib, os, time
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


def batch_decompose(queries: list[dict], tag: str = '', max_workers: int = 40) -> list[list[str]]:
    """Return list of sub-question lists (one per query)."""
    cache_key = hashlib.md5((tag + ''.join(q['question'] for q in queries)).encode()).hexdigest()[:12]
    cache_path = CACHE_DIR / f'llm_decompose_{cache_key}.json'
    if cache_path.exists():
        log.info(f'Loading cached LLM decompositions from {cache_path.name}')
        with open(cache_path) as f:
            return json.load(f)

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
    sbert = SentenceTransformer(EMBED_MODEL, device='cuda')
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
