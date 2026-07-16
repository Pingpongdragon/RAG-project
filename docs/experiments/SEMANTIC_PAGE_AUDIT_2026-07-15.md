# Semantic Page Cache Feasibility Audit

Audit version: `semantic-page-feasibility-v1`

This is a mechanism audit, not a main-paper result. `GoldDoc-LRU` and `GoldPage-LRU` are reactive gold-trace diagnostics, not prefetch upper bounds; only `RoutedPage-LRU` routes with the current query embedding. Capacity is equalized in documents, and page loads are also charged by document traffic. Writes are not capped in this feasibility run, so routed quality is optimistic while all resulting traffic remains visible.

## Command

```bash
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python benchmarks/audit_semantic_pages.py --datasets squad_direct hotpotqa_comparison 2wiki_comparison streamingqa_official --page-sizes 32 64 --route-widths 1 2 4 --cache-ratio 0.10 --output docs/experiments/SEMANTIC_PAGE_AUDIT_2026-07-15.json --report docs/experiments/SEMANTIC_PAGE_AUDIT_2026-07-15.md
```

## Results

| Dataset | Page | L | Route full support | Doc reuse | Page reuse | GoldDoc HA | GoldPage HA | RoutedPage HA | Gain | Doc write amp. | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| squad_direct | 32 | 1 | 50.2% | 79.8% | 93.6% | 84.8% | 15.4% | 13.2% | -71.6 pp | 154.17x | `no-page-locality-gain` |
| squad_direct | 32 | 2 | 66.0% | 79.8% | 93.6% | 84.8% | 15.4% | 12.8% | -72.0 pp | 333.78x | `no-page-locality-gain` |
| squad_direct | 32 | 4 | 79.6% | 79.8% | 93.6% | 84.8% | 15.4% | 12.0% | -72.8 pp | 716.71x | `no-page-locality-gain` |
| squad_direct | 64 | 1 | 55.2% | 79.8% | 96.8% | 84.8% | 5.0% | 6.4% | -78.4 pp | 345.48x | `no-page-locality-gain` |
| squad_direct | 64 | 2 | 74.0% | 79.8% | 96.8% | 84.8% | 5.0% | 5.8% | -79.0 pp | 726.41x | `no-page-locality-gain` |
| squad_direct | 64 | 4 | 84.8% | 79.8% | 96.8% | 84.8% | 5.0% | 10.2% | -74.6 pp | 1496.48x | `no-page-locality-gain` |
| hotpotqa_comparison | 32 | 1 | 10.0% | 26.8% | 42.2% | 26.5% | 9.5% | 16.0% | -10.5 pp | 19.17x | `no-page-locality-gain` |
| hotpotqa_comparison | 32 | 2 | 20.5% | 26.8% | 42.2% | 26.5% | 9.5% | 9.0% | -17.5 pp | 41.58x | `no-page-locality-gain` |
| hotpotqa_comparison | 32 | 4 | 28.5% | 26.8% | 42.2% | 26.5% | 9.5% | 9.5% | -17.0 pp | 77.64x | `no-page-locality-gain` |
| hotpotqa_comparison | 64 | 1 | 10.5% | 26.8% | 55.2% | 26.5% | 10.0% | 9.5% | -17.0 pp | 44.18x | `no-page-locality-gain` |
| hotpotqa_comparison | 64 | 2 | 21.5% | 26.8% | 55.2% | 26.5% | 10.0% | 11.0% | -15.5 pp | 80.89x | `no-page-locality-gain` |
| hotpotqa_comparison | 64 | 4 | 34.0% | 26.8% | 55.2% | 26.5% | 10.0% | 9.0% | -17.5 pp | 150.74x | `no-page-locality-gain` |
| 2wiki_comparison | 32 | 1 | 0.0% | 21.5% | 31.8% | 14.5% | 0.0% | 1.0% | -13.5 pp | 16.29x | `no-page-locality-gain` |
| 2wiki_comparison | 32 | 2 | 2.0% | 21.5% | 31.8% | 14.5% | 0.0% | 0.0% | -14.5 pp | 36.81x | `no-page-locality-gain` |
| 2wiki_comparison | 32 | 4 | 8.0% | 21.5% | 31.8% | 14.5% | 0.0% | 0.0% | -14.5 pp | 68.45x | `no-page-locality-gain` |
| 2wiki_comparison | 64 | 1 | 1.5% | 21.5% | 42.1% | 14.5% | 1.5% | 1.5% | -13.0 pp | 34.93x | `no-page-locality-gain` |
| 2wiki_comparison | 64 | 2 | 7.0% | 21.5% | 42.1% | 14.5% | 1.5% | 1.5% | -13.0 pp | 63.21x | `no-page-locality-gain` |
| 2wiki_comparison | 64 | 4 | 10.5% | 21.5% | 42.1% | 14.5% | 1.5% | 2.5% | -12.0 pp | 116.57x | `no-page-locality-gain` |
| streamingqa_official | 32 | 1 | 21.9% | 16.8% | 96.3% | 1.4% | 11.0% | 10.4% | +9.0 pp | 30.68x | `cost-limited` |
| streamingqa_official | 32 | 2 | 29.7% | 16.8% | 96.3% | 1.4% | 11.0% | 10.1% | +8.7 pp | 61.63x | `cost-limited` |
| streamingqa_official | 32 | 4 | 37.8% | 16.8% | 96.3% | 1.4% | 11.0% | 10.3% | +8.9 pp | 125.11x | `cost-limited` |
| streamingqa_official | 64 | 1 | 22.6% | 16.8% | 98.1% | 1.4% | 10.8% | 10.3% | +8.9 pp | 61.13x | `cost-limited` |
| streamingqa_official | 64 | 2 | 31.1% | 16.8% | 98.1% | 1.4% | 10.8% | 10.5% | +9.1 pp | 124.83x | `cost-limited` |
| streamingqa_official | 64 | 4 | 40.1% | 16.8% | 98.1% | 1.4% | 10.8% | 10.2% | +8.8 pp | 255.49x | `cost-limited` |

## Audit Conclusion

No tested page size/route width passes the joint quality-cost criterion. Whole-page caching should therefore not replace the document-level DRIP policy in its current form.

- **squad_direct:** best routed setting is page=32, L=1; Has-Answer 13.2% vs. 84.8% for GoldDoc-LRU, with 154.17x document traffic. Only 2.6% of loaded documents are reused within one window; active-page useful density is 10.1%.
- **hotpotqa_comparison:** best routed setting is page=32, L=1; Has-Answer 16.0% vs. 26.5% for GoldDoc-LRU, with 19.17x document traffic. Only 0.1% of loaded documents are reused within one window; active-page useful density is 4.3%.
- **2wiki_comparison:** best routed setting is page=64, L=4; Has-Answer 2.5% vs. 14.5% for GoldDoc-LRU, with 116.57x document traffic. Only 0.1% of loaded documents are reused within one window; active-page useful density is 2.2%.
- **streamingqa_official:** best routed setting is page=64, L=2; Has-Answer 10.5% vs. 1.4% for GoldDoc-LRU, with 124.83x document traffic. Only 1.4% of loaded documents are reused within one window; active-page useful density is 69.8%.

Recommended use: retain semantic partitions as cold-index routing or candidate-pruning metadata, then admit selected documents (or very small evidence bundles) with the existing replacement controller. Do not make an entire semantic page the cache replacement unit without a learned within-page selector.

## Reading The Verdict

- `no-page-locality-gain`: routed pages do not gain 2 pp and reactive gold pages expose no latent page benefit.
- `invalid-page-size`: at least one page is larger than the hot-tier document budget.
- `routing-limited`: pages contain useful future evidence, but query-to-page routing cannot recover it.
- `cost-limited`: routed pages improve answerability, but document traffic exceeds 4x.
- `promising`: routed pages gain at least 2 pp with no more than 4x document traffic.

The 2 pp and 4x thresholds are conservative audit heuristics, not statistical claims. A paper experiment must next compare against the repository's production LRU/ARC implementations, report multiple seeds, and assign measured byte/latency costs to page I/O.

Verdict counts: `cost-limited`=6, `no-page-locality-gain`=18
