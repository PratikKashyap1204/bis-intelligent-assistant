# Milestone 6 — Hybrid Retrieval + Retrieval Quality Hardening

Evaluation-driven. Dataset unchanged: `data/eval/retrieval_eval_dataset.json` (26 cases).
No commit, no push, no database mutation, no Docker changes.

## Selected configuration (evidence)

| Surface | Choice | Why |
|---|---|---|
| Default `search()` | keyword (unchanged) | Best Recall@5 (0.570) and multi-clause Recall@5 (0.444). Hybrid does not beat it overall. |
| RAG default | vector (unchanged) | Compatibility with M3/M4. |
| `method="vector"` (production factory) | cosine **similarity** floor **0.30** | Same Recall@1/3/5/10 as unthresholded vector; abstention 0.000 → 0.143. |
| `method="hybrid"` | RRF `k=60`, clause-only fusion, same vector floor | Best hybrid config: R@10 0.711 (beats keyword 0.702); R@1/3 unchanged vs naïve RRF; abstention 0.143. Not made default: R@5 0.474 vs keyword 0.570; multi-clause R@5 0.000 vs 0.444. |
| `VectorRetrievalBackend` constructor | `min_similarity=None` | Preserves M2/M5 always-return behaviour when constructed directly (tests, unthresholded mode). |

Score that is thresholded: **cosine similarity** `1 - pgvector cosine_distance`, stored on `RetrievalResult.relevance.score`. Not a confidence.

## Rejected (eval evidence)

- Weighted min-max fusion as default (R@5 still ≤ 0.500; only lexical-heavy 0.7/0.3 had any multi-clause R@5, still 0.167).
- RRF k ∈ {10, 20} (identical overall metrics to k=60).
- Vector floor 0.35/0.40/0.45 (abstention up to 0.429 but Recall@10 0.579 → 0.526/0.500).
- MMR / Jaccard diversification (hurt R@3 and R@5; multi-clause unchanged).
- Parser rewrite (q15 is a known sentence split; q16/q17 fail from ranking + table-row crowding, not from identity).
- Making hybrid the default search method.

## Metrics

See `m6_baseline_20260911T125022Z.json`, `m6_sweep_20260911T125205Z.json`, `m6_after_20260911T125816Z.json`.

Keyword is unchanged from the M5/M6 baseline. Vector recall/precision unchanged; abstention only. Hybrid is new.
