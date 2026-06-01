# TableMind — Restaurant Recommender

An item-to-item (I2I) restaurant recommender with cuisine steering, cold-start
handling, and group recommendations. Served via a FastAPI backend and a Gradio
demo UI.

The recommender represents every restaurant as a learned embedding and matches a
user's taste (built from the restaurants they liked) against the catalog by cosine
similarity, then applies geo, popularity, and quality reranking.

---

## Deployed model

The deployed embedding is the **content (cuisine) encoder** — a residual MLP
trained on restaurant content (text + structured features) with a category-
supervised contrastive objective, so restaurants in the same cuisine cluster
together. It is the best model for general cuisine-match discovery.

### Metrics (`eval_i2i.py`, 303 evaluation users, K=20)

| Metric | Score |
|---|---|
| Category Hit@20 | **0.9769** |
| Category Recall@20 | **0.9001** |
| Category NDCG@20 | **2.2485** |
| Hit@20 | 0.0792 |
| Recall@20 | 0.0749 |
| NDCG@20 | 0.0424 |

**Headline:** the content encoder lifts ranked cuisine relevance (Category NDCG)
**+32%** over the frozen baseline (1.70 → 2.25) and covers **90%** of the user's
relevant cuisines, while keeping recommendation quality high (avg rating 4.4,
86% of picks rated ≥ 4.0).

> Note: with a median of ~1 review per user this dataset is cold-start dominated,
> so exact next-item Hit Rate is ~0 for all models; chain (same brand) and
> category (same cuisine) are the meaningful relevance proxies.

---

## Performance

Operational metrics from `benchmark.py` (in-process serving path, 150 mixed
requests, single-threaded, CPU):

| Metric | Value |
|---|---|
| Latency (mean / p50) | 316 ms / 304 ms |
| Latency (p95 / p99 / max) | 463 / 609 / 619 ms |
| Throughput | 3.2 req/s (single-threaded) |
| Model load (cold start) | 0.44 s |
| Embedding matrix | 80,636 × 567 = 183 MB (float32) |
| Peak RSS | 767 MB |
| Empty / error rate | 0% / 0% |

Per request type: cold-start 217 ms, cuisine+filter 309 ms, history 444 ms.

**Optimization:** initial latency was ~1.8 s/req. Two changes brought it to
~316 ms (**5.8×**, 0.5 → 3.2 req/s) with no model change:
1. **Vectorized geo filter** — compute haversine over the whole catalog once
   (numpy) instead of a per-row pandas lookup over all 80K rows.
2. **Two-stage retrieval → ranking** — when a cuisine/filter/geo is active, cosine
   scores only that pre-filtered subset (e.g. ~3.5K vs 80K), not the full catalog.

```bash
I2I_EMBED_DIR=embeddings_i2i_content python benchmark.py
```

---

## Features

- **Personalized I2I recommendations** — taste vector from a user's liked
  restaurants, matched by embedding similarity.
- **Cuisine steering** — an explicit cuisine request (e.g. "Indian") filters and
  ranks within that cuisine, even when the user's history points elsewhere.
- **Cold-start** — new users get cuisine-seeded or popularity-based picks.
- **Auto-recommend on login** — returning users land directly on personalized picks.
- **Group recommendations** — find one place that works for a group with mixed
  tastes and constraints (least-misery / average / most-pleasure).
- **Geo + filters** — distance radius, price, dietary, and amenity filters.

---

## Quick start

```bash
pip install -r requirements.txt
./start.sh
```
- Demo UI:  http://localhost:7860
- API docs: http://localhost:8000/docs

**Viewing a remote demo locally** (SSH port forward from your laptop):
```bash
ssh -L 7860:localhost:7860 user@your-server     # then open http://localhost:7860
```
Or get a public link: `SHARE=1 ./start.sh`.

**Serve a different embedding model** (env var, no code change):
```bash
I2I_EMBED_DIR=embeddings_i2i        ./start.sh    # frozen baseline
I2I_EMBED_DIR=embeddings_i2i_chain  ./start.sh    # brand/chain model
# (default: embeddings_i2i_content — the deployed cuisine model)
```

---

## How it works

1. **Item embeddings** — each restaurant → a content vector (frozen MiniLM text +
   structured features), refined by the trained content encoder.
2. **User vector** — rating- and recency-weighted average of the embeddings of the
   restaurants a user liked (rating ≥ 4).
3. **Retrieve** — cosine similarity of the user vector against all items; restrict
   to requested cuisine / filters; geo-filter by distance.
4. **Rank** — blend similarity with popularity (cold-start `alpha`), then rerank by
   similarity + rating + popularity, dedupe chains, return top-N.

---

## Evaluate

```bash
python eval_i2i.py                                          # frozen baseline
I2I_EMBED_DIR=embeddings_i2i_content python eval_i2i.py         # deployed content model
I2I_EMBED_DIR=embeddings_i2i_content python eval_collaborative.py 2   # dense users
```

---

## Project layout

| Path | Role |
|---|---|
| `api.py`, `serving.py`, `recommend_engine.py` | FastAPI serving stack |
| `demo.py`, `start.sh` | Gradio UI + launcher |
| `embed.py` | Builds the base content embeddings |
| `train_i2i_content.py` | Trains the deployed cuisine encoder |
| `train_i2i_chain.py`, `train_i2i_encoder.py` | Alternative encoders (brand / interaction) |
| `eval_i2i.py`, `eval_collaborative.py` | Evaluation |


