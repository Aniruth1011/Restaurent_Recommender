"""
recommend_engine.py
Configurable item-to-item recommendation engine for ablation experiments.

This generalises recommend_i2i.py so every method can be toggled independently:

  candidate_mode  : how the user's I2I query is built
                      "avg" -> single weighted-average vector over liked items
                      "qe"  -> query expansion: one weighted vector per liked
                               item, candidates merged by item weight
                    (both use weighted business embeddings; weight = rating * recency)
  use_hybrid      : blend I2I with popularity by alpha = min(1, n_reviews / scale)
  pop_only        : pure-popularity baseline (ignores I2I entirely)
  use_mmr         : Maximal Marginal Relevance rerank vs plain relevance sort
  mmr_lambda      : 1.0 = pure relevance, 0.0 = pure diversity

The expensive part (cosine over the full item matrix) depends only on the
user and candidate_mode, so it is split into score_user() (cacheable) and
rank() (cheap, runs per config). experiment_mmr.py exploits this.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

OUTPUT_DIR = Path("output_review")
EMBED_DIR  = Path("embeddings_i2i")

N_CANDIDATES      = 200
N_RECOMMENDATIONS = 20


# ── config ─────────────────────────────────────────────────────────────────────

@dataclass
class RecConfig:
    name: str = "config"
    candidate_mode: str = "avg"     # "avg" | "qe"
    use_hybrid: bool = True
    pop_only: bool = False
    use_mmr: bool = False
    mmr_lambda: float = 1.0
    alpha_scale: int = 5
    n_candidates: int = N_CANDIDATES
    # final relevance score weights
    w_hybrid: float = 0.6
    w_rating: float = 0.2
    w_pop: float = 0.2
    max_per_chain: int = 1


# ── artifacts ────────────────────────────────────────────────────────────────

def load_all():
    item_embeddings = np.load(EMBED_DIR / "item_embeddings.npy")
    item_ids        = np.load(EMBED_DIR / "item_embedding_ids.npy", allow_pickle=True)
    restaurants     = pd.read_parquet(OUTPUT_DIR / "restaurants.parquet")
    reviews         = pd.read_parquet(OUTPUT_DIR / "reviews.parquet")
    id_to_idx       = {gid: i for i, gid in enumerate(item_ids)}
    return item_embeddings, item_ids, id_to_idx, restaurants, reviews


# ── geo helpers ──────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def get_user_centroid(user_id, reviews, restaurants):
    user_gids = reviews[reviews["user_id"] == user_id]["gmap_id"].tolist()
    locs = restaurants[restaurants["gmap_id"].isin(user_gids)][["latitude", "longitude"]].dropna()
    if locs.empty:
        return None, None
    return locs["latitude"].mean(), locs["longitude"].mean()


def _rest_locs(restaurants):
    rl = restaurants.set_index("gmap_id")[["latitude", "longitude"]]
    return rl[~rl.index.duplicated(keep="first")]


def _make_geo_filter(rest_locs, user_lat, user_lon, max_miles):
    def passes(gid):
        if user_lat is None or user_lon is None or max_miles is None:
            return True
        if gid not in rest_locs.index:
            return False
        row = rest_locs.loc[gid]
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            return False
        return haversine(user_lat, user_lon, row["latitude"], row["longitude"]) <= max_miles
    return passes


# ── I2I candidate building (weighted) ────────────────────────────────────────

def _liked_rows(user_id, reviews):
    user_reviews = reviews[reviews["user_id"] == user_id]
    liked = user_reviews[user_reviews["rating"] >= 4]
    return liked if not liked.empty else user_reviews


def build_user_vector(user_id, reviews, item_embeddings, id_to_idx):
    """Single weighted-average vector. weight = rating * recency_weight."""
    liked = _liked_rows(user_id, reviews)
    vectors, weights = [], []
    for _, row in liked.iterrows():
        idx = id_to_idx.get(row["gmap_id"])
        if idx is None:
            continue
        w = float(row["rating"]) * float(row.get("recency_weight", 1.0))
        vectors.append(item_embeddings[idx])
        weights.append(w)
    if not vectors:
        return None
    user_vec = np.average(np.array(vectors), axis=0, weights=np.array(weights))
    return normalize(user_vec.reshape(1, -1), norm="l2")


def _avg_i2i_scores(user_id, reviews, item_embeddings, item_ids, id_to_idx,
                    seen, passes_geo, n_candidates):
    uvec = build_user_vector(user_id, reviews, item_embeddings, id_to_idx)
    if uvec is None:
        return {}
    scores = cosine_similarity(uvec, item_embeddings)[0]
    for gid in seen:
        idx = id_to_idx.get(gid)
        if idx is not None:
            scores[idx] = -1.0
    top_idx = np.argsort(scores)[::-1][:n_candidates]
    return {item_ids[i]: float(scores[i]) for i in top_idx if passes_geo(item_ids[i])}


def _qe_i2i_scores(user_id, reviews, item_embeddings, item_ids, id_to_idx,
                   seen, passes_geo, n_candidates):
    """Query expansion: one weighted vector per liked item, merge by weight."""
    liked = _liked_rows(user_id, reviews)
    item_vecs = []
    for _, row in liked.iterrows():
        idx = id_to_idx.get(row["gmap_id"])
        if idx is None:
            continue
        w   = float(row["rating"]) * float(row.get("recency_weight", 1.0))
        vec = normalize(item_embeddings[idx].reshape(1, -1), norm="l2")
        item_vecs.append((vec, w))
    if not item_vecs:
        return {}

    total_w = sum(w for _, w in item_vecs) or 1.0
    combined = {}
    for vec, w in item_vecs:
        scores  = cosine_similarity(vec, item_embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:n_candidates]
        for i in top_idx:
            gid = item_ids[i]
            if gid in seen:
                continue
            combined[gid] = combined.get(gid, 0.0) + (w / total_w) * float(scores[i])
    return {g: s for g, s in combined.items() if passes_geo(g)}


# ── popularity ───────────────────────────────────────────────────────────────

def popularity_scores(restaurants, seen, passes_geo):
    df = restaurants[~restaurants["gmap_id"].isin(seen)].copy()
    if df.empty:
        return {}
    df = df[df["gmap_id"].apply(passes_geo)]
    if df.empty:
        return {}
    max_rating = df["weighted_avg_rating"].max() or 1.0
    max_count  = df["review_count"].max() or 1.0
    df["pop_score"] = (
        0.7 * df["weighted_avg_rating"].fillna(0) / max_rating +
        0.3 * df["review_count"].fillna(0) / max_count
    )
    return dict(zip(df["gmap_id"], df["pop_score"]))


# ── score / rank split ───────────────────────────────────────────────────────

def get_alpha(user_id, reviews, alpha_scale):
    n = len(reviews[reviews["user_id"] == user_id])
    return min(1.0, n / alpha_scale)


def score_user(user_id, reviews, item_embeddings, item_ids, id_to_idx, restaurants,
               candidate_mode, user_lat=None, user_lon=None, max_miles=None,
               n_candidates=N_CANDIDATES, rest_locs=None):
    """
    Compute the user-dependent, config-light signals once.
    Returns a dict reusable across all configs sharing the same candidate_mode.
    """
    if rest_locs is None:
        rest_locs = _rest_locs(restaurants)
    passes_geo = _make_geo_filter(rest_locs, user_lat, user_lon, max_miles)

    seen = set(reviews[reviews["user_id"] == user_id]["gmap_id"])

    if candidate_mode == "qe":
        i2i_raw = _qe_i2i_scores(user_id, reviews, item_embeddings, item_ids,
                                 id_to_idx, seen, passes_geo, n_candidates)
    else:
        i2i_raw = _avg_i2i_scores(user_id, reviews, item_embeddings, item_ids,
                                  id_to_idx, seen, passes_geo, n_candidates)

    pop_raw = popularity_scores(restaurants, seen, passes_geo)
    return {"seen": seen, "i2i_raw": i2i_raw, "pop_raw": pop_raw,
            "n_reviews": len(reviews[reviews["user_id"] == user_id])}


def _normalize_dict(d):
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi == lo:
        return {g: 1.0 for g in d}
    return {g: (v - lo) / (hi - lo) for g, v in d.items()}


def _mmr_rerank(cand_ids, relevance, item_embeddings, id_to_idx, depth, lambda_):
    valid = [(g, id_to_idx[g]) for g in cand_ids if g in id_to_idx]
    if not valid:
        return cand_ids[:depth]
    ids  = [g for g, _ in valid]
    embs = item_embeddings[[i for _, i in valid]]
    rel  = np.array([relevance.get(g, 0.0) for g in ids])

    selected, selected_emb, remaining = [], [], list(range(len(ids)))
    for _ in range(min(depth, len(ids))):
        if not selected_emb:
            best = max(remaining, key=lambda i: rel[i])
        else:
            sel = np.stack(selected_emb)
            sim = cosine_similarity(embs[remaining], sel).max(axis=1)
            mmr = lambda_ * rel[remaining] - (1 - lambda_) * sim
            best = remaining[int(np.argmax(mmr))]
        selected.append(ids[best])
        selected_emb.append(embs[best])
        remaining.remove(best)
    return selected


def _dedupe_chains_ordered(ordered_ids, name_map, max_per_chain):
    counts, out = {}, []
    for g in ordered_ids:
        c = name_map.get(g, "")
        c = c.lower().strip() if isinstance(c, str) else ""
        if counts.get(c, 0) >= max_per_chain:
            continue
        counts[c] = counts.get(c, 0) + 1
        out.append(g)
    return out


def rank(scored, cfg, item_embeddings, id_to_idx, restaurants_indexed, name_map,
         n=N_RECOMMENDATIONS):
    """
    Turn cached user scores into a final ordered gmap_id list under cfg.
    restaurants_indexed: restaurants set_index('gmap_id') with weighted_avg_rating, review_count.
    """
    i2i = _normalize_dict(scored["i2i_raw"])
    pop = _normalize_dict(scored["pop_raw"])

    if cfg.pop_only:
        combined = pop
    elif cfg.use_hybrid:
        alpha = min(1.0, scored["n_reviews"] / cfg.alpha_scale)
        ids = set(i2i) | set(pop)
        combined = {g: alpha * i2i.get(g, 0.0) + (1 - alpha) * pop.get(g, 0.0) for g in ids}
    else:
        combined = i2i

    if not combined:
        return []

    cand = sorted(combined, key=combined.get, reverse=True)[:cfg.n_candidates]

    # rating / popularity signals normalised over the candidate set
    sub = restaurants_indexed.reindex(cand)
    max_rating = sub["weighted_avg_rating"].max() or 1.0
    max_count  = sub["review_count"].max() or 1.0
    rating_score = (sub["weighted_avg_rating"].fillna(0) / max_rating).to_dict()
    pop_score    = (sub["review_count"].fillna(0) / max_count).to_dict()

    relevance = {
        g: cfg.w_hybrid * combined[g]
           + cfg.w_rating * rating_score.get(g, 0.0)
           + cfg.w_pop * pop_score.get(g, 0.0)
        for g in cand
    }

    depth = max(n * 3, 60)
    if cfg.use_mmr:
        ordered = _mmr_rerank(cand, relevance, item_embeddings, id_to_idx,
                              depth, cfg.mmr_lambda)
    else:
        ordered = sorted(cand, key=relevance.get, reverse=True)[:depth]

    ordered = _dedupe_chains_ordered(ordered, name_map, cfg.max_per_chain)
    return ordered[:n]


def recommend(user_id, reviews, item_embeddings, item_ids, id_to_idx, restaurants,
              cfg=None, user_lat=None, user_lon=None, max_miles=None,
              n=N_RECOMMENDATIONS):
    """Standalone convenience: score + rank in one call. Returns a DataFrame."""
    cfg = cfg or RecConfig()
    rest_locs = _rest_locs(restaurants)
    scored = score_user(user_id, reviews, item_embeddings, item_ids, id_to_idx,
                         restaurants, cfg.candidate_mode, user_lat, user_lon,
                         max_miles, cfg.n_candidates, rest_locs)
    ri = restaurants.set_index("gmap_id")
    ri = ri[~ri.index.duplicated(keep="first")]
    name_map = ri["name"].to_dict()
    ids = rank(scored, cfg, item_embeddings, id_to_idx, ri, name_map, n)
    return pd.DataFrame({"gmap_id": ids, "name": [name_map.get(g, "") for g in ids]})


if __name__ == "__main__":
    emb, item_ids, id_to_idx, restaurants, reviews = load_all()
    u = reviews["user_id"].iloc[0]
    lat, lon = get_user_centroid(u, reviews, restaurants)
    for cfg in [RecConfig(name="avg|no-mmr", candidate_mode="avg"),
                RecConfig(name="qe|mmr0.7", candidate_mode="qe", use_mmr=True, mmr_lambda=0.7)]:
        recs = recommend(u, reviews, emb, item_ids, id_to_idx, restaurants,
                         cfg=cfg, user_lat=lat, user_lon=lon, max_miles=50)
        print(f"\n=== {cfg.name} ===")
        print(recs.head(10))
