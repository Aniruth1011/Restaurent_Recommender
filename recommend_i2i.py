"""
recommend_i2i.py
Item-to-item recommendation using combined text + structured embeddings.

Inference flow:
  1. Take user's liked restaurants from train (rating >= 4)
  2. Weighted average of their embeddings (weight = rating * recency_weight)
  3. Cosine similarity against all item embeddings
  4. Geo filter within max_miles of user centroid
  5. Rerank by similarity + popularity
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

OUTPUT_DIR = Path("output_review")
MODELS_DIR = Path("models_review")
EMBED_DIR  = Path("embeddings_i2i")

N_CANDIDATES   = 100   # retrieve before geo filter
N_RECOMMENDATIONS = 20

# add this constant at the top
ALPHA_SCALE = 5   # at 5+ reviews, fully trust I2I


def get_alpha(user_id, reviews):
    """
    Alpha = how much to trust I2I vs popularity.
    0.0 = pure popularity, 1.0 = pure I2I
    """
    n = len(reviews[reviews["user_id"] == user_id])
    return min(1.0, n / ALPHA_SCALE)


def popularity_scores(restaurants, seen, user_lat, user_lon, max_miles, n):
    """Return popularity-ranked gmap_ids with normalized scores."""
    df = restaurants[~restaurants["gmap_id"].isin(seen)].copy()

    if user_lat is not None and user_lon is not None and max_miles is not None:
        df["distance_miles"] = df.apply(
            lambda r: haversine(user_lat, user_lon, r["latitude"], r["longitude"]), axis=1
        )
        df = df[df["distance_miles"] <= max_miles]

    if df.empty:
        return {}

    max_rating = df["weighted_avg_rating"].max() or 1.0
    max_count  = df["review_count"].max() or 1.0
    df["pop_score"] = (
        0.7 * df["weighted_avg_rating"].fillna(0) / max_rating +
        0.3 * df["review_count"].fillna(0) / max_count
    )
    return dict(zip(df["gmap_id"], df["pop_score"]))

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_artifacts():
    item_embeddings = np.load(EMBED_DIR / "item_embeddings.npy")
    item_ids        = np.load(EMBED_DIR / "item_embedding_ids.npy", allow_pickle=True)
    restaurants     = pd.read_parquet(OUTPUT_DIR / "restaurants.parquet")
    reviews         = pd.read_parquet(OUTPUT_DIR / "reviews.parquet")
    id_to_idx       = {gid: i for i, gid in enumerate(item_ids)}
    return item_embeddings, item_ids, id_to_idx, restaurants, reviews


def build_user_vector(user_id, reviews, item_embeddings, id_to_idx):
    """
    Weighted average of embeddings of restaurants the user liked (rating >= 4).
    Falls back to all reviews if no liked ones found.
    """
    user_reviews = reviews[reviews["user_id"] == user_id].copy()
    liked = user_reviews[user_reviews["rating"] >= 4]
    if liked.empty:
        liked = user_reviews   # fallback to all

    vectors = []
    weights = []
    for _, row in liked.iterrows():
        idx = id_to_idx.get(row["gmap_id"])
        if idx is None:
            continue
        w = float(row["rating"]) * float(row.get("recency_weight", 1.0))
        vectors.append(item_embeddings[idx])
        weights.append(w)

    if not vectors:
        return None

    vectors = np.array(vectors)
    weights = np.array(weights)
    user_vec = np.average(vectors, axis=0, weights=weights)
    return normalize(user_vec.reshape(1, -1), norm="l2")


def get_user_centroid(user_id, reviews, restaurants):
    user_gids = reviews[reviews["user_id"] == user_id]["gmap_id"].tolist()
    locs = restaurants[restaurants["gmap_id"].isin(user_gids)][["latitude", "longitude"]].dropna()
    if locs.empty:
        return None, None
    return locs["latitude"].mean(), locs["longitude"].mean()


def dedupe_chains(df, max_per_chain=1):
    df = df.copy()
    df["chain_name"] = df["name"].str.lower().str.strip()
    df = df.groupby("chain_name").head(max_per_chain).reset_index(drop=True)
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df = df.drop(columns=["chain_name"])
    return df


# def recommend_i2i(user_id, reviews, item_embeddings, item_ids, id_to_idx,
#                   restaurants, user_lat=None, user_lon=None,
#                   max_miles=None, n=N_RECOMMENDATIONS):

#     # build user vector
#     user_vec = build_user_vector(user_id, reviews, item_embeddings, id_to_idx)
#     if user_vec is None:
#         return pd.DataFrame()

#     # cosine similarity
#     scores = cosine_similarity(user_vec, item_embeddings)[0]

#     # exclude already seen
#     seen = set(reviews[reviews["user_id"] == user_id]["gmap_id"])
#     for gid in seen:
#         idx = id_to_idx.get(gid)
#         if idx is not None:
#             scores[idx] = -1.0

#     top_idx = np.argsort(scores)[::-1][:N_CANDIDATES]
#     candidates = [(item_ids[i], float(scores[i])) for i in top_idx]

#     # geo filter
#     if user_lat is not None and user_lon is not None and max_miles is not None:
#         rest_locs = restaurants.set_index("gmap_id")[["latitude", "longitude"]]
#         filtered = []
#         for gid, score in candidates:
#             if gid not in rest_locs.index:
#                 continue
#             row = rest_locs.loc[gid]
#             if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
#                 continue
#             dist = haversine(user_lat, user_lon, row["latitude"], row["longitude"])
#             if dist <= max_miles:
#                 filtered.append((gid, score, dist))
#         if not filtered:
#             return pd.DataFrame()
#         cand_df = pd.DataFrame(filtered, columns=["gmap_id", "sim_score", "distance_miles"])
#     else:
#         cand_df = pd.DataFrame(candidates, columns=["gmap_id", "sim_score"])
#         cand_df["distance_miles"] = np.nan

#     # merge restaurant info
#     cand_df = cand_df.merge(
#         restaurants[["gmap_id", "name", "weighted_avg_rating", "review_count", "recency_weight"]],
#         on="gmap_id", how="left"
#     )

#     # normalize popularity signals
#     max_rating = cand_df["weighted_avg_rating"].max() or 1.0
#     max_count  = cand_df["review_count"].max() or 1.0

#     cand_df["rating_score"]  = cand_df["weighted_avg_rating"].fillna(0) / max_rating
#     cand_df["popular_score"] = cand_df["review_count"].fillna(0) / max_count

#     # final score
#     cand_df["final_score"] = (
#         0.6 * cand_df["sim_score"] +
#         0.2 * cand_df["rating_score"] +
#         0.2 * cand_df["popular_score"]
#     )

#     cand_df = cand_df.sort_values("final_score", ascending=False)
#     cand_df = dedupe_chains(cand_df)

#     out_cols = ["gmap_id", "name", "final_score", "sim_score", "distance_miles"]
#     return cand_df[out_cols].head(n).reset_index(drop=True)


# convenience loader for eval script
def load_all():
    item_embeddings, item_ids, id_to_idx, restaurants, reviews = load_artifacts()
    return item_embeddings, item_ids, id_to_idx, restaurants, reviews


def recommend_i2i(user_id, reviews, item_embeddings, item_ids, id_to_idx,
                  restaurants, user_lat=None, user_lon=None,
                  max_miles=None, n=N_RECOMMENDATIONS):

    seen = set(reviews[reviews["user_id"] == user_id]["gmap_id"])
    alpha = get_alpha(user_id, reviews)

    # --- I2I scores ---
    user_vec = build_user_vector(user_id, reviews, item_embeddings, id_to_idx)
    i2i_scores = {}
    if user_vec is not None:
        scores = cosine_similarity(user_vec, item_embeddings)[0]
        for gid in seen:
            idx = id_to_idx.get(gid)
            if idx is not None:
                scores[idx] = -1.0
        top_idx = np.argsort(scores)[::-1][:N_CANDIDATES]

        if user_lat is not None and user_lon is not None and max_miles is not None:
            rest_locs = restaurants.set_index("gmap_id")[["latitude", "longitude"]]
            for i in top_idx:
                gid = item_ids[i]
                if gid in rest_locs.index:
                    row = rest_locs.loc[gid]
                    if not pd.isna(row["latitude"]) and not pd.isna(row["longitude"]):
                        dist = haversine(user_lat, user_lon, row["latitude"], row["longitude"])
                        if dist <= max_miles:
                            i2i_scores[gid] = float(scores[i])
        else:
            i2i_scores = {item_ids[i]: float(scores[i]) for i in top_idx}

    # normalize i2i scores to [0,1]
    if i2i_scores:
        min_s = min(i2i_scores.values())
        max_s = max(i2i_scores.values())
        if max_s > min_s:
            i2i_scores = {g: (s - min_s) / (max_s - min_s) for g, s in i2i_scores.items()}

    # --- popularity scores ---
    pop_scores = popularity_scores(restaurants, seen, user_lat, user_lon, max_miles, N_CANDIDATES)

    # normalize pop scores to [0,1]
    if pop_scores:
        min_s = min(pop_scores.values())
        max_s = max(pop_scores.values())
        if max_s > min_s:
            pop_scores = {g: (s - min_s) / (max_s - min_s) for g, s in pop_scores.items()}

    # --- blend ---
    all_ids = set(i2i_scores.keys()) | set(pop_scores.keys())
    combined = {
        gid: alpha * i2i_scores.get(gid, 0.0) + (1 - alpha) * pop_scores.get(gid, 0.0)
        for gid in all_ids
    }

    top_ids = sorted(combined, key=combined.get, reverse=True)[:N_CANDIDATES]

    # --- rerank ---
    cand_df = pd.DataFrame({
        "gmap_id":    top_ids,
        "hybrid_score": [combined[g] for g in top_ids]
    })
    cand_df = cand_df.merge(
        restaurants[["gmap_id", "name", "weighted_avg_rating", "review_count", "recency_weight"]],
        on="gmap_id", how="left"
    )

    max_rating = cand_df["weighted_avg_rating"].max() or 1.0
    max_count  = cand_df["review_count"].max() or 1.0
    cand_df["rating_score"]  = cand_df["weighted_avg_rating"].fillna(0) / max_rating
    cand_df["popular_score"] = cand_df["review_count"].fillna(0) / max_count

    cand_df["final_score"] = (
        0.6 * cand_df["hybrid_score"] +
        0.2 * cand_df["rating_score"] +
        0.2 * cand_df["popular_score"]
    )

    cand_df = cand_df.sort_values("final_score", ascending=False)
    cand_df = dedupe_chains(cand_df)

    # add distance
    if user_lat is not None and user_lon is not None:
        cand_df["distance_miles"] = cand_df.apply(
            lambda r: haversine(user_lat, user_lon,
                                restaurants.set_index("gmap_id").loc[r["gmap_id"], "latitude"],
                                restaurants.set_index("gmap_id").loc[r["gmap_id"], "longitude"])
            if r["gmap_id"] in restaurants.set_index("gmap_id").index else np.nan, axis=1
        )

    out_cols = ["gmap_id", "name", "final_score", "distance_miles"] \
               if "distance_miles" in cand_df.columns else ["gmap_id", "name", "final_score"]
    return cand_df[out_cols].head(n).reset_index(drop=True)
if __name__ == "__main__":
    # quick smoke test
    item_embeddings, item_ids, id_to_idx, restaurants, reviews = load_all()
    sample_user = reviews["user_id"].iloc[0]
    user_lat, user_lon = get_user_centroid(sample_user, reviews, restaurants)
    recs = recommend_i2i(sample_user, reviews, item_embeddings, item_ids,
                         id_to_idx, restaurants, user_lat=user_lat,
                         user_lon=user_lon, max_miles=50)
    print(recs) 


# """
# recommend_i2i.py
# Item-to-item recommendation with:
#   1. Hybrid blending  — I2I + popularity, weighted by user history size
#   2. Query expansion  — one vector per liked restaurant, union of candidates
#   3. MMR reranking    — Maximal Marginal Relevance for diversity + relevance
# """

# import numpy as np
# import pandas as pd
# from pathlib import Path
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import normalize

# OUTPUT_DIR = Path("output_review")
# MODELS_DIR = Path("models_review")
# EMBED_DIR  = Path("embeddings_i2i")

# N_CANDIDATES      = 200
# N_RECOMMENDATIONS = 20
# ALPHA_SCALE       = 5     # at 5+ reviews, fully trust I2I
# MMR_LAMBDA        = 1   # 0=max diversity, 1=max relevance


# # ── helpers ───────────────────────────────────────────────────────────────────

# def haversine(lat1, lon1, lat2, lon2):
#     R = 3958.8
#     lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
#     return R * 2 * np.arcsin(np.sqrt(a))


# def load_artifacts():
#     item_embeddings = np.load(EMBED_DIR / "item_embeddings.npy")
#     item_ids        = np.load(EMBED_DIR / "item_embedding_ids.npy", allow_pickle=True)
#     restaurants     = pd.read_parquet(OUTPUT_DIR / "restaurants.parquet")
#     reviews         = pd.read_parquet(OUTPUT_DIR / "reviews.parquet")
#     id_to_idx       = {gid: i for i, gid in enumerate(item_ids)}
#     return item_embeddings, item_ids, id_to_idx, restaurants, reviews


# def load_all():
#     return load_artifacts()


# def get_user_centroid(user_id, reviews, restaurants):
#     user_gids = reviews[reviews["user_id"] == user_id]["gmap_id"].tolist()
#     locs = restaurants[restaurants["gmap_id"].isin(user_gids)][["latitude", "longitude"]].dropna()
#     if locs.empty:
#         return None, None
#     return locs["latitude"].mean(), locs["longitude"].mean()


# def dedupe_chains(df, max_per_chain=1):
#     df = df.copy()
#     df["chain_name"] = df["name"].str.lower().str.strip()
#     df = df.groupby("chain_name").head(max_per_chain).reset_index(drop=True)
#     df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
#     return df.drop(columns=["chain_name"])


# def normalize_dict(d):
#     if not d:
#         return d
#     min_s = min(d.values())
#     max_s = max(d.values())
#     if max_s == min_s:
#         return {g: 1.0 for g in d}
#     return {g: (s - min_s) / (max_s - min_s) for g, s in d.items()}


# # ── 1. hybrid blending ────────────────────────────────────────────────────────

# def get_alpha(user_id, reviews):
#     """0.0 = pure popularity, 1.0 = pure I2I"""
#     n = len(reviews[reviews["user_id"] == user_id])
#     return min(1.0, n / ALPHA_SCALE)


# def get_popularity_scores(restaurants, seen, user_lat, user_lon, max_miles, n):
#     df = restaurants[~restaurants["gmap_id"].isin(seen)].copy()

#     if user_lat is not None and user_lon is not None and max_miles is not None:
#         df["distance_miles"] = df.apply(
#             lambda r: haversine(user_lat, user_lon, r["latitude"], r["longitude"]), axis=1
#         )
#         df = df[df["distance_miles"] <= max_miles]

#     if df.empty:
#         return {}

#     max_rating = df["weighted_avg_rating"].max() or 1.0
#     max_count  = df["review_count"].max() or 1.0
#     df["pop_score"] = (
#         0.7 * df["weighted_avg_rating"].fillna(0) / max_rating +
#         0.3 * df["review_count"].fillna(0) / max_count
#     )
#     return dict(zip(df["gmap_id"], df["pop_score"]))


# # ── 2. query expansion ────────────────────────────────────────────────────────

# def get_per_item_vectors(user_id, reviews, item_embeddings, id_to_idx):
#     """One embedding per liked restaurant, weighted by rating * recency."""
#     user_reviews = reviews[reviews["user_id"] == user_id].copy()
#     liked = user_reviews[user_reviews["rating"] >= 4]
#     if liked.empty:
#         liked = user_reviews

#     items = []
#     for _, row in liked.iterrows():
#         idx = id_to_idx.get(row["gmap_id"])
#         if idx is None:
#             continue
#         w   = float(row["rating"]) * float(row.get("recency_weight", 1.0))
#         vec = normalize(item_embeddings[idx].reshape(1, -1), norm="l2")
#         items.append((vec, w))
#     return items


# def query_expansion_scores(user_id, reviews, item_embeddings, item_ids,
#                             id_to_idx, seen):
#     """
#     For each liked restaurant, retrieve top-N similar items.
#     Merge candidate sets weighted by item importance.
#     Captures multi-faceted taste (likes Korean AND Italian).
#     """
#     item_vecs = get_per_item_vectors(user_id, reviews, item_embeddings, id_to_idx)
#     if not item_vecs:
#         return {}

#     total_weight    = sum(w for _, w in item_vecs)
#     combined_scores = {}

#     for vec, w in item_vecs:
#         scores  = cosine_similarity(vec, item_embeddings)[0]
#         top_idx = np.argsort(scores)[::-1][:N_CANDIDATES]
#         for i in top_idx:
#             gid = item_ids[i]
#             if gid in seen:
#                 continue
#             contribution = (w / total_weight) * float(scores[i])
#             combined_scores[gid] = combined_scores.get(gid, 0.0) + contribution

#     return combined_scores


# # ── 3. MMR reranking ──────────────────────────────────────────────────────────

# def mmr_rerank(candidates, scores, item_embeddings, id_to_idx,
#                n=N_RECOMMENDATIONS, lambda_=MMR_LAMBDA):
#     """
#     Maximal Marginal Relevance:
#     At each step pick the candidate maximizing:
#       lambda * relevance - (1-lambda) * max_similarity_to_already_selected
#     """
#     if not candidates:
#         return []

#     valid      = [(g, id_to_idx[g]) for g in candidates if g in id_to_idx]
#     cand_ids   = [g for g, _ in valid]
#     cand_idx   = [i for _, i in valid]

#     if not cand_ids:
#         return candidates[:n]

#     cand_embs = item_embeddings[cand_idx]
#     relevance = np.array([scores.get(g, 0.0) for g in cand_ids])

#     selected     = []
#     selected_emb = []
#     remaining    = list(range(len(cand_ids)))

#     for _ in range(min(n, len(cand_ids))):
#         if not remaining:
#             break
#         if not selected_emb:
#             best = max(remaining, key=lambda i: relevance[i])
#         else:
#             sel_embs        = np.stack(selected_emb)
#             sim_to_selected = cosine_similarity(cand_embs[remaining], sel_embs).max(axis=1)
#             mmr_scores      = lambda_ * relevance[remaining] - (1 - lambda_) * sim_to_selected
#             best            = remaining[int(np.argmax(mmr_scores))]

#         selected.append(cand_ids[best])
#         selected_emb.append(cand_embs[best])
#         remaining.remove(best)

#     return selected


# # ── main recommend function ───────────────────────────────────────────────────

# def recommend_i2i(user_id, reviews, item_embeddings, item_ids, id_to_idx,
#                   restaurants, user_lat=None, user_lon=None,
#                   max_miles=None, n=N_RECOMMENDATIONS):

#     seen  = set(reviews[reviews["user_id"] == user_id]["gmap_id"])
#     alpha = get_alpha(user_id, reviews)

#     # precompute restaurant locations for geo filter
#     rest_locs = restaurants.set_index("gmap_id")[["latitude", "longitude"]]
#     # dedupe index if needed
#     rest_locs = rest_locs[~rest_locs.index.duplicated(keep="first")]

#     def passes_geo(gid):
#         if user_lat is None or user_lon is None or max_miles is None:
#             return True
#         if gid not in rest_locs.index:
#             return False
#         row = rest_locs.loc[gid]
#         if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
#             return False
#         return haversine(user_lat, user_lon, row["latitude"], row["longitude"]) <= max_miles

#     # ── I2I via query expansion ──
#     i2i_raw    = query_expansion_scores(user_id, reviews, item_embeddings,
#                                         item_ids, id_to_idx, seen)
#     i2i_scores = {g: s for g, s in i2i_raw.items() if passes_geo(g)}
#     i2i_scores = normalize_dict(i2i_scores)

#     # ── popularity scores ──
#     pop_raw    = get_popularity_scores(restaurants, seen, user_lat, user_lon,
#                                        max_miles, N_CANDIDATES)
#     pop_scores = normalize_dict(pop_raw)

#     # ── blend ──
#     all_ids  = set(i2i_scores.keys()) | set(pop_scores.keys())
#     combined = {
#         gid: alpha * i2i_scores.get(gid, 0.0) + (1 - alpha) * pop_scores.get(gid, 0.0)
#         for gid in all_ids
#     }

#     if not combined:
#         return pd.DataFrame()

#     # ── MMR rerank ──
#     candidates  = sorted(combined, key=combined.get, reverse=True)[:N_CANDIDATES]
#     mmr_ordered = mmr_rerank(candidates, combined, item_embeddings, id_to_idx,
#                              n=n * 2, lambda_=MMR_LAMBDA)

#     if not mmr_ordered:
#         return pd.DataFrame()

#     # ── build output dataframe ──
#     cand_df = pd.DataFrame({
#         "gmap_id":      mmr_ordered,
#         "hybrid_score": [combined.get(g, 0.0) for g in mmr_ordered]
#     })
#     cand_df = cand_df.merge(
#         restaurants[["gmap_id", "name", "weighted_avg_rating",
#                      "review_count", "recency_weight"]],
#         on="gmap_id", how="left"
#     )

#     max_rating = cand_df["weighted_avg_rating"].max() or 1.0
#     max_count  = cand_df["review_count"].max() or 1.0
#     cand_df["rating_score"]  = cand_df["weighted_avg_rating"].fillna(0) / max_rating
#     cand_df["popular_score"] = cand_df["review_count"].fillna(0) / max_count

#     cand_df["final_score"] = (
#         0.6 * cand_df["hybrid_score"] +
#         0.2 * cand_df["rating_score"] +
#         0.2 * cand_df["popular_score"]
#     )

#     cand_df = cand_df.sort_values("final_score", ascending=False)
#     cand_df = dedupe_chains(cand_df)

#     # add distance
#     if user_lat is not None and user_lon is not None:
#         def get_dist(gid):
#             if gid not in rest_locs.index:
#                 return np.nan
#             row = rest_locs.loc[gid]
#             if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
#                 return np.nan
#             return haversine(user_lat, user_lon, row["latitude"], row["longitude"])
#         cand_df["distance_miles"] = cand_df["gmap_id"].apply(get_dist)

#     out_cols = ["gmap_id", "name", "final_score", "distance_miles"] \
#                if "distance_miles" in cand_df.columns else ["gmap_id", "name", "final_score"]
#     return cand_df[out_cols].head(n).reset_index(drop=True)


# if __name__ == "__main__":
#     item_embeddings, item_ids, id_to_idx, restaurants, reviews = load_all()
#     sample_user = reviews["user_id"].iloc[0]
#     user_lat, user_lon = get_user_centroid(sample_user, reviews, restaurants)
#     recs = recommend_i2i(sample_user, reviews, item_embeddings, item_ids,
#                          id_to_idx, restaurants, user_lat=user_lat,
#                          user_lon=user_lon, max_miles=50)
#     print(recs)