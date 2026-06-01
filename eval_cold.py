"""
eval_cold.py
Cold start evaluation using item-to-item similarity.
Cold users have no train history — we use their cold_start reviews
as if they just signed up and rated 1-2 places.

Strategy:
  - Split each cold user's reviews into "seed" (first half) and "held out" (second half)
  - Use seed reviews to build user vector (I2I)
  - Evaluate against held out reviews
  - Compare against popularity baseline
"""

import numpy as np
import pandas as pd
from pathlib import Path
import ast
from recommend_i2i import (load_all, recommend_i2i, get_user_centroid,
                           normalize_dict, dedupe_chains,
                            haversine, MMR_LAMBDA)
from recommend_i2i import popularity_score as get_popularity_score
OUTPUT_DIR = Path("output_review")

MAX_DISTANCE_MILES = 50
N_USERS            = 500
K                  = 10


# ── metrics ───────────────────────────────────────────────────────────────────

def get_chain_name(name):
    return name.lower().strip() if isinstance(name, str) else ""


def precision_at_k(recommended_ids, relevant_ids, k):
    return len(set(recommended_ids[:k]) & set(relevant_ids)) / k


def recall_at_k(recommended_ids, relevant_ids, k):
    hits = len(set(recommended_ids[:k]) & set(relevant_ids))
    return hits / len(relevant_ids) if relevant_ids else 0


def ndcg_at_k(recommended_ids, relevant_ids, k):
    dcg  = sum(1 / np.log2(i + 2) for i, g in enumerate(recommended_ids[:k]) if g in relevant_ids)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg > 0 else 0


def chain_recall_at_k(recommended_ids, relevant_chains, name_map, k):
    if not relevant_chains:
        return 0.0
    recommended_chains = set(get_chain_name(name_map.get(g, "")) for g in recommended_ids[:k])
    covered = recommended_chains & {c for c in relevant_chains if c != ""}
    return len(covered) / len(relevant_chains) if relevant_chains else 0.0


def chain_ndcg_at_k(recommended_ids, relevant_chains, name_map, k):
    if not relevant_chains:
        return 0.0
    dcg = sum(
        1 / np.log2(i + 2)
        for i, gid in enumerate(recommended_ids[:k])
        if get_chain_name(name_map.get(gid, "")) in relevant_chains
        and get_chain_name(name_map.get(gid, "")) != ""
    )
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_chains), k)))
    return dcg / idcg if idcg > 0 else 0.0


def get_categories(gmap_id, restaurants_indexed):
    if gmap_id not in restaurants_indexed.index:
        return set()
    cats = restaurants_indexed.loc[gmap_id, "category"]
    if isinstance(cats, pd.Series):
        cats = cats.iloc[0]
    if isinstance(cats, list):
        return set(c.lower().strip() for c in cats)
    return set()


def category_hit_at_k(recommended_ids, relevant_ids, restaurants_indexed, k):
    relevant_cats = set()
    for gid in relevant_ids:
        relevant_cats |= get_categories(gid, restaurants_indexed)
    if not relevant_cats:
        return 0
    for gid in recommended_ids[:k]:
        if get_categories(gid, restaurants_indexed) & relevant_cats:
            return 1
    return 0


def avg_rating_at_k(recommended_ids, restaurants_indexed, k):
    ratings = []
    for gid in recommended_ids[:k]:
        if gid in restaurants_indexed.index:
            r = restaurants_indexed.loc[gid, "weighted_avg_rating"]
            if isinstance(r, pd.Series):
                r = r.iloc[0]
            if pd.notna(r):
                ratings.append(float(r))
    return np.mean(ratings) if ratings else 0.0


def quality_frac_at_k(recommended_ids, restaurants_indexed, k, threshold=4.0):
    count = total = 0
    for gid in recommended_ids[:k]:
        if gid in restaurants_indexed.index:
            r = restaurants_indexed.loc[gid, "weighted_avg_rating"]
            if isinstance(r, pd.Series):
                r = r.iloc[0]
            if pd.notna(r):
                total += 1
                if float(r) >= threshold:
                    count += 1
    return count / total if total > 0 else 0.0


def compute_metrics(recommended_ids, relevant_ids, relevant_chains,
                    name_map, restaurants_indexed, k):
    recommended_chains = [get_chain_name(name_map.get(g, "")) for g in recommended_ids]
    return {
        "precision":    precision_at_k(recommended_ids, relevant_ids, k),
        "recall":       recall_at_k(recommended_ids, relevant_ids, k),
        "ndcg":         ndcg_at_k(recommended_ids, relevant_ids, k),
        "hit_strict":   1 if len(set(recommended_ids[:k]) & set(relevant_ids)) > 0 else 0,
        "chain_prec":   sum(1 for c in recommended_chains[:k] if c in relevant_chains and c != "") / k,
        "hit_chain":    1 if any(c in relevant_chains and c != "" for c in recommended_chains[:k]) else 0,
        "chain_recall": chain_recall_at_k(recommended_ids, relevant_chains, name_map, k),
        "chain_ndcg":   chain_ndcg_at_k(recommended_ids, relevant_chains, name_map, k),
        "cat_hit":      category_hit_at_k(recommended_ids, relevant_ids, restaurants_indexed, k),
        "avg_rating":   avg_rating_at_k(recommended_ids, restaurants_indexed, k),
        "quality_frac": quality_frac_at_k(recommended_ids, restaurants_indexed, k),
    }


def print_metrics(label, metrics_list, k):
    if not metrics_list:
        print(f"\n=== {label} === No users evaluated")
        return
    print(f"\n=== {label} ===")
    print(f"Evaluated on {len(metrics_list)} users")
    print(f"\n-- Strict Metrics --")
    print(f"Precision@{k}       (strict): {np.mean([m['precision']    for m in metrics_list]):.4f}")
    print(f"Recall@{k}          (strict): {np.mean([m['recall']       for m in metrics_list]):.4f}")
    print(f"NDCG@{k}            (strict): {np.mean([m['ndcg']         for m in metrics_list]):.4f}")
    print(f"Hit Rate@{k}        (strict): {np.mean([m['hit_strict']   for m in metrics_list]):.4f}")
    print(f"\n-- Chain Metrics --")
    print(f"Precision@{k}       (chain):  {np.mean([m['chain_prec']   for m in metrics_list]):.4f}")
    print(f"Recall@{k}          (chain):  {np.mean([m['chain_recall'] for m in metrics_list]):.4f}")
    print(f"NDCG@{k}            (chain):  {np.mean([m['chain_ndcg']   for m in metrics_list]):.4f}")
    print(f"Hit Rate@{k}        (chain):  {np.mean([m['hit_chain']    for m in metrics_list]):.4f}")
    print(f"\n-- Soft Metrics --")
    print(f"Category Hit@{k}:             {np.mean([m['cat_hit']      for m in metrics_list]):.4f}")
    print(f"Avg Rating@{k}:               {np.mean([m['avg_rating']   for m in metrics_list]):.4f}")
    print(f"Quality Frac@{k} (>=4.0):     {np.mean([m['quality_frac'] for m in metrics_list]):.4f}")


# ── cold start helpers ────────────────────────────────────────────────────────

def split_cold_user(user_reviews):
    """
    Split reviews into seed (first chronologically) and held-out (rest).
    Seed = what user provided at signup (earliest reviews)
    Held-out = what we try to predict
    """
    user_reviews = user_reviews.sort_values("review_dt")
    n = len(user_reviews)
    n_seed = max(1, n // 2)
    seed    = user_reviews.iloc[:n_seed]
    holdout = user_reviews.iloc[n_seed:]
    return seed, holdout


def get_centroid_from_seed(seed, restaurants):
    gids = seed["gmap_id"].tolist()
    locs = restaurants[restaurants["gmap_id"].isin(gids)][["latitude", "longitude"]].dropna()
    if locs.empty:
        return None, None
    return locs["latitude"].mean(), locs["longitude"].mean()


def filter_relevant_by_location(relevant_ids, user_lat, user_lon, restaurants, max_miles):
    if user_lat is None or max_miles is None:
        return relevant_ids
    rest_locs = restaurants.set_index("gmap_id")[["latitude", "longitude"]]
    rest_locs = rest_locs[~rest_locs.index.duplicated(keep="first")]
    filtered = []
    for gid in relevant_ids:
        if gid not in rest_locs.index:
            continue
        row = rest_locs.loc[gid]
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            continue
        if haversine(user_lat, user_lon, row["latitude"], row["longitude"]) <= max_miles:
            filtered.append(gid)
    return filtered


# ── popularity baseline for cold ─────────────────────────────────────────────

def popularity_recommend_cold(seen, restaurants, user_lat, user_lon, max_miles, n=20):
    pop_raw = get_popularity_scores(restaurants, seen, user_lat, user_lon, max_miles, n * 2)
    if not pop_raw:
        return []
    pop_scores = normalize_dict(pop_raw)
    top_ids = sorted(pop_scores, key=pop_scores.get, reverse=True)

    # build df for dedupe
    cand_df = pd.DataFrame({
        "gmap_id":     top_ids,
        "final_score": [pop_scores[g] for g in top_ids]
    }).merge(restaurants[["gmap_id", "name"]], on="gmap_id", how="left")
    cand_df = dedupe_chains(cand_df)
    return cand_df["gmap_id"].head(n).tolist()


# ── main eval ─────────────────────────────────────────────────────────────────

def evaluate_cold(n_users=N_USERS, k=K, max_miles=MAX_DISTANCE_MILES):
    print("Loading data...")
    cold        = pd.read_parquet(OUTPUT_DIR / "cold_start.parquet")
    restaurants = pd.read_parquet(OUTPUT_DIR / "restaurants.parquet")
    name_map    = restaurants.set_index("gmap_id")["name"].to_dict()

    print("Loading model artifacts...")
    item_embeddings, item_ids, id_to_idx, _, reviews_all = load_all()

    # load raw category info for soft metrics
    raw_restaurants = pd.read_csv(
        "restaurants_only.csv", engine="python", on_bad_lines="skip",
        usecols=["gmap_id", "category"]
    )
    raw_restaurants["category"] = raw_restaurants["category"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    raw_restaurants = raw_restaurants.drop_duplicates(subset=["gmap_id"])
    restaurants_indexed = raw_restaurants.set_index("gmap_id")
    rating_df = restaurants[["gmap_id", "weighted_avg_rating"]].set_index("gmap_id")
    restaurants_indexed = restaurants_indexed.join(rating_df, how="left")

    cold_users = cold["user_id"].unique()
    print(f"Total cold users: {len(cold_users)}")

    sample_users = np.random.default_rng(42).choice(
        cold_users, min(n_users, len(cold_users)), replace=False
    )

    i2i_metrics  = []
    pop_metrics  = []
    skipped      = 0
    debug_count  = 0

    for user_id in sample_users:
        user_reviews = cold[cold["user_id"] == user_id].copy()
        if len(user_reviews) < 2:
            skipped += 1
            continue

        seed, holdout = split_cold_user(user_reviews)

        # relevant = held-out reviews rated >= 4
        relevant_ids = holdout[holdout["rating"] >= 4]["gmap_id"].tolist()
        if not relevant_ids:
            skipped += 1
            continue

        # user location from seed
        user_lat, user_lon = get_centroid_from_seed(seed, restaurants)
        relevant_ids = filter_relevant_by_location(
            relevant_ids, user_lat, user_lon, restaurants, max_miles
        )
        if not relevant_ids:
            skipped += 1
            continue

        relevant_chains = set(get_chain_name(name_map.get(g, "")) for g in relevant_ids)
        seen = set(seed["gmap_id"].tolist())

        # ── I2I using seed reviews ──
        # temporarily build a fake reviews df with just seed data
        seed_reviews = seed.copy()
        seed_reviews["recency_weight"] = seed_reviews.get("recency_weight", 1.0)

        try:
            recs = recommend_i2i(
                user_id, seed_reviews, item_embeddings, item_ids, id_to_idx,
                restaurants, user_lat=user_lat, user_lon=user_lon, max_miles=max_miles
            )
            if recs.empty:
                i2i_recs = []
            else:
                i2i_recs = recs["gmap_id"].tolist()
        except Exception as e:
            print(f"I2I error user {user_id}: {e}")
            i2i_recs = []

        # ── popularity baseline ──
        pop_recs = popularity_recommend_cold(seen, restaurants, user_lat, user_lon, max_miles, n=k)

        # debug first 3
        if debug_count < 3:
            print(f"\n[DEBUG] Cold User {user_id}")
            print(f"  Seed:      {seed['gmap_id'].tolist()}")
            print(f"  Relevant:  {relevant_ids[:3]}")
            print(f"  I2I recs:  {i2i_recs[:3]}")
            print(f"  Pop recs:  {pop_recs[:3]}")
            debug_count += 1

        if i2i_recs:
            i2i_metrics.append(
                compute_metrics(i2i_recs, relevant_ids, relevant_chains,
                                name_map, restaurants_indexed, k)
            )
        if pop_recs:
            pop_metrics.append(
                compute_metrics(pop_recs, relevant_ids, relevant_chains,
                                name_map, restaurants_indexed, k)
            )

    print(f"\nSkipped: {skipped} | Evaluated: {len(i2i_metrics)}")
    print_metrics(f"Cold Start — I2I (max_miles={max_miles})", i2i_metrics, k)
    print_metrics(f"Cold Start — Popularity Baseline (max_miles={max_miles})", pop_metrics, k)


if __name__ == "__main__":
    evaluate_cold()
