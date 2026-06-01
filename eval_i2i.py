"""
eval_i2i.py
Evaluation script for item-to-item recommender.
Primary metric: chain hit rate (realistic for geo-distributed chains)
Secondary:      strict hit rate, NDCG, precision, recall
Soft metrics:   category match, rating quality, diversity
"""

import numpy as np
import pandas as pd
from pathlib import Path
from recommend_i2i import (load_all, recommend_i2i, get_user_centroid)

OUTPUT_DIR = Path("output_review")

MIN_HISTORY        = 1
MAX_DISTANCE_MILES = 50
N_USERS            = 500
K                  = 20


# ── strict metrics ────────────────────────────────────────────────────────────

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


# ── soft metrics ──────────────────────────────────────────────────────────────

def get_categories(gmap_id, restaurants_indexed):
    if gmap_id not in restaurants_indexed.index:
        return set()
    cats = restaurants_indexed.loc[gmap_id, "category"]
    # handle duplicate index
    if isinstance(cats, pd.Series):
        cats = cats.iloc[0]
    if isinstance(cats, list):
        return set(c.lower().strip() for c in cats)
    return set()

def category_hit_at_k(recommended_ids, relevant_ids, restaurants_indexed, k):
    """
    Did we recommend at least one restaurant sharing a category
    with any relevant restaurant?
    """
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
            # handle duplicate index returning Series
            if isinstance(r, pd.Series):
                r = r.iloc[0]
            if pd.notna(r):
                ratings.append(float(r))
    return np.mean(ratings) if ratings else 0.0


def rating_above_threshold_at_k(recommended_ids, restaurants_indexed, k, threshold=4.0):
    count = 0
    total = 0
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

def category_recall_at_k(recommended_ids, relevant_ids, restaurants_indexed, k):
    """
    Of all unique categories in relevant items, what fraction
    did we cover in recommendations?
    """
    relevant_cats = set()
    for gid in relevant_ids:
        relevant_cats |= get_categories(gid, restaurants_indexed)
    if not relevant_cats:
        return 0.0
    recommended_cats = set()
    for gid in recommended_ids[:k]:
        recommended_cats |= get_categories(gid, restaurants_indexed)
    return len(recommended_cats & relevant_cats) / len(relevant_cats)

def chain_recall_at_k(recommended_ids, relevant_chains, name_map, k):
    """
    Of all relevant chains, what fraction did we cover in recommendations?
    """
    if not relevant_chains:
        return 0.0
    recommended_chains = set(get_chain_name(name_map.get(g, "")) for g in recommended_ids[:k])
    covered = recommended_chains & {c for c in relevant_chains if c != ""}
    return len(covered) / len(relevant_chains) if relevant_chains else 0.0


def chain_ndcg_at_k(recommended_ids, relevant_chains, name_map, k):
    """
    NDCG where a hit is defined as recommending any restaurant
    from a relevant chain.
    """
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
def category_ndcg_at_k(recommended_ids, relevant_ids, restaurants_indexed, k):
    """
    NDCG where a recommendation is considered a hit if it shares
    a category with any relevant item.
    """
    relevant_cats = set()
    for gid in relevant_ids:
        relevant_cats |= get_categories(gid, restaurants_indexed)
    if not relevant_cats:
        return 0.0
    dcg = sum(
        1 / np.log2(i + 2)
        for i, gid in enumerate(recommended_ids[:k])
        if get_categories(gid, restaurants_indexed) & relevant_cats
    )
    # ideal: all k recommendations are category hits
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_cats), k)))
    return dcg / idcg if idcg > 0 else 0.0
def diversity_at_k(recommended_ids, restaurants_indexed, k):
    """
    Intra-list diversity: fraction of unique top-level categories
    across recommendations. Higher = more diverse.
    """
    all_cats = set()
    for gid in recommended_ids[:k]:
        all_cats |= get_categories(gid, restaurants_indexed)
    return len(all_cats) / k if k > 0 else 0.0


# def rating_above_threshold_at_k(recommended_ids, restaurants_indexed, k, threshold=4.0):
#     """Fraction of recommended restaurants with rating >= threshold."""
#     count = 0
#     total = 0
#     for gid in recommended_ids[:k]:
#         if gid in restaurants_indexed.index:
#             r = restaurants_indexed.loc[gid, "weighted_avg_rating"]
#             if pd.notna(r):
#                 total += 1
#                 if float(r) >= threshold:
#                     count += 1
#     return count / total if total > 0 else 0.0


# ── combined metrics ──────────────────────────────────────────────────────────

def compute_metrics(recommended_ids, relevant_ids, relevant_chains,
                    name_map, restaurants_indexed, k):
    recommended_chains = [get_chain_name(name_map.get(g, "")) for g in recommended_ids]
    return {
        # strict
        "precision":    precision_at_k(recommended_ids, relevant_ids, k),
        "recall":       recall_at_k(recommended_ids, relevant_ids, k),
        "ndcg":         ndcg_at_k(recommended_ids, relevant_ids, k),
        "hit_strict":   1 if len(set(recommended_ids[:k]) & set(relevant_ids)) > 0 else 0,
        # chain
        "chain_prec":   sum(1 for c in recommended_chains[:k] if c in relevant_chains and c != "") / k,
        "hit_chain":    1 if any(c in relevant_chains and c != "" for c in recommended_chains[:k]) else 0,
        "chain_recall": chain_recall_at_k(recommended_ids, relevant_chains, name_map, k),
        "chain_ndcg":   chain_ndcg_at_k(recommended_ids, relevant_chains, name_map, k),
        # soft
        "cat_hit":      category_hit_at_k(recommended_ids, relevant_ids, restaurants_indexed, k),
        "cat_recall":   category_recall_at_k(recommended_ids, relevant_ids, restaurants_indexed, k),
        "cat_ndcg":     category_ndcg_at_k(recommended_ids, relevant_ids, restaurants_indexed, k),
        "avg_rating":   avg_rating_at_k(recommended_ids, restaurants_indexed, k),
        "diversity":    diversity_at_k(recommended_ids, restaurants_indexed, k),
        "quality_frac": rating_above_threshold_at_k(recommended_ids, restaurants_indexed, k),
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
    print(f"Category Hit@{k}:             {np.mean([m['cat_hit']      for m in metrics_list]):.4f}  (rec shares cuisine with relevant)")
    print(f"Category Recall@{k}:          {np.mean([m['cat_recall']   for m in metrics_list]):.4f}  (fraction of relevant cuisines covered)")
    print(f"Category NDCG@{k}:            {np.mean([m['cat_ndcg']     for m in metrics_list]):.4f}  (ranked category relevance)")
    print(f"Avg Rating@{k}:               {np.mean([m['avg_rating']   for m in metrics_list]):.4f}  (avg rating of recommendations)")
    print(f"Quality Frac@{k} (>=4.0):     {np.mean([m['quality_frac'] for m in metrics_list]):.4f}  (frac of recs rated >=4)")
    print(f"Diversity@{k}:                {np.mean([m['diversity']    for m in metrics_list]):.4f}  (unique categories / k)")
    # ── location helpers ──────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def filter_relevant_by_location(relevant_ids, user_lat, user_lon, restaurants, max_miles):
    if user_lat is None or max_miles is None:
        return relevant_ids
    rest_locs = restaurants.set_index("gmap_id")[["latitude", "longitude"]]
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


# ── main eval ─────────────────────────────────────────────────────────────────

def evaluate_i2i(n_users=N_USERS, k=K, min_history=MIN_HISTORY, max_miles=MAX_DISTANCE_MILES):
    print("Loading data...")
    test  = pd.read_parquet(OUTPUT_DIR / "test.parquet")
    train = pd.read_parquet(OUTPUT_DIR / "train.parquet")

    print("Loading model artifacts...")
    item_embeddings, item_ids, id_to_idx, restaurants, reviews = load_all()
    name_map = restaurants.set_index("gmap_id")["name"].to_dict()

    # load raw restaurants for category info
    raw_restaurants = pd.read_csv(
        "restaurants_only.csv", engine="python", on_bad_lines="skip",
        usecols=["gmap_id", "category"]
    )
    import ast
    raw_restaurants["category"] = raw_restaurants["category"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    restaurants_indexed = raw_restaurants.set_index("gmap_id")

    # merge weighted_avg_rating into restaurants_indexed for soft metrics
    rating_df = restaurants[["gmap_id", "weighted_avg_rating"]].set_index("gmap_id")
    restaurants_indexed = restaurants_indexed.join(rating_df, how="left")

    # valid users
    train_user_counts = train.groupby("user_id").size()
    test_users  = set(test["user_id"].unique())
    train_users = set(train["user_id"].unique())
    overlap     = test_users & train_users
    valid_users = [u for u in overlap if train_user_counts.get(u, 0) >= min_history]
    print(f"Valid users (>={min_history} train reviews): {len(valid_users)}")

    sample_users = np.random.default_rng(42).choice(
        valid_users, min(n_users, len(valid_users)), replace=False
    )

    metrics_list        = []
    skipped_no_relevant = 0
    skipped_no_recs     = 0
    debug_count         = 0

    for user_id in sample_users:
        relevant_ids = test[
            (test["user_id"] == user_id) & (test["rating"] >= 4)
        ]["gmap_id"].tolist()

        if not relevant_ids:
            skipped_no_relevant += 1
            continue

        user_lat, user_lon = get_user_centroid(user_id, train, restaurants)
        relevant_ids = filter_relevant_by_location(
            relevant_ids, user_lat, user_lon, restaurants, max_miles
        )
        if not relevant_ids:
            skipped_no_relevant += 1
            continue

        relevant_chains = set(get_chain_name(name_map.get(g, "")) for g in relevant_ids)

        try:
            recs = recommend_i2i(
                user_id, reviews, item_embeddings, item_ids, id_to_idx,
                restaurants, user_lat=user_lat, user_lon=user_lon, max_miles=max_miles
            )
            if recs.empty:
                skipped_no_recs += 1
                continue
            recommended_ids = recs["gmap_id"].tolist()
        except Exception as e:
            print(f"User {user_id} error: {e}")
            skipped_no_recs += 1
            continue

        # debug first 3 users
        if debug_count < 3:
            print(f"\n[DEBUG] User {user_id}")
            print(f"  Relevant:    {relevant_ids[:5]}")
            print(f"  Recommended: {recommended_ids[:5]}")
            print(f"  Overlap:     {set(recommended_ids[:k]) & set(relevant_ids)}")
            debug_count += 1

        metrics_list.append(
            compute_metrics(recommended_ids, relevant_ids, relevant_chains,
                            name_map, restaurants_indexed, k)
        )

    print(f"\nSkipped (no relevant): {skipped_no_relevant} | Skipped (no recs): {skipped_no_recs}")
    print(f"Metrics collected: {len(metrics_list)}")
    print_metrics(f"Item-to-Item (min_history={min_history}, max_miles={max_miles})", metrics_list, k)
    return metrics_list


if __name__ == "__main__":
    evaluate_i2i()