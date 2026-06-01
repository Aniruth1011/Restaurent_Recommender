import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from recommend import (hybrid_recommend, rerank, build_interaction_matrix,
                       MODELS_DIR, OUTPUT_DIR)

MIN_HISTORY = 1
MAX_DISTANCE_MILES = None
N_USERS = 500
K = 10


def get_chain_name(name):
    return name.lower().strip() if isinstance(name, str) else ""


def precision_at_k(recommended_ids, relevant_ids, k):
    return len(set(recommended_ids[:k]) & set(relevant_ids)) / k


def recall_at_k(recommended_ids, relevant_ids, k):
    hits = len(set(recommended_ids[:k]) & set(relevant_ids))
    return hits / len(relevant_ids) if relevant_ids else 0


def ndcg_at_k(recommended_ids, relevant_ids, k):
    dcg = sum(
        1 / np.log2(i + 2)
        for i, gid in enumerate(recommended_ids[:k])
        if gid in relevant_ids
    )
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg > 0 else 0


def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def get_user_centroid(user_id, train, restaurants):
    user_gids = train[train["user_id"] == user_id]["gmap_id"].tolist()
    locs = restaurants[restaurants["gmap_id"].isin(user_gids)][["latitude", "longitude"]].dropna()
    if locs.empty:
        return None, None
    return locs["latitude"].mean(), locs["longitude"].mean()


def filter_relevant_by_location(relevant_ids, user_lat, user_lon, restaurants, max_miles):
    if user_lat is None:
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


def compute_metrics(recommended_ids, relevant_ids, relevant_chains, name_map, k):
    recommended_chains = [get_chain_name(name_map.get(g, "")) for g in recommended_ids]
    return {
        "precision":  precision_at_k(recommended_ids, relevant_ids, k),
        "recall":     recall_at_k(recommended_ids, relevant_ids, k),
        "ndcg":       ndcg_at_k(recommended_ids, relevant_ids, k),
        "chain_prec": sum(1 for c in recommended_chains[:k] if c in relevant_chains and c != "") / k,
        "hit_strict": 1 if len(set(recommended_ids[:k]) & set(relevant_ids)) > 0 else 0,
        "hit_chain":  1 if any(c in relevant_chains and c != "" for c in recommended_chains[:k]) else 0,
    }


def print_metrics(label, metrics_list, k):
    if not metrics_list:
        print(f"\n=== {label} === No users evaluated")
        return
    print(f"\n=== {label} ===")
    print(f"Evaluated on {len(metrics_list)} users")
    print(f"Precision@{k}  (strict): {np.mean([m['precision']  for m in metrics_list]):.4f}")
    print(f"Recall@{k}     (strict): {np.mean([m['recall']     for m in metrics_list]):.4f}")
    print(f"NDCG@{k}       (strict): {np.mean([m['ndcg']       for m in metrics_list]):.4f}")
    print(f"Precision@{k}  (chain):  {np.mean([m['chain_prec'] for m in metrics_list]):.4f}")
    print(f"Hit Rate@{k}   (strict): {np.mean([m['hit_strict'] for m in metrics_list]):.4f}")
    print(f"Hit Rate@{k}   (chain):  {np.mean([m['hit_chain']  for m in metrics_list]):.4f}")


def load_data():
    test        = pd.read_parquet(OUTPUT_DIR / "test.parquet")
    train       = pd.read_parquet(OUTPUT_DIR / "train.parquet")
    restaurants = pd.read_parquet(OUTPUT_DIR / "restaurants.parquet")  # has latest_review_days
    reviews     = pd.read_parquet(OUTPUT_DIR / "reviews.parquet")
    name_map    = restaurants.set_index("gmap_id")["name"].to_dict()
    train_user_counts = train.groupby("user_id").size()
    return test, train, train_user_counts, restaurants, reviews, name_map

def load_model_artifacts(reviews):
    with open(MODELS_DIR / "cf_model.pkl", "rb") as f:
        model = pickle.load(f)

    # CF matrix must match what the model was trained on (train split only)
    train_reviews = pd.read_parquet(OUTPUT_DIR / "train.parquet")
    matrix, _, item_map, user_to_idx, _ = build_interaction_matrix(train_reviews)

    item_features = pd.read_parquet(MODELS_DIR / "item_features.parquet")
    item_matrix   = np.load(MODELS_DIR / "item_matrix.npy")
    gmap_ids      = np.load(MODELS_DIR / "item_gmap_ids.npy", allow_pickle=True)
    return model, matrix, item_map, user_to_idx, item_features, item_matrix, gmap_ids

def evaluate_als(n_users=N_USERS, k=K, min_history=MIN_HISTORY, max_miles=MAX_DISTANCE_MILES):
    print("Loading data...")
    test, train, train_user_counts, restaurants, reviews, name_map = load_data()

    print("Loading model artifacts...")
    model, matrix, item_map, user_to_idx, item_features, item_matrix, gmap_ids = load_model_artifacts(reviews)

    test_users = test["user_id"].unique()
    valid_users = [u for u in test_users if train_user_counts.get(u, 0) >= min_history]
    print(f"Valid users (>={min_history} train reviews): {len(valid_users)}")
    sample_users = np.random.default_rng(42).choice(valid_users, min(n_users, len(valid_users)), replace=False)

    metrics_list = []
    skipped_no_relevant = 0
    skipped_no_recs = 0

    for user_id in sample_users:
        relevant_ids = test[(test["user_id"] == user_id) & (test["rating"] >= 4)]["gmap_id"].tolist()
        if not relevant_ids:
            skipped_no_relevant += 1
            continue

        user_lat, user_lon = get_user_centroid(user_id, train, restaurants)
        relevant_ids = filter_relevant_by_location(relevant_ids, user_lat, user_lon, restaurants, max_miles)
        if not relevant_ids:
            skipped_no_relevant += 1
            continue

        relevant_chains = set(get_chain_name(name_map.get(g, "")) for g in relevant_ids)

        try:
            recs_raw = hybrid_recommend(user_id, model, matrix, user_to_idx, item_map,
                                        reviews, item_features, item_matrix, gmap_ids)
            if not recs_raw:
                skipped_no_recs += 1
                continue
            recs = rerank(recs_raw, restaurants, user_lat, user_lon, max_miles)
            if recs.empty:
                skipped_no_recs += 1
                continue
            recommended_ids = recs["gmap_id"].tolist()
        except Exception as e:
            print(f"User {user_id} error: {e}")
            skipped_no_recs += 1
            continue

        metrics_list.append(compute_metrics(recommended_ids, relevant_ids, relevant_chains, name_map, k))

    print(f"Skipped (no relevant): {skipped_no_relevant} | Skipped (no recs): {skipped_no_recs}")
    print(f"Metrics collected: {len(metrics_list)}")
    print_metrics(f"ALS (min_history={min_history}, max_miles={max_miles})", metrics_list, k)
    return metrics_list


if __name__ == "__main__":
    evaluate_als()