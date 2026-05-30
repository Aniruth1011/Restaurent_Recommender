import pandas as pd
import numpy as np
import pickle
import torch
from pathlib import Path
from recommend import recommend
from enc2 import TwoTowerModel, EMBEDDING_DIM, DEVICE

MODELS_DIR = Path("models")
EMBEDDINGS_DIR = Path("embeddings")
OUTPUT_DIR = Path("output")

MIN_HISTORY = 3
MAX_DISTANCE_MILES =  50
N_USERS = 200
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
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def get_user_centroid(user_id, train, restaurants):
    user_gids = train[train["user_id"]==user_id]["gmap_id"].tolist()
    locs = restaurants[restaurants["gmap_id"].isin(user_gids)][["latitude","longitude"]].dropna()
    if locs.empty:
        return None, None
    return locs["latitude"].mean(), locs["longitude"].mean()


def filter_relevant_by_location(relevant_ids, user_lat, user_lon, restaurants, max_miles):
    if user_lat is None:
        return relevant_ids
    rest_locs = restaurants.set_index("gmap_id")[["latitude","longitude"]]
    filtered = []
    for gid in relevant_ids:
        if gid not in rest_locs.index:
            continue
        row = rest_locs.loc[gid]
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            continue
        dist = haversine(user_lat, user_lon, row["latitude"], row["longitude"])
        if dist <= max_miles:
            filtered.append(gid)
    return filtered


def load_eval_data():
    test = pd.read_parquet(OUTPUT_DIR / "test.parquet")
    train = pd.read_parquet(OUTPUT_DIR / "train.parquet")
    restaurants = pd.read_parquet(OUTPUT_DIR / "restaurants_geo.parquet")
    name_map = restaurants.set_index("gmap_id")["name"].to_dict()
    train_user_counts = train.groupby("user_id").size()
    return test, train, train_user_counts, restaurants, name_map


def get_valid_sample(test, train_user_counts, n_users, min_history):
    test_users = test["user_id"].unique()
    valid = [u for u in test_users if train_user_counts.get(u, 0) >= min_history]
    print(f"Valid users (>={min_history} train reviews): {len(valid)}")
    return np.random.default_rng(42).choice(valid, min(n_users, len(valid)), replace=False)


def compute_metrics(recommended_ids, relevant_ids, relevant_chains, name_map, k):
    recommended_chains = [get_chain_name(name_map.get(g,"")) for g in recommended_ids]
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
    print(f"Precision@{k}  (strict): {np.mean([m['precision']    for m in metrics_list]):.4f}")
    print(f"Recall@{k}     (strict): {np.mean([m['recall']       for m in metrics_list]):.4f}")
    print(f"NDCG@{k}       (strict): {np.mean([m['ndcg']         for m in metrics_list]):.4f}")
    print(f"Precision@{k}  (chain):  {np.mean([m['chain_prec']   for m in metrics_list]):.4f}")
    print(f"Hit Rate@{k}   (strict): {np.mean([m['hit_strict']   for m in metrics_list]):.4f}")
    print(f"Hit Rate@{k}   (chain):  {np.mean([m['hit_chain']    for m in metrics_list]):.4f}")


# ─── ALS EVALUATION ───────────────────────────────────────────────────────────

def evaluate_als(n_users=N_USERS, k=K, min_history=MIN_HISTORY, max_miles=MAX_DISTANCE_MILES):
    test, train, train_user_counts, restaurants, name_map = load_eval_data()
    sample_users = get_valid_sample(test, train_user_counts, n_users, min_history)
    metrics_list = []
    skipped_no_relevant = 0
    skipped_no_recs = 0

    for user_id in sample_users:
        relevant_ids = test[(test["user_id"]==user_id) & (test["rating"]>=4)]["gmap_id"].tolist()
        if not relevant_ids:
            skipped_no_relevant += 1
            continue

        user_lat, user_lon = get_user_centroid(user_id, train, restaurants)
        relevant_ids = filter_relevant_by_location(relevant_ids, user_lat, user_lon, restaurants, max_miles)
        if not relevant_ids:
            skipped_no_relevant += 1
            continue

        relevant_chains = set(get_chain_name(name_map.get(g,"")) for g in relevant_ids)

        try:
            recs = recommend(user_id, user_lat=user_lat, user_lon=user_lon, max_miles=max_miles)
            if recs.empty:
                skipped_no_recs += 1
                continue
            recommended_ids = recs["gmap_id"].tolist()
        except Exception:
            skipped_no_recs += 1
            continue

        metrics_list.append(compute_metrics(recommended_ids, relevant_ids, relevant_chains, name_map, k))

    print(f"Skipped (no relevant): {skipped_no_relevant} | Skipped (no recs): {skipped_no_recs}")
    print_metrics(f"ALS Baseline (min_history={min_history}, max_miles={max_miles})", metrics_list, k)
    return metrics_list


# ─── TWO TOWER EVALUATION ─────────────────────────────────────────────────────

def load_two_tower():
    with open(MODELS_DIR / "user_emb_map.pkl", "rb") as f:
        user_emb_map = pickle.load(f)
    with open(MODELS_DIR / "user_struct_map.pkl", "rb") as f:
        user_struct_map = pickle.load(f)
    item_struct = pd.read_parquet(MODELS_DIR / "item_struct_features.parquet")
    user_struct_df = pd.read_parquet(MODELS_DIR / "user_struct_features.parquet")
    item_struct_dim = len(item_struct.columns) - 1
    user_struct_dim = len(user_struct_df.columns) - 1
    model = TwoTowerModel(user_struct_dim, item_struct_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / "two_tower.pt", map_location=DEVICE))
    model.eval()
    item_vecs = np.load(EMBEDDINGS_DIR / "item_index_vecs.npy")
    item_ids = np.load(EMBEDDINGS_DIR / "item_index_ids.npy", allow_pickle=True)
    return model, user_emb_map, user_struct_map, item_vecs, item_ids


def tt_recommend_fast(user_id, model, user_emb_map, user_struct_map,
                      item_vecs, item_ids, restaurants,
                      user_lat=None, user_lon=None, max_miles=None, n=20):
    user_emb = user_emb_map.get(user_id, np.zeros(EMBEDDING_DIM, dtype=np.float32))
    user_struct = user_struct_map.get(user_id)
    if user_struct is None:
        return []

    u_emb_t = torch.tensor(user_emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    u_struct_t = torch.tensor(user_struct, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        user_vec = model.encode_user(u_emb_t, u_struct_t).cpu().numpy()

    scores = (item_vecs @ user_vec.T).squeeze()
    top_idx = np.argsort(scores)[::-1]

    # geo filter
    if user_lat and user_lon and max_miles:
        rest_locs = restaurants.set_index("gmap_id")[["latitude","longitude"]]
        filtered = []
        for i in top_idx:
            gid = item_ids[i]
            if gid in rest_locs.index:
                row = rest_locs.loc[gid]
                if not pd.isna(row["latitude"]) and not pd.isna(row["longitude"]):
                    dist = haversine(user_lat, user_lon, row["latitude"], row["longitude"])
                    if dist <= max_miles:
                        filtered.append((gid, float(scores[i])))
                        if len(filtered) >= n:
                            break
        return filtered
    else:
        return [(item_ids[i], float(scores[i])) for i in top_idx[:n]]


def evaluate_two_tower(n_users=N_USERS, k=K, min_history=MIN_HISTORY, max_miles=MAX_DISTANCE_MILES):
    test, train, train_user_counts, restaurants, name_map = load_eval_data()
    sample_users = get_valid_sample(test, train_user_counts, n_users, min_history)

    print("Loading two tower model...")
    model, user_emb_map, user_struct_map, item_vecs, item_ids = load_two_tower()
    metrics_list = []
    skipped_no_relevant = 0
    skipped_no_recs = 0

    for user_id in sample_users:
        relevant_ids = test[(test["user_id"]==user_id) & (test["rating"]>=4)]["gmap_id"].tolist()
        if not relevant_ids:
            skipped_no_relevant += 1
            continue
        if user_id not in user_emb_map:
            continue

        user_lat, user_lon = get_user_centroid(user_id, train, restaurants)
        relevant_ids = filter_relevant_by_location(relevant_ids, user_lat, user_lon, restaurants, max_miles)
        if not relevant_ids:
            skipped_no_relevant += 1
            continue

        relevant_chains = set(get_chain_name(name_map.get(g,"")) for g in relevant_ids)

        try:
            recs_raw = tt_recommend_fast(user_id, model, user_emb_map, user_struct_map,
                                         item_vecs, item_ids, restaurants,
                                         user_lat=user_lat, user_lon=user_lon,
                                         max_miles=max_miles, n=k)
            if not recs_raw:
                skipped_no_recs += 1
                continue
            recommended_ids = [g for g,_ in recs_raw]
        except Exception:
            skipped_no_recs += 1
            continue

        metrics_list.append(compute_metrics(recommended_ids, relevant_ids, relevant_chains, name_map, k))

    print(f"Skipped (no relevant): {skipped_no_relevant} | Skipped (no recs): {skipped_no_recs}")
    print_metrics(f"Two Tower (min_history={min_history}, max_miles={max_miles})", metrics_list, k)
    return metrics_list


# ─── COMPARE ──────────────────────────────────────────────────────────────────

def load_lightgcn():
    with open(MODELS_DIR / "lgcn_user_map.pkl", "rb") as f:
        _, user_to_idx = pickle.load(f)
    with open(MODELS_DIR / "lgcn_item_map.pkl", "rb") as f:
        item_map, _ = pickle.load(f)

    user_vecs = np.load(MODELS_DIR / "lgcn_user_vecs.npy")
    item_vecs = np.load(MODELS_DIR / "lgcn_item_vecs.npy")
    return user_to_idx, item_map, user_vecs, item_vecs

def load_lightgcn2(models_dir):
    models_dir = Path(models_dir)
    with open(models_dir / "lgcn_user_map.pkl", "rb") as f:
        _, user_to_idx = pickle.load(f)
    with open(models_dir / "lgcn_item_map.pkl", "rb") as f:
        item_map, _ = pickle.load(f)
    user_vecs = np.load(models_dir / "lgcn_user_vecs.npy")
    item_vecs = np.load(models_dir / "lgcn_item_vecs.npy")
    return user_to_idx, item_map, user_vecs, item_vecs


def lgcn_recommend_fast(user_id, user_to_idx, item_map, user_vecs, item_vecs,
                        restaurants, user_lat=None, user_lon=None, max_miles=None, n=20):
    if user_id not in user_to_idx:
        return []
    u_idx = user_to_idx[user_id]
    u_vec = user_vecs[u_idx]
    scores = item_vecs @ u_vec
    top_idx = np.argsort(scores)[::-1]

    if user_lat and user_lon and max_miles:
        rest_locs = restaurants.set_index("gmap_id")[["latitude","longitude"]]
        filtered = []
        for i in top_idx:
            gid = item_map[i]
            if gid in rest_locs.index:
                row = rest_locs.loc[gid]
                if not pd.isna(row["latitude"]) and not pd.isna(row["longitude"]):
                    dist = haversine(user_lat, user_lon, row["latitude"], row["longitude"])
                    if dist <= max_miles:
                        filtered.append((gid, float(scores[i])))
                        if len(filtered) >= n:
                            break
        return filtered
    else:
        return [(item_map[i], float(scores[i])) for i in top_idx[:n]]


def evaluate_lightgcn(model_dir , n_users=N_USERS, k=K, min_history=MIN_HISTORY, max_miles=MAX_DISTANCE_MILES ):
    test, train, train_user_counts, restaurants, name_map = load_eval_data()
    sample_users = get_valid_sample(test, train_user_counts, n_users, min_history)

    print("Loading LightGCN...")
    user_to_idx, item_map, user_vecs, item_vecs = load_lightgcn2(model_dir)
    metrics_list = []
    skipped_no_relevant = 0
    skipped_no_recs = 0

    for user_id in sample_users:
        relevant_ids = test[(test["user_id"]==user_id) & (test["rating"]>=4)]["gmap_id"].tolist()
        if not relevant_ids:
            skipped_no_relevant += 1
            continue
        if user_id not in user_to_idx:
            continue

        user_lat, user_lon = get_user_centroid(user_id, train, restaurants)
        relevant_ids = filter_relevant_by_location(relevant_ids, user_lat, user_lon, restaurants, max_miles)
        if not relevant_ids:
            skipped_no_relevant += 1
            continue

        relevant_chains = set(get_chain_name(name_map.get(g,"")) for g in relevant_ids)

        try:
            recs_raw = lgcn_recommend_fast(user_id, user_to_idx, item_map, user_vecs, item_vecs,
                                           restaurants, user_lat=user_lat, user_lon=user_lon,
                                           max_miles=max_miles, n=k)
            if not recs_raw:
                skipped_no_recs += 1
                continue
            recommended_ids = [g for g,_ in recs_raw]
        except Exception:
            skipped_no_recs += 1
            continue

        metrics_list.append(compute_metrics(recommended_ids, relevant_ids, relevant_chains, name_map, k))

    print(f"Skipped (no relevant): {skipped_no_relevant} | Skipped (no recs): {skipped_no_recs}")
    print_metrics(f"LightGCN (min_history={min_history}, max_miles={max_miles})", metrics_list, k)
    return metrics_list

def compare(n_users=N_USERS, k=K, min_history=MIN_HISTORY, max_miles=MAX_DISTANCE_MILES):
    als_metrics  = evaluate_als(n_users, k, min_history, max_miles)
    tt_metrics   = evaluate_two_tower(n_users, k, min_history, max_miles)
    lgcn_metrics1 = evaluate_lightgcn(n_users, k, min_history, max_miles , "model")


    print("\n=== Comparison Summary ===")
    for metric, label in [("hit_chain",  "Hit Rate@K  (chain) "),
                           ("hit_strict", "Hit Rate@K  (strict)"),
                           ("chain_prec", "Precision@K (chain) "),
                           ("ndcg",       "NDCG@K      (strict)")]:
        als_val  = np.mean([m[metric] for m in als_metrics])  if als_metrics  else 0
        tt_val   = np.mean([m[metric] for m in tt_metrics])   if tt_metrics   else 0
        lgcn_val = np.mean([m[metric] for m in lgcn_metrics]) if lgcn_metrics else 0
        print(f"{label}  ALS: {als_val:.4f}  TwoTower: {tt_val:.4f}  LightGCN: {lgcn_val:.4f}")

def compare2(n_users=N_USERS, k=K, min_history=MIN_HISTORY, max_miles=MAX_DISTANCE_MILES):
    als_metrics   = evaluate_als(n_users, k, min_history, max_miles)
    tt_metrics    = evaluate_two_tower(n_users, k, min_history, max_miles)
    lgcn64_metrics  = evaluate_lightgcn("models" , n_users, k, min_history, max_miles)
    lgcn128_metrics = evaluate_lightgcn("models/lgcn_enriched" , n_users, k, min_history, max_miles)

    print("\n=== Comparison Summary ===")
    all_models = [
        ("ALS",         als_metrics),
        ("TwoTower",    tt_metrics),
        ("LightGCN-64", lgcn64_metrics),
        ("LightGCN-128",lgcn128_metrics),
    ]
    for metric, label in [("hit_strict", "Hit Rate@K  (strict)"),
                           ("hit_chain",  "Hit Rate@K  (chain) "),
                           ("ndcg",       "NDCG@K      (strict)"),
                           ("chain_prec", "Precision@K (chain) ")]:
        vals = "  ".join(
            f"{name}: {np.mean([m[metric] for m in metrics]):.4f}" if metrics else f"{name}: N/A"
            for name, metrics in all_models
        )
        print(f"{label}  {vals}")

if __name__ == "__main__":
    compare2()
