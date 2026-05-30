import pandas as pd
import numpy as np
import scipy.sparse as sparse
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import implicit
import pickle

OUTPUT_DIR = Path("output")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

N_FACTORS = 50
N_ITERATIONS = 20
N_RECOMMENDATIONS = 20
CF_ALPHA_SCALE = 20

def build_feature_store():

    restaurants = pd.read_parquet(OUTPUT_DIR / "restaurants.parquet")
    reviews = pd.read_parquet(OUTPUT_DIR / "reviews.parquet")
    user_profiles = pd.read_parquet(OUTPUT_DIR / "user_profiles.parquet")
    topic_dist = (reviews.groupby("gmap_id")["topic_id"].value_counts(normalize=True).unstack(fill_value=0).add_prefix("topic_dist_"))
    restaurants = restaurants.merge(topic_dist, on="gmap_id", how="left")

    misc_cols = ["has_delivery","has_dine_in","has_takeout","has_drive_through",
                 "serves_alcohol","has_outdoor_seating","is_wheelchair",
                 "accepts_reservations","has_live_music","kid_friendly"]

    cat_cols = [c for c in restaurants.columns if c.startswith("cat_")]
    topic_cols = [c for c in restaurants.columns if c.startswith("topic_dist_")]
    num_cols = ["price_numeric","avg_sentiment","weighted_avg_rating",
                "review_count","has_response_rate"]

    feature_cols = misc_cols + cat_cols + topic_cols + num_cols
    feature_cols = [c for c in feature_cols if c in restaurants.columns]

    item_features = restaurants[["gmap_id"] + feature_cols].copy()
    item_features[feature_cols] = item_features[feature_cols].fillna(0)

    item_features.to_parquet(MODELS_DIR / "item_features.parquet", index=False)
    user_profiles.to_parquet(MODELS_DIR / "user_features.parquet", index=False)

    return restaurants, reviews, user_profiles, item_features

def build_interaction_matrix(reviews):
    reviews = reviews.copy()
    reviews["user_idx"] = reviews["user_id"].astype("category").cat.codes
    reviews["item_idx"] = reviews["gmap_id"].astype("category").cat.codes

    user_map = dict(enumerate(reviews["user_id"].astype("category").cat.categories))
    item_map = dict(enumerate(reviews["gmap_id"].astype("category").cat.categories))
    user_to_idx = {v:k for k,v in user_map.items()}
    item_to_idx = {v:k for k,v in item_map.items()}

    matrix = sparse.csr_matrix( (reviews["recency_weight"] * reviews["rating"], (reviews["user_idx"], reviews["item_idx"])))

    with open(MODELS_DIR / "user_map.pkl", "wb") as f:
        pickle.dump((user_map, user_to_idx), f)
    with open(MODELS_DIR / "item_map.pkl", "wb") as f:
        pickle.dump((item_map, item_to_idx), f)

    return matrix, user_map, item_map, user_to_idx, item_to_idx

def train_cf_model(matrix):

    model = implicit.als.AlternatingLeastSquares(
        factors=N_FACTORS,
        iterations=N_ITERATIONS,
        calculate_training_loss=True,
        random_state=42
    )
    model.fit(matrix)
    with open(MODELS_DIR / "cf_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model


def get_cf_recommendations(user_id, model, matrix, user_to_idx, item_map, n=N_RECOMMENDATIONS):
    if user_id not in user_to_idx:
        return []
    user_idx = user_to_idx[user_id]
    ids, scores = model.recommend(user_idx, matrix[user_idx], N=n, filter_already_liked_items=True)
    return [(item_map[i], float(s)) for i, s in zip(ids, scores)]


def build_item_matrix(item_features):
    gmap_ids = item_features["gmap_id"].values
    feature_cols = [c for c in item_features.columns if c != "gmap_id"]
    matrix = item_features[feature_cols].values.astype(float)
    matrix = normalize(matrix, norm="l2")
    np.save(MODELS_DIR / "item_matrix.npy", matrix)
    np.save(MODELS_DIR / "item_gmap_ids.npy", gmap_ids)
    return matrix, gmap_ids

def build_user_content_vector(user_id, reviews, item_features):
    user_reviews = reviews[reviews["user_id"]==user_id].copy()
    if len(user_reviews) == 0:
        return None
    user_reviews = user_reviews.merge(item_features, on="gmap_id", how="left")
    feature_cols = [c for c in item_features.columns if c != "gmap_id"]
    weights = user_reviews["recency_weight"] * user_reviews["rating"]
    weighted = user_reviews[feature_cols].multiply(weights, axis=0)
    user_vector = weighted.sum() / weights.sum()
    return user_vector.values.reshape(1,-1)

def get_cb_recommendations(user_id, reviews, item_features, item_matrix, gmap_ids, n=N_RECOMMENDATIONS):
    user_vector = build_user_content_vector(user_id, reviews, item_features)
    if user_vector is None:
        return []
    user_vector = normalize(user_vector, norm="l2")
    scores = cosine_similarity(user_vector, item_matrix)[0]
    top_idx = np.argsort(scores)[::-1][:n]
    seen = set(reviews[reviews["user_id"]==user_id]["gmap_id"])
    results = [(gmap_ids[i], float(scores[i])) for i in top_idx if gmap_ids[i] not in seen]
    return results[:n]

def get_cb_recommendations_cold(preference_vector, item_matrix, gmap_ids, n=N_RECOMMENDATIONS):
    pv = normalize(np.array(preference_vector).reshape(1,-1), norm="l2")
    scores = cosine_similarity(pv, item_matrix)[0]
    top_idx = np.argsort(scores)[::-1][:n]
    return [(gmap_ids[i], float(scores[i])) for i in top_idx]

def get_cf_alpha(user_id, reviews):
    n = len(reviews[reviews["user_id"]==user_id])
    return min(1.0, n / CF_ALPHA_SCALE)

def normalize_scores(recommendations):
    if not recommendations:
        return {}
    scores = np.array([s for _,s in recommendations])
    min_s, max_s = scores.min(), scores.max()
    if max_s == min_s:
        return {gmap_id: 1.0 for gmap_id, _ in recommendations}
    return {gmap_id: (s - min_s) / (max_s - min_s) for gmap_id, s in recommendations}


def hybrid_recommend(user_id, model, matrix, user_to_idx, item_map,
                     reviews, item_features, item_matrix, gmap_ids, n=N_RECOMMENDATIONS):
    alpha = get_cf_alpha(user_id, reviews)

    cf_recs = get_cf_recommendations(user_id, model, matrix, user_to_idx, item_map, n=n*2)
    cb_recs = get_cb_recommendations(user_id, reviews, item_features, item_matrix, gmap_ids, n=n*2)

    cf_scores = normalize_scores(cf_recs)
    cb_scores = normalize_scores(cb_recs)

    all_ids = set(cf_scores.keys()) | set(cb_scores.keys())
    combined = {}
    for gmap_id in all_ids:
        cf_s = cf_scores.get(gmap_id, 0.0)
        cb_s = cb_scores.get(gmap_id, 0.0)
        combined[gmap_id] = alpha * cf_s + (1 - alpha) * cb_s

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return ranked[:n]


def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def rerank(recommendations, restaurants, user_lat=None, user_lon=None, max_miles=None):
    gmap_ids = [g for g,_ in recommendations]
    scores = {g:s for g,s in recommendations}

    df = restaurants[restaurants["gmap_id"].isin(gmap_ids)].copy()

    if user_lat and user_lon:
        df["distance_miles"] = df.apply(
            lambda r: haversine(user_lat, user_lon, r["latitude"], r["longitude"]), axis=1
        )
        if max_miles:
            df = df[df["distance_miles"] <= max_miles]
        max_dist = df["distance_miles"].max() or 1
        df["distance_score"] = 1 - (df["distance_miles"] / max_dist)
    else:
        df["distance_score"] = 1.0

    max_days = df["latest_review_days"].max() or 1
    df["recency_score"] = 1 - (df["latest_review_days"].fillna(max_days) / max_days)
    df["response_score"] = df["has_response_rate"].fillna(0)

    df["hybrid_score"] = df["gmap_id"].map(scores)
    df["final_score"] = (0.6 * df["hybrid_score"] + 0.2 * df["distance_score"] + 0.1 * df["recency_score"] + 0.1 * df["response_score"] )

    df = df.sort_values("final_score", ascending=False) 

    df = dedupe_chains(df)

    return df[["gmap_id","name","final_score","distance_miles"] if "distance_miles" in df.columns
              else ["gmap_id","name","final_score"]].reset_index(drop=True)



# def recommend(user_id, user_lat=None, user_lon=None, max_miles=None):
#     with open(MODELS_DIR / "cf_model.pkl", "rb") as f:
#         model = pickle.load(f)
#     with open(MODELS_DIR / "user_map.pkl", "rb") as f:
#         _, user_to_idx = pickle.load(f)
#     with open(MODELS_DIR / "item_map.pkl", "rb") as f:
#         item_map, _ = pickle.load(f)

#     reviews = pd.read_parquet(OUTPUT_DIR / "reviews.parquet")
#     restaurants = pd.read_parquet(OUTPUT_DIR / "restaurants_geo.parquet")
#     item_features = pd.read_parquet(MODELS_DIR / "item_features.parquet")
#     item_matrix = np.load(MODELS_DIR / "item_matrix.npy")
#     gmap_ids = np.load(MODELS_DIR / "item_gmap_ids.npy", allow_pickle=True)

#     matrix, _, _, user_to_idx, item_map_inv = build_interaction_matrix(reviews)

#     recs = hybrid_recommend(user_id, model, matrix, user_to_idx, item_map_inv,
#                             reviews, item_features, item_matrix, gmap_ids)
#     return rerank(recs, restaurants, user_lat, user_lon, max_miles)


def dedupe_chains(df, max_per_chain=1):
    df["chain_name"] = df["name"].str.lower().str.strip()
    df = df.groupby("chain_name").head(max_per_chain).reset_index(drop=True)
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df = df.drop(columns=["chain_name"])
    return df
    
def recommend(user_id, user_lat=None, user_lon=None, max_miles=None):
    try:
        user_id = float(user_id)
    except (ValueError, TypeError):
        pass

    with open(MODELS_DIR / "cf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "user_map.pkl", "rb") as f:
        _, user_to_idx = pickle.load(f)
    with open(MODELS_DIR / "item_map.pkl", "rb") as f:
        item_map, _ = pickle.load(f)

    matrix = sparse.load_npz(MODELS_DIR / "interaction_matrix.npz")
    reviews = pd.read_parquet(OUTPUT_DIR / "reviews.parquet")
    restaurants = pd.read_parquet(OUTPUT_DIR / "restaurants_geo.parquet")
    item_features = pd.read_parquet(MODELS_DIR / "item_features.parquet")
    item_matrix = np.load(MODELS_DIR / "item_matrix.npy")
    gmap_ids = np.load(MODELS_DIR / "item_gmap_ids.npy", allow_pickle=True)

    recs = hybrid_recommend(user_id, model, matrix, user_to_idx, item_map,
                            reviews, item_features, item_matrix, gmap_ids)
    return rerank(recs, restaurants, user_lat, user_lon, max_miles)
    
def run():
    print("Building feature store...")
    restaurants, reviews, user_profiles, item_features = build_feature_store()

    print("Loading train split...")
    train_reviews = pd.read_parquet(OUTPUT_DIR / "train.parquet")

    print("Building interaction matrix...")
    matrix, user_map, item_map, user_to_idx, item_to_idx = build_interaction_matrix(train_reviews)

    sparse.save_npz(MODELS_DIR / "interaction_matrix.npz", matrix)

    print("Training CF model...")
    model = train_cf_model(matrix)

    print("Building item content matrix...")
    item_matrix, gmap_ids = build_item_matrix(item_features)

    print("Done.")

if __name__ == "__main__":
    run()