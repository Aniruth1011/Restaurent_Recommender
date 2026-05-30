import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from recommend import build_interaction_matrix, hybrid_recommend, rerank, build_item_matrix, get_cb_recommendations_cold, normalize_scores

OUTPUT_DIR = Path("output")
MODELS_DIR = Path("models")

CONSTRAINT_TO_COL = {
    "vegan": None,
    "vegetarian": None,
    "gluten_free": None,
    "halal": None,
    "alcohol": "serves_alcohol",
    "outdoor": "has_outdoor_seating",
    "wheelchair": "is_wheelchair",
    "live_music": "has_live_music",
    "kid_friendly":"kid_friendly",
    "reservations":"accepts_reservations",
}

DIETARY_TOPIC_KEYWORDS = {
    "vegan": "vegan",
    "vegetarian":  "vegetarian",
    "gluten_free": "gluten",}


def load_topic_info():
    topic_info = pd.read_parquet(MODELS_DIR / "topic_info.parquet") if (
        MODELS_DIR / "topic_info.parquet").exists() else None
    return topic_info

def find_dietary_topic_ids(topic_info, dietary_constraints):
    if topic_info is None:
        return {}
    result = {}
    for constraint in dietary_constraints:
        keyword = DIETARY_TOPIC_KEYWORDS.get(constraint)
        if not keyword:
            continue
        mask = topic_info["Representation"].astype(str).str.lower().str.contains(keyword)
        matched = topic_info[mask]["Topic"].tolist()
        if matched:
            result[constraint] = matched
    return result

def apply_hard_constraints(restaurants, constraints, topic_info=None):
    df = restaurants.copy()
    dietary = [c for c in constraints if c in DIETARY_TOPIC_KEYWORDS]
    misc_constraints = [c for c in constraints if c not in DIETARY_TOPIC_KEYWORDS]

    for constraint in misc_constraints:
        col = CONSTRAINT_TO_COL.get(constraint)
        if col and col in df.columns:
            df = df[df[col] == 1]

    if dietary and topic_info is not None:
        dietary_topic_map = find_dietary_topic_ids(topic_info, dietary)
        topic_dist_cols = [c for c in df.columns if c.startswith("topic_dist_")]
        for constraint, topic_ids in dietary_topic_map.items():
            matching_cols = [f"topic_dist_{t}" for t in topic_ids if f"topic_dist_{t}" in df.columns]
            if matching_cols:
                df = df[df[matching_cols].sum(axis=1) > 0.05]

    return df

def build_member_profile(member):
    return { "user_id": member.get("user_id"), "constraints": member.get("constraints", []),
        "price_max": member.get("price_max", 4), "lat": member.get("lat"), "lon": member.get("lon"),}

def score_for_member(member_profile, restaurants_filtered, reviews,
                     model, matrix, user_to_idx, item_map,
                     item_features, item_matrix, gmap_ids, n=50):
    user_id = member_profile["user_id"]

    if user_id:
        recs = hybrid_recommend(user_id, model, matrix, user_to_idx, item_map,reviews, item_features, item_matrix, gmap_ids, n=n)
    else:
        pref_vector = member_profile.get("preference_vector")
        if pref_vector is None:
            return {}
        recs = get_cb_recommendations_cold(pref_vector, item_matrix, gmap_ids, n=n)

    valid_ids = set(restaurants_filtered["gmap_id"])
    recs = [(g, s) for g, s in recs if g in valid_ids]
    return normalize_scores(recs)

def aggregate_least_misery(member_scores):
    all_ids = set()
    for scores in member_scores:
        all_ids.update(scores.keys())
    result = {}
    for gmap_id in all_ids:
        individual = [s.get(gmap_id, 0.0) for s in member_scores]
        result[gmap_id] = min(individual)
    return result

def aggregate_average(member_scores):
    all_ids = set()
    for scores in member_scores:
        all_ids.update(scores.keys())
    result = {}
    for gmap_id in all_ids:
        individual = [s.get(gmap_id, 0.0) for s in member_scores]
        result[gmap_id] = np.mean(individual)
    return result

def aggregate_most_pleasure(member_scores):
    all_ids = set()
    for scores in member_scores:
        all_ids.update(scores.keys())
    result = {}
    for gmap_id in all_ids:
        individual = [s.get(gmap_id, 0.0) for s in member_scores]
        result[gmap_id] = max(individual)
    return result

AGGREGATION_STRATEGIES = {
    "least_misery":  aggregate_least_misery,
    "average": aggregate_average,
    "most_pleasure": aggregate_most_pleasure,
}

def build_explanation(gmap_id, restaurants, members, member_scores, topic_info=None):
    row = restaurants[restaurants["gmap_id"]==gmap_id]
    if row.empty:
        return {}

    row = row.iloc[0]
    explanations = []

    for i, member in enumerate(members):
        name = member.get("name", f"Member {i+1}")
        score = member_scores[i].get(gmap_id, 0.0)
        constraints = member.get("constraints", [])
        constraint_str = ", ".join(constraints) if constraints else "general preference"
        explanations.append({"member": name, "score": round(score, 3), "matched_constraints": constraint_str, })

    misc_highlights = []
    for col, label in [("serves_alcohol","Bar/drinks available"), ("has_outdoor_seating","Outdoor seating"),
                       ("has_live_music","Live music"), ("kid_friendly","Kid friendly"), ("accepts_reservations","Takes reservations")]:
        if row.get(col, 0) == 1:
            misc_highlights.append(label)

    return {
        "gmap_id": gmap_id, "name": row.get("name",""),
        "avg_rating": round(row.get("weighted_avg_rating", 0), 2), "price": row.get("price",""),
        "highlights": misc_highlights, "member_breakdown": explanations,
    }

def group_recommend(members, strategy="least_misery", user_lat=None, user_lon=None, max_miles=None, n=10):
    with open(MODELS_DIR / "cf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "user_map.pkl", "rb") as f:
        _, user_to_idx = pickle.load(f)
    with open(MODELS_DIR / "item_map.pkl", "rb") as f:
        item_map, _ = pickle.load(f)

    reviews = pd.read_parquet(OUTPUT_DIR / "reviews.parquet")
    restaurants = pd.read_parquet(OUTPUT_DIR / "restaurants_geo.parquet")
    item_features = pd.read_parquet(MODELS_DIR / "item_features.parquet")
    item_matrix = np.load(MODELS_DIR / "item_matrix.npy")
    gmap_ids = np.load(MODELS_DIR / "item_gmap_ids.npy", allow_pickle=True)
    topic_info = load_topic_info()

    matrix, _, _, user_to_idx, item_map_inv = build_interaction_matrix(reviews)

    all_constraints = list(set(c for m in members for c in m.get("constraints", [])))
    price_max = min(m.get("price_max", 4) for m in members)

    restaurants_filtered = apply_hard_constraints(restaurants, all_constraints, topic_info)
    restaurants_filtered = restaurants_filtered[ restaurants_filtered["price_numeric"].fillna(4) <= price_max ]

    if user_lat and user_lon and max_miles:
        from phase2 import haversine
        restaurants_filtered["distance_miles"] = restaurants_filtered.apply( lambda r: haversine(user_lat, user_lon, r["latitude"], r["longitude"]), axis=1)
        restaurants_filtered = restaurants_filtered[ restaurants_filtered["distance_miles"] <= max_miles]

    print(f"Candidates after filtering: {len(restaurants_filtered)}")

    member_profiles = [build_member_profile(m) for m in members]
    member_scores = [ score_for_member(p, restaurants_filtered, reviews, model, matrix, user_to_idx, item_map_inv, item_features, item_matrix, gmap_ids) for p in member_profiles ]

    agg_fn = AGGREGATION_STRATEGIES.get(strategy, aggregate_least_misery)
    group_scores = agg_fn(member_scores)

    ranked = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)[:n*2]
    recs_df = rerank(ranked, restaurants_filtered, user_lat, user_lon, max_miles)
    recs_df = recs_df.head(n)

    results = []
    for _, row in recs_df.iterrows():
        explanation = build_explanation( row["gmap_id"], restaurants, members, member_scores, topic_info )
        explanation["final_score"] = round(row["final_score"], 3)
        results.append(explanation)

    return results

if __name__ == "__main__":
    group = [
        {
            "name": "Maya",
            "user_id": "some_user_id_1",
            "constraints": ["vegan"],
            "price_max": 3,
        },
        {
            "name": "James",
            "user_id": "some_user_id_2",
            "constraints": ["alcohol"],
            "price_max": 4,
        },
        {
            "name": "Ravi",
            "user_id": "some_user_id_3",
            "constraints": [],
            "price_max": 2,
        },
    ]

    results = group_recommend(
        members=group,
        strategy="least_misery",
        user_lat=37.7749,
        user_lon=-122.4194,
        max_miles=5,
        n=10
    )

    for r in results:
        print(r)