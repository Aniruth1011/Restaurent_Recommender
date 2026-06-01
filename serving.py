"""
serving.py
Serving layer for the demo app, built on top of recommend_engine.py (the I2I
semantic-embedding recommender in this folder).

A recommendation always needs a QUERY VECTOR; only its source changes:
  1. dataset history  -> engine score_user() (users present in reviews.parquet)
  2. logged likes     -> weighted avg of liked items' embeddings (new users warm up)
  3. cold-start prefs -> avg embeddings of top-rated restaurants in stated cuisines
  4. nothing          -> pure popularity (engine fallback)

The vector feeds the engine's existing cosine -> popularity-hybrid -> rank() path
(chain dedup, geo filter, rating/pop blend). Everything reads only from this
folder's artifacts (output_review/, embeddings_i2i/).
"""

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from recommend_engine import (
    load_all as _engine_load_all,
    RecConfig, rank, score_user,
    _rest_locs, _make_geo_filter, popularity_scores, build_user_vector,
    haversine, N_CANDIDATES,
)

# ── cuisine label -> cat_ columns ────────────────────────────────────────────
CUISINE_TO_CATS = {
    "Mexican":       ["cat_mexican_restaurant", "cat_taco_restaurant", "cat_tex-mex_restaurant", "cat_burrito_restaurant"],
    "Italian":       ["cat_italian_restaurant", "cat_pizza_restaurant", "cat_pasta_shop"],
    "Chinese":       ["cat_chinese_restaurant", "cat_dim_sum_restaurant", "cat_cantonese_restaurant"],
    "Japanese":      ["cat_japanese_restaurant", "cat_sushi_restaurant", "cat_ramen_restaurant"],
    "Indian":        ["cat_indian_restaurant", "cat_south_asian_restaurant", "cat_pakistani_restaurant"],
    "American":      ["cat_american_restaurant", "cat_new_american_restaurant", "cat_traditional_american_restaurant", "cat_diner"],
    "Thai":          ["cat_thai_restaurant"],
    "Mediterranean": ["cat_mediterranean_restaurant", "cat_greek_restaurant", "cat_middle_eastern_restaurant"],
    "Korean":        ["cat_korean_restaurant", "cat_korean_barbecue_restaurant"],
    "Vietnamese":    ["cat_vietnamese_restaurant", "cat_pho_restaurant"],
    "Sushi":         ["cat_sushi_restaurant", "cat_japanese_restaurant"],
    "Pizza":         ["cat_pizza_restaurant", "cat_pizza_delivery", "cat_pizza_takeout"],
    "Burger":        ["cat_hamburger_restaurant", "cat_fast_food_restaurant"],
    "Seafood":       ["cat_seafood_restaurant"],
    "Vegan":         ["cat_vegan_restaurant", "cat_vegetarian_restaurant", "cat_health_food_restaurant"],
    "BBQ":           ["cat_barbecue_restaurant"],
    "Cafe":          ["cat_cafe", "cat_coffee_shop", "cat_espresso_bar", "cat_bakery"],
    "Bar":           ["cat_bar", "cat_cocktail_bar", "cat_wine_bar", "cat_sports_bar", "cat_gastropub", "cat_pub"],
}
CUISINE_OPTIONS = ["Any"] + list(CUISINE_TO_CATS.keys())

AMENITY_COLS = {
    "serves_alcohol": "serves_alcohol",
    "has_outdoor_seating": "has_outdoor_seating",
    "is_wheelchair": "is_wheelchair",
    "kid_friendly": "kid_friendly",
    "has_live_music": "has_live_music",
    "accepts_reservations": "accepts_reservations",
}
DIETARY_COLS = {
    "vegan": "cat_vegan_restaurant",
    "vegetarian": "cat_vegetarian_restaurant",
    "gluten_free": "cat_gluten-free_restaurant",
    "halal": "cat_halal_restaurant",
}

# Active serving model: qe|pure with weights (w_hybrid, w_rating, w_pop)=(0.9,0,0.1).
#   candidate_mode="qe" -> query-expansion I2I for users with multi-item history
#   use_hybrid=False    -> "pure" I2I (no popularity alpha-blend into `combined`)
_CFG = RecConfig(name="serving", candidate_mode="qe", use_hybrid=False,
                 use_mmr=False, w_hybrid=0.9, w_rating=0.0, w_pop=0.1)

# Fallback when there is no I2I query vector at all (true cold popularity case):
# a pure-pure config returns nothing because `combined` (=I2I) is empty, so rank
# by popularity instead.
_POP_CFG = RecConfig(name="popularity", pop_only=True,
                     w_hybrid=0.9, w_rating=0.0, w_pop=0.1)

_G = {}  # loaded artifacts


# ── load ──────────────────────────────────────────────────────────────────────

def load_all():
    if _G:
        return
    print("Loading I2I engine artifacts...")
    emb, item_ids, id_to_idx, restaurants, reviews = _engine_load_all()
    rest_locs = _rest_locs(restaurants)
    ri = restaurants.set_index("gmap_id")[["weighted_avg_rating", "review_count"]]
    ri = ri[~ri.index.duplicated(keep="first")]
    rd = restaurants.set_index("gmap_id")
    rd = rd[~rd.index.duplicated(keep="first")]
    # Some amenity columns (serves_alcohol, outdoor, live_music) are all-zero in
    # this dataset; filtering on them would empty the results. Only treat a flag
    # column as usable if it has at least one positive value.
    candidate_cols = set(AMENITY_COLS.values()) | set(DIETARY_COLS.values())
    usable_cols = {c for c in candidate_cols
                   if c in restaurants.columns and (restaurants[c] > 0).any()}
    _G.update(dict(
        emb=emb, item_ids=item_ids, id_to_idx=id_to_idx,
        restaurants=restaurants, reviews=reviews,
        rest_locs=rest_locs, ri=ri, rd=rd,
        name_map=restaurants.set_index("gmap_id")["name"].to_dict(),
        dataset_users=set(reviews["user_id"].unique()),
        usable_cols=usable_cols,
    ))
    skipped = candidate_cols - usable_cols
    if skipped:
        print(f"Note: no data for {sorted(skipped)} — those filters are no-ops.")
    print(f"Loaded: {len(restaurants)} restaurants, {emb.shape[0]} embeddings, "
          f"{len(_G['dataset_users'])} known users")


# ── sanitize ────────────────────────────────────────────────────────────────

def _san(v):
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 5)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


# ── query vectors ─────────────────────────────────────────────────────────────

def vector_from_interactions(liked, neg_weight=0.5):
    """liked: list of (gmap_id, rating) -> normalized profile embedding.

    Positive interactions (rating >= 4) pull the query vector toward those
    items; negative feedback (rating <= 2) pushes it away. Disliked items are
    also in the caller's `seen` set, so they never reappear in results.
    Returns None when there is no positive signal to anchor on.
    """
    emb, id_to_idx = _G["emb"], _G["id_to_idx"]
    pos_vecs, pos_ws, neg_vecs = [], [], []
    for gid, rating in liked:
        idx = id_to_idx.get(gid)
        if idx is None:
            continue
        r = float(rating) if rating is not None else 5.0
        if r <= 2.0:
            neg_vecs.append(emb[idx])
        else:
            pos_vecs.append(emb[idx])
            pos_ws.append(max(r, 0.1))
    if not pos_vecs:
        return None
    v = np.average(np.array(pos_vecs), axis=0, weights=np.array(pos_ws))
    if neg_vecs:
        v = v - neg_weight * np.mean(np.array(neg_vecs), axis=0)
    return normalize(v.reshape(1, -1), norm="l2")


def build_cold_start_vector(prefs):
    """Average embeddings of top-rated restaurants in the user's stated cuisines."""
    restaurants, emb, id_to_idx = _G["restaurants"], _G["emb"], _G["id_to_idx"]
    cuisines = (prefs or {}).get("cuisines") or []
    cats = []
    for c in cuisines:
        cats += CUISINE_TO_CATS.get(c, [])
    cats = [c for c in cats if c in restaurants.columns]
    if not cats:
        return None
    mask = (restaurants[cats].max(axis=1) > 0) & (restaurants["review_count"] >= 20)
    seed = restaurants[mask].sort_values("weighted_avg_rating", ascending=False).head(60)
    idxs = [id_to_idx[g] for g in seed["gmap_id"] if g in id_to_idx]
    if not idxs:
        return None
    return normalize(emb[idxs].mean(axis=0).reshape(1, -1), norm="l2")


def query_scores(qvec, seen, passes_geo, n_candidates=N_CANDIDATES, allowed=None):
    """Cosine of a query vector over all item embeddings -> top geo-valid candidates.

    If `allowed` is given, rank within that set (cuisine/hard-filter) BEFORE
    truncating, so an explicit cuisine request returns that cuisine ranked by the
    user's taste - not an empty list because the global top-N were other cuisines.
    """
    emb, item_ids, id_to_idx = _G["emb"], _G["item_ids"], _G["id_to_idx"]
    scores = cosine_similarity(qvec, emb)[0]
    for gid in seen:
        i = id_to_idx.get(gid)
        if i is not None:
            scores[i] = -1.0
    out = {}
    for i in np.argsort(scores)[::-1]:
        gid = item_ids[i]
        if allowed is not None and gid not in allowed:
            continue
        if not passes_geo(gid):
            continue
        out[gid] = float(scores[i])
        if len(out) >= n_candidates:
            break
    return out


def _cuisine_ids(cuisines):
    """gmap_ids matching any of the requested cuisines, or None if no usable cuisine."""
    if not cuisines:
        return None
    restaurants = _G["restaurants"]
    cats = [c for cu in cuisines for c in CUISINE_TO_CATS.get(cu, [])]
    cats = [c for c in cats if c in restaurants.columns]
    if not cats:
        return None
    return set(restaurants[restaurants[cats].max(axis=1) > 0]["gmap_id"])


# ── hard filters ──────────────────────────────────────────────────────────────

def apply_filters(df, filters):
    filters = filters or {}
    usable = _G.get("usable_cols", set())
    if filters.get("max_price"):
        df = df[df["price_numeric"].fillna(4) <= filters["max_price"]]
    for key, col in AMENITY_COLS.items():
        if filters.get(key) and col in usable:
            df = df[df[col] == 1]
    for key, col in DIETARY_COLS.items():
        if filters.get(key) and col in usable:
            df = df[df[col] > 0]
    cat = filters.get("category")
    if cat and cat != "Any":
        cols = [c for c in CUISINE_TO_CATS.get(cat, []) if c in df.columns]
        if cols:
            df = df[df[cols].max(axis=1) > 0]
    return df


# ── record building ───────────────────────────────────────────────────────────

def _pretty_cat(row):
    grp = row.get("category_standardized")
    if isinstance(grp, str) and grp:
        return grp.replace("_", " ").title()
    for c in row.index:
        if c.startswith("cat_") and row[c] == 1 and c != "cat_restaurant":
            return c.replace("cat_", "").replace("_", " ").title()
    return "Restaurant"


def _records(rec_ids, raw_scores, user_lat, user_lon):
    """Join restaurant fields; attach a 0..1 display 'match' score from raw_scores."""
    rd, name_map = _G["rd"], _G["name_map"]
    vals = [raw_scores.get(g, 0.0) for g in rec_ids]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    out = []
    for gid, raw in zip(rec_ids, vals):
        if gid not in rd.index:
            continue
        row = rd.loc[gid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        lat, lon = row.get("latitude"), row.get("longitude")
        dist = None
        if user_lat is not None and user_lon is not None and pd.notna(lat) and pd.notna(lon):
            dist = float(haversine(user_lat, user_lon, lat, lon))
        match = (raw - lo) / (hi - lo) if hi > lo else 1.0
        rec = {
            "gmap_id": gid,
            "name": row.get("name", ""),
            "address": _san(row.get("address", "")),
            "latitude": _san(lat),
            "longitude": _san(lon),
            "avg_rating": _san(row.get("weighted_avg_rating")),
            "price": row.get("price") if isinstance(row.get("price"), str) else "",
            "category": _pretty_cat(row),
            "final_score": round(float(match), 3),
            "distance_miles": _san(dist),
        }
        out.append(rec)
    return out


# ── public: single-user recommend ───────────────────────────────────────────

def recommend(user_id=None, prefs=None, filters=None,
              user_lat=None, user_lon=None, max_miles=None, n=20):
    load_all()
    restaurants = _G["restaurants"]
    passes_geo = _make_geo_filter(_G["rest_locs"], user_lat, user_lon, max_miles)

    # Allowed candidate set from hard filters + explicit cuisine request. An
    # explicit cuisine steers the result ("show me Indian" -> Indian) even when
    # the user's taste profile points elsewhere (e.g. Thai/Japanese).
    allowed = None
    if filters:
        allowed = set(apply_filters(restaurants, filters)["gmap_id"])
    cuisine_ids = _cuisine_ids((prefs or {}).get("cuisines"))
    if cuisine_ids is not None:
        allowed = cuisine_ids if allowed is None else (allowed & cuisine_ids)

    source = "popularity"
    # 1. dataset history
    if user_id and user_id in _G["dataset_users"]:
        scored = score_user(user_id, _G["reviews"], _G["emb"], _G["item_ids"],
                            _G["id_to_idx"], restaurants, candidate_mode=_CFG.candidate_mode,
                            user_lat=user_lat, user_lon=user_lon, max_miles=max_miles,
                            rest_locs=_G["rest_locs"])
        source = "history"
    else:
        # 2. logged likes  3. cold-start prefs  4. popularity
        liked = []
        if user_id:
            try:
                from database import get_user_interactions
                liked = [(i["gmap_id"], i["rating"]) for i in get_user_interactions(user_id)]
            except Exception:
                liked = []
        seen = set(g for g, _ in liked)
        qvec = vector_from_interactions(liked)
        if qvec is not None:
            source = "interactions"
        if qvec is None and prefs:
            qvec = build_cold_start_vector(prefs)
            if qvec is not None:
                source = "cold_start"
        i2i_raw = query_scores(qvec, seen, passes_geo, allowed=allowed) if qvec is not None else {}
        pop_raw = popularity_scores(restaurants, seen, passes_geo)
        # full I2I weight for synthesized vectors (alpha = n_reviews/scale)
        n_reviews = _CFG.alpha_scale if qvec is not None else 0
        scored = {"seen": seen, "i2i_raw": i2i_raw, "pop_raw": pop_raw, "n_reviews": n_reviews}

    # restrict candidate id sets to the allowed (filters + cuisine) set
    if allowed is not None:
        scored["i2i_raw"] = {g: v for g, v in scored["i2i_raw"].items() if g in allowed}
        scored["pop_raw"] = {g: v for g, v in scored["pop_raw"].items() if g in allowed}

    cfg = _CFG if scored["i2i_raw"] else _POP_CFG  # pure cfg needs an I2I vector
    rec_ids = rank(scored, cfg, _G["emb"], _G["id_to_idx"], _G["ri"], _G["name_map"], n=n)
    raw = scored["i2i_raw"] if scored["i2i_raw"] else scored["pop_raw"]
    recs = _records(rec_ids, raw, user_lat, user_lon)
    for r in recs:
        r["source"] = source
        r["is_personalised"] = source in ("history", "interactions")
    return recs


# ── public: group recommend (the USP) ────────────────────────────────────────

_AGG = {
    "least_misery": min,
    "average": lambda v: sum(v) / len(v),
    "most_pleasure": max,
}


def _member_query_vec(member):
    """Per-member query vector: dataset history -> logged likes -> cold-start prefs."""
    uid = member.get("user_id")
    if uid and uid in _G["dataset_users"]:
        return build_user_vector(uid, _G["reviews"], _G["emb"], _G["id_to_idx"])
    liked = []
    if uid:
        try:
            from database import get_user_interactions
            liked = [(i["gmap_id"], i["rating"]) for i in get_user_interactions(uid)]
        except Exception:
            liked = []
    v = vector_from_interactions(liked)
    if v is not None:
        return v
    return build_cold_start_vector({"cuisines": member.get("cuisines", [])})


def group_recommend(members, strategy="least_misery",
                    user_lat=None, user_lon=None, max_miles=None, n=10):
    load_all()
    restaurants = _G["restaurants"]

    # union of hard constraints + strictest (min) price
    merged, price_max = {}, 4
    for m in members:
        for k, v in (m.get("filters") or {}).items():
            if v:
                merged[k] = v
        if m.get("price_max"):
            price_max = min(price_max, m["price_max"])
    merged["max_price"] = price_max

    cand = apply_filters(restaurants, merged)
    if user_lat is not None and user_lon is not None and max_miles is not None:
        lat, lon = cand["latitude"].values, cand["longitude"].values
        dist = haversine(user_lat, user_lon, lat, lon)
        cand = cand.assign(distance_miles=dist)
        cand = cand[cand["distance_miles"] <= max_miles]
    if len(cand) == 0:
        return []

    cand_ids = list(cand["gmap_id"])
    cand_idx = [_G["id_to_idx"].get(g) for g in cand_ids]
    valid = [(g, i) for g, i in zip(cand_ids, cand_idx) if i is not None]
    if not valid:
        return []
    vids = [g for g, _ in valid]
    vmat = _G["emb"][[i for _, i in valid]]

    pop = popularity_scores(restaurants, set(), lambda g: True)
    member_scores = []
    for m in members:
        qvec = _member_query_vec(m)
        if qvec is not None:
            sims = cosine_similarity(qvec, vmat)[0]
            scores = {g: float(s) for g, s in zip(vids, sims)}
        else:
            scores = {g: float(pop.get(g, 0.0)) for g in vids}
        if scores:
            lo, hi = min(scores.values()), max(scores.values())
            if hi > lo:
                scores = {g: (s - lo) / (hi - lo) for g, s in scores.items()}
        member_scores.append(scores)

    agg = _AGG.get(strategy, min)
    combined = {g: agg([ms.get(g, 0.0) for ms in member_scores]) for g in vids}
    ranked = sorted(combined, key=combined.get, reverse=True)

    # chain-dedupe (one per chain name) then take n
    name_map = _G["name_map"]
    seen_chain, ordered = set(), []
    for g in ranked:
        c = str(name_map.get(g, "")).lower().strip()
        if c in seen_chain:
            continue
        seen_chain.add(c)
        ordered.append(g)
        if len(ordered) >= n:
            break

    recs = _records(ordered, combined, user_lat, user_lon)
    for r in recs:
        gid = r["gmap_id"]
        r["member_breakdown"] = [
            {"name": m.get("name", f"Member {i+1}"),
             "score": round(float(member_scores[i].get(gid, 0.0)), 3)}
            for i, m in enumerate(members)
        ]
    return recs


# ── interactions (warm-up) ────────────────────────────────────────────────────

def log_like(user_id, gmap_id, rating=5.0):
    from database import upsert_user, log_interaction
    upsert_user(user_id)
    log_interaction(user_id, gmap_id, float(rating), "explicit")


# ── feedback -> profile update ────────────────────────────────────────────────

POS_RATING = 5.0   # 👍
NEG_RATING = 1.0   # 👎


def _cuisine_of(gmap_id):
    """Best-guess cuisine label for a restaurant from its cat_ columns."""
    rd = _G.get("rd")
    if rd is None or gmap_id not in rd.index:
        return None
    row = rd.loc[gmap_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    for cuisine, cats in CUISINE_TO_CATS.items():
        for c in cats:
            if c in row.index and row.get(c) == 1:
                return cuisine
    return None


def _update_cuisine_affinity(user_id, gmap_id, delta):
    """Bump the user's stored cuisine_affinity for this restaurant's cuisine."""
    from database import get_user, upsert_user
    cuisine = _cuisine_of(gmap_id)
    if not cuisine:
        return None
    u = get_user(user_id) or {}
    prefs = u.get("preferences") or {}
    affinity = dict(prefs.get("cuisine_affinity") or {})
    affinity[cuisine] = affinity.get(cuisine, 0) + delta
    prefs["cuisine_affinity"] = affinity
    upsert_user(user_id, preferences=prefs)
    return cuisine


def record_feedback(user_id, gmap_id, sentiment):
    """Record 👍/👎 on a recommendation and update the user's profile.

    Two profile signals are updated:
      1. an interaction row (drives the I2I query vector next recommend call)
      2. cuisine_affinity in the user's stored preferences (explicit profile)
    """
    from database import upsert_user, log_interaction
    up = sentiment == "up"
    rating = POS_RATING if up else NEG_RATING
    upsert_user(user_id)
    log_interaction(user_id, gmap_id, rating, source="feedback")
    cuisine = _update_cuisine_affinity(user_id, gmap_id, 1 if up else -1)
    return {"rating": rating, "cuisine": cuisine}


if __name__ == "__main__":
    load_all()
    print("\n== cold-start: Italian + Sushi, LA, 10mi ==")
    for r in recommend(prefs={"cuisines": ["Italian", "Sushi"]},
                       user_lat=34.0522, user_lon=-118.2437, max_miles=10, n=5):
        print(f"  {r['name'][:34]:34s} r={r['avg_rating']} {r['category']:18s} "
              f"{round(r['distance_miles'],1)}mi src={r['source']}")
    print("\n== group: vegan + bar, average ==")
    g = group_recommend(
        members=[{"name": "Maya", "cuisines": ["Vegan"], "filters": {"vegan": True}},
                 {"name": "Jay", "cuisines": ["Bar"], "filters": {"serves_alcohol": True}}],
        strategy="average", user_lat=34.0522, user_lon=-118.2437, max_miles=10, n=5)
    for r in g:
        bd = " ".join(f"{b['name']}={b['score']}" for b in r["member_breakdown"])
        print(f"  {r['name'][:30]:30s} {r['category']:16s} [{bd}]")
