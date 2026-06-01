import sys
import numpy as np
import pandas as pd
from pathlib import Path

from recommend_engine import (
    RecConfig, load_all, score_user, rank, get_user_centroid, _rest_locs,
)
from eval_i2i import (
    compute_metrics, get_chain_name, filter_relevant_by_location,
)

OUTPUT_DIR = Path("output_review")

MIN_HISTORY        = 1
MAX_DISTANCE_MILES = 50
N_USERS            = 500
K                  = 20


def build_grid():
    """Every combination of the methods, plus a popularity baseline."""
    configs = [RecConfig(name="popularity", pop_only=True, candidate_mode="avg")]
    rerank_opts = [("no-mmr", False, 1.0),
                   ("mmr1.0", True, 1.0),
                   ("mmr0.9", True, 0.9),
                   ("mmr0.7", True, 0.7),
                   ("mmr0.5", True, 0.5)]
    for mode in ["avg", "qe"]:
        for hybrid in [True, False]:
            for rname, use_mmr, lam in rerank_opts:
                hyb_tag = "hyb" if hybrid else "pure"
                configs.append(RecConfig(
                    name=f"{mode}|{hyb_tag}|{rname}",
                    candidate_mode=mode,
                    use_hybrid=hybrid,
                    use_mmr=use_mmr,
                    mmr_lambda=lam,
                ))
    return configs


def load_eval_data():
    test  = pd.read_parquet(OUTPUT_DIR / "test.parquet")
    train = pd.read_parquet(OUTPUT_DIR / "train.parquet")
    item_embeddings, item_ids, id_to_idx, restaurants, reviews = load_all()
    name_map = restaurants.set_index("gmap_id")["name"].to_dict()

    # categories from raw csv for soft metrics (matches eval_i2i)
    import ast
    raw = pd.read_csv("restaurants_only.csv", engine="python",
                      on_bad_lines="skip", usecols=["gmap_id", "category"])
    raw["category"] = raw["category"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    restaurants_indexed = raw.set_index("gmap_id")
    rating_df = restaurants[["gmap_id", "weighted_avg_rating"]].set_index("gmap_id")
    restaurants_indexed = restaurants_indexed.join(rating_df, how="left")

    return (test, train, item_embeddings, item_ids, id_to_idx, restaurants,
            reviews, name_map, restaurants_indexed)


def pick_users(test, train, n_users, min_history):
    train_counts = train.groupby("user_id").size()
    overlap = set(test["user_id"].unique()) & set(train["user_id"].unique())
    valid = [u for u in overlap if train_counts.get(u, 0) >= min_history]
    rng = np.random.default_rng(42)
    return rng.choice(valid, min(n_users, len(valid)), replace=False)


def build_score_cache(sample_users, test, train, reviews, restaurants,
                      item_embeddings, item_ids, id_to_idx, name_map,
                      modes_needed, max_miles, rest_locs):
    """
    Run the expensive cosine pass once per (user, candidate_mode) and cache it.
    Returns (user_ctx, score_cache) where
      user_ctx[user]   = (relevant_ids, relevant_chains)
      score_cache[user][mode] = scored dict from score_user()
    Both experiment_mmr and tune_weights reuse this so scoring is never duplicated.
    """
    score_cache, user_ctx = {}, {}
    print("Scoring users (cached per candidate_mode)...")
    for ui, user_id in enumerate(sample_users):
        relevant_ids = test[(test["user_id"] == user_id) & (test["rating"] >= 4)]["gmap_id"].tolist()
        if not relevant_ids:
            continue
        user_lat, user_lon = get_user_centroid(user_id, train, restaurants)
        relevant_ids = filter_relevant_by_location(relevant_ids, user_lat, user_lon,
                                                    restaurants, max_miles)
        if not relevant_ids:
            continue
        relevant_chains = set(get_chain_name(name_map.get(g, "")) for g in relevant_ids)
        user_ctx[user_id] = (relevant_ids, relevant_chains)

        score_cache[user_id] = {}
        for mode in modes_needed:
            score_cache[user_id][mode] = score_user(
                user_id, reviews, item_embeddings, item_ids, id_to_idx, restaurants,
                candidate_mode=mode, user_lat=user_lat, user_lon=user_lon,
                max_miles=max_miles, rest_locs=rest_locs)
        if (ui + 1) % 50 == 0:
            print(f"  scored {ui + 1}/{len(sample_users)}")
    return user_ctx, score_cache


def run(n_users=N_USERS, k=K, min_history=MIN_HISTORY, max_miles=MAX_DISTANCE_MILES):
    print("Loading data + artifacts...")
    (test, train, item_embeddings, item_ids, id_to_idx, restaurants,
     reviews, name_map, restaurants_indexed) = load_eval_data()

    # rating/count index for rank()
    ri = restaurants.set_index("gmap_id")[["weighted_avg_rating", "review_count"]]
    ri = ri[~ri.index.duplicated(keep="first")]
    rest_locs = _rest_locs(restaurants)

    sample_users = pick_users(test, train, n_users, min_history)
    print(f"Evaluating {len(sample_users)} users")

    configs = build_grid()
    modes_needed = sorted({c.candidate_mode for c in configs})
    print(f"Configs: {len(configs)} | candidate modes to score: {modes_needed}")

    user_ctx, score_cache = build_score_cache(
        sample_users, test, train, reviews, restaurants,
        item_embeddings, item_ids, id_to_idx, name_map,
        modes_needed, max_miles, rest_locs)

    evaluable = list(user_ctx.keys())
    print(f"Users with relevant items: {len(evaluable)}")

    # ── evaluate each config (cheap: reuses cached scores) ──
    rows = []
    for ci, cfg in enumerate(configs):
        mlist = []
        for user_id in evaluable:
            relevant_ids, relevant_chains = user_ctx[user_id]
            scored = score_cache[user_id][cfg.candidate_mode]
            rec_ids = rank(scored, cfg, item_embeddings, id_to_idx, ri, name_map, n=k)
            if not rec_ids:
                continue
            mlist.append(compute_metrics(rec_ids, relevant_ids, relevant_chains,
                                         name_map, restaurants_indexed, k))
        if not mlist:
            continue
        row = {"config": cfg.name, "n_eval": len(mlist)}
        for key in mlist[0]:
            row[key] = float(np.mean([m[key] for m in mlist]))
        rows.append(row)
        print(f"  [{ci + 1}/{len(configs)}] {cfg.name:24s} "
              f"hit_chain={row['hit_chain']:.4f} chain_ndcg={row['chain_ndcg']:.4f} "
              f"chain_recall={row['chain_recall']:.4f} hit_strict={row['hit_strict']:.4f}")

    df = pd.DataFrame(rows)
    df = df.sort_values("chain_ndcg", ascending=False).reset_index(drop=True)

    out_path = "experiment_results.csv"
    df.to_csv(out_path, index=False)

    # ── pretty print, chain metrics first ──
    cols = ["config", "n_eval", "hit_chain", "chain_ndcg", "chain_prec", "chain_recall",
            "hit_strict", "ndcg", "recall", "precision",
            "cat_hit", "cat_recall", "cat_ndcg", "diversity", "avg_rating", "quality_frac"]
    cols = [c for c in cols if c in df.columns]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(f"\n{'='*100}\nRESULTS (sorted by chain NDCG@{k}) — {len(evaluable)} users\n{'='*100}")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nFull results written to {out_path}")

    best = df.iloc[0]
    print(f"\nBest config by chain NDCG: {best['config']}  "
          f"(hit_chain={best['hit_chain']:.4f}, chain_ndcg={best['chain_ndcg']:.4f})")
    return df


if __name__ == "__main__":
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else N_USERS
    k       = int(sys.argv[2]) if len(sys.argv) > 2 else K
    run(n_users=n_users, k=k)
