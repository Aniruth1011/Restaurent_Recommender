"""
eval_collaborative.py
Companion to eval_i2i.py that restricts evaluation to DENSE users - those with
enough training history for a collaborative (co-like) signal to plausibly help.

Motivation:
  The dataset is cold-start dominated (~1 review/user). eval_i2i.py evaluates
  everyone, where content similarity dominates and a trained encoder cannot beat
  frozen content embeddings. This script asks the narrower question: among users
  who actually have a history, does the learned (co-like) embedding space help?

  Reality check on this dataset (train&test overlap, >=1 relevant test item):
      >=2 train reviews -> 103 users
      >=3 train reviews ->  27 users
      >=5 train reviews ->   2 users  (not enough to be meaningful)
  So small MIN_HISTORY thresholds are the only viable ones, and results at high
  thresholds are noisy by construction - that thinness is itself a finding.

Usage:
  python eval_collaborative.py            # default MIN_HISTORY=2
  python eval_collaborative.py 3          # require >=3 train reviews
  I2I_EMBED_DIR=embeddings_i2i_learned2 python eval_collaborative.py 2   # learned
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# reuse the metric + helper functions from eval_i2i so the two evals stay identical
from eval_i2i import (
    get_chain_name, compute_metrics, print_metrics,
    filter_relevant_by_location,
)
from recommend_i2i import load_all, recommend_i2i, get_user_centroid

OUTPUT_DIR = Path("output_review")

MIN_HISTORY        = 2     # default: users with >=2 training reviews
MAX_DISTANCE_MILES = 50
N_USERS            = 1000  # cover the whole (small) dense cohort
K                  = 20


def evaluate_collaborative(n_users=N_USERS, k=K, min_history=MIN_HISTORY,
                           max_miles=MAX_DISTANCE_MILES):
    print("Loading data...")
    test  = pd.read_parquet(OUTPUT_DIR / "test.parquet")
    train = pd.read_parquet(OUTPUT_DIR / "train.parquet")

    print("Loading model artifacts...")
    item_embeddings, item_ids, id_to_idx, restaurants, reviews = load_all()
    name_map = restaurants.set_index("gmap_id")["name"].to_dict()

    # category info for soft metrics
    import ast
    raw_restaurants = pd.read_csv(
        "restaurants_only.csv", engine="python", on_bad_lines="skip",
        usecols=["gmap_id", "category"]
    )
    raw_restaurants["category"] = raw_restaurants["category"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    restaurants_indexed = raw_restaurants.set_index("gmap_id")
    rating_df = restaurants[["gmap_id", "weighted_avg_rating"]].set_index("gmap_id")
    restaurants_indexed = restaurants_indexed.join(rating_df, how="left")

    # DENSE users only: in both splits, with >= min_history training reviews
    train_user_counts = train.groupby("user_id").size()
    overlap = set(test["user_id"].unique()) & set(train["user_id"].unique())
    valid_users = [u for u in overlap if train_user_counts.get(u, 0) >= min_history]
    print(f"Dense users (>={min_history} train reviews): {len(valid_users)}")

    if not valid_users:
        print("No users meet the history threshold - cannot evaluate.")
        return []

    sample_users = np.random.default_rng(42).choice(
        valid_users, min(n_users, len(valid_users)), replace=False
    )

    metrics_list = []
    skipped_no_relevant = skipped_no_recs = 0

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

        metrics_list.append(
            compute_metrics(recommended_ids, relevant_ids, relevant_chains,
                            name_map, restaurants_indexed, k)
        )

    print(f"\nSkipped (no relevant): {skipped_no_relevant} | Skipped (no recs): {skipped_no_recs}")
    print(f"Metrics collected: {len(metrics_list)}")
    print_metrics(f"Collaborative / dense users (min_history={min_history}, "
                  f"max_miles={max_miles})", metrics_list, k)
    return metrics_list


if __name__ == "__main__":
    mh = int(sys.argv[1]) if len(sys.argv) > 1 else MIN_HISTORY
    evaluate_collaborative(min_history=mh)
