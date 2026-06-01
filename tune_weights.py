import sys
import numpy as np
import pandas as pd

from recommend_engine import RecConfig, rank, _rest_locs, _normalize_dict
from experiment_mmr import (
    load_eval_data, pick_users, build_score_cache,
    MIN_HISTORY, MAX_DISTANCE_MILES, N_USERS, K,
)
from eval_i2i import compute_metrics

# base configs to tune weights for (the chain-metric winners from the sweep).
# mmr off here to isolate the weight effect; flip use_mmr if you want.
BASE_CONFIGS = [
    # RecConfig(name="avg|pure", candidate_mode="avg", use_hybrid=False, use_mmr=False),
    # RecConfig(name="qe|pure",  candidate_mode="qe",  use_hybrid=False, use_mmr=False),
    RecConfig(name="avg|hyb",  candidate_mode="avg", use_hybrid=True,  use_mmr=False),
    RecConfig(name="qe|hyb",   candidate_mode="qe",  use_hybrid=True,  use_mmr=False),
]

OBJECTIVE = "chain_ndcg"   # what we optimise


def weight_simplex(step=0.1):
    """All (w_hybrid, w_rating, w_pop) on a grid that sum to 1."""
    vals = [round(i * step, 4) for i in range(int(round(1 / step)) + 1)]
    grid = []
    for a in vals:
        for b in vals:
            c = round(1.0 - a - b, 4)
            if c < -1e-9 or c > 1 + 1e-9:
                continue
            grid.append((round(a, 4), round(b, 4), round(max(c, 0.0), 4)))
    return sorted(set(grid))


def prep_user(base_cfg, scored, ri, name_map):
    """
    Precompute everything in rank() that does NOT depend on the final-score
    weights, so a weight sweep never re-runs the dict normalisation, the
    alpha-blend, or the pandas .reindex(). Returns numpy arrays aligned to a
    candidate list, plus per-candidate chain keys for dedupe. None if no cands.
    """
    i2i = _normalize_dict(scored["i2i_raw"])
    pop = _normalize_dict(scored["pop_raw"])
    if base_cfg.pop_only:
        combined = pop
    elif base_cfg.use_hybrid:
        alpha = min(1.0, scored["n_reviews"] / base_cfg.alpha_scale)
        ids = set(i2i) | set(pop)
        combined = {g: alpha * i2i.get(g, 0.0) + (1 - alpha) * pop.get(g, 0.0) for g in ids}
    else:
        combined = i2i
    if not combined:
        return None

    cand = sorted(combined, key=combined.get, reverse=True)[:base_cfg.n_candidates]
    sub = ri.reindex(cand)
    max_rating = sub["weighted_avg_rating"].max() or 1.0
    max_count  = sub["review_count"].max() or 1.0
    combined_arr = np.fromiter((combined[g] for g in cand), dtype=float, count=len(cand))
    rating_arr   = (sub["weighted_avg_rating"].fillna(0) / max_rating).to_numpy()
    pop_arr      = (sub["review_count"].fillna(0) / max_count).to_numpy()
    chain_keys = [c.lower().strip() if isinstance(c, str) else ""
                  for c in (name_map.get(g, "") for g in cand)]
    return cand, combined_arr, rating_arr, pop_arr, chain_keys


def _rank_from_prep(prep, weights, max_per_chain, k):
    """Replicate rank()'s non-MMR path from precomputed arrays (cheap, no pandas)."""
    cand, combined_arr, rating_arr, pop_arr, chain_keys = prep
    wh, wr, wp = weights
    relevance = wh * combined_arr + wr * rating_arr + wp * pop_arr
    depth = max(k * 3, 60)
    order = np.argsort(-relevance, kind="stable")[:depth]
    counts, out = {}, []
    for idx in order:
        c = chain_keys[idx]
        if counts.get(c, 0) >= max_per_chain:
            continue
        counts[c] = counts.get(c, 0) + 1
        out.append(cand[idx])
        if len(out) >= k:
            break
    return out


def eval_weights(prep_cache, users, base_cfg, weights, user_ctx,
                 name_map, restaurants_indexed, k):
    """Mean metrics for base_cfg with a given (w_hybrid, w_rating, w_pop)."""
    mlist = []
    for user_id in users:
        prep = prep_cache.get(user_id)
        if prep is None:
            continue
        rec_ids = _rank_from_prep(prep, weights, base_cfg.max_per_chain, k)
        if not rec_ids:
            continue
        relevant_ids, relevant_chains = user_ctx[user_id]
        mlist.append(compute_metrics(rec_ids, relevant_ids, relevant_chains,
                                     name_map, restaurants_indexed, k))
    if not mlist:
        return None
    return {key: float(np.mean([m[key] for m in mlist])) for key in mlist[0]}


def run(n_users=N_USERS, step=0.1, k=K, min_history=MIN_HISTORY, max_miles=MAX_DISTANCE_MILES):
    print("Loading data + artifacts...")
    (test, train, item_embeddings, item_ids, id_to_idx, restaurants,
     reviews, name_map, restaurants_indexed) = load_eval_data()

    ri = restaurants.set_index("gmap_id")[["weighted_avg_rating", "review_count"]]
    ri = ri[~ri.index.duplicated(keep="first")]
    rest_locs = _rest_locs(restaurants)

    sample_users = pick_users(test, train, n_users, min_history)
    modes_needed = sorted({c.candidate_mode for c in BASE_CONFIGS})
    user_ctx, score_cache = build_score_cache(
        sample_users, test, train, reviews, restaurants,
        item_embeddings, item_ids, id_to_idx, name_map,
        modes_needed, max_miles, rest_locs)

    evaluable = list(user_ctx.keys())
    rng = np.random.default_rng(7)
    rng.shuffle(evaluable)
    half = len(evaluable) // 2
    val_users, test_users = evaluable[:half], evaluable[half:]
    print(f"Evaluable: {len(evaluable)} | val={len(val_users)} test={len(test_users)}")

    grid = weight_simplex(step)
    print(f"Weight grid points: {len(grid)} (step={step}), objective={OBJECTIVE}\n")

    fixed = (0.6, 0.2, 0.2)
    summary = []

    def m(metrics, key):
        return metrics[key] if metrics else float("nan")

    for base in BASE_CONFIGS:
        if base.use_mmr:
            raise NotImplementedError(
                f"{base.name}: weight tuning fast-path assumes use_mmr=False")

        # precompute weight-independent arrays once per (base, user); the grid
        # sweep below then costs only a weighted sum + argsort per evaluation.
        prep_cache = {}
        for user_id in evaluable:
            prep = prep_user(base, score_cache[user_id][base.candidate_mode], ri, name_map)
            if prep is not None:
                prep_cache[user_id] = prep

        # search on VAL
        best_w, best_obj = None, -1.0
        for w in grid:
            res = eval_weights(prep_cache, val_users, base, w, user_ctx,
                               name_map, restaurants_indexed, k)
            if res and res[OBJECTIVE] > best_obj:
                best_obj, best_w = res[OBJECTIVE], w

        # report chosen + fixed baseline on held-out TEST
        tuned_test = eval_weights(prep_cache, test_users, base, best_w, user_ctx,
                                  name_map, restaurants_indexed, k)
        fixed_test = eval_weights(prep_cache, test_users, base, fixed, user_ctx,
                                  name_map, restaurants_indexed, k)

        print(f"── {base.name} ──")
        print(f"  best weights (val):   w_hybrid={best_w[0]}, w_rating={best_w[1]}, w_pop={best_w[2]}  "
              f"(val {OBJECTIVE}={best_obj:.4f})")
        print(f"  TUNED  on test: hit_chain={m(tuned_test,'hit_chain'):.4f}  "
              f"chain_ndcg={m(tuned_test,'chain_ndcg'):.4f}  "
              f"chain_recall={m(tuned_test,'chain_recall'):.4f}")
        print(f"  FIXED  on test: hit_chain={m(fixed_test,'hit_chain'):.4f}  "
              f"chain_ndcg={m(fixed_test,'chain_ndcg'):.4f}  "
              f"chain_recall={m(fixed_test,'chain_recall'):.4f}  (0.6/0.2/0.2)")
        delta = m(tuned_test, OBJECTIVE) - m(fixed_test, OBJECTIVE)
        print(f"  lift on test {OBJECTIVE}: {delta:+.4f}\n")

        row = {"base": base.name,
               "w_hybrid": best_w[0], "w_rating": best_w[1], "w_pop": best_w[2],
               f"val_{OBJECTIVE}": best_obj}
        for key in (tuned_test or {}):
            row[f"test_tuned_{key}"] = tuned_test[key]
            row[f"test_fixed_{key}"] = fixed_test[key] if fixed_test else float("nan")
        summary.append(row)

    df = pd.DataFrame(summary).sort_values(f"test_tuned_{OBJECTIVE}", ascending=False)
    df.to_csv("tuned_weights_results.csv", index=False)
    print("Full results written to tuned_weights_results.csv")
    best = df.iloc[0]
    print(f"\nBest overall: {best['base']} with "
          f"w_hybrid={best['w_hybrid']}, w_rating={best['w_rating']}, w_pop={best['w_pop']} "
          f"-> test {OBJECTIVE}={best[f'test_tuned_{OBJECTIVE}']:.4f}")
    return df


if __name__ == "__main__":
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else N_USERS
    step    = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    k       = int(sys.argv[3]) if len(sys.argv) > 3 else K
    run(n_users=n_users, step=step, k=k)
