"""
train_i2i_encoder.py
Deep item-to-item encoder for an extreme cold-start dataset (~1 review/user).

Why this design:
  The data has a median of 1 review per user, so per-user models (NCF, sequential,
  user-embedding two-towers) have nothing to learn. The only trainable signal is
  ITEM-to-ITEM: "people who liked A also liked B". We learn a projection of the
  existing 567-dim content vectors (frozen MiniLM text + structured features) into
  a space where co-liked restaurants are close, using a Siamese encoder trained
  with in-batch contrastive loss (InfoNCE / CLIP-style).

  RESIDUAL + ANCHOR design:
    A plain encoder that REPLACES the content vector discards the category/chain
    structure that already works, and (with only ~8K co-like pairs) underperforms
    the frozen baseline. Instead we learn a RESIDUAL on top of the content vector:
        output = normalize(x + gate * delta(x))
    with `gate` initialized to 0, so the model STARTS exactly at the frozen
    baseline and only moves where the co-like signal demands it. An ANCHOR loss
        lambda * (1 - cos(output, x))
    penalizes drifting away from the content position. lambda is swept:
        lambda large  -> recovers the baseline (safety floor)
        lambda small  -> co-likes nudge the space; look for a lift above baseline

  Because the encoder consumes CONTENT (not a user/item id), every restaurant -
  including never-reviewed ones - gets a learned embedding, so cold-start is
  preserved. Output is saved in the same format as embeddings_i2i/ so it drops
  straight into recommend_i2i.py and eval_i2i.py.

Run:
  python train_i2i_encoder.py
Then evaluate the winning lambda against the frozen baseline:
  python eval_i2i.py                                  # frozen baseline
  I2I_EMBED_DIR=embeddings_i2i_learned2 python eval_i2i.py   # learned
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

OUTPUT_DIR   = Path("output_review")
EMBED_DIR    = Path("embeddings_i2i")          # input: frozen 567-dim content vectors
OUT_DIR      = Path("embeddings_i2i_learned2")  # output: best learned embeddings
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- hyperparameters ---
HIDDEN       = 256
DROPOUT      = 0.1
EPOCHS       = 200
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-5
TEMPERATURE  = 0.07
VAL_FRAC     = 0.1
WARMUP_EPOCHS = 10       # linear LR warmup, then cosine anneal to ~0
EVAL_EVERY    = 10       # epochs between val checks / best-checkpoint updates
LAMBDAS       = [0.0, 0.3, 1.0, 3.0]   # anchor-loss weights to sweep
SEED          = 42

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


# ── data ────────────────────────────────────────────────────────────────────

def load_item_features():
    """Frozen 567-dim content vectors + their gmap_ids."""
    vecs = np.load(EMBED_DIR / "item_embeddings.npy").astype(np.float32)
    ids  = np.load(EMBED_DIR / "item_embedding_ids.npy", allow_pickle=True)
    id_to_row = {gid: i for i, gid in enumerate(ids)}
    return vecs, ids, id_to_row


def mine_colike_pairs(id_to_row):
    """
    Positive pairs = restaurants co-liked (rating>=4) by the same user.
    Returns an (P, 2) int array of row indices into the feature matrix.
    """
    train = pd.read_parquet(OUTPUT_DIR / "train.parquet")
    liked = train[train["rating"] >= 4]
    liked = liked[liked["gmap_id"].isin(id_to_row.keys())]

    per_user = liked.groupby("user_id")["gmap_id"].apply(lambda s: sorted(set(s)))
    per_user = per_user[per_user.apply(len) >= 2]

    pairs = set()
    for items in per_user:
        for a, b in combinations(items, 2):
            ra, rb = id_to_row[a], id_to_row[b]
            pairs.add((ra, rb) if ra < rb else (rb, ra))

    return np.array(sorted(pairs), dtype=np.int64)


class PairDataset(Dataset):
    def __init__(self, feats, pairs):
        self.feats = feats
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b = self.pairs[i]
        return self.feats[a], self.feats[b]


# ── model ───────────────────────────────────────────────────────────────────

class ResidualItemEncoder(nn.Module):
    """
    Shared (Siamese) residual encoder:
        output = normalize(x + gate * delta(x))
    `gate` starts at 0 -> output == content vector == frozen baseline at init.
    Output dim == input dim (567), so content structure is preserved by default.
    """
    def __init__(self, dim, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return F.normalize(x + self.gate * self.delta(x), dim=-1)


def info_nce(z_a, z_b, temperature=TEMPERATURE):
    """Symmetric in-batch contrastive loss (CLIP-style)."""
    logits = z_a @ z_b.T / temperature
    targets = torch.arange(z_a.size(0), device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, targets) +
                  F.cross_entropy(logits.T, targets))


def anchor_loss(z, x):
    """Penalize drift from the content position. x is already L2-normalized."""
    return (1.0 - (z * x).sum(dim=-1)).mean()


# ── evaluation: learned vs frozen baseline ───────────────────────────────────

@torch.no_grad()
def retrieval_metrics(query_emb, gallery_emb, pairs, ks=(10, 20, 50)):
    """
    For each val pair (a,b): rank b among the full gallery by similarity to a.
    recall@k = fraction of pairs whose partner lands in the top k.
    """
    a_idx, b_idx = pairs[:, 0], pairs[:, 1]
    sims = query_emb[a_idx] @ gallery_emb.T
    sims[np.arange(len(a_idx)), a_idx] = -np.inf   # exclude the query item itself

    order = np.argsort(-sims, axis=1)
    ranks = np.array([np.where(order[i] == b)[0][0] for i, b in enumerate(b_idx)])

    out = {f"recall@{k}": float(np.mean(ranks < k)) for k in ks}
    out["median_rank"] = float(np.median(ranks))
    return out


# ── one training run for a given anchor weight ────────────────────────────────

def train_one(lambda_anchor, feats, feats_np, train_pairs, val_pairs):
    loader = DataLoader(PairDataset(feats, train_pairs),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = ResidualItemEncoder(dim=feats_np.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    best_recall, best_state, best_epoch = -1.0, None, 0
    print(f"\n=== lambda_anchor = {lambda_anchor} ===")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xa, xb in loader:
            za, zb = model(xa), model(xb)
            loss = info_nce(za, zb) + lambda_anchor * (
                anchor_loss(za, xa) + anchor_loss(zb, xb))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        scheduler.step()

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                learned = model(feats).numpy()
            m = retrieval_metrics(learned, learned, val_pairs)
            print(f"  epoch {epoch:>3}  lr {scheduler.get_last_lr()[0]:.2e}  "
                  f"loss {total/len(loader):.4f}  gate {model.gate.item():+.3f}  "
                  f"recall@20 {m['recall@20']:.3f}  median_rank {m['median_rank']:.0f}")
            if m["recall@20"] > best_recall:
                best_recall = m["recall@20"]
                best_epoch  = epoch
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        emb = model(feats).numpy().astype(np.float32)
    final = retrieval_metrics(emb, emb, val_pairs)
    print(f"  -> best epoch {best_epoch}  gate {model.gate.item():+.3f}  "
          f"recall@20 {final['recall@20']:.3f}")
    return emb, model, final


# ── main: sweep lambda ────────────────────────────────────────────────────────

def main():
    print("Loading frozen content vectors...")
    feats_np, ids, id_to_row = load_item_features()
    print(f"  items: {feats_np.shape[0]}  dim: {feats_np.shape[1]}")

    print("Mining co-like pairs...")
    pairs = mine_colike_pairs(id_to_row)
    rng.shuffle(pairs)
    n_val = int(len(pairs) * VAL_FRAC)
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"  pairs: {len(pairs)}  (train {len(train_pairs)} / val {len(val_pairs)})")

    feats = torch.from_numpy(feats_np)

    base = retrieval_metrics(feats_np, feats_np, val_pairs)
    print(f"\n[baseline frozen] recall@20 {base['recall@20']:.3f}  "
          f"recall@50 {base['recall@50']:.3f}  median_rank {base['median_rank']:.0f}")

    results = []
    best_overall = (-1.0, None, None)   # (recall@20, lambda, emb)
    for lam in LAMBDAS:
        emb, model, m = train_one(lam, feats, feats_np, train_pairs, val_pairs)
        results.append((lam, m))
        if m["recall@20"] > best_overall[0]:
            best_overall = (m["recall@20"], lam, emb)

    # ── summary table ──
    print("\n" + "=" * 56)
    print(f"{'config':>22}  {'recall@20':>10}  {'recall@50':>10}  {'med_rank':>9}")
    print(f"{'frozen baseline':>22}  {base['recall@20']:>10.3f}  "
          f"{base['recall@50']:>10.3f}  {base['median_rank']:>9.0f}")
    for lam, m in results:
        print(f"{'lambda=' + str(lam):>22}  {m['recall@20']:>10.3f}  "
              f"{m['recall@50']:>10.3f}  {m['median_rank']:>9.0f}")

    best_recall, best_lam, best_emb = best_overall
    print("=" * 56)
    print(f"Best: lambda={best_lam}  recall@20={best_recall:.3f}  "
          f"(baseline {base['recall@20']:.3f}, lift {best_recall - base['recall@20']:+.3f})")

    np.save(OUT_DIR / "item_embeddings.npy", best_emb)
    np.save(OUT_DIR / "item_embedding_ids.npy", ids)
    print(f"\nSaved best learned embeddings to {OUT_DIR}/  (dim {best_emb.shape[1]})")
    print(f"Evaluate full pipeline:  I2I_EMBED_DIR={OUT_DIR} python eval_i2i.py")


if __name__ == "__main__":
    main()
