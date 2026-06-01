"""
train_i2i_content.py
CONTENT-supervised deep item encoder (deep learning applied to content, NOT interactions).

Rationale:
  The interaction signal is far too sparse to train on (~8K co-like pairs; see
  train_i2i_encoder.py, which never beats the frozen baseline). But the CONTENT
  side is abundant: 80,715 restaurants, every one labeled, 160 categories with
  >=20 members -> effectively unlimited training pairs. So we apply deep learning
  where the data actually is: learn an embedding in which restaurants that share
  a cuisine/category are close.

  Positive pair = two restaurants sharing a SPECIFIC category (the generic
  "Restaurant" tag is dropped so the model learns real cuisine structure, not the
  catch-all). Trained with in-batch contrastive loss on the residual encoder
  (output = normalize(x + gate*delta(x))) + anchor loss, so content structure is
  preserved and only refined.

  Honest caveat: eval_i2i.py's metrics are category/chain based, so a category-
  trained encoder is partly ALIGNED with the eval (unlike the interaction model).
  Watch the CHAIN metrics especially - chains are not the training label, so a
  chain-metric lift is the least circular evidence that this helped.

Run:
  python train_i2i_content.py
  I2I_EMBED_DIR=embeddings_i2i_content python eval_i2i.py
"""

import ast
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# reuse the proven encoder + losses
from train_i2i_encoder import (
    ResidualItemEncoder, info_nce, anchor_loss, load_item_features,
)

EMBED_DIR = Path("embeddings_i2i")
OUT_DIR   = Path("embeddings_i2i_content")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_RESTAURANTS = "restaurants_only.csv"

# --- hyperparameters ---
EPOCHS            = 40
PAIRS_PER_EPOCH   = 100_000
BATCH_SIZE        = 512
LR                = 1e-3
WEIGHT_DECAY      = 1e-5
WARMUP_EPOCHS     = 3
EVAL_EVERY        = 5         # epochs between the (expensive) category-coherence proxy
LAMBDA_ANCHOR     = 1.0       # keep close to content; this signal is dense so less drift needed
GENERIC_TAGS      = {"restaurant", "store", "establishment"}
MIN_CAT_MEMBERS   = 5
SEED              = 42

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


# ── content labels ────────────────────────────────────────────────────────────

def load_categories(ids):
    """Return cats_by_row (list[set]) aligned to the embedding row order."""
    df = pd.read_csv(RAW_RESTAURANTS, engine="python", on_bad_lines="skip",
                     usecols=["gmap_id", "category"])
    df["category"] = df["category"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    cat_map = dict(zip(df["gmap_id"], df["category"]))

    cats_by_row = []
    for gid in ids:
        raw = cat_map.get(gid, []) or []
        specific = {c.lower().strip() for c in raw
                    if c.lower().strip() not in GENERIC_TAGS}
        cats_by_row.append(specific)
    return cats_by_row


def build_category_index(cats_by_row):
    """cat -> np.array(rows), keeping only categories with >= MIN_CAT_MEMBERS."""
    cat_to_rows = {}
    for row, cats in enumerate(cats_by_row):
        for c in cats:
            cat_to_rows.setdefault(c, []).append(row)
    cat_to_rows = {c: np.array(r) for c, r in cat_to_rows.items()
                   if len(r) >= MIN_CAT_MEMBERS}
    # anchor rows = those that belong to at least one usable category
    usable = set(cat_to_rows.keys())
    anchor_rows = [row for row, cats in enumerate(cats_by_row) if cats & usable]
    return cat_to_rows, np.array(anchor_rows)


class CategoryPairDataset(Dataset):
    """On each draw: anchor + a positive that shares one of the anchor's categories."""
    def __init__(self, feats, cats_by_row, cat_to_rows, anchor_rows, length):
        self.feats = feats
        self.cats_by_row = cats_by_row
        self.cat_to_rows = cat_to_rows
        self.usable = set(cat_to_rows.keys())
        self.anchor_rows = anchor_rows
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        a = int(rng.choice(self.anchor_rows))
        shared = list(self.cats_by_row[a] & self.usable)
        cat = shared[rng.integers(len(shared))]
        pool = self.cat_to_rows[cat]
        b = int(pool[rng.integers(len(pool))])
        # avoid the degenerate self-pair
        tries = 0
        while b == a and tries < 5:
            b = int(pool[rng.integers(len(pool))])
            tries += 1
        return self.feats[a], self.feats[b]


# ── content-coherence proxy ───────────────────────────────────────────────────

@torch.no_grad()
def category_neighbor_precision(emb, cats_by_row, n_query=1000, k=10):
    """For sampled items: fraction of top-k neighbors sharing >=1 category. Higher = better."""
    q = rng.choice(len(emb), min(n_query, len(emb)), replace=False)
    sims = emb[q] @ emb.T                       # (Q, N) single BLAS matmul
    sims[np.arange(len(q)), q] = -np.inf        # exclude self
    topk = np.argpartition(-sims, k, axis=1)[:, :k]
    hits = total = 0
    for qi_idx, i in enumerate(q):
        qi = cats_by_row[i]
        if not qi:
            continue
        for j in topk[qi_idx]:
            total += 1
            if qi & cats_by_row[j]:
                hits += 1
    return hits / total if total else 0.0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading frozen content vectors...")
    feats_np, ids, _ = load_item_features()
    print(f"  items: {feats_np.shape[0]}  dim: {feats_np.shape[1]}")

    print("Loading category labels...")
    cats_by_row = load_categories(ids)
    cat_to_rows, anchor_rows = build_category_index(cats_by_row)
    print(f"  usable categories (>= {MIN_CAT_MEMBERS}): {len(cat_to_rows)}  "
          f"anchor items: {len(anchor_rows)}")

    feats = torch.from_numpy(feats_np)
    ds = CategoryPairDataset(feats, cats_by_row, cat_to_rows, anchor_rows, PAIRS_PER_EPOCH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    base_prec = category_neighbor_precision(feats_np, cats_by_row)
    print(f"\n[baseline frozen] category-neighbor precision@10: {base_prec:.3f}")

    model = ResidualItemEncoder(dim=feats_np.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    best_prec, best_state = -1.0, None
    print("\nTraining (content / category-supervised)...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xa, xb in loader:
            za, zb = model(xa), model(xb)
            loss = info_nce(za, zb) + LAMBDA_ANCHOR * (anchor_loss(za, xa) + anchor_loss(zb, xb))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        scheduler.step()

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                emb = model(feats).numpy()
            prec = category_neighbor_precision(emb, cats_by_row)
            print(f"  epoch {epoch:>2}  lr {scheduler.get_last_lr()[0]:.2e}  "
                  f"loss {total/len(loader):.4f}  gate {model.gate.item():+.3f}  "
                  f"cat_nn_prec@10 {prec:.3f}")
            if prec > best_prec:
                best_prec = prec
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            print(f"  epoch {epoch:>2}  lr {scheduler.get_last_lr()[0]:.2e}  "
                  f"loss {total/len(loader):.4f}  gate {model.gate.item():+.3f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        emb = model(feats).numpy().astype(np.float32)

    print(f"\n[learned best] cat_nn_prec@10 {best_prec:.3f}  "
          f"(baseline {base_prec:.3f}, lift {best_prec - base_prec:+.3f})")
    np.save(OUT_DIR / "item_embeddings.npy", emb)
    np.save(OUT_DIR / "item_embedding_ids.npy", ids)
    print(f"\nSaved to {OUT_DIR}/  (dim {emb.shape[1]})")
    print(f"Evaluate full pipeline:  I2I_EMBED_DIR={OUT_DIR} python eval_i2i.py")


if __name__ == "__main__":
    main()
