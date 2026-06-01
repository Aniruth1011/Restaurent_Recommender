import ast
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train_i2i_encoder import ResidualItemEncoder, anchor_loss, load_item_features

OUTPUT_DIR = Path("output_review")
EMBED_DIR  = Path("embeddings_i2i")
OUT_DIR    = Path("embeddings_i2i_chain")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_RESTAURANTS = "restaurants_only.csv"

# --- hyperparameters ---
EPOCHS          = 40
PAIRS_PER_EPOCH = 100_000
BATCH_SIZE      = 512
LR              = 1e-3
WEIGHT_DECAY    = 1e-5
WARMUP_EPOCHS   = 3
EVAL_EVERY      = 5
TEMPERATURE     = 0.07
LAMBDA_ANCHOR   = 0.5        # lighter anchor than the cuisine model: let brands tighten
GENERIC_TAGS    = {"restaurant", "store", "establishment"}
SEED            = 42

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


# ── labels: brand name (positives) + category (hard negatives) ────────────────

def load_labels(ids):
    rest = pd.read_parquet(OUTPUT_DIR / "restaurants.parquet")[["gmap_id", "name"]]
    name_map = dict(zip(rest["gmap_id"], rest["name"]))

    df = pd.read_csv(RAW_RESTAURANTS, engine="python", on_bad_lines="skip",
                     usecols=["gmap_id", "category"])
    df["category"] = df["category"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    cat_map = dict(zip(df["gmap_id"], df["category"]))

    names_by_row, cats_by_row = [], []
    for gid in ids:
        nm = name_map.get(gid, "")
        names_by_row.append(nm.lower().strip() if isinstance(nm, str) else "")
        raw = cat_map.get(gid, []) or []
        cats_by_row.append({c.lower().strip() for c in raw
                            if c.lower().strip() not in GENERIC_TAGS})
    return names_by_row, cats_by_row


def build_indexes(names_by_row, cats_by_row):
    name_to_rows, cat_to_rows = {}, {}
    for row, nm in enumerate(names_by_row):
        if nm:
            name_to_rows.setdefault(nm, []).append(row)
    for row, cats in enumerate(cats_by_row):
        for c in cats:
            cat_to_rows.setdefault(c, []).append(row)
    name_to_rows = {n: np.array(r) for n, r in name_to_rows.items() if len(r) >= 2}
    cat_to_rows  = {c: np.array(r) for c, r in cat_to_rows.items() if len(r) >= 2}
    chain_anchor_rows = np.array(sorted({r for rows in name_to_rows.values() for r in rows}))
    return name_to_rows, cat_to_rows, chain_anchor_rows


class ChainTripletDataset(Dataset):
    """anchor a, positive b (same brand), hard negative h (same cuisine, different brand)."""
    def __init__(self, feats, names_by_row, cats_by_row,
                 name_to_rows, cat_to_rows, anchor_rows, length):
        self.feats = feats
        self.names_by_row = names_by_row
        self.cats_by_row = cats_by_row
        self.name_to_rows = name_to_rows
        self.cat_to_rows = cat_to_rows
        self.anchor_rows = anchor_rows
        self.length = length

    def __len__(self):
        return self.length

    def _sample_hard_neg(self, a):
        cats = list(self.cats_by_row[a] & self.cat_to_rows.keys())
        if not cats:
            return int(rng.integers(len(self.feats)))      # fall back to random negative
        pool = self.cat_to_rows[cats[rng.integers(len(cats))]]
        for _ in range(6):
            h = int(pool[rng.integers(len(pool))])
            if self.names_by_row[h] != self.names_by_row[a]:
                return h
        return int(rng.integers(len(self.feats)))

    def __getitem__(self, _):
        a = int(rng.choice(self.anchor_rows))
        pool = self.name_to_rows[self.names_by_row[a]]
        b = int(pool[rng.integers(len(pool))])
        tries = 0
        while b == a and tries < 5:
            b = int(pool[rng.integers(len(pool))]); tries += 1
        h = self._sample_hard_neg(a)
        return self.feats[a], self.feats[b], self.feats[h]


def info_nce_with_hard(z_a, z_b, z_h, temperature=TEMPERATURE):
    """InfoNCE where candidates = in-batch positives PLUS each row's hard negative."""
    cand = torch.cat([z_b, z_h], dim=0)            # (2B, D): first B are the true partners
    logits = z_a @ cand.T / temperature            # (B, 2B)
    targets = torch.arange(z_a.size(0), device=z_a.device)
    return F.cross_entropy(logits, targets)


# ── proxy: chain-neighbor precision ───────────────────────────────────────────

@torch.no_grad()
def chain_neighbor_precision(emb, names_by_row, anchor_rows, n_query=1000, k=10):
    q = rng.choice(anchor_rows, min(n_query, len(anchor_rows)), replace=False)
    sims = emb[q] @ emb.T
    sims[np.arange(len(q)), q] = -np.inf
    topk = np.argpartition(-sims, k, axis=1)[:, :k]
    hits = total = 0
    for qi, i in enumerate(q):
        nm = names_by_row[i]
        for j in topk[qi]:
            total += 1
            if names_by_row[j] == nm:
                hits += 1
    return hits / total if total else 0.0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading frozen content vectors...")
    feats_np, ids, _ = load_item_features()
    print(f"  items: {feats_np.shape[0]}  dim: {feats_np.shape[1]}")

    print("Loading brand/category labels...")
    names_by_row, cats_by_row = load_labels(ids)
    name_to_rows, cat_to_rows, anchor_rows = build_indexes(names_by_row, cats_by_row)
    print(f"  chains (>=2 locations): {len(name_to_rows)}  chain items: {len(anchor_rows)}")

    feats = torch.from_numpy(feats_np)
    ds = ChainTripletDataset(feats, names_by_row, cats_by_row,
                             name_to_rows, cat_to_rows, anchor_rows, PAIRS_PER_EPOCH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    base = chain_neighbor_precision(feats_np, names_by_row, anchor_rows)
    print(f"\n[baseline frozen] chain-neighbor precision@10: {base:.3f}")

    model = ResidualItemEncoder(dim=feats_np.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    best_prec, best_state = -1.0, None
    print("\nTraining (chain / brand-supervised)...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xa, xb, xh in loader:
            za, zb, zh = model(xa), model(xb), model(xh)
            loss = info_nce_with_hard(za, zb, zh) + LAMBDA_ANCHOR * (
                anchor_loss(za, xa) + anchor_loss(zb, xb))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        scheduler.step()

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                emb = model(feats).numpy()
            prec = chain_neighbor_precision(emb, names_by_row, anchor_rows)
            print(f"  epoch {epoch:>2}  lr {scheduler.get_last_lr()[0]:.2e}  "
                  f"loss {total/len(loader):.4f}  gate {model.gate.item():+.3f}  "
                  f"chain_nn_prec@10 {prec:.3f}")
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

    print(f"\n[learned best] chain_nn_prec@10 {best_prec:.3f}  "
          f"(baseline {base:.3f}, lift {best_prec - base:+.3f})")
    np.save(OUT_DIR / "item_embeddings.npy", emb)
    np.save(OUT_DIR / "item_embedding_ids.npy", ids)
    print(f"\nSaved to {OUT_DIR}/  (dim {emb.shape[1]})")
    print(f"Evaluate:  I2I_EMBED_DIR={OUT_DIR} python eval_i2i.py")


if __name__ == "__main__":
    main()
