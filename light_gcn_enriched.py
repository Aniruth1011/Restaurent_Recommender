import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
import scipy.sparse as sp

OUTPUT_DIR = Path("output")
MODELS_DIR = Path("models/lgcn_enriched")
EMBEDDINGS_DIR = Path("embeddings")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIM = 64
N_LAYERS = 3
BATCH_SIZE = 2048
EPOCHS = 50
LR = 1e-3
HARD_NEG_POOL = 20
HARD_NEG_RATIO = 0.25
HARD_NEG_REFRESH = 10  # refresh pools every N epochs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── STEP 1: LOAD SIDE FEATURES ───────────────────────────────────────────────

def load_side_features():
    biz_ids = np.load(EMBEDDINGS_DIR / "business_gmap_ids.npy", allow_pickle=True)
    biz_embs = np.load(EMBEDDINGS_DIR / "business_embeddings.npy")
    biz_emb_map = dict(zip(biz_ids, biz_embs))

    user_ids = np.load(EMBEDDINGS_DIR / "user_ids.npy", allow_pickle=True)
    user_embs = np.load(EMBEDDINGS_DIR / "user_embeddings.npy")
    user_emb_map = dict(zip(user_ids, user_embs))

    item_struct = pd.read_parquet(Path("models") / "item_struct_features.parquet")
    user_struct = pd.read_parquet(Path("models") / "user_struct_features.parquet")

    item_struct_map = {row["gmap_id"]: row.drop("gmap_id").values.astype(np.float32)
                       for _, row in item_struct.iterrows()}
    user_struct_map = {row["user_id"]: row.drop("user_id").values.astype(np.float32)
                       for _, row in user_struct.iterrows()}

    text_dim = biz_embs.shape[1]
    item_struct_dim = len(item_struct.columns) - 1
    user_struct_dim = len(user_struct.columns) - 1

    print(f"Text dim: {text_dim}, item struct dim: {item_struct_dim}, user struct dim: {user_struct_dim}")
    return biz_emb_map, user_emb_map, item_struct_map, user_struct_map, text_dim, item_struct_dim, user_struct_dim


# ─── STEP 2: BUILD GRAPH ──────────────────────────────────────────────────────

def build_graph(train_df):
    print("Building user-item graph...")
    users = train_df["user_id"].astype("category")
    items = train_df["gmap_id"].astype("category")

    user_map = dict(enumerate(users.cat.categories))
    item_map = dict(enumerate(items.cat.categories))
    user_to_idx = {v:k for k,v in user_map.items()}
    item_to_idx = {v:k for k,v in item_map.items()}

    n_users = len(user_map)
    n_items = len(item_map)

    user_idx = train_df["user_id"].map(user_to_idx).values
    item_idx = train_df["gmap_id"].map(item_to_idx).values
    ratings  = (train_df["recency_weight"] * train_df["rating"]).values

    row  = np.concatenate([user_idx, item_idx + n_users])
    col  = np.concatenate([item_idx + n_users, user_idx])
    data = np.concatenate([ratings, ratings])

    adj = sp.csr_matrix((data, (row, col)), shape=(n_users + n_items, n_users + n_items))

    degree = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    norm_adj = d_mat @ adj @ d_mat

    norm_adj = norm_adj.tocoo()
    indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
    values  = torch.FloatTensor(norm_adj.data)
    shape   = torch.Size(norm_adj.shape)
    adj_tensor = torch.sparse_coo_tensor(indices, values, shape).to(DEVICE)

    with open(MODELS_DIR / "lgcn_user_map.pkl", "wb") as f:
        pickle.dump((user_map, user_to_idx), f)
    with open(MODELS_DIR / "lgcn_item_map.pkl", "wb") as f:
        pickle.dump((item_map, item_to_idx), f)

    print(f"Graph: {n_users} users, {n_items} items, {len(norm_adj.data)} edges")
    return adj_tensor, n_users, n_items, user_map, item_map, user_to_idx, item_to_idx


# ─── STEP 3: BUILD INITIAL EMBEDDINGS FROM SIDE FEATURES ─────────────────────

def build_initial_embeddings(n_users, n_items, user_map, item_map,
                              biz_emb_map, user_emb_map,
                              item_struct_map, user_struct_map,
                              text_dim, item_struct_dim, user_struct_dim,
                              embedding_dim):
    # item: concat(text_emb, struct) -> project to embedding_dim
    item_features = []
    for i in range(n_items):
        gmap_id = item_map[i]
        text = biz_emb_map.get(gmap_id, np.zeros(text_dim, dtype=np.float32))
        struct = item_struct_map.get(gmap_id, np.zeros(item_struct_dim, dtype=np.float32))
        item_features.append(np.concatenate([text, struct]))
    item_features = np.stack(item_features).astype(np.float32)

    # user: concat(text_emb, struct) -> project to embedding_dim
    user_features = []
    for i in range(n_users):
        user_id = user_map[i]
        text = user_emb_map.get(user_id, np.zeros(text_dim, dtype=np.float32))
        struct = user_struct_map.get(user_id, np.zeros(user_struct_dim, dtype=np.float32))
        user_features.append(np.concatenate([text, struct]))
    user_features = np.stack(user_features).astype(np.float32)

    print(f"Item feature dim: {item_features.shape[1]}, User feature dim: {user_features.shape[1]}")
    return torch.tensor(item_features, dtype=torch.float32), torch.tensor(user_features, dtype=torch.float32)


# ─── STEP 4: FEATURE-ENRICHED LIGHTGCN ───────────────────────────────────────

class FeatureEnrichedLightGCN(nn.Module):
    def __init__(self, n_users, n_items, adj,
                 user_features, item_features,
                 embedding_dim=EMBEDDING_DIM, n_layers=N_LAYERS):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.adj = adj
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

        user_feat_dim = user_features.shape[1]
        item_feat_dim = item_features.shape[1]

        # feature projection layers
        self.user_proj = nn.Sequential(
            nn.Linear(user_feat_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.item_proj = nn.Sequential(
            nn.Linear(item_feat_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # register features as buffers (not trained, just stored)
        self.register_buffer("user_features", user_features)
        self.register_buffer("item_features", item_features)

        # residual ID embeddings — learned on top of features
        self.user_id_emb = nn.Embedding(n_users, embedding_dim)
        self.item_id_emb = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_id_emb.weight, std=0.01)
        nn.init.normal_(self.item_id_emb.weight, std=0.01)

    def get_base_embeddings(self):
        user_emb = self.user_proj(self.user_features) + self.user_id_emb.weight
        item_emb = self.item_proj(self.item_features) + self.item_id_emb.weight
        return user_emb, item_emb

    def forward(self):
        user_emb, item_emb = self.get_base_embeddings()
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.adj, all_emb)
            embs.append(all_emb)

        final_emb = torch.stack(embs, dim=1).mean(dim=1)
        return final_emb[:self.n_users], final_emb[self.n_users:]


# ─── STEP 5: BPR DATASET WITH HARD NEGATIVES ─────────────────────────────────

class BPRDataset(Dataset):
    def __init__(self, train_df, user_to_idx, item_to_idx, n_items, hard_neg_pools=None):
        pos = train_df[train_df["rating"] >= 4][["user_id","gmap_id"]].copy()
        pos["user_idx"] = pos["user_id"].map(user_to_idx)
        pos["item_idx"] = pos["gmap_id"].map(item_to_idx)
        pos = pos.dropna(subset=["user_idx","item_idx"])
        self.pairs = pos[["user_idx","item_idx"]].values.astype(np.int64)
        self.n_items = n_items
        self.hard_neg_pools = hard_neg_pools
        self.n_hard = int(1 * HARD_NEG_RATIO)
        self.user_liked = pos.groupby("user_idx")["item_idx"].apply(set).to_dict()
        self.all_items = np.arange(n_items)
        self.rng = np.random.default_rng(42)

    def _get_neg(self, user_idx):
        liked = self.user_liked.get(user_idx, set())
        if self.hard_neg_pools and user_idx in self.hard_neg_pools:
            pool = self.hard_neg_pools[user_idx]
            if pool:
                return int(self.rng.choice(pool))
        neg = int(self.rng.integers(0, self.n_items))
        while neg in liked:
            neg = int(self.rng.integers(0, self.n_items))
        return neg

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        user_idx, pos_idx = self.pairs[idx]
        neg_idx = self._get_neg(user_idx)
        return (torch.tensor(user_idx, dtype=torch.long),
                torch.tensor(pos_idx,  dtype=torch.long),
                torch.tensor(neg_idx,  dtype=torch.long))


# ─── STEP 6: HARD NEGATIVE MINING ────────────────────────────────────────────

def precompute_hard_neg_pools(model, n_users, user_liked):
    print("Precomputing hard negative pools...")
    model.eval()
    with torch.no_grad():
        _, item_vecs = model()
        item_vecs = item_vecs.cpu().numpy()

    pools = {}
    batch_size = 1000

    with torch.no_grad():
        user_vecs_all, _ = model()
        user_vecs_all = user_vecs_all.cpu().numpy()

    for i in range(0, n_users, batch_size):
        batch_u_vecs = user_vecs_all[i:i+batch_size]
        scores = batch_u_vecs @ item_vecs.T

        for j, u_idx in enumerate(range(i, min(i+batch_size, n_users))):
            liked = user_liked.get(u_idx, set())
            top_idx = np.argsort(scores[j])[::-1]
            pool = [int(k) for k in top_idx if k not in liked][:HARD_NEG_POOL]
            pools[u_idx] = pool

    model.train()
    return pools


# ─── STEP 7: BPR LOSS ─────────────────────────────────────────────────────────

def bpr_loss(user_emb, pos_emb, neg_emb, l2_reg=1e-4):
    pos_scores = (user_emb * pos_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_emb).sum(dim=-1)
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    l2 = (user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / 2
    return loss + l2_reg * l2 / user_emb.shape[0]


# ─── STEP 8: TRAINING ─────────────────────────────────────────────────────────

def train(train_df, adj, n_users, n_items,
          user_map, item_map, user_to_idx, item_to_idx,
          user_features, item_features):

    user_liked = (train_df[train_df["rating"] >= 4]
                  .assign(user_idx=train_df["user_id"].map(user_to_idx),
                          item_idx=train_df["gmap_id"].map(item_to_idx))
                  .dropna(subset=["user_idx","item_idx"])
                  .groupby("user_idx")["item_idx"].apply(set).to_dict())

    model = FeatureEnrichedLightGCN(
        n_users, n_items, adj,
        user_features.to(DEVICE),
        item_features.to(DEVICE),
        embedding_dim=EMBEDDING_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    checkpoint_path = MODELS_DIR / "checkpoint.pt"
    start_epoch = 0
    hard_neg_pools = None

    if checkpoint_path.exists():
        print("Resuming from checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        if epoch % HARD_NEG_REFRESH == 0:
            hard_neg_pools = precompute_hard_neg_pools(model, n_users, user_liked)

        dataset = BPRDataset(train_df, user_to_idx, item_to_idx, n_items, hard_neg_pools)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, pin_memory=False)

        model.train()
        total_loss = 0

        for user_idx, pos_idx, neg_idx in loader:
            user_idx = user_idx.to(DEVICE)
            pos_idx  = pos_idx.to(DEVICE)
            neg_idx  = neg_idx.to(DEVICE)

            user_final, item_final = model()
            u_emb   = user_final[user_idx]
            pos_emb = item_final[pos_idx]
            neg_emb = item_final[neg_idx]

            loss = bpr_loss(u_emb, pos_emb, neg_emb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.4f}")
        scheduler.step()

        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss":      avg_loss,
        }, checkpoint_path)

    torch.save(model.state_dict(), MODELS_DIR / "lightgcn_enriched.pt")
    return model


# ─── STEP 9: SAVE EMBEDDINGS ──────────────────────────────────────────────────

def save_embeddings(model):
    model.eval()
    with torch.no_grad():
        user_vecs, item_vecs = model()
    user_vecs = user_vecs.cpu().numpy()
    item_vecs = item_vecs.cpu().numpy()
    np.save(MODELS_DIR / "lgcn_user_vecs.npy", user_vecs)
    np.save(MODELS_DIR / "lgcn_item_vecs.npy", item_vecs)
    print(f"Saved: user_vecs {user_vecs.shape}, item_vecs {item_vecs.shape}")
    return user_vecs, item_vecs


# ─── STEP 10: INFERENCE ───────────────────────────────────────────────────────

def lgcn_enriched_recommend(user_id, n=20):
    with open(MODELS_DIR / "lgcn_user_map.pkl", "rb") as f:
        _, user_to_idx = pickle.load(f)
    with open(MODELS_DIR / "lgcn_item_map.pkl", "rb") as f:
        item_map, _ = pickle.load(f)

    if user_id not in user_to_idx:
        return []

    user_vecs = np.load(MODELS_DIR / "lgcn_user_vecs.npy")
    item_vecs = np.load(MODELS_DIR / "lgcn_item_vecs.npy")

    u_idx  = user_to_idx[user_id]
    u_vec  = user_vecs[u_idx]
    scores = item_vecs @ u_vec
    top_idx = np.argsort(scores)[::-1][:n]
    return [(item_map[i], float(scores[i])) for i in top_idx]


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading side features...")
    (biz_emb_map, user_emb_map, item_struct_map, user_struct_map,
     text_dim, item_struct_dim, user_struct_dim) = load_side_features()

    train_df = pd.read_parquet(OUTPUT_DIR / "train.parquet")
    print(f"Train rows: {len(train_df)}")

    adj, n_users, n_items, user_map, item_map, user_to_idx, item_to_idx = build_graph(train_df)

    print("Building initial embeddings from side features...")
    item_features, user_features = build_initial_embeddings(
        n_users, n_items, user_map, item_map,
        biz_emb_map, user_emb_map,
        item_struct_map, user_struct_map,
        text_dim, item_struct_dim, user_struct_dim,
        EMBEDDING_DIM
    )

    print("Training Feature-Enriched LightGCN...")
    model = train(
        train_df, adj, n_users, n_items,
        user_map, item_map, user_to_idx, item_to_idx,
        user_features, item_features
    )

    print("Saving embeddings...")
    save_embeddings(model)
    print("Feature-Enriched LightGCN complete.")


if __name__ == "__main__":
    run()
