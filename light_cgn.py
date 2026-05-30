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
MODELS_DIR = Path("models/lightcgn128")
EMBEDDINGS_DIR = Path("embeddings")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIM =  128
N_LAYERS = 3
BATCH_SIZE = 2048
EPOCHS = 50
LR = 1e-3
NEG_SAMPLES = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── STEP 1: BUILD GRAPH ──────────────────────────────────────────────────────

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

    # build symmetric adjacency matrix
    # top-right block: users -> items
    # bottom-left block: items -> users
    row = np.concatenate([user_idx, item_idx + n_users])
    col = np.concatenate([item_idx + n_users, user_idx])
    data = np.concatenate([ratings, ratings])

    adj = sp.csr_matrix((data, (row, col)), shape=(n_users + n_items, n_users + n_items))

    # normalize: D^(-1/2) * A * D^(-1/2)
    degree = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    norm_adj = d_mat @ adj @ d_mat

    # convert to torch sparse
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


# ─── STEP 2: ADD RELATIVE_RESULTS EDGES ──────────────────────────────────────

def add_item_item_edges(train_df, item_to_idx):
    print("Adding item-item edges from relative_results...")
    try:
        restaurants = pd.read_parquet(OUTPUT_DIR / "restaurants.parquet")
        if "relative_results" not in restaurants.columns:
            return None
        edges = []
        for _, row in restaurants.iterrows():
            src = row["gmap_id"]
            if src not in item_to_idx:
                continue
            neighbors = row.get("relative_results", [])
            if not isinstance(neighbors, list):
                continue
            for tgt in neighbors:
                if tgt in item_to_idx:
                    edges.append((item_to_idx[src], item_to_idx[tgt]))
        if edges:
            print(f"Added {len(edges)} item-item edges")
            return edges
    except Exception as e:
        print(f"Could not add item-item edges: {e}")
    return None


# ─── STEP 3: LIGHTGCN MODEL ───────────────────────────────────────────────────

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, adj, embedding_dim=EMBEDDING_DIM, n_layers=N_LAYERS):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.adj = adj
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)

        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.adj, all_emb)
            embs.append(all_emb)

        # average across layers — key LightGCN design choice
        final_emb = torch.stack(embs, dim=1).mean(dim=1)
        user_final = final_emb[:self.n_users]
        item_final = final_emb[self.n_users:]
        return user_final, item_final

    def get_user_embedding(self, user_idx):
        user_final, _ = self.forward()
        return user_final[user_idx]

    def get_item_embedding(self, item_idx):
        _, item_final = self.forward()
        return item_final[item_idx]


# ─── STEP 4: BPR DATASET ──────────────────────────────────────────────────────

class BPRDataset(Dataset):
    def __init__(self, train_df, user_to_idx, item_to_idx, n_items):
        pos = train_df[train_df["rating"] >= 4][["user_id","gmap_id"]].copy()
        pos["user_idx"] = pos["user_id"].map(user_to_idx)
        pos["item_idx"] = pos["gmap_id"].map(item_to_idx)
        pos = pos.dropna(subset=["user_idx","item_idx"])
        self.pairs = pos[["user_idx","item_idx"]].values.astype(np.int64)
        self.n_items = n_items
        self.user_liked = (pos.groupby("user_idx")["item_idx"]
                           .apply(set).to_dict())
        self.rng = np.random.default_rng(42)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        user_idx, pos_idx = self.pairs[idx]
        liked = self.user_liked.get(user_idx, set())
        neg_idx = self.rng.integers(0, self.n_items)
        while neg_idx in liked:
            neg_idx = self.rng.integers(0, self.n_items)
        return (torch.tensor(user_idx, dtype=torch.long),
                torch.tensor(pos_idx,  dtype=torch.long),
                torch.tensor(neg_idx,  dtype=torch.long))


# ─── STEP 5: BPR LOSS ─────────────────────────────────────────────────────────

def bpr_loss(user_emb, pos_emb, neg_emb, l2_reg=1e-4):
    pos_scores = (user_emb * pos_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_emb).sum(dim=-1)
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    l2 = (user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / 2
    return loss + l2_reg * l2 / user_emb.shape[0]


# ─── STEP 6: TRAINING ─────────────────────────────────────────────────────────

def train_lightgcn(train_df, adj, n_users, n_items, user_to_idx, item_to_idx):
    dataset = BPRDataset(train_df, user_to_idx, item_to_idx, n_items)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=False)

    model     = LightGCN(n_users, n_items, adj).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    checkpoint_path = MODELS_DIR / "lightgcn_checkpoint.pt"
    start_epoch = 0

    if checkpoint_path.exists():
        print("Resuming from checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
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

    torch.save(model.state_dict(), MODELS_DIR / "lightgcn.pt")
    return model


# ─── STEP 7: PRECOMPUTE EMBEDDINGS ────────────────────────────────────────────

def save_embeddings(model, n_users, n_items):
    model.eval()
    with torch.no_grad():
        user_final, item_final = model()

    user_vecs = user_final.cpu().numpy()
    item_vecs = item_final.cpu().numpy()

    np.save(MODELS_DIR / "lgcn_user_vecs.npy", user_vecs)
    np.save(MODELS_DIR / "lgcn_item_vecs.npy", item_vecs)
    print(f"Saved: user_vecs {user_vecs.shape}, item_vecs {item_vecs.shape}")
    return user_vecs, item_vecs


# ─── STEP 8: INFERENCE ────────────────────────────────────────────────────────

def lgcn_recommend(user_id, n=20):
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

    train_df = pd.read_parquet(OUTPUT_DIR / "train.parquet")
    print(f"Train rows: {len(train_df)}")

    adj, n_users, n_items, user_map, item_map, user_to_idx, item_to_idx = build_graph(train_df)

    print("Training LightGCN...")
    model = train_lightgcn(train_df, adj, n_users, n_items, user_to_idx, item_to_idx)

    print("Saving embeddings...")
    save_embeddings(model, n_users, n_items)

    print("LightGCN complete.")


if __name__ == "__main__":
    run()
