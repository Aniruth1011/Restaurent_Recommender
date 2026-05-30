import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path("output")
MODELS_DIR = Path("models")
EMBEDDINGS_DIR = Path("embeddings")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIM = 384
HIDDEN_DIM = 256
OUTPUT_DIM = 128
BATCH_SIZE = 1024
EPOCHS = 20
LR = 1e-3
NEG_SAMPLES = 4
HARD_NEG_RATIO = 0.5  # 50% hard negatives, 50% random
HARD_NEG_POOL = 50    # sample from top-50 scoring items as hard negative candidates
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── STEP 1: BUSINESS EMBEDDINGS ─────────────────────────────────────────────

def build_business_embeddings():
    import pyarrow.parquet as pq
    import gc

    print(f"Building business embeddings on {DEVICE}...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    business_embeddings = {}

    pf = pq.ParquetFile("processed/raw_reviews.parquet")
    now = pd.Timestamp.now().timestamp() * 1000

    for batch in pf.iter_batches(batch_size=10000, columns=["gmap_id","text","time","rating"]):
        chunk = batch.to_pandas()
        chunk["text"] = chunk["text"].fillna("no review").apply(
            lambda x: x if isinstance(x, str) and x.strip() else "no review"
        )
        chunk["days"] = (now - chunk["time"]) / (1000 * 86400)
        chunk["recency_weight"] = np.exp(-0.001 * chunk["days"])
        chunk["weight"] = (chunk["recency_weight"] * chunk["rating"]).fillna(1.0)

        embeddings = encoder.encode(chunk["text"].tolist(), batch_size=64,
                                    show_progress_bar=False, convert_to_numpy=True)
        embeddings = np.nan_to_num(embeddings, nan=0.0)

        for i, (_, row) in enumerate(chunk.iterrows()):
            gid = row["gmap_id"]
            w = row["weight"]
            if gid not in business_embeddings:
                business_embeddings[gid] = [np.zeros(EMBEDDING_DIM), 0.0]
            business_embeddings[gid][0] += w * embeddings[i]
            business_embeddings[gid][1] += w

        del embeddings, chunk
        gc.collect()

    gmap_ids = np.array(list(business_embeddings.keys()))
    embeddings_matrix = np.stack([
        v[0] / v[1] if v[1] > 0 else np.zeros(EMBEDDING_DIM)
        for v in business_embeddings.values()
    ]).astype(np.float32)

    np.save(EMBEDDINGS_DIR / "business_gmap_ids.npy", gmap_ids)
    np.save(EMBEDDINGS_DIR / "business_embeddings.npy", embeddings_matrix)
    print(f"Business embeddings: {embeddings_matrix.shape}")
    return gmap_ids, embeddings_matrix
# ─── STEP 2: USER EMBEDDINGS ──────────────────────────────────────────────────

def build_user_embeddings():
    import pyarrow.parquet as pq
    import gc

    biz_ids = np.load(EMBEDDINGS_DIR / "business_gmap_ids.npy", allow_pickle=True)
    biz_embs = np.load(EMBEDDINGS_DIR / "business_embeddings.npy")
    biz_emb_map = dict(zip(biz_ids, biz_embs))

    pf = pq.ParquetFile(OUTPUT_DIR / "train.parquet")
    user_embeddings = {}

    for batch in pf.iter_batches(batch_size=50000, columns=["user_id","gmap_id","rating","recency_weight"]):
        chunk = batch.to_pandas()
        chunk["weight"] = chunk["rating"] * chunk["recency_weight"]
        for _, row in chunk.iterrows():
            uid = row["user_id"]
            gid = row["gmap_id"]
            w = row["weight"]
            if gid not in biz_emb_map:
                continue
            if uid not in user_embeddings:
                user_embeddings[uid] = [np.zeros(EMBEDDING_DIM), 0.0]
            user_embeddings[uid][0] += w * biz_emb_map[gid]
            user_embeddings[uid][1] += w
        del chunk
        gc.collect()

    user_ids = np.array(list(user_embeddings.keys()))
    user_embs = np.stack([
        v[0] / v[1] if v[1] > 0 else np.zeros(EMBEDDING_DIM)
        for v in user_embeddings.values()
    ]).astype(np.float32)

    np.save(EMBEDDINGS_DIR / "user_ids.npy", user_ids)
    np.save(EMBEDDINGS_DIR / "user_embeddings.npy", user_embs)
    print(f"User embeddings: {user_embs.shape}")
    return user_ids, user_embs


# ─── STEP 3: STRUCTURED FEATURES ─────────────────────────────────────────────

def build_structured_features():
    restaurants = pd.read_parquet(OUTPUT_DIR / "restaurants.parquet")
    user_profiles = pd.read_parquet(OUTPUT_DIR / "user_profiles.parquet")

    misc_cols = ["has_delivery","has_dine_in","has_takeout","has_drive_through",
                 "serves_alcohol","has_outdoor_seating","is_wheelchair",
                 "accepts_reservations","has_live_music","kid_friendly"]
    cat_cols = [c for c in restaurants.columns if c.startswith("cat_")]
    num_cols = ["price_numeric","weighted_avg_rating","review_count","has_response_rate"]
    item_struct_cols = [c for c in misc_cols + cat_cols + num_cols if c in restaurants.columns]

    item_struct = restaurants[["gmap_id"] + item_struct_cols].fillna(0)
    user_struct_cols = [c for c in user_profiles.columns if c not in ["user_id","fav_category"]]
    user_struct = user_profiles[["user_id"] + user_struct_cols].fillna(0)

    item_struct.to_parquet(MODELS_DIR / "item_struct_features.parquet", index=False)
    user_struct.to_parquet(MODELS_DIR / "user_struct_features.parquet", index=False)
    return item_struct, user_struct


# ─── STEP 4: DATASET WITH HARD NEGATIVE MINING ───────────────────────────────

class TwoTowerDataset(Dataset):
    def __init__(self, train_df, user_emb_map, item_emb_map,
                 user_struct_map, item_struct_map,
                 all_item_ids, user_struct_dim, item_struct_dim,
                 item_vecs=None, n_neg=NEG_SAMPLES):
        self.pairs = train_df[train_df["rating"] >= 4][["user_id","gmap_id"]].values
        self.user_emb_map = user_emb_map
        self.item_emb_map = item_emb_map
        self.user_struct_map = user_struct_map
        self.item_struct_map = item_struct_map
        self.all_item_ids = np.array(all_item_ids)
        self.user_struct_dim = user_struct_dim
        self.item_struct_dim = item_struct_dim
        self.item_vecs = item_vecs  # (n_items, output_dim) — used for hard negatives
        self.item_id_to_idx = {gid: i for i, gid in enumerate(all_item_ids)}
        self.n_neg = n_neg
        self.n_hard = int(n_neg * HARD_NEG_RATIO)
        self.n_random = n_neg - self.n_hard
        self.rng = np.random.default_rng(42)

        # build user -> set of liked items for exclusion
        self.user_liked = train_df[train_df["rating"] >= 4].groupby("user_id")["gmap_id"].apply(set).to_dict()

    # def _get_hard_negatives(self, user_id, n):
    #     if self.item_vecs is None:
    #         return self.rng.choice(self.all_item_ids, n, replace=False)

    #     user_emb = self.user_emb_map.get(user_id, np.zeros(EMBEDDING_DIM, dtype=np.float32))
    #     user_struct = self.user_struct_map.get(user_id, np.zeros(self.user_struct_dim, dtype=np.float32))
    #     user_vec = np.concatenate([user_emb, user_struct])

    #     # approximate user vector as raw concat for fast scoring during data loading
    #     # real scoring uses the towers but that's too slow here
    #     scores = self.item_vecs @ user_emb  # (n_items,) fast dot product
    #     liked = self.user_liked.get(user_id, set())

    #     # get top HARD_NEG_POOL indices, exclude liked items
    #     top_idx = np.argsort(scores)[::-1]
    #     hard_pool = [self.all_item_ids[i] for i in top_idx if self.all_item_ids[i] not in liked][:HARD_NEG_POOL]

    #     if len(hard_pool) < n:
    #         return self.rng.choice(self.all_item_ids, n, replace=False)
    #     return self.rng.choice(hard_pool, n, replace=False)

    def _get_hard_negatives(self, user_id, n):
        if self.item_vecs is None:
            return self.rng.choice(self.all_item_ids, n, replace=False)
    
        user_emb = self.user_emb_map.get(user_id, np.zeros(EMBEDDING_DIM, dtype=np.float32))
        
        # item_vecs could be 128-dim (tower output) or 384-dim (raw embeddings)
        # normalize user_emb to match item_vecs dimension
        if self.item_vecs.shape[1] == EMBEDDING_DIM:
            query = user_emb
        else:
            # item_vecs are tower outputs (128-dim), project user_emb down
            # use raw embedding dot product as approximation instead
            query = user_emb[:self.item_vecs.shape[1]]
    
        scores = self.item_vecs @ query
        liked = self.user_liked.get(user_id, set())
        top_idx = np.argsort(scores)[::-1]
        hard_pool = [self.all_item_ids[i] for i in top_idx if self.all_item_ids[i] not in liked][:HARD_NEG_POOL]
    
        if len(hard_pool) < n:
            return self.rng.choice(self.all_item_ids, n, replace=False)
        return self.rng.choice(hard_pool, n, replace=False)
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        user_id, pos_item_id = self.pairs[idx]
        liked = self.user_liked.get(user_id, set())

        user_emb = self.user_emb_map.get(user_id, np.zeros(EMBEDDING_DIM, dtype=np.float32))
        user_struct = self.user_struct_map.get(user_id, np.zeros(self.user_struct_dim, dtype=np.float32))
        pos_emb = self.item_emb_map.get(pos_item_id, np.zeros(EMBEDDING_DIM, dtype=np.float32))
        pos_struct = self.item_struct_map.get(pos_item_id, np.zeros(self.item_struct_dim, dtype=np.float32))

        # hard negatives
        hard_ids = self._get_hard_negatives(user_id, self.n_hard)
        # random negatives
        rand_ids = []
        while len(rand_ids) < self.n_random:
            cand = self.rng.choice(self.all_item_ids)
            if cand not in liked:
                rand_ids.append(cand)

        neg_ids = list(hard_ids) + rand_ids

        neg_embs = np.stack([self.item_emb_map.get(n, np.zeros(EMBEDDING_DIM, dtype=np.float32)) for n in neg_ids])
        neg_structs = np.stack([self.item_struct_map.get(n, np.zeros(self.item_struct_dim, dtype=np.float32)) for n in neg_ids])

        return {
            "user_emb": torch.tensor(user_emb, dtype=torch.float32),
            "user_struct": torch.tensor(user_struct, dtype=torch.float32),
            "pos_emb": torch.tensor(pos_emb, dtype=torch.float32),
            "pos_struct": torch.tensor(pos_struct, dtype=torch.float32),
            "neg_embs": torch.tensor(neg_embs, dtype=torch.float32),
            "neg_structs": torch.tensor(neg_structs, dtype=torch.float32),
        }


# ─── STEP 5: TWO TOWER MODEL ──────────────────────────────────────────────────

class Tower(nn.Module):
    def __init__(self, emb_dim, struct_dim, output_dim=OUTPUT_DIM):
        super().__init__()
        input_dim = emb_dim + struct_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, output_dim),
        )

    def forward(self, emb, struct):
        x = torch.cat([emb, struct], dim=-1)
        return F.normalize(self.net(x), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, user_struct_dim, item_struct_dim):
        super().__init__()
        self.user_tower = Tower(EMBEDDING_DIM, user_struct_dim)
        self.item_tower = Tower(EMBEDDING_DIM, item_struct_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, user_emb, user_struct, item_emb, item_struct):
        u = self.user_tower(user_emb, user_struct)
        v = self.item_tower(item_emb, item_struct)
        return (u * v).sum(dim=-1) / self.temperature.clamp(min=0.01)

    def encode_user(self, user_emb, user_struct):
        return self.user_tower(user_emb, user_struct)

    def encode_item(self, item_emb, item_struct):
        return self.item_tower(item_emb, item_struct)


# ─── STEP 6: BPR LOSS + TRAINING ──────────────────────────────────────────────

def bpr_loss(pos_scores, neg_scores):
    return -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()


# def train_two_tower(train_df, user_emb_map, item_emb_map,
#                     user_struct_map, item_struct_map,
#                     user_struct_dim, item_struct_dim):
#     all_item_ids = list(item_emb_map.keys())
#     item_vecs_raw = np.stack([item_emb_map[gid] for gid in all_item_ids]).astype(np.float32)

#     # warm up: first few epochs with random negatives only
#     # then switch to hard negatives after model has learned something
#     def make_dataset(use_hard):
#         vecs = item_vecs_raw if use_hard else None
#         ds = TwoTowerDataset(train_df, user_emb_map, item_emb_map,
#                              user_struct_map, item_struct_map,
#                              all_item_ids, user_struct_dim, item_struct_dim,
#                              item_vecs=vecs)
#         return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
#                           num_workers=4, pin_memory=True)

#     model = TwoTowerModel(user_struct_dim, item_struct_dim).to(DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

#     warmup_epochs = 5  # random negatives for first 5 epochs

#     for epoch in range(EPOCHS):
#         use_hard = epoch >= warmup_epochs
#         loader = make_dataset(use_hard)

#         # refresh item vecs for hard negative mining every 5 epochs
#         if use_hard and (epoch - warmup_epochs) % 5 == 0:
#             import time
            
#             print("Building tensors...")
#             t0 = time.time()
            
#             all_embs = torch.tensor(item_vecs_raw, dtype=torch.float32).to(DEVICE)
#             all_structs = torch.tensor(
#                 np.stack([
#                     item_struct_map.get(gid, np.zeros(item_struct_dim))
#                     for gid in all_item_ids
#                 ]),
#                 dtype=torch.float32
#             ).to(DEVICE)
            
#             print(f"Tensor creation took {time.time()-t0:.2f}s")
            
#             t0 = time.time()
#             with torch.no_grad():
#                 updated_vecs = model.encode_item(all_embs, all_structs)
            
#             print(f"encode_item took {time.time()-t0:.2f}s")
            
#             t0 = time.time()
#             updated_vecs = updated_vecs.cpu().numpy()
            
#             print(f"cpu().numpy() took {time.time()-t0:.2f}s")

#             # print(f"Epoch {epoch+1}: refreshing item vectors...")
#             # model.eval()
#             # all_embs = torch.tensor(item_vecs_raw, dtype=torch.float32).to(DEVICE)
#             # all_structs = torch.tensor(
#             #     np.stack([item_struct_map.get(gid, np.zeros(item_struct_dim)) for gid in all_item_ids]),
#             #     dtype=torch.float32
#             # ).to(DEVICE)
#             # with torch.no_grad():
#             #     updated_vecs = model.encode_item(all_embs, all_structs).cpu().numpy()  # 128-dim
#             # loader.dataset.item_vecs = item_vecs_raw  # keep using 384-dim raw for scoring

#         model.train()
#         total_loss = 0
#         for batch in loader:
#             user_emb = batch["user_emb"].to(DEVICE)
#             user_struct = batch["user_struct"].to(DEVICE)
#             pos_emb = batch["pos_emb"].to(DEVICE)
#             pos_struct = batch["pos_struct"].to(DEVICE)
#             neg_embs = batch["neg_embs"].to(DEVICE)
#             neg_structs = batch["neg_structs"].to(DEVICE)

#             pos_scores = model(user_emb, user_struct, pos_emb, pos_struct)
#             neg_scores = torch.stack([
#                 model(user_emb, user_struct, neg_embs[:,k,:], neg_structs[:,k,:])
#                 for k in range(NEG_SAMPLES)
#             ], dim=1)

#             loss = bpr_loss(pos_scores, neg_scores)
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             total_loss += loss.item()

#         mode = "hard+random" if use_hard else "random"
#         print(f"Epoch {epoch+1}/{EPOCHS} [{mode}] — loss: {total_loss/len(loader):.4f}")
#         scheduler.step()

#     torch.save(model.state_dict(), MODELS_DIR / "two_tower.pt")
#     return model


def train_two_tower(train_df, user_emb_map, item_emb_map,
                    user_struct_map, item_struct_map,
                    user_struct_dim, item_struct_dim):
    all_item_ids = list(item_emb_map.keys())
    item_vecs_raw = np.stack([item_emb_map[gid] for gid in all_item_ids]).astype(np.float32)

    def make_dataset(use_hard):
        vecs = item_vecs_raw if use_hard else None
        ds = TwoTowerDataset(train_df, user_emb_map, item_emb_map,
                             user_struct_map, item_struct_map,
                             all_item_ids, user_struct_dim, item_struct_dim,
                             item_vecs=vecs)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)

    model = TwoTowerModel(user_struct_dim, item_struct_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    checkpoint_path = MODELS_DIR / "two_tower_checkpoint.pt"
    start_epoch = 0

    if checkpoint_path.exists():
        print("Resuming from checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    warmup_epochs = 5

    for epoch in range(start_epoch, EPOCHS):
        use_hard = epoch >= warmup_epochs
        loader = make_dataset(use_hard)

        if use_hard and (epoch - warmup_epochs) % 5 == 0:
            print(f"Epoch {epoch+1}: refreshing item vectors...")
            model.eval()
            all_embs = torch.tensor(item_vecs_raw, dtype=torch.float32).to(DEVICE)
            all_structs = torch.tensor(
                np.stack([item_struct_map.get(gid, np.zeros(item_struct_dim)) for gid in all_item_ids]),
                dtype=torch.float32
            ).to(DEVICE)
            with torch.no_grad():
                updated_vecs = model.encode_item(all_embs, all_structs).cpu().numpy()
            loader.dataset.item_vecs = item_vecs_raw

        model.train()
        total_loss = 0
        for batch in loader:
            user_emb = batch["user_emb"].to(DEVICE)
            user_struct = batch["user_struct"].to(DEVICE)
            pos_emb = batch["pos_emb"].to(DEVICE)
            pos_struct = batch["pos_struct"].to(DEVICE)
            neg_embs = batch["neg_embs"].to(DEVICE)
            neg_structs = batch["neg_structs"].to(DEVICE)

            pos_scores = model(user_emb, user_struct, pos_emb, pos_struct)
            neg_scores = torch.stack([
                model(user_emb, user_struct, neg_embs[:,k,:], neg_structs[:,k,:])
                for k in range(NEG_SAMPLES)
            ], dim=1)

            loss = bpr_loss(pos_scores, neg_scores)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        mode = "hard+random" if use_hard else "random"
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} [{mode}] — loss: {avg_loss:.4f}")
        scheduler.step()

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss": avg_loss,
        }, checkpoint_path)

    torch.save(model.state_dict(), MODELS_DIR / "two_tower.pt")
    return model
# ─── STEP 7: BUILD ITEM INDEX ─────────────────────────────────────────────────

def build_item_index(model, item_emb_map, item_struct_map, item_struct_dim):
    model.eval()
    gmap_ids = list(item_emb_map.keys())

    all_embs = torch.tensor(
        np.stack([item_emb_map.get(gid, np.zeros(EMBEDDING_DIM)) for gid in gmap_ids]),
        dtype=torch.float32
    ).to(DEVICE)
    all_structs = torch.tensor(
        np.stack([item_struct_map.get(gid, np.zeros(item_struct_dim)) for gid in gmap_ids]),
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        item_vecs = model.encode_item(all_embs, all_structs).cpu().numpy()

    np.save(EMBEDDINGS_DIR / "item_index_vecs.npy", item_vecs)
    np.save(EMBEDDINGS_DIR / "item_index_ids.npy", np.array(gmap_ids))

    try:
        import faiss
        index = faiss.IndexFlatIP(OUTPUT_DIM)
        index.add(item_vecs)
        faiss.write_index(index, str(EMBEDDINGS_DIR / "faiss_index.bin"))
        print(f"Faiss index: {index.ntotal} items")
    except ImportError:
        print("Faiss not installed, using numpy search")

    return item_vecs, gmap_ids


# ─── STEP 8: INFERENCE ────────────────────────────────────────────────────────

def tt_recommend(user_id, n=20):
    with open(MODELS_DIR / "user_emb_map.pkl", "rb") as f:
        user_emb_map = pickle.load(f)
    with open(MODELS_DIR / "user_struct_map.pkl", "rb") as f:
        user_struct_map = pickle.load(f)

    user_struct_dim = next(iter(user_struct_map.values())).shape[0]
    item_struct_dim = pd.read_parquet(MODELS_DIR / "item_struct_features.parquet").shape[1] - 1

    model = TwoTowerModel(user_struct_dim, item_struct_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / "two_tower.pt", map_location=DEVICE))
    model.eval()

    item_vecs = np.load(EMBEDDINGS_DIR / "item_index_vecs.npy")
    item_ids = np.load(EMBEDDINGS_DIR / "item_index_ids.npy", allow_pickle=True)

    user_emb = user_emb_map.get(user_id, np.zeros(EMBEDDING_DIM, dtype=np.float32))
    user_struct = user_struct_map.get(user_id)
    if user_struct is None:
        return []

    u_emb_t = torch.tensor(user_emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    u_struct_t = torch.tensor(user_struct, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        user_vec = model.encode_user(u_emb_t, u_struct_t).cpu().numpy()

    try:
        import faiss
        index = faiss.read_index(str(EMBEDDINGS_DIR / "faiss_index.bin"))
        scores, indices = index.search(user_vec, n)
        return [(item_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
    except ImportError:
        scores = (item_vecs @ user_vec.T).squeeze()
        top_idx = np.argsort(scores)[::-1][:n]
        return [(item_ids[i], float(scores[i])) for i in top_idx]
def run():
    if not (EMBEDDINGS_DIR / "business_embeddings.npy").exists():
        biz_ids, biz_embs = build_business_embeddings()
    else:
        print("Loading existing business embeddings...")
        biz_ids = np.load(EMBEDDINGS_DIR / "business_gmap_ids.npy", allow_pickle=True)
        biz_embs = np.load(EMBEDDINGS_DIR / "business_embeddings.npy")
    biz_emb_map = dict(zip(biz_ids, biz_embs))

    if not (EMBEDDINGS_DIR / "user_embeddings.npy").exists():
        user_ids, user_embs = build_user_embeddings()
    else:
        print("Loading existing user embeddings...")
        user_ids = np.load(EMBEDDINGS_DIR / "user_ids.npy", allow_pickle=True)
        user_embs = np.load(EMBEDDINGS_DIR / "user_embeddings.npy")
    user_emb_map = dict(zip(user_ids, user_embs))

    if not (MODELS_DIR / "item_struct_map.pkl").exists():
        item_struct, user_struct = build_structured_features()
        item_struct_map = {row["gmap_id"]: row.drop("gmap_id").values.astype(np.float32)
                           for _, row in item_struct.iterrows()}
        user_struct_map = {row["user_id"]: row.drop("user_id").values.astype(np.float32)
                           for _, row in user_struct.iterrows()}
        with open(MODELS_DIR / "item_struct_map.pkl", "wb") as f:
            pickle.dump(item_struct_map, f)
        with open(MODELS_DIR / "user_struct_map.pkl", "wb") as f:
            pickle.dump(user_struct_map, f)
        with open(MODELS_DIR / "user_emb_map.pkl", "wb") as f:
            pickle.dump(user_emb_map, f)
        with open(MODELS_DIR / "biz_emb_map.pkl", "wb") as f:
            pickle.dump(biz_emb_map, f)
    else:
        print("Loading existing structured features...")
        with open(MODELS_DIR / "item_struct_map.pkl", "rb") as f:
            item_struct_map = pickle.load(f)
        with open(MODELS_DIR / "user_struct_map.pkl", "rb") as f:
            user_struct_map = pickle.load(f)
        item_struct = pd.read_parquet(MODELS_DIR / "item_struct_features.parquet")
        user_struct = pd.read_parquet(MODELS_DIR / "user_struct_features.parquet")

    item_struct_dim = len(item_struct.columns) - 1
    user_struct_dim = len(user_struct.columns) - 1

    print("=== Training two tower ===")
    train_df = pd.read_parquet(OUTPUT_DIR / "train.parquet")
    model = train_two_tower(train_df, user_emb_map, biz_emb_map,
                            user_struct_map, item_struct_map,
                            user_struct_dim, item_struct_dim)

    print("=== Building item index ===")
    build_item_index(model, biz_emb_map, item_struct_map, item_struct_dim)
    print("Two tower complete.")
if __name__ == "__main__":
    run()