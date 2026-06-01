# """
# embed.py
# Builds rich item embeddings for each restaurant by combining:
#   - Text embeddings (name + description + aggregated review text) via all-MiniLM-L6-v2
#   - Structured features (category, price, misc flags, topic distributions)
# Saves:
#   - embeddings_i2i/item_embeddings.npy   : combined vectors (N, D)
#   - embeddings_i2i/item_embedding_ids.npy: gmap_ids in same order
# """

# import numpy as np
# import pandas as pd
# from pathlib import Path
# from sentence_transformers import SentenceTransformer
# from sklearn.preprocessing import normalize
# import torch
# import gc

# OUTPUT_DIR    = Path("output_review")
# MODELS_DIR    = Path("models_review")
# EMBED_DIR     = Path("embeddings")
# EMBED_DIR.mkdir(parents=True, exist_ok=True)

# RAW_REVIEWS_PATH = "recommendations_restaurants.csv"

# TEXT_MODEL       = "all-MiniLM-L6-v2"
# BATCH_SIZE       = 128
# MAX_REVIEW_CHARS = 1000


# def build_restaurant_text(restaurants, raw_reviews):
#     """
#     For each restaurant, build a single text string:
#       "<name>. <description>. <aggregated review snippets>"
#     """
#     print("Aggregating review text per restaurant...")
#     review_text = (
#         raw_reviews[raw_reviews["text"].notna()]
#         .groupby("gmap_id")["text"]
#         .apply(lambda texts: " ".join(texts.dropna().astype(str))[:MAX_REVIEW_CHARS])
#         .reset_index()
#         .rename(columns={"text": "review_agg"})
#     )

#     df = restaurants[["gmap_id", "name", "description"]].copy()
#     df = df.merge(review_text, on="gmap_id", how="left")

#     df["name"]        = df["name"].fillna("")
#     df["description"] = df["description"].fillna("")
#     df["review_agg"]  = df["review_agg"].fillna("")

#     df["full_text"] = (
#         df["name"] + ". " +
#         df["description"] + ". " +
#         df["review_agg"]
#     ).str.strip()

#     return df[["gmap_id", "full_text"]]


# def encode_text(texts, model, batch_size=BATCH_SIZE):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Encoding {len(texts)} texts on {device}...")
#     embeddings = model.encode(
#         texts,
#         batch_size=batch_size,
#         show_progress_bar=True,
#         convert_to_numpy=True,
#         device=device,
#     )
#     return embeddings.astype(np.float32)


# def load_structured_features():
#     item_features = pd.read_parquet(MODELS_DIR / "item_features.parquet")
#     feature_cols  = [c for c in item_features.columns if c != "gmap_id"]
#     matrix = item_features[feature_cols].values.astype(np.float32)
#     matrix = np.nan_to_num(matrix, nan=0.0)
#     return item_features["gmap_id"].values, matrix


# # def build_embeddings():
# #     print("Loading restaurants...")
# #     restaurants = pd.read_parquet(OUTPUT_DIR / "restaurants.parquet")

# #     print("Loading raw reviews for text...")
# #     raw_reviews = pd.read_csv(
# #         RAW_REVIEWS_PATH,
# #         engine="python",
# #         on_bad_lines="skip",
# #         usecols=["gmap_id", "text"],
# #     )
# #     print(f"Raw reviews loaded: {len(raw_reviews):,}")

# #     # align on gmap_ids in item_features
# #     struct_gmap_ids, struct_matrix = load_structured_features()
# #     struct_id_set = set(struct_gmap_ids)
# #     restaurants   = restaurants[restaurants["gmap_id"].isin(struct_id_set)].copy()
# #     restaurants   = restaurants.set_index("gmap_id").loc[struct_gmap_ids].reset_index()

# #     print(f"Restaurants to embed: {len(restaurants)}")

# #     # filter raw reviews to only restaurants we care about
# #     raw_reviews = raw_reviews[raw_reviews["gmap_id"].isin(struct_id_set)]

# #     # --- text embeddings ---
# #     text_df  = build_restaurant_text(restaurants, raw_reviews)
# #     texts    = text_df["full_text"].tolist()

# #     print("Loading sentence transformer...")
# #     model    = SentenceTransformer(TEXT_MODEL)
# #     text_emb = encode_text(texts, model)
# #     text_emb = normalize(text_emb, norm="l2")

# #     del model, raw_reviews
# #     gc.collect()

# #     # --- structured features ---
# #     struct_norm = normalize(struct_matrix, norm="l2")

# #     # --- combine: 60% text + 40% structured ---
# #     combined = np.concatenate([0.6 * text_emb, 0.4 * struct_norm], axis=1)
# #     combined = normalize(combined, norm="l2")

# #     print(f"Combined embedding shape: {combined.shape}")

# #     np.save(EMBED_DIR / "item_embeddings.npy",    combined)
# #     np.save(EMBED_DIR / "item_embedding_ids.npy", struct_gmap_ids)
# #     print(f"Saved to {EMBED_DIR}/")


# def build_embeddings():
#     print("Loading restaurants...")
#     restaurants = pd.read_parquet(OUTPUT_DIR / "restaurants.parquet")

#     # load structured features
#     struct_gmap_ids, struct_matrix = load_structured_features()
#     struct_id_set = set(struct_gmap_ids)

#     # load prebuilt business (review text) embeddings
#     BUSINESS_EMBED_DIR = Path("embeddings")  # update to wherever yours are saved
#     biz_gmap_ids, biz_matrix = load_business_embeddings(BUSINESS_EMBED_DIR)
#     biz_id_to_idx = {gid: i for i, gid in enumerate(biz_gmap_ids)}

#     # align all three on common gmap_ids
#     common_ids = [gid for gid in struct_gmap_ids if gid in biz_id_to_idx]
#     print(f"Restaurants with all features: {len(common_ids)}")

#     struct_idx = [list(struct_gmap_ids).index(gid) for gid in common_ids]
#     biz_idx    = [biz_id_to_idx[gid] for gid in common_ids]

#     struct_aligned = struct_matrix[struct_idx]
#     biz_aligned    = biz_matrix[biz_idx]

#     # normalize each
#     struct_norm = normalize(struct_aligned, norm="l2")
#     biz_norm    = normalize(biz_aligned,    norm="l2")

#     # combine: 60% review text + 40% structured
#     combined = np.concatenate([0.6 * biz_norm, 0.4 * struct_norm], axis=1)
#     combined = normalize(combined, norm="l2")

#     print(f"Combined embedding shape: {combined.shape}")

#     common_ids_arr = np.array(common_ids)
#     np.save(EMBED_DIR / "item_embeddings.npy",    combined)
#     np.save(EMBED_DIR / "item_embedding_ids.npy", common_ids_arr)
#     print(f"Saved to {EMBED_DIR}/")



# if __name__ == "__main__":
#     build_embeddings() 

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import torch
import gc

OUTPUT_DIR    = Path("output_review")
MODELS_DIR    = Path("models_review")
EMBED_DIR     = Path("embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)

RAW_REVIEWS_PATH = "recommendations_restaurants.csv"

TEXT_MODEL       = "all-MiniLM-L6-v2"
BATCH_SIZE       = 128
MAX_REVIEW_CHARS = 1000


def build_restaurant_text(restaurants, raw_reviews):
    """
    For each restaurant, build a single text string:
      "<name>. <description>. <aggregated review snippets>"
    """
    print("Aggregating review text per restaurant...")
    review_text = (
        raw_reviews[raw_reviews["text"].notna()]
        .groupby("gmap_id")["text"]
        .apply(lambda texts: " ".join(texts.dropna().astype(str))[:MAX_REVIEW_CHARS])
        .reset_index()
        .rename(columns={"text": "review_agg"})
    )

    df = restaurants[["gmap_id", "name", "description"]].copy()
    df = df.merge(review_text, on="gmap_id", how="left")

    df["name"]        = df["name"].fillna("")
    df["description"] = df["description"].fillna("")
    df["review_agg"]  = df["review_agg"].fillna("")

    df["full_text"] = (
        df["name"] + ". " +
        df["description"] + ". " +
        df["review_agg"]
    ).str.strip()

    return df[["gmap_id", "full_text"]]


def encode_text(texts, model, batch_size=BATCH_SIZE):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding {len(texts)} texts on {device}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
    )
    return embeddings.astype(np.float32)


def load_structured_features():
    item_features = pd.read_parquet(MODELS_DIR / "item_features.parquet")
    feature_cols  = [c for c in item_features.columns if c != "gmap_id"]
    matrix = item_features[feature_cols].values.astype(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0)
    return item_features["gmap_id"].values, matrix


def load_business_embeddings():
    BUSINESS_EMBED_DIR = Path("embeddings")
    gmap_ids   = np.load(BUSINESS_EMBED_DIR / "business_gmap_ids.npy", allow_pickle=True)
    embeddings = np.load(BUSINESS_EMBED_DIR / "business_embeddings.npy")
    print(f"Business embeddings loaded: {embeddings.shape}")
    return gmap_ids, embeddings
    
# def build_embeddings():
#     print("Loading restaurants...")
#     restaurants = pd.read_parquet(OUTPUT_DIR / "restaurants.parquet")

#     print("Loading raw reviews for text...")
#     raw_reviews = pd.read_csv(
#         RAW_REVIEWS_PATH,
#         engine="python",
#         on_bad_lines="skip",
#         usecols=["gmap_id", "text"],
#     )
#     print(f"Raw reviews loaded: {len(raw_reviews):,}")

#     # align on gmap_ids in item_features
#     struct_gmap_ids, struct_matrix = load_structured_features()
#     struct_id_set = set(struct_gmap_ids)
#     restaurants   = restaurants[restaurants["gmap_id"].isin(struct_id_set)].copy()
#     restaurants   = restaurants.set_index("gmap_id").loc[struct_gmap_ids].reset_index()

#     print(f"Restaurants to embed: {len(restaurants)}")

#     # filter raw reviews to only restaurants we care about
#     raw_reviews = raw_reviews[raw_reviews["gmap_id"].isin(struct_id_set)]

#     # --- text embeddings ---
#     text_df  = build_restaurant_text(restaurants, raw_reviews)
#     texts    = text_df["full_text"].tolist()

#     print("Loading sentence transformer...")
#     model    = SentenceTransformer(TEXT_MODEL)
#     text_emb = encode_text(texts, model)
#     text_emb = normalize(text_emb, norm="l2")

#     del model, raw_reviews
#     gc.collect()

#     # --- structured features ---
#     struct_norm = normalize(struct_matrix, norm="l2")

#     # --- combine: 60% text + 40% structured ---
#     combined = np.concatenate([0.6 * text_emb, 0.4 * struct_norm], axis=1)
#     combined = normalize(combined, norm="l2")

#     print(f"Combined embedding shape: {combined.shape}")

#     np.save(EMBED_DIR / "item_embeddings.npy",    combined)
#     np.save(EMBED_DIR / "item_embedding_ids.npy", struct_gmap_ids)
#     print(f"Saved to {EMBED_DIR}/")


def build_embeddings():
    print("Loading restaurants...")
    restaurants = pd.read_parquet(OUTPUT_DIR / "restaurants.parquet")

    # load structured features
    struct_gmap_ids, struct_matrix = load_structured_features()
    struct_id_set = set(struct_gmap_ids)

    # load prebuilt business (review text) embeddings
    BUSINESS_EMBED_DIR = Path("embeddings")  # update to wherever yours are saved
    #biz_gmap_ids, biz_matrix = load_business_embeddings(BUSINESS_EMBED_DIR)
    biz_gmap_ids, biz_matrix = load_business_embeddings()

    biz_id_to_idx = {gid: i for i, gid in enumerate(biz_gmap_ids)}

    # align all three on common gmap_ids
    common_ids = [gid for gid in struct_gmap_ids if gid in biz_id_to_idx]
    print(f"Restaurants with all features: {len(common_ids)}")

    struct_idx = [list(struct_gmap_ids).index(gid) for gid in common_ids]
    biz_idx    = [biz_id_to_idx[gid] for gid in common_ids]

    struct_aligned = struct_matrix[struct_idx]
    biz_aligned    = biz_matrix[biz_idx]

    # normalize each
    struct_norm = normalize(struct_aligned, norm="l2")
    biz_norm    = normalize(biz_aligned,    norm="l2")

    # combine: 60% review text + 40% structured
    combined = np.concatenate([0.6 * biz_norm, 0.4 * struct_norm], axis=1)
    combined = normalize(combined, norm="l2")

    print(f"Combined embedding shape: {combined.shape}")

    common_ids_arr = np.array(common_ids)
    np.save(EMBED_DIR / "item_embeddings.npy",    combined)
    np.save(EMBED_DIR / "item_embedding_ids.npy", common_ids_arr)
    print(f"Saved to {EMBED_DIR}/")



if __name__ == "__main__":
    build_embeddings()
