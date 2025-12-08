import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.metrics import ndcg_score

def compute_ndcg_by_query(df_val: pd.DataFrame, preds: np.ndarray, k: int = 10) -> float:
    df_val = df_val.copy()
    df_val["pred"] = preds
    scores = []
    for qid, group in df_val.groupby("query_id"):
        y_true = group["relevance"].values.reshape(1, -1)
        y_score = group["pred"].values.reshape(1, -1)
        ndcg = ndcg_score(y_true, y_score, k=k)
        scores.append(ndcg)
    if not scores:
        return 0.0
    return float(np.mean(scores))

def normalize_none(x: str) -> str:
    NONE = {"none", "nan", "null", "unknown", "unknown_brand"}
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in NONE:
        return ""
    return s

def clean_text(x: str) -> str:
    HTML_TAG_RE = re.compile(r"<[^>]+>")
    WHITESPACE_RE = re.compile(r"\s+")

    s = normalize_none(x)
    if not s:
        return ""
    s = HTML_TAG_RE.sub(" ", s)
    s = WHITESPACE_RE.sub(" ", s)
    return s.strip().lower()

def build_product_text(df: pd.DataFrame,
                       max_desc_tokens: int = 256,
                       max_bullet_tokens: int = 128) -> pd.Series:
    def truncate_by_tokens(text: str, max_tokens: int) -> str:
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text
        return " ".join(tokens[:max_tokens])

    title = df["product_title_clean"]
    brand = df["product_brand_clean"].apply(lambda s: f"[brand] {s}" if s else "")
    color = df["product_color_clean"].apply(lambda s: f"[color] {s}" if s else "")

    bullets = df["product_bullet_point_clean"].apply(
        lambda s: truncate_by_tokens(s, max_bullet_tokens) if s else ""
    )
    desc = df["product_description_clean"].apply(
        lambda s: truncate_by_tokens(s, max_desc_tokens) if s else ""
    )

    product_text = (
            "[title] " + title + " " +
            brand + " " +
            color + " " +
            "[bullet] " + bullets + " " +
            "[desc] " + desc
    )
    return product_text

def truncate_words(text: str, max_words: int = 128) -> str:
    tokens = text.split()
    if len(tokens) <= max_words:
        return text
    return " ".join(tokens[:max_words])

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["len_query"] = df["query_clean"].str.split().apply(len)
    df["len_title"] = df["product_title_clean"].str.split().apply(len)
    df["len_desc"] = df["product_description_clean"].str.split().apply(len)
    df["len_bullet"] = df["product_bullet_point_clean"].str.split().apply(len)

    df["has_desc"] = df["product_description_clean"].apply(lambda s: int(bool(s)))
    df["has_bullet"] = df["product_bullet_point_clean"].apply(lambda s: int(bool(s)))
    df["has_brand"] = df["product_brand_clean"].apply(lambda s: int(bool(s)))
    df["has_color"] = df["product_color_clean"].apply(lambda s: int(bool(s)))
    return df


def compute_bm25_for_df(df: pd.DataFrame) -> np.ndarray:
    """BM25 score(query, product_text) по каждому query_id."""
    bm25_scores = np.zeros(len(df), dtype=np.float32)

    for qid, group in df.groupby("query_id"):
        idx = group.index.values
        docs_tokens = group["product_text"].str.split().tolist()
        bm25 = BM25Okapi(docs_tokens)
        query_tokens = group["query_clean"].iloc[0].split()
        scores = bm25.get_scores(query_tokens)
        bm25_scores[idx] = scores.astype(np.float32)

    return bm25_scores
