"""
Основной файл с решением соревнования
"""
import os
os.environ["WANDB_DISABLED"] = "true"  # отключаем wandb, чтобы не просил API ключ
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer

from catboost import CatBoostRanker, Pool

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

from utils import clean_text, build_product_text, truncate_words, add_basic_features
from utils import compute_bm25_for_df, compute_ndcg_by_query

def create_submission(test: pd.DataFrame):
    """
    Cоздание файла submission.csv в папку results
    """

    submission = test[["id", "prediction"]].copy()
    submission.to_csv("results/submission.csv", index=False)
    print("Saved submission to: results/submission.csv")

def preprocess_data(train: pd.DataFrame, test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # очищаем все текстовые поля
    train["query_clean"] = train["query"].apply(clean_text)
    test["query_clean"] = test["query"].apply(clean_text)

    for col in ["product_title", "product_description",
                "product_bullet_point", "product_brand", "product_color"]:
        train[col + "_clean"] = train[col].apply(clean_text)
        test[col + "_clean"] = test[col].apply(clean_text)

    train["product_text"] = build_product_text(train)
    test["product_text"] = build_product_text(test)

    # короткая версия текста для CrossEncoder
    train["product_text_ce"] = train["product_text"].apply(lambda s: truncate_words(s, 128))
    test["product_text_ce"] = test["product_text"].apply(lambda s: truncate_words(s, 128))

    train = add_basic_features(train)
    test = add_basic_features(test)

    # фильтруем шум: почти пустой товар
    noise_mask = (
            (train["len_title"] < 3) &
            (train["len_desc"] == 0) &
            (train["len_bullet"] == 0)
    )
    print("Noisy rows to drop:", noise_mask.sum())

    train = train.loc[~noise_mask].reset_index(drop=True)

    return train, test

def process_features_for_models(train: pd.DataFrame, test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # векторизация и токенизация признаков
    print("Computing BM25 for train...")
    train["bm25"] = compute_bm25_for_df(train)
    print("Computing BM25 for test...")
    test["bm25"] = compute_bm25_for_df(test)

    print("Building joint TF-IDF for cosine similarity...")
    joint_corpus = pd.concat([
        train["query_clean"], train["product_text"],
        test["query_clean"], test["product_text"]
    ], axis=0).astype(str)

    tfidf_joint = TfidfVectorizer(
        max_features=80000,
        ngram_range=(1, 2),
        min_df=3
    )
    tfidf_joint.fit(joint_corpus)

    Xq_train = tfidf_joint.transform(train["query_clean"])
    Xd_train = tfidf_joint.transform(train["product_text"])

    Xq_test = tfidf_joint.transform(test["query_clean"])
    Xd_test = tfidf_joint.transform(test["product_text"])

    cos_train = np.asarray((Xq_train.multiply(Xd_train)).sum(axis=1)).ravel().astype(np.float32)
    cos_test = np.asarray((Xq_test.multiply(Xd_test)).sum(axis=1)).ravel().astype(np.float32)

    train["cosine_joint"] = cos_train
    test["cosine_joint"] = cos_test

    del Xq_train, Xd_train, Xq_test, Xd_test  # освобождаем память

    # Match-фичи для усиления обучающей способности модели
    print("Building match features...")

    train["query_tokens"] = train["query_clean"].str.split()
    train["title_tokens"] = train["product_title_clean"].str.split()
    train["product_tokens"] = train["product_text"].str.split()

    test["query_tokens"] = test["query_clean"].str.split()
    test["title_tokens"] = test["product_title_clean"].str.split()
    test["product_tokens"] = test["product_text"].str.split()

    def count_overlap(list1, list2):
        return len(set(list1) & set(list2))

    train["match_title"] = [
        count_overlap(q, t) for q, t in zip(train["query_tokens"], train["title_tokens"])
    ]
    train["match_product"] = [
        count_overlap(q, p) for q, p in zip(train["query_tokens"], train["product_tokens"])
    ]

    test["match_title"] = [
        count_overlap(q, t) for q, t in zip(test["query_tokens"], test["title_tokens"])
    ]
    test["match_product"] = [
        count_overlap(q, p) for q, p in zip(test["query_tokens"], test["product_tokens"])
    ]

    # точное вхождение бренда в запрос
    train["exact_brand_in_query"] = (
            (train["product_brand_clean"] != "") &
            train.apply(lambda row: row["product_brand_clean"] in row["query_clean"], axis=1)
    ).astype(int)

    test["exact_brand_in_query"] = (
            (test["product_brand_clean"] != "") &
            test.apply(lambda row: row["product_brand_clean"] in row["query_clean"], axis=1)
    ).astype(int)

    return train, test

def main():

    # Загрузка данных
    RANDOM_STATE = 993
    train = pd.read_csv("data/train.csv", engine="python")
    test = pd.read_csv("data/test.csv", engine="python")
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    # Создание сабмита
    create_submission(test)

    # Подготовка данных перед обучением
    train, test = preprocess_data(train, test)

    train, test = process_features_for_models(train, test)

    # 7. Подготовка для обучения CatBoostRanker
    numeric_cols = [
        "len_query", "len_title", "len_desc", "len_bullet",
        "has_desc", "has_bullet", "has_brand", "has_color",
        "bm25", "cosine_joint",
        "match_title", "match_product", "exact_brand_in_query"
    ]
    X_train = train[numeric_cols].astype("float32").values
    X_test = test[numeric_cols].astype("float32").values
    y = train["relevance"].values.astype("float32")
    groups = train["query_id"].values
    groups_test = test["query_id"].values

    print("Feature matrix shapes:", X_train.shape, X_test.shape)

    # 8. Валидация и обучение CatBoostRanker (YetiRank)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss.split(X_train, y, groups=groups))

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    qid_tr, qid_val = groups[train_idx], groups[val_idx]

    train_part = train.iloc[train_idx].reset_index(drop=True)
    val_part = train.iloc[val_idx].reset_index(drop=True)
    pool_tr = Pool(X_tr, label=y_tr, group_id=qid_tr)
    pool_val = Pool(X_val, label=y_val, group_id=qid_val)

    ranker = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric="NDCG:top=10",
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        random_seed=RANDOM_STATE,
        task_type="GPU",
        verbose=100,
    )

    print("Training CatBoostRanker...")
    ranker.fit(pool_tr, eval_set=pool_val, use_best_model=True)
    val_pred_cb = ranker.predict(pool_val)
    ndcg_cb = compute_ndcg_by_query(val_part, val_pred_cb, k=10)
    print(f"CatBoostRanker validation nDCG@10: {ndcg_cb:.5f}")

    # Обучение CrossEncoder (BERT)
    MAX_CE_TRAIN_SAMPLES = 150_000

    ce_train_df = train.copy()
    if len(ce_train_df) > MAX_CE_TRAIN_SAMPLES:
        ce_train_df = ce_train_df.sample(MAX_CE_TRAIN_SAMPLES, random_state=RANDOM_STATE)

    ce_train_samples = [
        InputExample(
            texts=[row.query_clean, row.product_text_ce],
            label=float(row.relevance) / 3.0
        )
        for row in ce_train_df.itertuples()
    ]

    ce_train_loader = DataLoader(ce_train_samples, batch_size=16, shuffle=True)
    ce_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ce_model = CrossEncoder(
        ce_model_name,
        num_labels=1,
        max_length=512
    )

    print("Training CrossEncoder...")
    ce_model.fit(
        train_dataloader=ce_train_loader,
        epochs=1,
        warmup_steps=int(0.1 * len(ce_train_loader)),
        output_path=None,
    )

    # предсказания CrossEncoder на валидации
    val_pairs = [
        [q, p] for q, p in zip(val_part["query_clean"], val_part["product_text_ce"])
    ]
    ce_val_scores = ce_model.predict(val_pairs).reshape(-1)

    ndcg_ce = compute_ndcg_by_query(val_part, ce_val_scores, k=10)
    print(f"CrossEncoder validation nDCG@10: {ndcg_ce:.5f}")

    # подбор лучшего alpha для ансамбля
    alphas = np.linspace(0.0, 1.0, 21)
    best_alpha = None
    best_ndcg = -1.0
    for a in alphas:
        ens_scores = a * val_pred_cb + (1.0 - a) * ce_val_scores
        nd = compute_ndcg_by_query(val_part, ens_scores, k=10)
        print(f"alpha={a:.2f} -> nDCG@10={nd:.5f}")
        if nd > best_ndcg:
            best_ndcg = nd
            best_alpha = a
    print(f"Best alpha: {best_alpha:.3f} with nDCG@10={best_ndcg:.5f}")

    # 10. Обучаем CatBoost на всём train
    pool_full = Pool(X_train, label=y, group_id=groups)
    pool_test = Pool(X_test, group_id=groups_test)

    print("Training CatBoost on full train...")
    ranker_full = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric="NDCG:top=10",
        iterations=ranker.tree_count_,
        learning_rate=ranker.learning_rate_,
        depth=ranker.get_param("depth"),
        random_seed=RANDOM_STATE,
        task_type="GPU",
        verbose=100,
    )
    ranker_full.fit(pool_full)

    cb_test_scores = ranker_full.predict(pool_test)

    # Производим re-ranking top-K на тесте CrossEncoder'ом
    print("Re-ranking test with CrossEncoder...")
    test["cb_score"] = cb_test_scores
    final_scores = cb_test_scores.copy()

    TOP_K = 50
    for qid, group in test.groupby("query_id"):
        idx = group.index.values
        scores = group["cb_score"].values
        order = np.argsort(-scores)
        top_local = order[:TOP_K]
        top_idx = idx[top_local]

        pairs = [
            [test.loc[i, "query_clean"], test.loc[i, "product_text_ce"]]
            for i in top_idx
        ]
        ce_scores = ce_model.predict(pairs).reshape(-1)

        mixed_model = best_alpha * final_scores[top_idx] + (1.0 - best_alpha) * ce_scores
        final_scores[top_idx] = mixed_model

    # Формируем предсказание
    test["prediction"] = final_scores

if __name__ == "__main__":
    main()
    print("Для просмотра анализа данных, откройте файл `eda.ipynb`")

