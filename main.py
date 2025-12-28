import os, random
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from catboost import CatBoostRegressor, Pool

SEED = 322

def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(SEED)

# Load
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

train["dt"] = pd.to_datetime(train["dt"])
test["dt"]  = pd.to_datetime(test["dt"])

assert "row_id" in test.columns

# Base features
CAT_COLS = [
    "product_id", "management_group_id",
    "first_category_id", "second_category_id", "third_category_id",
    "dow", "month"
]
NUM_COLS = [
    "n_stores", "precpt", "avg_temperature", "avg_humidity", "avg_wind_level",
    "day_of_month", "week_of_year", "holiday_flag", "activity_flag"
]

# AR lag features
LAG_COLS = [
    "p05_lag1", "p95_lag1",
    "p05_roll7_mean", "p95_roll7_mean"
]

FEATURES = CAT_COLS + NUM_COLS + LAG_COLS

def build_history_from_df(df_known: pd.DataFrame):
    """
    df_known must have: dt, product_id, price_p05, price_p95
    Returns dicts + deques for roll7.
    """
    last_p05 = {}
    last_p95 = {}
    q05 = defaultdict(lambda: deque(maxlen=7))
    q95 = defaultdict(lambda: deque(maxlen=7))

    tmp = df_known.sort_values(["product_id", "dt"])
    for pid, g in tmp.groupby("product_id", sort=False):
        v05 = g["price_p05"].values
        v95 = g["price_p95"].values
        if len(v05):
            last_p05[pid] = float(v05[-1])
            last_p95[pid] = float(v95[-1])
            for x in v05[-7:]:
                q05[pid].append(float(x))
            for x in v95[-7:]:
                q95[pid].append(float(x))
    return last_p05, last_p95, q05, q95


def add_ar_lags(batch: pd.DataFrame, last_p05, last_p95, q05, q95) -> pd.DataFrame:
    """
    Adds AR lag features using current history state.
    batch is for one date, but can contain many products.
    """
    b = batch.copy()
    pid_arr = b["product_id"].values

    p05_l1 = np.empty(len(b), dtype=float)
    p95_l1 = np.empty(len(b), dtype=float)
    p05_r7 = np.empty(len(b), dtype=float)
    p95_r7 = np.empty(len(b), dtype=float)

    for i, pid in enumerate(pid_arr):
        lp05 = last_p05.get(pid, np.nan)
        lp95 = last_p95.get(pid, np.nan)
        p05_l1[i] = lp05
        p95_l1[i] = lp95

        dq05 = q05[pid]
        dq95 = q95[pid]
        p05_r7[i] = float(np.mean(dq05)) if len(dq05) else np.nan
        p95_r7[i] = float(np.mean(dq95)) if len(dq95) else np.nan

    b["p05_lag1"] = p05_l1
    b["p95_lag1"] = p95_l1
    b["p05_roll7_mean"] = p05_r7
    b["p95_roll7_mean"] = p95_r7
    return b


def update_history_with_preds(batch: pd.DataFrame, p05_pred, p95_pred, last_p05, last_p95, q05, q95):
    for pid, v05, v95 in zip(batch["product_id"].values, p05_pred, p95_pred):
        v05 = float(v05); v95 = float(v95)
        last_p05[pid] = v05
        last_p95[pid] = v95
        q05[pid].append(v05)
        q95[pid].append(v95)

train_sorted = train.sort_values(["product_id", "dt"]).copy()

train_sorted["p05_lag1"] = train_sorted.groupby("product_id")["price_p05"].shift(1)
train_sorted["p95_lag1"] = train_sorted.groupby("product_id")["price_p95"].shift(1)

train_sorted["p05_roll7_mean"] = (
    train_sorted.groupby("product_id")["price_p05"]
    .shift(1).rolling(7, min_periods=1).mean()
    .reset_index(level=0, drop=True)
)
train_sorted["p95_roll7_mean"] = (
    train_sorted.groupby("product_id")["price_p95"]
    .shift(1).rolling(7, min_periods=1).mean()
    .reset_index(level=0, drop=True)
)

X_tr = train_sorted[FEATURES]

y_tr_p05 = train_sorted["price_p05"]
y_tr_p95 = train_sorted["price_p95"]

# for RMSE head: mid + logw
EPS_W = 1e-6
y_tr_mid = (y_tr_p05.values + y_tr_p95.values) / 2.0
y_tr_w   = np.maximum(y_tr_p95.values - y_tr_p05.values, EPS_W)
y_tr_logw = np.log(y_tr_w)

# CatBoost params from Optuna
BEST = {
    "depth": 8,
    "learning_rate": 0.03651139083379959,
    "l2_leaf_reg": 3.279703912387428,
    "subsample": 0.8333382250020284,
    "iterations": 1500,
    "od_wait": 160
}

COMMON = dict(
    iterations=int(BEST["iterations"]),
    learning_rate=float(BEST["learning_rate"]),
    depth=int(BEST["depth"]),
    l2_leaf_reg=float(BEST["l2_leaf_reg"]),
    random_seed=SEED,
    task_type="GPU",
    od_type="Iter",
    od_wait=int(BEST["od_wait"]),
    verbose=300,
)

pool_tr05 = Pool(X_tr, y_tr_p05, cat_features=CAT_COLS)
pool_tr95 = Pool(X_tr, y_tr_p95, cat_features=CAT_COLS)
pool_tr_mid = Pool(X_tr, y_tr_mid, cat_features=CAT_COLS)
pool_tr_logw = Pool(X_tr, y_tr_logw, cat_features=CAT_COLS)

print("Training 4 models (Quantile p05/p95 + RMSE mid/logw) ...")

m_q05 = CatBoostRegressor(**COMMON, loss_function="Quantile:alpha=0.05")
m_q95 = CatBoostRegressor(**COMMON, loss_function="Quantile:alpha=0.95")

m_mid  = CatBoostRegressor(**COMMON, loss_function="RMSE")
m_logw = CatBoostRegressor(**COMMON, loss_function="RMSE")

m_q05.fit(pool_tr05)
m_q95.fit(pool_tr95)
m_mid.fit(pool_tr_mid)
m_logw.fit(pool_tr_logw)

# history from full train
last_p05, last_p95, q05, q95 = build_history_from_df(train)

test_sorted = test.sort_values(["dt", "product_id"]).copy()

# storage (by original index)
# pred_q_lo = np.empty(len(test_sorted), dtype=float)
# pred_q_hi = np.empty(len(test_sorted), dtype=float)

pred_e_lo = np.empty(len(test_sorted), dtype=float)
pred_e_hi = np.empty(len(test_sorted), dtype=float)

# ensemble weights
W_Q = 0.75   # вес quantile
W_R = 0.25   # вес rmse(mid/logw)
MIN_WIDEN = 0.0

# width scaling
K_WIDTH = 1.10

print("AR predicting test by date ...")

for d, idx in test_sorted.groupby("dt").groups.items():
    batch = test_sorted.loc[idx].copy().sort_values(["product_id"])
    batch = add_ar_lags(batch, last_p05, last_p95, q05, q95)

    pool_b = Pool(batch[FEATURES], cat_features=CAT_COLS)

    # --- Quantile preds ---
    q05p = m_q05.predict(pool_b)
    q95p = m_q95.predict(pool_b)
    q_lo = np.minimum(q05p, q95p)
    q_hi = np.maximum(q05p, q95p)

    # --- RMSE mid/logw preds -> interval ---
    midp = m_mid.predict(pool_b)
    logwp = m_logw.predict(pool_b)
    wp = np.exp(np.clip(logwp, -12, 6))  # clip to avoid overflow
    # width scaling
    wp = wp * K_WIDTH

    r_lo = midp - wp / 2.0
    r_hi = midp + wp / 2.0
    r_lo, r_hi = np.minimum(r_lo, r_hi), np.maximum(r_lo, r_hi)

    # store quantile-only AR output
    q_hi = np.maximum(q_hi, q_lo + MIN_WIDEN)

    # ensemble via mid/logw blending (стабильнее, чем смешивать p05/p95)
    q_mid = (q_lo + q_hi) / 2.0
    q_w   = np.maximum(q_hi - q_lo, EPS_W)
    q_logw = np.log(q_w)

    r_mid = (r_lo + r_hi) / 2.0
    r_w   = np.maximum(r_hi - r_lo, EPS_W)
    r_logw = np.log(r_w)

    ens_mid  = W_Q * q_mid  + W_R * r_mid
    ens_logw = W_Q * q_logw + W_R * r_logw

    ens_w = np.exp(np.clip(ens_logw, -12, 6))
    ens_lo = ens_mid - ens_w / 2.0
    ens_hi = ens_mid + ens_w / 2.0
    ens_lo, ens_hi = np.minimum(ens_lo, ens_hi), np.maximum(ens_lo, ens_hi)
    ens_hi = np.maximum(ens_hi, ens_lo + MIN_WIDEN)

    pos = test_sorted.index.get_indexer(batch.index)

    # pred_q_lo[pos] = q_lo
    # pred_q_hi[pos] = q_hi
    pred_e_lo[pos] = ens_lo
    pred_e_hi[pos] = ens_hi

    update_history_with_preds(batch, ens_lo, ens_hi, last_p05, last_p95, q05, q95)

# Save submissions
# out = test_sorted[["row_id"]].copy()
# out["price_p05"] = pred_q_lo
# out["price_p95"] = pred_q_hi
# out = out.sort_values("row_id")
# out.to_csv("submission_quantile_ar.csv", index=False)
# print("Saved submission_quantile_ar.csv")

sub = test_sorted[["row_id"]].copy()
sub["price_p05"] = pred_e_lo
sub["price_p95"] = pred_e_hi
sub = sub.sort_values("row_id")
sub.to_csv("results/submission.csv", index=False)
print("Saved submission.csv")
# score: 0.25