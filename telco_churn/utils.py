# telco_churn/utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

def prepare_xy(df: pd.DataFrame, schema, random_state=42):
    y = (df[schema.target].astype(str) == schema.target_pos).astype(int)
    X = df.drop(columns=[schema.target] + schema.drop_cols, errors='ignore')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    return X, y, X_train, X_test, y_train, y_test

def pos_weight(y):
    pos = y.sum()
    neg = len(y) - pos
    return float(neg / max(pos, 1))

def evaluate(pipe, X, y, X_tr, y_tr, X_te, y_te, cv_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    # 为了加速、减少外部并发问题，这里不再并行 n_jobs，不想改可自行加回
    from sklearn.model_selection import cross_val_score
    auc_cv = float(cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc').mean())
    ap_cv  = float(cross_val_score(pipe, X, y, cv=cv, scoring='average_precision').mean())
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    auc_hold = float(roc_auc_score(y_te, proba))
    ap_hold  = float(average_precision_score(y_te, proba))
    return {
        "auc_cv": auc_cv, "ap_cv": ap_cv,
        "auc_hold": auc_hold, "ap_hold": ap_hold,
    }, pipe
