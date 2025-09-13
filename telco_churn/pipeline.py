# telco_churn/pipeline.py
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import json
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from telco_churn.features.business import BusinessFeatureAdder

@dataclass
class FeatureSchema:
    # 来自 notebook 的列定义
    cat_multi: List[str]
    cat_bin:   List[str]
    # BusinessFeatureAdder 会生成 is_new_customer，所以 num_base 与你 notebook 一致
    num_base:  List[str]
    # 训练时需要丢弃的列（比如 ID、弱特征），以及目标列名/数据源约定等也可放这里
    drop_cols: List[str]
    target:    str = "Churn"        # 原列名
    target_pos: str = "Yes"         # 阳性取值

def build_preprocessor_for_lr(schema: FeatureSchema, use_bins: bool=False) -> ColumnTransformer:
    if use_bins:
        # 可选：如果你要给 LR 用分箱
        cats = schema.cat_multi + schema.cat_bin + ['tenure_bin','monthly_bin']
        return ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), cats),
                ('num', StandardScaler(), schema.num_base),
            ],
            remainder='drop'
        )
    else:
        return ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'),
                 schema.cat_multi + schema.cat_bin),
                ('num', StandardScaler(), schema.num_base),
            ],
            remainder='drop'
        )

def build_preprocessor_for_tree(schema: FeatureSchema) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ('cat_multi', OneHotEncoder(handle_unknown='ignore'), schema.cat_multi),
            ('cat_bin',   OneHotEncoder(drop='if_binary', handle_unknown='ignore'), schema.cat_bin),
            ('num', 'passthrough', schema.num_base),
        ], remainder='drop'
    )

def build_lr_pipeline(schema: FeatureSchema) -> Pipeline:
    pre = build_preprocessor_for_lr(schema, use_bins=False)
    return Pipeline([
        ('fe', BusinessFeatureAdder(add_bins_for_lr=False)),
        ('pre', pre),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1))
    ])

def build_rf_pipeline(schema: FeatureSchema, random_state: int=42) -> Pipeline:
    pre = build_preprocessor_for_tree(schema)
    return Pipeline([
        ('fe', BusinessFeatureAdder(add_bins_for_lr=False)),
        ('pre', pre),
        ('clf',  RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_split=4, min_samples_leaf=2,
            class_weight='balanced', n_jobs=-1, random_state=random_state
        ))
    ])

def build_xgb_pipeline(schema: FeatureSchema, scale_pos_weight: float, random_state: int=42) -> Pipeline:
    pre = build_preprocessor_for_tree(schema)
    clf = XGBClassifier(
        n_estimators=700, learning_rate=0.05, max_depth=3,
        min_child_weight=2.0, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, objective='binary:logistic', eval_metric='auc',
        scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=random_state
    )
    return Pipeline([
        ('fe', BusinessFeatureAdder(add_bins_for_lr=False)),
        ('pre', pre),
        ('clf', clf)
    ])

def save_schema(schema: FeatureSchema, path: str):
    with open(path, 'w') as f:
        json.dump(asdict(schema), f, ensure_ascii=False, indent=2)

def load_schema(path: str) -> FeatureSchema:
    with open(path, 'r') as f:
        d = json.load(f)
    return FeatureSchema(**d)

def save_pipeline(pipe: Pipeline, path: str):
    joblib.dump(pipe, path)

def load_pipeline(path: str) -> Pipeline:
    # 只要 import 过 BusinessFeatureAdder，就能顺利反序列化
    return joblib.load(path)
