# train_xgb.py
import os
import pandas as pd

from telco_churn.pipeline import (
    FeatureSchema, build_xgb_pipeline,
    save_pipeline, save_schema
)
from telco_churn.utils import prepare_xy, pos_weight, evaluate

RANDOM_STATE = 42

if __name__ == "__main__":
    # 1) 读数据（与你 notebook 一致）
    df = pd.read_parquet("data/clean_telco.parquet")  # 或改成 CSV

    # 2) 列定义（直接来自你的 notebook）
    schema = FeatureSchema(
        cat_multi=['Contract','PaymentMethod','InternetService'],
        cat_bin=['PaperlessBilling','SeniorCitizen','Partner','Dependents'],
        num_base=['tenure','MonthlyCharges','is_new_customer'],
        drop_cols=['customerID', 'gender'],  # 训练时丢弃
        target='Churn', target_pos='Yes'
    )

    # 3) 切分 + 计算 scale_pos_weight
    X, y, X_tr, X_te, y_tr, y_te = prepare_xy(df, schema, random_state=RANDOM_STATE)
    spw = pos_weight(y_tr)

    # 4) 构建 XGB pipeline（你最后选择了 XGBoost）
    pipe_xgb = build_xgb_pipeline(schema, scale_pos_weight=spw, random_state=RANDOM_STATE)

    # 5) 评估（CV + Holdout）
    metrics, fitted = evaluate(pipe_xgb, X, y, X_tr, y_tr, X_te, y_te, cv_splits=5, random_state=RANDOM_STATE)
    print(f"[XGB] CV AUC={metrics['auc_cv']:.3f} | CV PR-AUC={metrics['ap_cv']:.3f} | "
          f"Holdout AUC={metrics['auc_hold']:.3f} | Holdout PR-AUC={metrics['ap_hold']:.3f}")

    # 6) 导出
    os.makedirs("models", exist_ok=True)
    save_pipeline(fitted, "models/churn_pipeline.pkl")
    save_schema(schema, "models/feature_schema.json")
    print("✅ 导出完成：models/churn_pipeline.pkl & models/feature_schema.json")
