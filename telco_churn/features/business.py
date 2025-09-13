# telco_churn/features/business.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class BusinessFeatureAdder(BaseEstimator, TransformerMixin):
    """
    与 notebook 同名同参：
    add_bins_for_lr: 是否生成分箱（仅给 LR 用）
    drop_totalcharges: 是否删除 TotalCharges 以避免共线
    """
    def __init__(self, add_bins_for_lr=False, drop_totalcharges=True):
        self.add_bins_for_lr = add_bins_for_lr
        self.drop_totalcharges = drop_totalcharges

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 新客标记
        if 'tenure' in X:
            X['is_new_customer'] = (pd.to_numeric(X['tenure'], errors='coerce') <= 6).astype(int)

        # 仅给 LR 用的分箱（需要时再开）
        if self.add_bins_for_lr:
            if 'tenure' in X:
                X['tenure_bin'] = pd.cut(
                    pd.to_numeric(X['tenure'], errors='coerce'),
                    [-1, 6, 24, 120],
                    labels=['new_0_6','mid_7_24','old_25p']
                )
            if 'MonthlyCharges' in X:
                X['monthly_bin'] = pd.cut(
                    pd.to_numeric(X['MonthlyCharges'], errors='coerce'),
                    [0, 40, 70, 999],
                    labels=['m_low_0_40','m_med_41_70','m_high_71p'],
                    include_lowest=True
                )

        if self.drop_totalcharges and 'TotalCharges' in X.columns:
            X = X.drop(columns=['TotalCharges'])

        return X
