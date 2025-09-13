# app/streamlit_app.py
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# 让页面更早设置（避免某些版本的警告）
st.set_page_config(page_title="Telco Churn Targeting", layout="wide")

# ------------------------------------------------------------
# 关键：确保自定义类在 sys.modules 中可见，便于 joblib 反序列化
# 如果你的包名/路径不同，请把 'telco_churn' 改成你的实际包路径
# ------------------------------------------------------------
try:
    # 这行导入即足够让 joblib 找到 BusinessFeatureAdder
    from telco_churn.features.business import BusinessFeatureAdder  # noqa: F401
except Exception:
    st.warning(
        "提示：未能从 telco_churn.features.business 导入 BusinessFeatureAdder。"
        "若你的模型是用该类训练的，请确保模块存在。"
    )

# ------------------------------------------------------------
# 路径：以当前文件为基准，稳健拼出 models 目录（位于项目根目录）
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent   # 如果放在根目录，这里就是根
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "churn_pipeline.pkl"
SCHEMA_PATH = MODELS_DIR / "feature_schema.json"

# -----------------------------
# 1) 载入模型 & schema
# -----------------------------
@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型文件：{MODEL_PATH}")
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"找不到特征 schema 文件：{SCHEMA_PATH}")

    pipe = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    return pipe, schema

try:
    pipe, feature_schema = load_assets()
except Exception as e:
    st.error(f"加载模型/Schema 失败：{e}")
    st.stop()

# -----------------------------
# 2) 工具函数
# -----------------------------
YESNO_MAP = {
    "yes": "Yes", "y": "Yes", "1": "Yes", 1: "Yes", True: "Yes",
    "no": "No", "n": "No", "0": "No", 0: "No", False: "No",
    "Yes": "Yes", "No": "No"
}

CAT_COLS = ["Contract","InternetService","PaymentMethod","PaperlessBilling","Partner","Dependents"]
NUM_COLS = ["tenure","MonthlyCharges","TotalCharges"]

def _norm_yesno(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    return YESNO_MAP.get(s, YESNO_MAP.get(s.lower(), s))  # 未命中映射则原样返回

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """把输入 df 的类型/取值拉齐到训练习惯；缺列填默认；确保数值列是 float、类别列是 object。"""
    df = df.copy()

    # 补齐必需列
    required = ["tenure","MonthlyCharges","Contract","InternetService","PaymentMethod",
                "PaperlessBilling","SeniorCitizen","Partner","Dependents"]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    # 数值列：统一 float（防止 isnan 遇到 object 报错）
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # 二值列：规范为 "Yes"/"No" 的字符串（而不是 True/False），避免混型
    for c in ["PaperlessBilling","Partner","Dependents"]:
        if c in df.columns:
            df[c] = df[c].map(_norm_yesno)

    # SeniorCitizen：统一成 0/1 的 int（容忍 Yes/True/1）
    if "SeniorCitizen" in df.columns:
        sr = df["SeniorCitizen"].astype(str).str.strip().str.lower()
        df["SeniorCitizen"] = np.where(sr.isin(["1","yes","true"]), 1,
                                np.where(sr.isin(["0","no","false"]), 0, 0)).astype(int)

    # 类别空值 → "Unknown"，并显式设为 object（避免 pandas FutureWarning & 保持和旧 pipeline 一致）
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype("object")

    # 如果提供了 feature_schema，按 schema 补齐缺失列并排序（可选）
    if isinstance(feature_schema, dict) and "columns" in feature_schema:
        cols = feature_schema["columns"]
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[[c for c in cols if c in df.columns]]

    return df

def reason_hints(row: pd.Series) -> list:
    """给每条用户出 1-3 条‘运营理由’提示（不依赖 SHAP）。"""
    hints = []
    if str(row.get("Contract","")) == "Month-to-month":
        hints.append("月付合约：退出成本低")
    if pd.to_numeric(row.get("MonthlyCharges", np.nan), errors="coerce") >= 90:
        hints.append("账单高：建议降档或组合包")
    if str(row.get("PaymentMethod","")) == "Electronic check":
        hints.append("电子支票：建议迁移自动扣款")
    if int(row.get("SeniorCitizen",0)) == 1:
        hints.append("老年用户：简化流程/客服关怀")
    if str(row.get("InternetService","")) == "Fiber optic":
        hints.append("光纤：价格/体验敏感")
    return hints[:3]

def precision_at_k(y_true, y_score, k_pct):
    if y_true is None:
        return None
    k = max(1, int(len(y_score) * k_pct / 100.0))
    order = np.argsort(-y_score)
    topk_idx = order[:k]
    return float(np.mean(y_true.iloc[topk_idx]))

def lift_at_k(precision_k, base_rate):
    if precision_k is None or base_rate is None or base_rate <= 0:
        return None
    return float(precision_k / base_rate)

def make_precision_curve(y_true, y_score, ks=(1,2,5,7,10,12,15,20,25,30,35,40,45,50)):
    rows = []
    for k in ks:
        p = precision_at_k(y_true, y_score, k)
        rows.append({"k":k, "precision":p})
    return pd.DataFrame(rows)

# -----------------------------
# 3) 页面布局
# -----------------------------
st.title("📉 Telco Churn Targeting – Top-K & Precision/Lift")
show_debug = st.checkbox("显示调试信息", value=False)

tab_single, tab_batch = st.tabs(["🧍 单个预测", "📦 批量评分（CSV 上传）"])

# ============================
# A) 单个预测
# ============================
with tab_single:
    st.subheader("1) 手动输入一个用户，返回流失概率")
    col1, col2, col3 = st.columns(3)

    tenure = col1.number_input("tenure（月）", min_value=0, max_value=120, value=3)
    monthly = col2.number_input("MonthlyCharges", min_value=0.0, max_value=200.0, value=85.0, step=0.1)
    contract = col3.selectbox("Contract", ["Month-to-month","One year","Two year"])

    col4, col5, col6 = st.columns(3)
    internet = col4.selectbox("InternetService", ["DSL","Fiber optic","No"])
    pay = col5.selectbox("PaymentMethod", [
        "Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"
    ])
    paperless = col6.selectbox("PaperlessBilling", ["Yes","No"])

    col7, col8, col9 = st.columns(3)
    senior = col7.selectbox("SeniorCitizen", [0,1])
    partner = col8.selectbox("Partner", ["Yes","No"])
    depend = col9.selectbox("Dependents", ["Yes","No"])

    # 组装一条记录
    one = pd.DataFrame([{
        "tenure":tenure, "MonthlyCharges":monthly, "Contract":contract,
        "InternetService":internet, "PaymentMethod":pay, "PaperlessBilling":paperless,
        "SeniorCitizen":senior, "Partner":partner, "Dependents":depend
    }])
    one = coerce_types(one)

    if show_debug:
        st.write("DEBUG dtypes:", one.dtypes.astype(str).to_dict())
        st.write("DEBUG row:", one.iloc[0].to_dict())

    if st.button("计算流失概率", use_container_width=True):
        try:
            proba = pipe.predict_proba(one)[0,1]
            st.metric("Churn Probability", f"{proba:.2%}")
            st.write("**原因提示（供运营话术参考）**：", "、".join(reason_hints(one.iloc[0])) or "—")
        except Exception as e:
            st.error(f"预测失败：{e}")

# ============================
# B) 批量评分 + Top-K 评估
# ============================
with tab_batch:
    st.subheader("2) 上传 CSV 进行批量评分（可含 Churn 标签用于线下评估）")

    uploaded = st.file_uploader("上传 CSV（列名需与训练一致；缺列会自动兜底）", type=["csv"])
    if uploaded is None:
        st.info("你也可以先用示例：")
        demo = pd.DataFrame([
            {"customerID":"0001","tenure":3,"MonthlyCharges":88.5,"Contract":"Month-to-month","InternetService":"Fiber optic",
             "PaymentMethod":"Electronic check","PaperlessBilling":"Yes","SeniorCitizen":0,"Partner":"No","Dependents":"No","Churn":"Yes"},
            {"customerID":"0002","tenure":18,"MonthlyCharges":65.0,"Contract":"One year","InternetService":"DSL",
             "PaymentMethod":"Bank transfer (automatic)","PaperlessBilling":"No","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","Churn":"No"},
            {"customerID":"0003","tenure":55,"MonthlyCharges":102.2,"Contract":"Month-to-month","InternetService":"Fiber optic",
             "PaymentMethod":"Electronic check","PaperlessBilling":"Yes","SeniorCitizen":1,"Partner":"No","Dependents":"No","Churn":"Yes"},
            {"customerID":"0004","tenure":2,"MonthlyCharges":25.2,"Contract":"Month-to-month","InternetService":"No",
             "PaymentMethod":"Mailed check","PaperlessBilling":"No","SeniorCitizen":0,"Partner":"No","Dependents":"No","Churn":"No"},
            {"customerID":"0005","tenure":40,"MonthlyCharges":72.0,"Contract":"Two year","InternetService":"DSL",
             "PaymentMethod":"Credit card (automatic)","PaperlessBilling":"No","SeniorCitizen":0,"Partner":"Yes","Dependents":"Yes","Churn":"No"},
        ])
        df_raw = demo
    else:
        df_raw = pd.read_csv(uploaded)

    df = coerce_types(df_raw)

    if show_debug:
        st.write("DEBUG head dtypes:", df.dtypes.astype(str).to_dict())

    # 评分
    with st.spinner("Scoring..."):
        try:
            X_infer = df.drop(columns=[c for c in ["Churn"] if c in df.columns])
            proba = pipe.predict_proba(X_infer)[:,1]
            scored = df.copy()
            scored["churn_proba"] = proba
            scored = scored.sort_values("churn_proba", ascending=False).reset_index(drop=True)
            scored["rank"] = np.arange(1, len(scored)+1)
        except Exception as e:
            st.error(f"批量评分失败：{e}")
            st.stop()

    # Top-K 控件
    st.markdown("### 选择预算：Top-K%")
    k_pct = st.slider("Top-K（%）", 1, 50, 10, 1)
    k = max(1, int(len(scored) * k_pct / 100.0))
    scored["topk_flag"] = (scored["rank"] <= k).astype(int)
    scored["reason_hints"] = scored.apply(reason_hints, axis=1)

    # 指标：Precision@K & Lift@K（仅当有标签时计算）
    base_rate = None
    y_true = None
    if "Churn" in df.columns:
        y_true = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
        base_rate = float(y_true.mean())
        prec_k = precision_at_k(y_true, scored["churn_proba"], k_pct)
        lift_k = lift_at_k(prec_k, base_rate)
    else:
        prec_k, lift_k = None, None

    c1,c2,c3 = st.columns(3)
    c1.metric("Top-K 人数", f"{k}")
    c2.metric("Precision@K", f"{prec_k:.3f}" if prec_k is not None else "—（无标签）")
    c3.metric("Lift@K", f"{lift_k:.2f}x" if lift_k is not None else "—（无标签）")

    # 曲线（仅有标签时画）
    if y_true is not None:
        st.markdown("#### Precision@K 曲线（用于选择预算点）")
        df_curve = make_precision_curve(y_true, scored["churn_proba"])
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(df_curve["k"], df_curve["precision"], marker="o", label="Precision@K")
        ax.set_xlabel("Top-K (%)"); ax.set_ylabel("Precision"); ax.grid(True, alpha=0.3); ax.legend()
        st.pyplot(fig)
        st.caption(f"基线流失率（整包数据）：{base_rate:.3f}")

    # Top-K 名单 + 下载
    st.markdown("#### Top-K 高风险名单（含理由提示）")
    cols_show = [c for c in ["customerID","tenure","MonthlyCharges","Contract",
                             "InternetService","PaymentMethod","PaperlessBilling",
                             "SeniorCitizen","Partner","Dependents","churn_proba","rank","reason_hints"]
                 if c in scored.columns]
    st.dataframe(scored.loc[scored["topk_flag"]==1, cols_show].head(100), use_container_width=True)

    csv_topk = scored.loc[scored["topk_flag"]==1].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ 下载 Top-K 名单 CSV", data=csv_topk, file_name=f"top_{k_pct}pct_churn_list.csv", mime="text/csv")
