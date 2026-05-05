# -*- coding: utf-8 -*-
from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from model_inference import (
    DEFAULT_ORDERED_FEATURES,
    HIGH_RISK_THRESHOLD,
    SENSITIVITY_LEVEL,
    ModelInference,
    display_name,
    rename_for_display,
)

st.set_page_config(
    page_title="BDRR Calculator",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stMetric {background: #f8fafc; border: 1px solid #e5e7eb; padding: 1rem; border-radius: 0.85rem;}
    div[data-testid="stAlert"] {border-radius: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("BDRR Calculator")
st.caption(
    "A web-based calculator for estimating the probability of 21-gene recurrence score ≥ 26 "
    "in HR-positive/HER2-negative early breast cancer."
)

with st.expander("Important note", expanded=False):
    st.write(
        "This calculator is intended for research and decision-support use only. "
        "It should not replace the 21-gene assay, multidisciplinary assessment, or guideline-based clinical judgment."
    )


@st.cache_resource(show_spinner="Loading model...")
def load_inference(threshold: float) -> ModelInference:
    return ModelInference(model_dir=None, threshold=threshold)


with st.sidebar:
    st.header("Prediction settings")
    threshold = st.number_input(
        "Classification threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(HIGH_RISK_THRESHOLD),
        step=0.001,
        format="%.3f",
        help="Default threshold is the 90% sensitivity threshold exported from the training pipeline.",
    )
    st.write(f"Sensitivity strategy: **{SENSITIVITY_LEVEL}**")
    st.divider()

inference = load_inference(threshold)

if inference.model is None or inference.pipeline_data is None or inference.load_error:
    st.error(inference.load_error or "Model files are not available.")
    st.info(
        "Please place your exported model files under `saved_models/best_model_*/`: "
        "`best_model.pkl`, `preprocessing_pipeline.pkl`, and preferably `model_metadata.json`."
    )
    st.stop()

info = inference.model_info()
with st.sidebar:
    st.success("Model loaded")
    st.caption(f"Model: {info.get('model_type')}")
    md = info.get("metadata", {}) or {}
    if md.get("test_auc") is not None:
        st.caption(f"Test AUC: {float(md.get('test_auc')):.4f}")
    if md.get("test_accuracy") is not None:
        st.caption(f"Test accuracy: {float(md.get('test_accuracy')):.4f}")
    st.caption(f"Directory: {info.get('model_dir')}")

# ------------------------ Manual calculator ------------------------
st.sidebar.header("Patient characteristics")

age = st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=50, step=1)
family_history = st.sidebar.selectbox("Family history of breast cancer", options=[0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
ls_ratio = st.sidebar.number_input("Sonographic long-to-short axis ratio", min_value=0.0, max_value=10.0, value=1.20, step=0.05, format="%.2f")
grade = st.sidebar.selectbox("Histologic grade", options=[1, 2, 3], index=1, format_func=lambda x: {1: "I", 2: "II", 3: "III"}[x])
er_prop = st.sidebar.number_input("ER positive percentage (%)", min_value=0, max_value=100, value=60, step=1)
pr_status = st.sidebar.selectbox("PR status", options=[0, 1], index=1, format_func=lambda x: "Positive" if x == 1 else "Negative")
pr_prop = st.sidebar.number_input("PR positive percentage (%)", min_value=0, max_value=100, value=50, step=1)
ki67 = st.sidebar.number_input("Ki-67 labeling index (%)", min_value=0, max_value=100, value=20, step=1)
her2 = st.sidebar.selectbox("HER2 status by IHC", options=[0, 1, 2, 3], index=0, format_func=lambda x: {0: "0", 1: "1+", 2: "2+", 3: "3+"}[x])
p53 = st.sidebar.selectbox("p53", options=[0, 1], index=0, format_func=lambda x: "Aberrant pattern" if x == 1 else "Wild-type pattern")

input_dict = {
    "Age": age,
    "Family history": family_history,
    "Preoperative Long /short diameter": ls_ratio,
    "Histologic grade": grade,
    "ER proportion": er_prop,
    "PR status": pr_status,
    "PR proportion": pr_prop,
    "Ki-67 proportion": ki67,
    "HER-2 IHC": her2,
    "p53": p53,
}

feature_names = inference.feature_names
row_for_model = {k: input_dict.get(k, 0) for k in feature_names}
row_for_display = {k: input_dict.get(k, 0) for k in DEFAULT_ORDERED_FEATURES}

col_input, col_pred = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("Current inputs")
    st.dataframe(rename_for_display(pd.DataFrame([row_for_display])), use_container_width=True, hide_index=True)

with col_pred:
    st.subheader("Single-patient prediction")
    if st.button("Predict", type="primary", use_container_width=True):
        try:
            y_pred = inference.predict(row_for_model)
            y_proba = inference.predict_proba(row_for_model)
            proba = float(y_proba[0][1]) if y_proba.shape[1] >= 2 else float(np.max(y_proba[0]))
            pred = int(y_pred[0])
            risk_label = "High RS" if pred == 1 else "Low RS"

            st.metric("Risk category", risk_label)
            st.progress(min(max(proba, 0.0), 1.0))
            st.markdown(f"**Predicted probability of High RS: {proba * 100:.1f}%**")
            st.markdown(f"Classification threshold: **{inference.threshold:.3f}**")

            if pred == 1:
                st.warning("The model classifies this case as High RS under the current threshold. 21-gene testing is recommended where feasible.")
            else:
                st.success("The model classifies this case as Low RS under the current threshold. Interpret together with clinical context.")

            out_df = pd.DataFrame([row_for_model])
            out_df["Prediction"] = risk_label
            out_df["Predicted Probability of High RS"] = proba
            out_df["Predicted Probability of High RS (%)"] = proba * 100
            out_df["Threshold"] = inference.threshold
            csv = rename_for_display(out_df).to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download this prediction as CSV",
                data=csv,
                file_name="single_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.subheader("Instance-level SHAP explanation")
            with st.spinner("Computing SHAP values..."):
                x_processed = inference.preprocess_data(row_for_model)
                model = inference.model
                model_type = str(type(model)).lower()
                try:
                    if any(key in model_type for key in ["xgboost", "lightgbm", "catboost", "forest", "gradientboosting", "histgradientboosting"]):
                        explainer = shap.TreeExplainer(model)
                    elif any(key in model_type for key in ["logistic", "linear", "sgd"]):
                        explainer = shap.LinearExplainer(model, x_processed)
                    else:
                        explainer = shap.Explainer(model, x_processed)
                    shap_values = explainer(x_processed)
                    exp_one = shap_values[0]
                    exp_one.feature_names = [display_name(str(name)) for name in x_processed.columns]

                    fig = plt.figure(figsize=(8, 5))
                    shap.plots.waterfall(exp_one, max_display=12, show=False)
                    st.pyplot(fig, clear_figure=True)
                except Exception as exc:  # noqa: BLE001
                    st.info(f"SHAP explanation could not be generated for this model: {exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")

st.divider()

# ------------------------ Batch prediction ------------------------
st.subheader("Batch prediction")
st.write("Upload a CSV or Excel file containing the model variables. Extra columns will be kept in the output; target columns are ignored.")
uploaded_file = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix == ".csv":
            batch_df = pd.read_csv(uploaded_file)
        else:
            batch_df = pd.read_excel(uploaded_file)

        st.write("Preview of uploaded data")
        st.dataframe(rename_for_display(batch_df.head(10)), use_container_width=True, hide_index=True)

        if st.button("Run batch prediction", use_container_width=True):
            result_df, _ = inference.predict_dataframe(batch_df)
            result_disp = rename_for_display(result_df)
            st.success(f"Batch prediction completed: {len(result_df)} rows")
            st.dataframe(result_disp.head(50), use_container_width=True, hide_index=True)

            csv = result_disp.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download batch results as CSV",
                data=csv,
                file_name="batch_prediction_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read or predict this file: {exc}")

with st.expander("Required variable names for batch upload"):
    st.write("The app uses the feature order saved in `preprocessing_pipeline.pkl`. Current feature names are:")
    st.code("\n".join(feature_names))
