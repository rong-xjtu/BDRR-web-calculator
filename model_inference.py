# -*- coding: utf-8 -*-
"""Model loading and inference utilities for the BDRR web calculator.

Expected model directory structure:

saved_models/
  best_model_YYYYMMDD_HHMMSS/
    best_model.pkl
    preprocessing_pipeline.pkl
    model_metadata.json          # optional but recommended
    feature_importance.csv       # optional

The preprocessing_pipeline.pkl file is expected to be a dict containing at least:
- feature_names: list[str]
- scaler: fitted sklearn scaler or None
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

HIGH_RISK_THRESHOLD = 0.116
SENSITIVITY_LEVEL = "90%"

FEATURE_NAME_MAP = {
    "Ki-67 proportion": "Ki-67 labeling index",
    "Ki-67 proportiong": "Ki-67 labeling index",
    "PR proportion": "PR positive percentage",
    "PR proportiong": "PR positive percentage",
    "Preoperative Long /short diameter": "Sonographic long-to-short axis ratio",
    "Histologic grade": "Histologic grade",
    "ER proportion": "ER positive percentage",
    "HER-2 IHC": "HER2 status",
    "Family history": "Family history of breast cancer",
}

DEFAULT_ORDERED_FEATURES = [
    "Age",
    "Family history",
    "Preoperative Long /short diameter",
    "Histologic grade",
    "ER proportion",
    "PR status",
    "PR proportion",
    "Ki-67 proportion",
    "HER-2 IHC",
    "p53",
]


def display_name(name: str) -> str:
    return FEATURE_NAME_MAP.get(name, name)


def rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=FEATURE_NAME_MAP)


class ModelInference:
    """Load a saved sklearn/xgboost-style classifier and run inference."""

    def __init__(self, model_dir: Optional[Union[str, Path]] = None, threshold: float = HIGH_RISK_THRESHOLD):
        self.threshold = float(threshold)
        self.model_dir = Path(model_dir) if model_dir else self.find_latest_model()
        self.model = None
        self.pipeline_data: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.load_error: Optional[str] = None

        if self.model_dir is None:
            self.load_error = "未找到 saved_models/best_model_* 模型目录。"
            return
        self._load_model_and_pipeline()

    @staticmethod
    def find_latest_model(base_dir: Union[str, Path] = "saved_models") -> Optional[Path]:
        base = Path(base_dir)
        if not base.exists():
            return None
        candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("best_model_")]
        if not candidates:
            return None
        return sorted(candidates, key=lambda p: p.name, reverse=True)[0]

    def _load_model_and_pipeline(self) -> bool:
        try:
            model_path = self.model_dir / "best_model.pkl"
            pipeline_path = self.model_dir / "preprocessing_pipeline.pkl"
            metadata_path = self.model_dir / "model_metadata.json"

            if not model_path.exists():
                self.load_error = f"模型文件不存在：{model_path}"
                return False
            if not pipeline_path.exists():
                self.load_error = f"预处理管道文件不存在：{pipeline_path}"
                return False

            self.model = joblib.load(model_path)
            loaded_pipeline = joblib.load(pipeline_path)
            self.pipeline_data = loaded_pipeline if isinstance(loaded_pipeline, dict) else {"pipeline": loaded_pipeline}

            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
            return True
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"加载模型/预处理文件失败：{exc}"
            return False

    @property
    def feature_names(self) -> list[str]:
        names = self.pipeline_data.get("feature_names") or self.pipeline_data.get("selected_features")
        return list(names) if names else list(DEFAULT_ORDERED_FEATURES)

    def preprocess_data(self, data: Union[pd.DataFrame, Dict[str, Any], Iterable[Any]]) -> pd.DataFrame:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            values = list(data)
            if len(values) != len(self.feature_names):
                raise ValueError(f"输入变量数量为 {len(values)}，但模型需要 {len(self.feature_names)} 个变量。")
            df = pd.DataFrame([values], columns=self.feature_names)

        # Remove common target columns if accidentally uploaded.
        for target_col in ["Risk group", "risk group", "Target", "target"]:
            if target_col in df.columns:
                df = df.drop(columns=[target_col])

        # Fill missing features with 0 so older input templates remain usable.
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        df = df[self.feature_names].copy()

        # Convert all variables to numeric. The web UI uses numeric encodings.
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        scaler = self.pipeline_data.get("scaler")
        model_name = self.metadata.get("model_name", "")
        needs_scaling = model_name in ["Logistic Regression", "Neural Network", "SGD Classifier"]
        if needs_scaling and scaler is not None:
            scaled = scaler.transform(df)
            df = pd.DataFrame(scaled, columns=self.feature_names, index=df.index)

        return df

    def predict_proba(self, data: Union[pd.DataFrame, Dict[str, Any], Iterable[Any]]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(self.load_error or "模型未加载。")
        x = self.preprocess_data(data)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(x)
            probs_pos = 1 / (1 + np.exp(-scores))
            return np.vstack([1 - probs_pos, probs_pos]).T
        raise RuntimeError("当前模型不支持 predict_proba 或 decision_function。")

    def predict(self, data: Union[pd.DataFrame, Dict[str, Any], Iterable[Any]]) -> np.ndarray:
        probs = self.predict_proba(data)
        high_risk_probs = probs[:, 1] if probs.shape[1] >= 2 else np.max(probs, axis=1)
        return (high_risk_probs >= self.threshold).astype(int)

    def predict_dataframe(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        x_raw = data.copy()
        probs = self.predict_proba(x_raw)
        high_risk_probs = probs[:, 1] if probs.shape[1] >= 2 else np.max(probs, axis=1)
        pred = (high_risk_probs >= self.threshold).astype(int)

        result = x_raw.copy()
        result["Prediction"] = np.where(pred == 1, "High RS", "Low RS")
        result["Predicted Probability of High RS"] = high_risk_probs
        result["Predicted Probability of High RS (%)"] = high_risk_probs * 100
        result["Threshold"] = self.threshold
        return result, self.preprocess_data(x_raw)

    def model_info(self) -> Dict[str, Any]:
        return {
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "model_type": type(self.model).__name__ if self.model is not None else None,
            "threshold": self.threshold,
            "sensitivity_level": SENSITIVITY_LEVEL,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
            "load_error": self.load_error,
        }
