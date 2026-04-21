from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_supernova_dataset import (
    HISTORY_START,
    NY_TZ,
    chunked,
    compute_ttm_squeeze,
    fetch_stockanalysis_symbols,
    get_info_snapshot,
    get_reverse_splits,
    normalize_download_frame,
    rs_features_for_event,
)


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "clean_recovery_supernova_labeled_dataset.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "spike_model.joblib"
METRICS_PATH = MODELS_DIR / "spike_model_metrics.json"
DEFAULT_SCREEN_PATH = ROOT / "spike_scan_results.csv"

POSITIVE_LABELS = {"CLEAN_RECOVERY_2X", "NEAR_POSITIVE", "OPEN_DRIVE_SQUEEZE"}
TOXIC_LABEL = "TOXIC_EXCLUDE"

# These pages are intentionally used as a compact candidate universe, not as a full listed-symbol universe.
DEFAULT_UNIVERSE_URLS = [
    "https://stockanalysis.com/markets/active/",
    "https://stockanalysis.com/markets/gainers/",
    "https://stockanalysis.com/markets/gainers/week/",
    "https://stockanalysis.com/markets/gainers/month/",
    "https://stockanalysis.com/markets/gainers/ytd/",
    "https://stockanalysis.com/markets/losers/",
    "https://stockanalysis.com/markets/losers/week/",
    "https://stockanalysis.com/markets/losers/month/",
]

YES_NO_COLUMNS = [
    "reverse_split_in_last_24m",
    "has_never_reverse_split",
    "former_runner_history",
    "clean_collapse_recovery_candidate",
    "price_above_sma50_yes_no",
    "price_above_sma200_yes_no",
    "vcp_like_contraction_yes_no",
    "ttm_squeeze_on",
    "ttm_squeeze_fired",
]

MODEL_FEATURES = [
    "log_prev_close",
    "log_avg_volume_20d",
    "reverse_split_recent",
    "never_reverse_split",
    "former_runner_history_flag",
    "clean_collapse_recovery_flag",
    "prior_explosive_days",
    "highest_intraday_expansion_252d_pct",
    "max_volume_shock_252d",
    "drawdown_from_major_high_pct",
    "years_since_major_peak",
    "distance_from_52w_low_pct",
    "distance_to_20d_high_pct",
    "distance_to_50d_high_pct",
    "price_vs_sma20_pct",
    "price_vs_sma50_pct",
    "price_vs_sma150_pct",
    "price_vs_sma200_pct",
    "price_above_sma50_flag",
    "price_above_sma200_flag",
    "vcp_like_contraction_flag",
    "ttm_squeeze_on_flag",
    "ttm_squeeze_fired_flag",
    "atr_pct",
    "log_amihud_illiquidity",
]


def yes_no_to_int(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    if text == "YES":
