from __future__ import annotations

import gzip
import json
import math
import ssl
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NY_TZ = ZoneInfo("America/New_York")
HISTORY_START = "2023-01-01"


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def fetch_url_text(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=30, context=ssl.create_default_context()) as response:
        data = response.read()
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data).decode("utf-8", "ignore")
    return data.decode("utf-8", "ignore")


def fetch_stockanalysis_symbols(url: str) -> list[str]:
    html = fetch_url_text(url)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return []
    rows = table.find_all("tr")[1:]
    symbols = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        symbol = cells[1].get_text(" ", strip=True).upper()
        if symbol.isascii():
            symbols.append(symbol)
    return symbols


def normalize_download_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker not in raw.columns.levels[0]:
            return pd.DataFrame()
        frame = raw[ticker].copy()
    else:
        frame = raw.copy()
    frame = frame.rename_axis("Date").reset_index()
    if "Date" not in frame.columns:
        return pd.DataFrame()
    parsed = pd.to_datetime(frame["Date"])
    if getattr(parsed.dt, "tz", None) is None:
        parsed = parsed.dt.tz_localize(NY_TZ)
    else:
        parsed = parsed.dt.tz_convert(NY_TZ)
    frame["Date"] = parsed
    needed = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    available = [col for col in needed if col in frame.columns]
    frame = frame[available].copy()
    return frame.dropna(subset=["Open", "High", "Low", "Close"], how="any")


def get_reverse_splits(ticker: str) -> pd.Series:
    try:
        splits = yf.Ticker(ticker).splits
    except Exception:
        return pd.Series(dtype=float)
    if splits is None or len(splits) == 0:
        return pd.Series(dtype=float)
    splits = splits[splits < 1]
    if len(splits) == 0:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(splits.index)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize(NY_TZ)
    else:
        idx = idx.tz_convert(NY_TZ)
    out = pd.Series(splits.values, index=idx)
    out.index.name = "Date"
    return out.sort_index()


def get_info_snapshot(ticker: str) -> dict[str, Any]:
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return {}
    if not isinstance(info, dict):
        return {}
    return info


def compute_ttm_squeeze(close: pd.Series, high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    basis = close.shift(1).rolling(20).mean()
    std = close.shift(1).rolling(20).std()
    bb_upper = basis + 2 * std
    bb_lower = basis - 2 * std

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr20 = tr.shift(1).rolling(20).mean()
    kc_upper = basis + 1.5 * atr20
    kc_lower = basis - 1.5 * atr20

    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    squeeze_fired = (~squeeze_on) & squeeze_on.shift(1).fillna(False)
    return squeeze_on, squeeze_fired


def rs_features_for_event(reverse_splits: pd.Series, event_date: pd.Timestamp) -> tuple[bool, str, bool]:
    if reverse_splits.empty:
        return False, "", True
    event_date = pd.Timestamp(event_date)
    if event_date.tzinfo is None:
        event_date = event_date.tz_localize(NY_TZ)
    else:
        event_date = event_date.tz_convert(NY_TZ)
    recent_cut = event_date - pd.Timedelta(days=730)
    up_to_event = reverse_splits[reverse_splits.index <= event_date]
    rs_last_24m = bool(((up_to_event.index >= recent_cut) & (up_to_event.index <= event_date)).any())
    rs_dates = ";".join(d.strftime("%Y-%m-%d") for d in up_to_event.index)
    has_never = len(up_to_event) == 0
    return rs_last_24m, rs_dates, has_never


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "clean_recovery_supernova_labeled_dataset.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "spike_model.joblib"
METRICS_PATH = MODELS_DIR / "spike_model_metrics.json"
DEFAULT_SCREEN_PATH = ROOT / "spike_scan_results.csv"

CONTINUATION_LABELS = {"CLEAN_RECOVERY_2X", "NEAR_POSITIVE"}
SUPERNOVA_LABELS = {"CLEAN_RECOVERY_2X", "OPEN_DRIVE_SQUEEZE"}
POSITIVE_LABELS = {"CLEAN_RECOVERY_2X", "NEAR_POSITIVE", "OPEN_DRIVE_SQUEEZE"}
TOXIC_LABEL = "TOXIC_EXCLUDE"
FAILED_SETUP_LABEL = "FAILED_SETUP"

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
    "log_avg_dollar_volume_20d",
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
    "rvol_raw",
    "close_strength_raw",
    "atr_pct",
    "log_amihud_illiquidity",
]

FOCUS_MAX_PICKS_DEFAULT = 2
CONTINUATION_MAX_PICKS_DEFAULT = 2
SUPERNOVA_ALERT_MAX_PICKS_DEFAULT = 1

BUDGET_PRICE_MIN = 0.75
BUDGET_PRICE_MAX = 25.0
SUPERNOVA_MARKET_CAP_MAX = 2_500_000_000
CONTINUATION_MARKET_CAP_MAX = 6_000_000_000
SUPERNOVA_FLOAT_MAX = 80_000_000
CONTINUATION_FLOAT_MAX = 220_000_000
SUPERNOVA_SCORE_THRESHOLD = 64.0
CONTINUATION_SCORE_THRESHOLD = 67.0
SUPERNOVA_ALERT_THRESHOLD = 79.0

MIN_TRAIN_AVG_DOLLAR_VOLUME = 500_000
MIN_CONTINUATION_AVG_DOLLAR_VOLUME = 5_000_000
MIN_CONTINUATION_DAY_DOLLAR_VOLUME = 10_000_000
MIN_CONTINUATION_PREMARKET_DOLLAR_VOLUME = 500_000
MIN_SUPERNOVA_AVG_DOLLAR_VOLUME = 3_000_000
MIN_SUPERNOVA_DAY_DOLLAR_VOLUME = 8_000_000
MIN_SUPERNOVA_PREMARKET_DOLLAR_VOLUME = 1_000_000
MAX_CONTINUATION_FAILED_SETUP_PROBABILITY = 0.32
MAX_SUPERNOVA_FAILED_SETUP_PROBABILITY = 0.28


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def to_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def yes_no_flag(value: Any) -> bool:
    return str(value).strip().upper() == "YES"


def scale_0_1(value: Any, low: float, high: float) -> float:
    numeric = to_float(value)
    if numeric is None or high <= low:
        return 0.0
    return clamp((numeric - low) / (high - low), 0.0, 1.0)


def breakout_pressure_score(distance_pct: Any, max_below_pct: float = 12.0) -> float:
    numeric = to_float(distance_pct)
    if numeric is None:
        return 0.0
    if numeric >= 0:
        return 1.0
    return clamp(1.0 - abs(numeric) / max_below_pct, 0.0, 1.0)


def hold_quality_score(value: Any) -> float:
    text = str(value or "").strip().upper()
    if text == "GOOD":
        return 1.0
    if text == "FAIR":
        return 0.65
    if text == "POOR":
        return 0.15
    return 0.35


def yes_no_to_int(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    if text == "YES":
        return 1.0
    if text == "NO":
        return 0.0
    return np.nan


def safe_ratio_pct(numerator: float | None, denominator: float | None) -> float | None:
    if denominator in (0, None) or numerator is None:
        return None
    if pd.isna(numerator) or pd.isna(denominator):
        return None
    return (float(numerator) / float(denominator) - 1.0) * 100.0


def log1p_safe(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(np.log1p(max(float(value), 0.0)))


def coalesce_numeric(raw: pd.DataFrame, primary: str, fallback: str | None = None) -> pd.Series:
    if primary in raw.columns:
        primary_series = pd.to_numeric(raw[primary], errors="coerce")
    else:
        primary_series = pd.Series(np.nan, index=raw.index, dtype=float)
    if fallback is None:
        return primary_series
    if fallback in raw.columns:
        fallback_series = pd.to_numeric(raw[fallback], errors="coerce")
    else:
        fallback_series = pd.Series(np.nan, index=raw.index, dtype=float)
    return primary_series.fillna(fallback_series)


def filter_training_rows(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    prev_close = pd.to_numeric(df["prev_close"], errors="coerce")
    avg_volume_20d = pd.to_numeric(df["avg_volume_20d"], errors="coerce")
    avg_dollar_volume_20d = prev_close * avg_volume_20d
    keep = (
        prev_close.notna()
        & avg_volume_20d.notna()
        & (prev_close > 0)
        & (avg_volume_20d > 0)
        & (avg_dollar_volume_20d >= MIN_TRAIN_AVG_DOLLAR_VOLUME)
    )
    return df.loc[keep].reset_index(drop=True)


def load_labeled_dataset(csv_path: Path | str = DATASET_PATH) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, parse_dates=["event_date"])
    frame = filter_training_rows(frame)
    frame["target_spike"] = frame["label"].isin(POSITIVE_LABELS).astype(int)
    frame["target_continuation"] = frame["label"].isin(CONTINUATION_LABELS).astype(int)
    frame["target_supernova"] = frame["label"].isin(SUPERNOVA_LABELS).astype(int)
    frame["target_toxic"] = (frame["label"] == TOXIC_LABEL).astype(int)
    frame["target_failed_setup"] = (frame["label"] == FAILED_SETUP_LABEL).astype(int)
    return frame


def engineer_model_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    features = pd.DataFrame(index=df.index)

    prev_close = pd.to_numeric(df["prev_close"], errors="coerce")
    avg_volume_20d = pd.to_numeric(df["avg_volume_20d"], errors="coerce")
    avg_dollar_volume_20d = prev_close * avg_volume_20d
    sma20 = pd.to_numeric(df["sma20"], errors="coerce")
    sma50 = pd.to_numeric(df["sma50"], errors="coerce")
    sma150 = pd.to_numeric(df["sma150"], errors="coerce")
    sma200 = pd.to_numeric(df["sma200"], errors="coerce")
    atr_14 = pd.to_numeric(df["atr_14"], errors="coerce")
    amihud = pd.to_numeric(df["amihud_illiquidity_proxy"], errors="coerce")
    rvol = coalesce_numeric(df, "rvol", "latest_rvol")
    close_strength = coalesce_numeric(df, "close_strength", "latest_close_strength")

    features["log_prev_close"] = prev_close.map(log1p_safe)
    features["log_avg_volume_20d"] = avg_volume_20d.map(log1p_safe)
    features["log_avg_dollar_volume_20d"] = avg_dollar_volume_20d.map(log1p_safe)
    features["reverse_split_recent"] = df["reverse_split_in_last_24m"].map(yes_no_to_int)
    features["never_reverse_split"] = df["has_never_reverse_split"].map(yes_no_to_int)
    features["former_runner_history_flag"] = df["former_runner_history"].map(yes_no_to_int)
    features["clean_collapse_recovery_flag"] = df["clean_collapse_recovery_candidate"].map(yes_no_to_int)
    features["prior_explosive_days"] = pd.to_numeric(df["number_of_prior_explosive_days"], errors="coerce")
    features["highest_intraday_expansion_252d_pct"] = pd.to_numeric(
        df["highest_intraday_expansion_last_252d_pct"], errors="coerce"
    )
    features["max_volume_shock_252d"] = pd.to_numeric(df["max_volume_shock_last_252d"], errors="coerce")
    features["drawdown_from_major_high_pct"] = pd.to_numeric(
        df["drawdown_from_major_high_pct"], errors="coerce"
    )
    features["years_since_major_peak"] = pd.to_numeric(df["years_since_major_peak"], errors="coerce")
    features["distance_from_52w_low_pct"] = pd.to_numeric(df["distance_from_52w_low_pct"], errors="coerce")
    features["distance_to_20d_high_pct"] = pd.to_numeric(df["distance_to_20d_high_pct"], errors="coerce")
    features["distance_to_50d_high_pct"] = pd.to_numeric(df["distance_to_50d_high_pct"], errors="coerce")
    features["price_vs_sma20_pct"] = (prev_close / sma20 - 1.0) * 100.0
    features["price_vs_sma50_pct"] = (prev_close / sma50 - 1.0) * 100.0
    features["price_vs_sma150_pct"] = (prev_close / sma150 - 1.0) * 100.0
    features["price_vs_sma200_pct"] = (prev_close / sma200 - 1.0) * 100.0
    features["price_above_sma50_flag"] = df["price_above_sma50_yes_no"].map(yes_no_to_int)
    features["price_above_sma200_flag"] = df["price_above_sma200_yes_no"].map(yes_no_to_int)
    features["vcp_like_contraction_flag"] = df["vcp_like_contraction_yes_no"].map(yes_no_to_int)
    features["ttm_squeeze_on_flag"] = df["ttm_squeeze_on"].map(yes_no_to_int)
    features["ttm_squeeze_fired_flag"] = df["ttm_squeeze_fired"].map(yes_no_to_int)
    features["rvol_raw"] = rvol
    features["close_strength_raw"] = close_strength
    features["atr_pct"] = (atr_14 / prev_close) * 100.0
    features["log_amihud_illiquidity"] = amihud.map(log1p_safe)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features[MODEL_FEATURES]


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    C=0.5,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def build_dummy_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", DummyClassifier(strategy="prior")),
        ]
    )


def fit_binary_estimator(X: pd.DataFrame, y: pd.Series) -> Any:
    if y.nunique() < 2:
        return build_dummy_pipeline().fit(X, y)
    return build_pipeline().fit(X, y)


def extract_binary_coefficients(estimator: Any) -> np.ndarray:
    if isinstance(estimator, Pipeline):
        model = estimator.named_steps["model"]
        if hasattr(model, "coef_"):
            return model.coef_[0]
    return np.zeros(len(MODEL_FEATURES))


def predict_positive_probability(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    proba = estimator.predict_proba(X)
    classes = getattr(estimator, "classes_", None)
    if classes is None and isinstance(estimator, Pipeline):
        classes = getattr(estimator.named_steps["model"], "classes_", None)
    if proba.shape[1] == 1:
        if classes is not None and len(classes) == 1 and int(classes[0]) == 1:
            return np.ones(len(X))
        return np.zeros(len(X))
    if classes is not None:
        class_list = list(classes)
        if 1 in class_list:
            return proba[:, class_list.index(1)]
    return proba[:, -1]


def compute_binary_metrics(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    if y_true.nunique() < 2:
        return {
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "brier_score": float("nan"),
        }
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "brier_score": float(brier_score_loss(y_true, proba)),
    }


def evaluate_model(raw: pd.DataFrame, target_col: str) -> dict[str, Any]:
    X = engineer_model_frame(raw)
    y = raw[target_col].astype(int)
    if y.nunique() < 2:
        return {"cross_validation": compute_binary_metrics(y, np.zeros(len(y)))}
    pipeline = build_pipeline()

    min_class_size = int(y.value_counts().min())
    n_splits = max(3, min(5, min_class_size))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    metrics = {"cross_validation": compute_binary_metrics(y, cv_proba)}

    holdout_cutoff = pd.Timestamp("2026-01-01")
    holdout_mask = raw["event_date"] >= holdout_cutoff
    if holdout_mask.any() and (~holdout_mask).any():
        y_holdout = y[holdout_mask]
        y_train = y[~holdout_mask]
        if y_holdout.nunique() > 1 and y_train.nunique() > 1:
            fitted = build_pipeline().fit(X.loc[~holdout_mask], y_train)
            holdout_proba = fitted.predict_proba(X.loc[holdout_mask])[:, 1]
            metrics["time_holdout_2026"] = compute_binary_metrics(y_holdout, holdout_proba)

    return metrics


def fit_final_models(raw: pd.DataFrame) -> dict[str, Any]:
    X = engineer_model_frame(raw)
    spike_pipeline = fit_binary_estimator(X, raw["target_spike"].astype(int))
    continuation_pipeline = fit_binary_estimator(X, raw["target_continuation"].astype(int))
    supernova_pipeline = fit_binary_estimator(X, raw["target_supernova"].astype(int))
    toxic_pipeline = fit_binary_estimator(X, raw["target_toxic"].astype(int))
    failure_pipeline = fit_binary_estimator(X, raw["target_failed_setup"].astype(int))

    spike_coef = extract_binary_coefficients(spike_pipeline)
    continuation_coef = extract_binary_coefficients(continuation_pipeline)
    supernova_coef = extract_binary_coefficients(supernova_pipeline)
    toxic_coef = extract_binary_coefficients(toxic_pipeline)
    failure_coef = extract_binary_coefficients(failure_pipeline)
    feature_importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "spike_logit_coef": spike_coef,
            "continuation_logit_coef": continuation_coef,
            "supernova_logit_coef": supernova_coef,
            "toxic_logit_coef": toxic_coef,
            "failed_setup_logit_coef": failure_coef,
        }
    ).sort_values("spike_logit_coef", ascending=False)

    return {
        "spike_pipeline": spike_pipeline,
        "continuation_pipeline": continuation_pipeline,
        "supernova_pipeline": supernova_pipeline,
        "toxic_pipeline": toxic_pipeline,
        "failure_pipeline": failure_pipeline,
        "feature_names": MODEL_FEATURES,
        "positive_labels": sorted(POSITIVE_LABELS),
        "continuation_labels": sorted(CONTINUATION_LABELS),
        "supernova_labels": sorted(SUPERNOVA_LABELS),
        "toxic_label": TOXIC_LABEL,
        "failed_setup_label": FAILED_SETUP_LABEL,
        "trained_at": datetime.now(tz=NY_TZ).isoformat(),
        "training_rows": int(len(raw)),
        "feature_importance": feature_importance.to_dict(orient="records"),
    }


def save_artifact(artifact: dict[str, Any], path: Path | str = MODEL_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: Path | str = MODEL_PATH) -> dict[str, Any]:
    artifact = joblib.load(path)
    required_keys = {
        "continuation_pipeline",
        "supernova_pipeline",
        "toxic_pipeline",
        "failure_pipeline",
    }
    missing_keys = required_keys - set(artifact.keys())
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Incompatible model artifact, missing keys: {missing}")
    return artifact


def save_metrics(metrics: dict[str, Any], path: Path | str = METRICS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def get_candidate_universe(extra_tickers: list[str] | None = None) -> list[str]:
    symbols: list[str] = []
    for url in DEFAULT_UNIVERSE_URLS:
        try:
            symbols.extend(fetch_stockanalysis_symbols(url))
        except Exception as exc:
            print(f"[warn] could not parse {url}: {exc}")
    if extra_tickers:
        symbols.extend(extra_tickers)
    deduped = []
    seen = set()
    for symbol in symbols:
        ticker = symbol.strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        deduped.append(ticker)
    return deduped


def download_daily_universe(tickers: list[str]) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for batch in chunked(tickers, 25):
        try:
            raw = yf.download(
                batch,
                start=HISTORY_START,
                end=(pd.Timestamp.now(tz=NY_TZ) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as exc:
            print(f"[warn] batch download failed for {batch[:3]}...: {exc}")
            continue
        for ticker in batch:
            frame = normalize_download_frame(raw, ticker)
            if len(frame) >= 220:
                histories[ticker] = frame
    return histories


def build_scan_snapshot_from_history(ticker: str, frame: pd.DataFrame) -> dict[str, Any] | None:
    df = frame.copy().sort_values("Date").reset_index(drop=True)
    if len(df) < 220:
        return None

    last_close = float(df["Close"].iloc[-1])
    last_open = float(df["Open"].iloc[-1])
    last_high = float(df["High"].iloc[-1])
    last_low = float(df["Low"].iloc[-1])
    last_volume = float(df["Volume"].iloc[-1])
    if last_close <= 0:
        return None

    avg_volume_20d = float(df["Volume"].tail(20).mean())
    major_high_idx = int(df["High"].idxmax())
    major_high = float(df["High"].iloc[major_high_idx])
    major_high_date = pd.Timestamp(df["Date"].iloc[major_high_idx]).tz_convert(NY_TZ)
    high_20d = float(df["High"].tail(20).max())
    high_50d = float(df["High"].tail(50).max())

    prior_close = df["Close"].shift(1)
    expansion_pct_series = (df["High"] / prior_close - 1.0) * 100.0
    avg_volume_prior = df["Volume"].shift(1).rolling(20).mean()
    rvol_series = df["Volume"] / avg_volume_prior

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    squeeze_on, squeeze_fired = compute_ttm_squeeze(df["Close"], df["High"], df["Low"])
    daily_range = df["High"] - df["Low"]
    atr_14 = float(true_range.tail(14).mean()) if len(true_range.dropna()) >= 14 else None
    rolling_52w_low = float(df["Low"].tail(252).min())

    ret_abs = (df["Close"] / df["Close"].shift(1) - 1.0).abs()
    dollar_volume = df["Close"] * df["Volume"]
    amihud_series = (ret_abs / dollar_volume.replace(0, np.nan)).rolling(20).mean() * 1e6

    reverse_splits = get_reverse_splits(ticker)
    as_of_date = pd.Timestamp.now(tz=NY_TZ)
    rs_last_24m, rs_dates, has_never = rs_features_for_event(reverse_splits, as_of_date)

    info = get_info_snapshot(ticker)

    snapshot = {
        "ticker": ticker,
        "event_date": as_of_date.strftime("%Y-%m-%d"),
        "prev_close": last_close,
        "latest_open": last_open,
        "latest_high": last_high,
        "latest_low": last_low,
        "latest_day_volume": last_volume,
        "latest_dollar_volume": last_close * last_volume,
        "latest_rvol": (last_volume / avg_volume_20d) if avg_volume_20d else None,
        "latest_close_strength": ((last_close - last_low) / (last_high - last_low))
        if last_high > last_low
        else None,
        "latest_expansion_pct": ((last_high / float(df["Close"].iloc[-2])) - 1.0) * 100.0 if len(df) >= 2 else None,
        "avg_volume_20d": avg_volume_20d,
        "market_cap": info.get("marketCap"),
        "free_float": info.get("floatShares"),
        "short_float_pct": info.get("shortPercentOfFloat"),
        "days_to_cover": info.get("shortRatio"),
        "reverse_split_in_last_24m": "YES" if rs_last_24m else "NO",
        "reverse_split_dates": rs_dates,
        "has_never_reverse_split": "YES" if has_never else "NO",
        "former_runner_history": "YES" if bool((expansion_pct_series.tail(252) >= 80).sum()) else "NO",
        "number_of_prior_explosive_days": int((expansion_pct_series.tail(252) >= 80).sum()),
        "highest_intraday_expansion_last_252d_pct": float(expansion_pct_series.tail(252).max()),
        "max_volume_shock_last_252d": float(rvol_series.tail(252).max()),
        "clean_collapse_recovery_candidate": "YES"
        if (
            major_high > 0
            and (1.0 - last_close / major_high) * 100.0 >= 60
            and safe_ratio_pct(last_close, rolling_52w_low) is not None
            and safe_ratio_pct(last_close, rolling_52w_low) <= 50
            and float(df["Volume"].iloc[-1]) > avg_volume_20d * 3
        )
        else "NO",
        "years_since_major_peak": round((as_of_date - major_high_date).days / 365.25, 2),
        "distance_from_52w_low_pct": safe_ratio_pct(last_close, rolling_52w_low),
        "drawdown_from_major_high_pct": (1.0 - last_close / major_high) * 100.0 if major_high else None,
        "distance_to_20d_high_pct": safe_ratio_pct(last_close, high_20d),
        "distance_to_50d_high_pct": safe_ratio_pct(last_close, high_50d),
        "prior_20d_high": high_20d,
        "prior_50d_high": high_50d,
        "sma20": float(df["Close"].tail(20).mean()),
        "sma50": float(df["Close"].tail(50).mean()),
        "sma150": float(df["Close"].tail(150).mean()) if len(df) >= 150 else np.nan,
        "sma200": float(df["Close"].tail(200).mean()) if len(df) >= 200 else np.nan,
        "price_above_sma50_yes_no": "YES"
        if last_close > float(df["Close"].tail(50).mean())
        else "NO",
        "price_above_sma200_yes_no": "YES"
        if len(df) >= 200 and last_close > float(df["Close"].tail(200).mean())
        else "NO",
        "vcp_like_contraction_yes_no": "YES"
        if len(df) >= 60 and float(daily_range.tail(20).mean()) < float(daily_range.tail(60).mean()) * 0.75
        else "NO",
        "ttm_squeeze_on": "YES" if bool(squeeze_on.iloc[-1]) else "NO",
        "ttm_squeeze_fired": "YES" if bool(squeeze_fired.iloc[-1]) else "NO",
        "atr_14": atr_14,
        "amihud_illiquidity_proxy": float(amihud_series.iloc[-1]) if pd.notna(amihud_series.iloc[-1]) else None,
    }
    return snapshot


def fetch_current_premarket_context(ticker: str, prev_close: float | None) -> dict[str, Any]:
    today = pd.Timestamp.now(tz=NY_TZ).normalize()
    try:
        intraday = yf.Ticker(ticker).history(
            start=today.strftime("%Y-%m-%d"),
            end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1m",
            auto_adjust=False,
            prepost=True,
        )
    except Exception:
        return {}
    if intraday is None or intraday.empty:
        return {}

    intraday = intraday.reset_index()
    dt_col = "Datetime" if "Datetime" in intraday.columns else intraday.columns[0]
    intraday[dt_col] = pd.to_datetime(intraday[dt_col], utc=True).dt.tz_convert(NY_TZ)
    bars = intraday[intraday[dt_col].dt.normalize() == today]
    if bars.empty:
        return {}

    pre = bars[
        (bars[dt_col].dt.time >= datetime.strptime("04:00", "%H:%M").time())
        & (bars[dt_col].dt.time < datetime.strptime("09:30", "%H:%M").time())
    ].copy()
    if pre.empty:
        return {}

    pre_open = float(pre.iloc[0]["Open"])
    pre_high = float(pre["High"].max())
    pre_low = float(pre["Low"].min())
    pre_volume = float(pre["Volume"].sum())
    if pre_volume <= 0:
        return {}
    pre_dollar = float((pre["Close"] * pre["Volume"]).sum())
    pre_vwap = pre_dollar / pre_volume if pre_volume else None

    hold_quality = ""
    if pre_high > 0:
        last_pre_close = float(pre.iloc[-1]["Close"])
        hold_ratio = last_pre_close / pre_high
        if hold_ratio >= 0.9:
            hold_quality = "GOOD"
        elif hold_ratio >= 0.75:
            hold_quality = "FAIR"
        else:
            hold_quality = "POOR"

    return {
        "premarket_open": pre_open,
        "premarket_high": pre_high,
        "premarket_low": pre_low,
        "premarket_last": float(pre.iloc[-1]["Close"]),
        "premarket_volume": int(pre_volume),
        "premarket_dollar_volume": round(pre_dollar, 2),
        "premarket_vwap": round(pre_vwap, 6) if pre_vwap else None,
        "premarket_gap_pct": safe_ratio_pct(pre_open, prev_close),
        "premarket_hold_quality": hold_quality,
        "premarket_range_pct": safe_ratio_pct(pre_high, pre_low),
    }


def market_cap_penalty(row: pd.Series, ceiling: float) -> float:
    market_cap = to_float(row.get("market_cap"))
    if market_cap is None:
        return 0.0
    return scale_0_1(market_cap, ceiling, ceiling * 3.0)


def float_penalty(row: pd.Series, ceiling: float) -> float:
    free_float = to_float(row.get("free_float"))
    if free_float is None:
        return 0.0
    return scale_0_1(free_float, ceiling, ceiling * 3.0)


def short_squeeze_pressure(row: pd.Series) -> float:
    short_score = scale_0_1(row.get("short_float_pct"), 0.05, 0.30)
    cover_score = scale_0_1(row.get("days_to_cover"), 1.5, 8.0)
    float_score = 1.0 - scale_0_1(row.get("free_float"), 20_000_000, 180_000_000)
    return clamp(short_score * 0.5 + cover_score * 0.25 + float_score * 0.25, 0.0, 1.0)


def premarket_confirmation(row: pd.Series) -> float:
    gap_score = scale_0_1(row.get("premarket_gap_pct"), 2.0, 25.0)
    dollar_score = scale_0_1(row.get("premarket_dollar_volume"), 250_000, 10_000_000)
    hold_score = hold_quality_score(row.get("premarket_hold_quality"))
    return clamp(gap_score * 0.35 + dollar_score * 0.35 + hold_score * 0.30, 0.0, 1.0)


def structural_runner_score(row: pd.Series) -> float:
    former_runner = 1.0 if yes_no_flag(row.get("former_runner_history")) else 0.0
    explosive_history = scale_0_1(row.get("highest_intraday_expansion_last_252d_pct"), 40.0, 250.0)
    volume_shock = scale_0_1(row.get("max_volume_shock_last_252d"), 3.0, 20.0)
    return clamp(former_runner * 0.45 + explosive_history * 0.30 + volume_shock * 0.25, 0.0, 1.0)


def continuation_tape_score(row: pd.Series) -> float:
    trend_score = 0.0
    if yes_no_flag(row.get("price_above_sma50_yes_no")):
        trend_score += 0.4
    if yes_no_flag(row.get("price_above_sma200_yes_no")):
        trend_score += 0.2
    if yes_no_flag(row.get("vcp_like_contraction_yes_no")):
        trend_score += 0.15
    if yes_no_flag(row.get("ttm_squeeze_on")):
        trend_score += 0.10
    if yes_no_flag(row.get("ttm_squeeze_fired")):
        trend_score += 0.15
    close_strength = scale_0_1(row.get("latest_close_strength"), 0.45, 0.9)
    latest_rvol = scale_0_1(row.get("latest_rvol"), 1.1, 4.0)
    breakout_score = breakout_pressure_score(row.get("distance_to_20d_high_pct"), max_below_pct=8.0)
    return clamp(trend_score * 0.40 + close_strength * 0.25 + latest_rvol * 0.20 + breakout_score * 0.15, 0.0, 1.0)


def budget_liquidity_score(row: pd.Series) -> float:
    price = to_float(row.get("prev_close"))
    avg_volume = to_float(row.get("avg_volume_20d"))
    if price is None or avg_volume is None:
        return 0.0
    if price < BUDGET_PRICE_MIN or price > BUDGET_PRICE_MAX:
        return 0.0
    price_score = 1.0 - abs(clamp((price - 7.5) / 17.5, -1.0, 1.0))
    volume_score = scale_0_1(avg_volume, 800_000, 20_000_000)
    return clamp(price_score * 0.35 + volume_score * 0.65, 0.0, 1.0)


def avg_dollar_volume_20d(row: pd.Series) -> float | None:
    prev_close = to_float(row.get("prev_close"))
    avg_volume = to_float(row.get("avg_volume_20d"))
    if prev_close is None or avg_volume is None:
        return None
    return prev_close * avg_volume


def latest_dollar_volume(row: pd.Series) -> float | None:
    latest = to_float(row.get("latest_dollar_volume"))
    if latest is not None:
        return latest
    prev_close = to_float(row.get("prev_close"))
    day_volume = to_float(row.get("latest_day_volume"))
    if prev_close is None or day_volume is None:
        return None
    return prev_close * day_volume


def compute_supernova_probability(row: pd.Series) -> float:
    supernova_base = clamp(to_float(row.get("supernova_base_probability")) or 0.0, 0.0, 1.0)
    toxic = clamp(to_float(row.get("toxic_probability")) or 0.0, 0.0, 1.0)
    failed_setup = clamp(to_float(row.get("failed_setup_probability")) or 0.0, 0.0, 1.0)
    runner = structural_runner_score(row)
    squeeze = short_squeeze_pressure(row)
    breakout = breakout_pressure_score(row.get("distance_to_20d_high_pct"), max_below_pct=12.0)
    premarket = premarket_confirmation(row)
    volatility = scale_0_1(
        (to_float(row.get("atr_14")) or 0.0) / max(to_float(row.get("prev_close")) or 1.0, 0.01),
        0.04,
        0.20,
    )
    liquidity = budget_liquidity_score(row)
    avg_dollar_score = scale_0_1(avg_dollar_volume_20d(row), 3_000_000, 30_000_000)
    day_dollar_score = scale_0_1(latest_dollar_volume(row), 8_000_000, 100_000_000)
    rs_penalty = 1.0 if yes_no_flag(row.get("reverse_split_in_last_24m")) else 0.0
    legacy_rs_penalty = 0.4 if not yes_no_flag(row.get("has_never_reverse_split")) else 0.0
    size_penalty = market_cap_penalty(row, SUPERNOVA_MARKET_CAP_MAX)
    float_cap_penalty = float_penalty(row, SUPERNOVA_FLOAT_MAX)

    probability = (
        supernova_base * 0.32
        + runner * 0.16
        + squeeze * 0.14
        + breakout * 0.10
        + premarket * 0.12
        + volatility * 0.05
        + liquidity * 0.05
        + avg_dollar_score * 0.04
        + day_dollar_score * 0.05
        - toxic * 0.30
        - failed_setup * 0.18
        - rs_penalty * 0.22
        - legacy_rs_penalty * 0.08
        - size_penalty * 0.10
        - float_cap_penalty * 0.06
    )
    return clamp(probability, 0.0, 1.0)


def compute_continuation_probability(row: pd.Series) -> float:
    continuation_base = clamp(to_float(row.get("continuation_base_probability")) or 0.0, 0.0, 1.0)
    toxic = clamp(to_float(row.get("toxic_probability")) or 0.0, 0.0, 1.0)
    failed_setup = clamp(to_float(row.get("failed_setup_probability")) or 0.0, 0.0, 1.0)
    tape = continuation_tape_score(row)
    squeeze = short_squeeze_pressure(row)
    runner = structural_runner_score(row)
    premarket = premarket_confirmation(row)
    liquidity = budget_liquidity_score(row)
    avg_dollar_score = scale_0_1(avg_dollar_volume_20d(row), 5_000_000, 40_000_000)
    day_dollar_score = scale_0_1(latest_dollar_volume(row), 10_000_000, 120_000_000)
    rs_penalty = 1.0 if yes_no_flag(row.get("reverse_split_in_last_24m")) else 0.0
    legacy_rs_penalty = 0.30 if not yes_no_flag(row.get("has_never_reverse_split")) else 0.0
    size_penalty = market_cap_penalty(row, CONTINUATION_MARKET_CAP_MAX)
    float_cap_penalty = float_penalty(row, CONTINUATION_FLOAT_MAX)

    probability = (
        continuation_base * 0.34
        + tape * 0.24
        + squeeze * 0.05
        + runner * 0.06
        + premarket * 0.10
        + liquidity * 0.05
        + avg_dollar_score * 0.07
        + day_dollar_score * 0.06
        - toxic * 0.28
        - failed_setup * 0.28
        - rs_penalty * 0.16
        - legacy_rs_penalty * 0.10
        - size_penalty * 0.08
        - float_cap_penalty * 0.05
    )
    return clamp(probability, 0.0, 1.0)


def passes_supernova_filters(row: pd.Series) -> bool:
    price = to_float(row.get("prev_close"))
    avg_volume = to_float(row.get("avg_volume_20d"))
    market_cap = to_float(row.get("market_cap"))
    free_float = to_float(row.get("free_float"))
    latest_rvol = to_float(row.get("latest_rvol"))
    close_strength = to_float(row.get("latest_close_strength"))
    gap = to_float(row.get("premarket_gap_pct"))
    toxic = to_float(row.get("toxic_probability")) or 0.0
    failed_setup = to_float(row.get("failed_setup_probability")) or 0.0
    avg_dollar = avg_dollar_volume_20d(row)
    day_dollar = latest_dollar_volume(row)
    premarket_dollar = to_float(row.get("premarket_dollar_volume"))

    if price is None or price < BUDGET_PRICE_MIN or price > 20.0:
        return False
    if avg_volume is None or avg_volume < 750_000:
        return False
    if avg_dollar is None or avg_dollar < MIN_SUPERNOVA_AVG_DOLLAR_VOLUME:
        return False
    if day_dollar is None or day_dollar < MIN_SUPERNOVA_DAY_DOLLAR_VOLUME:
        return False
    if market_cap is not None and market_cap > SUPERNOVA_MARKET_CAP_MAX:
        return False
    if free_float is not None and free_float > SUPERNOVA_FLOAT_MAX:
        return False
    if yes_no_flag(row.get("reverse_split_in_last_24m")):
        return False
    if not yes_no_flag(row.get("has_never_reverse_split")):
        return False
    if toxic > 0.30:
        return False
    if failed_setup > MAX_SUPERNOVA_FAILED_SETUP_PROBABILITY:
        return False
    if breakout_pressure_score(row.get("distance_to_20d_high_pct"), 12.0) < 0.35:
        return False
    if structural_runner_score(row) < 0.35:
        return False
    if latest_rvol is not None and latest_rvol < 1.35:
        return False
    if close_strength is not None and close_strength < 0.62:
        return False
    if gap is not None and gap < -5.0:
        return False
    if premarket_dollar is not None and premarket_dollar < MIN_SUPERNOVA_PREMARKET_DOLLAR_VOLUME:
        return False
    if str(row.get("premarket_hold_quality") or "").upper() == "POOR":
        return False
    return True


def passes_continuation_filters(row: pd.Series) -> bool:
    price = to_float(row.get("prev_close"))
    avg_volume = to_float(row.get("avg_volume_20d"))
    market_cap = to_float(row.get("market_cap"))
    free_float = to_float(row.get("free_float"))
    latest_rvol = to_float(row.get("latest_rvol"))
    close_strength = to_float(row.get("latest_close_strength"))
    gap = to_float(row.get("premarket_gap_pct"))
    toxic = to_float(row.get("toxic_probability")) or 0.0
    failed_setup = to_float(row.get("failed_setup_probability")) or 0.0
    avg_dollar = avg_dollar_volume_20d(row)
    day_dollar = latest_dollar_volume(row)
    premarket_dollar = to_float(row.get("premarket_dollar_volume"))

    if price is None or price < 1.0 or price > BUDGET_PRICE_MAX:
        return False
    if avg_volume is None or avg_volume < 1_000_000:
        return False
    if avg_dollar is None or avg_dollar < MIN_CONTINUATION_AVG_DOLLAR_VOLUME:
        return False
    if day_dollar is None or day_dollar < MIN_CONTINUATION_DAY_DOLLAR_VOLUME:
        return False
    if market_cap is not None and market_cap > 3_500_000_000:
        return False
    if free_float is not None and free_float > 120_000_000:
        return False
    if yes_no_flag(row.get("reverse_split_in_last_24m")):
        return False
    if not yes_no_flag(row.get("has_never_reverse_split")):
        return False
    if toxic > 0.18:
        return False
    if failed_setup > MAX_CONTINUATION_FAILED_SETUP_PROBABILITY:
        return False
    if not yes_no_flag(row.get("price_above_sma50_yes_no")):
        return False
    if close_strength is not None and close_strength < 0.68:
        return False
    if latest_rvol is not None and latest_rvol < 1.40:
        return False
    if breakout_pressure_score(row.get("distance_to_20d_high_pct"), 6.0) < 0.55:
        return False
    if str(row.get("premarket_hold_quality") or "").upper() == "POOR":
        return False
    if premarket_dollar is not None and premarket_dollar < MIN_CONTINUATION_PREMARKET_DOLLAR_VOLUME:
        return False
    if gap is not None and gap < -4.0:
        return False
    return True


def derive_breakout_reference(row: pd.Series, profile: str) -> float | None:
    candidates: list[float] = []
    prev_close = to_float(row.get("prev_close"))
    premarket_high = to_float(row.get("premarket_high"))
    prior_20d_high = to_float(row.get("prior_20d_high"))
    prior_50d_high = to_float(row.get("prior_50d_high"))

    if premarket_high is not None:
        candidates.append(premarket_high)
    if prior_20d_high is not None:
        candidates.append(prior_20d_high)
    if profile == "NEXT_DAY_CONTINUATION" and prior_50d_high is not None:
        candidates.append(prior_50d_high)
    if prev_close is not None:
        candidates.append(prev_close * (1.02 if profile == "SUPERNOVA_IGNITION" else 1.01))
    if not candidates:
        return None
    return max(candidates)


def build_selection_reason(row: pd.Series, profile: str) -> str:
    parts: list[str] = []
    if yes_no_flag(row.get("has_never_reverse_split")):
        parts.append("clean capital structure")
    if structural_runner_score(row) >= 0.6:
        parts.append("former runner memory")
    if short_squeeze_pressure(row) >= 0.55:
        parts.append("short squeeze pressure")
    if premarket_confirmation(row) >= 0.65:
        parts.append("strong premarket confirmation")
    if continuation_tape_score(row) >= 0.6 and profile == "NEXT_DAY_CONTINUATION":
        parts.append("tight continuation tape")
    if breakout_pressure_score(row.get("distance_to_20d_high_pct")) >= 0.75:
        parts.append("near breakout trigger")
    if not parts:
        parts.append("model edge exceeds focus threshold")
    return ", ".join(parts[:3])


def build_trade_plan(row: pd.Series, profile: str) -> dict[str, Any]:
    entry_reference = derive_breakout_reference(row, profile)
    prev_close = to_float(row.get("prev_close"))
    premarket_vwap = to_float(row.get("premarket_vwap"))
    atr_14 = to_float(row.get("atr_14")) or 0.0
    premarket_low = to_float(row.get("premarket_low"))

    if entry_reference is None or prev_close is None:
        return {
            "entry_price": None,
            "stop_price": None,
            "target_price": None,
            "expected_move_pct": None,
            "risk_reward_ratio": None,
            "no_trade_below": None,
        }

    entry_buffer = 1.002 if profile == "SUPERNOVA_IGNITION" else 1.001
    entry_price = entry_reference * entry_buffer

    atr_pct = (atr_14 / max(prev_close, 0.01)) * 100.0 if atr_14 else 0.0
    stop_pct = clamp(
        max(atr_pct * (0.70 if profile == "SUPERNOVA_IGNITION" else 0.55), 5.5 if profile == "SUPERNOVA_IGNITION" else 4.5),
        5.5 if profile == "SUPERNOVA_IGNITION" else 4.5,
        11.0 if profile == "SUPERNOVA_IGNITION" else 8.5,
    )

    no_trade_floor = max(
        premarket_vwap if premarket_vwap is not None else 0.0,
        premarket_low if premarket_low is not None else 0.0,
        prev_close * (0.97 if profile == "SUPERNOVA_IGNITION" else 0.985),
    )
    stop_price = max(entry_price * (1.0 - stop_pct / 100.0), no_trade_floor * 0.995)
    if stop_price >= entry_price:
        stop_price = entry_price * 0.94

    profile_probability = (
        to_float(row.get("supernova_probability")) if profile == "SUPERNOVA_IGNITION" else to_float(row.get("continuation_probability"))
    ) or 0.0
    expected_move_pct = clamp(
        (13.0 if profile == "SUPERNOVA_IGNITION" else 12.0) + profile_probability * (22.0 if profile == "SUPERNOVA_IGNITION" else 18.0),
        15.0,
        30.0,
    )
    target_price = entry_price * (1.0 + expected_move_pct / 100.0)
    risk_reward_ratio = (target_price - entry_price) / max(entry_price - stop_price, 0.0001)

    return {
        "entry_price": round(entry_price, 4),
        "stop_price": round(stop_price, 4),
        "target_price": round(target_price, 4),
        "expected_move_pct": round(expected_move_pct, 2),
        "risk_reward_ratio": round(risk_reward_ratio, 2),
        "no_trade_below": round(no_trade_floor, 4) if no_trade_floor > 0 else None,
    }


def enrich_focus_rows(scored: pd.DataFrame) -> pd.DataFrame:
    enriched = scored.copy()
    for col in [
        "latest_rvol",
        "latest_close_strength",
        "latest_day_volume",
        "latest_dollar_volume",
        "prior_20d_high",
        "prior_50d_high",
        "premarket_high",
        "premarket_low",
        "premarket_vwap",
        "premarket_gap_pct",
        "premarket_dollar_volume",
        "premarket_hold_quality",
    ]:
        if col not in enriched.columns:
            enriched[col] = np.nan
    enriched["avg_dollar_volume_20d"] = enriched.apply(avg_dollar_volume_20d, axis=1)
    enriched["supernova_probability"] = enriched.apply(compute_supernova_probability, axis=1)
    enriched["continuation_probability"] = enriched.apply(compute_continuation_probability, axis=1)
    enriched["supernova_focus_score"] = (enriched["supernova_probability"] * 100.0).round(2)
    enriched["continuation_focus_score"] = (enriched["continuation_probability"] * 100.0).round(2)

    profiles = []
    scores = []
    for _, row in enriched.iterrows():
        supernova_score = to_float(row.get("supernova_focus_score")) or 0.0
        continuation_score = to_float(row.get("continuation_focus_score")) or 0.0
        if supernova_score >= continuation_score:
            profiles.append("SUPERNOVA_IGNITION")
            scores.append(supernova_score)
        else:
            profiles.append("NEXT_DAY_CONTINUATION")
            scores.append(continuation_score)
    enriched["trade_profile"] = profiles
    enriched["scan_priority_score"] = scores
    enriched["setup_grade"] = enriched["scan_priority_score"].map(priority_to_grade)
    return enriched


def attach_selection_payload(
    df: pd.DataFrame,
    *,
    profile: str,
    selection_bucket: str,
    reset_rank: bool = True,
) -> pd.DataFrame:
    selected = df.copy()
    if selected.empty:
        if "selection_rank" not in selected.columns:
            selected["selection_rank"] = pd.Series(dtype=int)
        selected["trade_profile"] = profile
        selected["selection_bucket"] = selection_bucket
        return selected

    selected["trade_profile"] = profile
    selected["selection_bucket"] = selection_bucket
    selected["selection_reason"] = selected.apply(
        lambda row: build_selection_reason(row, profile),
        axis=1,
    )
    trade_plans = selected.apply(lambda row: build_trade_plan(row, profile), axis=1)
    trade_plan_df = pd.DataFrame(list(trade_plans), index=selected.index)
    selected = pd.concat([selected, trade_plan_df], axis=1)
    selected["scan_priority_score"] = (
        selected["continuation_focus_score"] if profile == "NEXT_DAY_CONTINUATION" else selected["supernova_focus_score"]
    )
    selected["setup_grade"] = selected["scan_priority_score"].map(priority_to_grade)
    selected = selected.sort_values("scan_priority_score", ascending=False).reset_index(drop=True)
    if reset_rank:
        selected["selection_rank"] = np.arange(1, len(selected) + 1)
    return selected


def continuation_fallback_filter(row: pd.Series) -> bool:
    price = to_float(row.get("prev_close"))
    avg_volume = to_float(row.get("avg_volume_20d"))
    toxic = to_float(row.get("toxic_probability")) or 0.0
    failed_setup = to_float(row.get("failed_setup_probability")) or 0.0
    if price is None or price < 1.0 or price > BUDGET_PRICE_MAX:
        return False
    if avg_volume is None or avg_volume < 800_000:
        return False
    if yes_no_flag(row.get("reverse_split_in_last_24m")):
        return False
    if toxic > 0.20:
        return False
    if failed_setup > 0.28:
        return False
    return True


def select_continuation_picks(
    enriched: pd.DataFrame,
    max_picks: int = CONTINUATION_MAX_PICKS_DEFAULT,
) -> pd.DataFrame:
    if enriched.empty:
        return enriched.copy()

    continuation_pool = enriched[enriched.apply(passes_continuation_filters, axis=1)].copy()
    continuation_pool = continuation_pool[continuation_pool["continuation_focus_score"] >= CONTINUATION_SCORE_THRESHOLD]
    continuation_pool["continuation_quality_score"] = (
        continuation_pool["continuation_focus_score"]
        - continuation_pool["failed_setup_probability"] * 35.0
        - continuation_pool["toxic_probability"] * 25.0
    )
    continuation_pool = continuation_pool.sort_values(
        ["continuation_quality_score", "continuation_focus_score", "clean_spike_probability", "latest_rvol"],
        ascending=False,
    )

    if continuation_pool.empty:
        selected = attach_selection_payload(
            continuation_pool.head(0),
            profile="NEXT_DAY_CONTINUATION",
            selection_bucket="NO_TRADE",
        )
        selected.attrs["eligible_focus_count"] = 0
        selected.attrs["used_fallback"] = False
        return selected

    selected = attach_selection_payload(
        continuation_pool.head(max_picks),
        profile="NEXT_DAY_CONTINUATION",
        selection_bucket="CONTINUATION_PRIMARY",
    )
    selected.attrs["eligible_focus_count"] = int(len(continuation_pool))
    selected.attrs["used_fallback"] = False
    return selected


def select_supernova_alerts(
    enriched: pd.DataFrame,
    max_alerts: int = SUPERNOVA_ALERT_MAX_PICKS_DEFAULT,
) -> pd.DataFrame:
    if enriched.empty:
        return enriched.copy()

    supernova_pool = enriched[enriched.apply(passes_supernova_filters, axis=1)].copy()
    supernova_pool = supernova_pool[supernova_pool["supernova_focus_score"] >= SUPERNOVA_ALERT_THRESHOLD]
    supernova_pool["supernova_quality_score"] = (
        supernova_pool["supernova_focus_score"]
        - supernova_pool["failed_setup_probability"] * 30.0
        - supernova_pool["toxic_probability"] * 25.0
    )
    supernova_pool = supernova_pool.sort_values(
        ["supernova_quality_score", "supernova_focus_score", "clean_spike_probability", "latest_rvol"],
        ascending=False,
    )

    selected = attach_selection_payload(
        supernova_pool.head(max_alerts),
        profile="SUPERNOVA_IGNITION",
        selection_bucket="SUPERNOVA_ALERT",
    )
    selected.attrs["eligible_focus_count"] = int(len(supernova_pool))
    selected.attrs["used_fallback"] = False
    return selected


def select_focus_picks(scored: pd.DataFrame, max_picks: int = FOCUS_MAX_PICKS_DEFAULT) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    enriched = enrich_focus_rows(scored)
    continuation = select_continuation_picks(enriched, max_picks=max_picks)
    continuation.attrs["universe_size"] = int(len(enriched))
    return continuation


def score_snapshot_rows(rows: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    features = engineer_model_frame(rows)
    spike_proba = predict_positive_probability(artifact["spike_pipeline"], features)
    continuation_proba = predict_positive_probability(artifact["continuation_pipeline"], features)
    supernova_proba = predict_positive_probability(artifact["supernova_pipeline"], features)
    toxic_proba = predict_positive_probability(artifact["toxic_pipeline"], features)
    failed_setup_proba = predict_positive_probability(artifact["failure_pipeline"], features)

    scored = rows.copy()
    scored["spike_probability"] = spike_proba
    scored["continuation_base_probability"] = continuation_proba
    scored["supernova_base_probability"] = supernova_proba
    scored["toxic_probability"] = toxic_proba
    scored["failed_setup_probability"] = failed_setup_proba
    scored["clean_spike_probability"] = (
        np.maximum(scored["continuation_base_probability"], scored["supernova_base_probability"])
        * (1.0 - scored["toxic_probability"])
        * (1.0 - scored["failed_setup_probability"] * 0.60)
    )
    scored["scan_priority_score"] = scored["clean_spike_probability"] * 100.0
    scored["setup_grade"] = scored["scan_priority_score"].map(priority_to_grade)
    return scored.sort_values("scan_priority_score", ascending=False).reset_index(drop=True)


def run_scan(
    artifact: dict[str, Any] | None = None,
    artifact_path: Path | str = MODEL_PATH,
    tickers: list[str] | None = None,
    append_default_universe: bool = False,
    skip_premarket: bool = False,
    top_context_rows: int = 30,
    max_picks: int = FOCUS_MAX_PICKS_DEFAULT,
) -> pd.DataFrame:
    if artifact is None:
        artifact = load_artifact(artifact_path)

    if tickers and not append_default_universe:
        universe = []
        seen = set()
        for ticker in tickers:
            cleaned = ticker.strip().upper()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                universe.append(cleaned)
    else:
        universe = get_candidate_universe(extra_tickers=tickers or [])

    daily_histories = download_daily_universe(universe)
    snapshots = []
    for ticker, frame in daily_histories.items():
        snapshot = build_scan_snapshot_from_history(ticker, frame)
        if snapshot is not None:
            snapshots.append(snapshot)

    if not snapshots:
        return pd.DataFrame()

    scan_df = pd.DataFrame(snapshots)
    scored = score_snapshot_rows(scan_df, artifact)

    if not skip_premarket:
        top_context_rows = max(1, top_context_rows)
        top_for_context = scored.head(top_context_rows).copy()
        context_rows = []
        for _, row in top_for_context.iterrows():
            merged = row.to_dict()
            merged.update(fetch_current_premarket_context(row["ticker"], row["prev_close"]))
            context_rows.append(merged)

        refreshed = pd.DataFrame(context_rows)
        base_rest = scored[~scored["ticker"].isin(refreshed["ticker"])].copy()
        rescored_top = score_snapshot_rows(refreshed, artifact)
        scored = pd.concat([rescored_top, base_rest], ignore_index=True)
        scored = scored.sort_values("scan_priority_score", ascending=False).reset_index(drop=True)

    focus_picks = select_focus_picks(scored, max_picks=max_picks)
    focus_picks.attrs["universe_size"] = int(len(scored))
    return focus_picks


def run_dual_scan(
    artifact: dict[str, Any] | None = None,
    artifact_path: Path | str = MODEL_PATH,
    tickers: list[str] | None = None,
    append_default_universe: bool = False,
    skip_premarket: bool = False,
    top_context_rows: int = 60,
    continuation_max_picks: int = CONTINUATION_MAX_PICKS_DEFAULT,
    supernova_max_alerts: int = SUPERNOVA_ALERT_MAX_PICKS_DEFAULT,
) -> dict[str, Any]:
    if artifact is None:
        artifact = load_artifact(artifact_path)

    if tickers and not append_default_universe:
        universe = []
        seen = set()
        for ticker in tickers:
            cleaned = ticker.strip().upper()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                universe.append(cleaned)
    else:
        universe = get_candidate_universe(extra_tickers=tickers or [])

    daily_histories = download_daily_universe(universe)
    snapshots = []
    for ticker, frame in daily_histories.items():
        snapshot = build_scan_snapshot_from_history(ticker, frame)
        if snapshot is not None:
            snapshots.append(snapshot)

    if not snapshots:
        return {
            "scanned_universe_size": 0,
            "continuation_picks": pd.DataFrame(),
            "supernova_alerts": pd.DataFrame(),
        }

    scan_df = pd.DataFrame(snapshots)
    scored = score_snapshot_rows(scan_df, artifact)

    if not skip_premarket:
        top_context_rows = max(1, top_context_rows)
        top_for_context = scored.head(top_context_rows).copy()
        context_rows = []
        for _, row in top_for_context.iterrows():
            merged = row.to_dict()
            merged.update(fetch_current_premarket_context(row["ticker"], row["prev_close"]))
            context_rows.append(merged)

        refreshed = pd.DataFrame(context_rows)
        base_rest = scored[~scored["ticker"].isin(refreshed["ticker"])].copy()
        rescored_top = score_snapshot_rows(refreshed, artifact)
        scored = pd.concat([rescored_top, base_rest], ignore_index=True)
        scored = scored.sort_values("scan_priority_score", ascending=False).reset_index(drop=True)

    enriched = enrich_focus_rows(scored)
    continuation_picks = select_continuation_picks(enriched, max_picks=continuation_max_picks)
    supernova_alerts = select_supernova_alerts(enriched, max_alerts=supernova_max_alerts)

    scanned_universe_size = int(len(enriched))
    continuation_picks.attrs["universe_size"] = scanned_universe_size
    supernova_alerts.attrs["universe_size"] = scanned_universe_size

    return {
        "scanned_universe_size": scanned_universe_size,
        "continuation_picks": continuation_picks,
        "supernova_alerts": supernova_alerts,
    }


def priority_to_grade(score: float) -> str:
    if score >= 80:
        return "A+"
    if score >= 65:
        return "A"
    if score >= 50:
        return "B"
    if score >= 35:
        return "C"
    return "D"
