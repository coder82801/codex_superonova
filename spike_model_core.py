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

POSITIVE_LABELS = {"CLEAN_RECOVERY_2X", "NEAR_POSITIVE", "OPEN_DRIVE_SQUEEZE"}
TOXIC_LABEL = "TOXIC_EXCLUDE"

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


def load_labeled_dataset(csv_path: Path | str = DATASET_PATH) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, parse_dates=["event_date"])
    frame["target_spike"] = frame["label"].isin(POSITIVE_LABELS).astype(int)
    frame["target_toxic"] = (frame["label"] == TOXIC_LABEL).astype(int)
    return frame


def engineer_model_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    features = pd.DataFrame(index=df.index)

    prev_close = pd.to_numeric(df["prev_close"], errors="coerce")
    avg_volume_20d = pd.to_numeric(df["avg_volume_20d"], errors="coerce")
    sma20 = pd.to_numeric(df["sma20"], errors="coerce")
    sma50 = pd.to_numeric(df["sma50"], errors="coerce")
    sma150 = pd.to_numeric(df["sma150"], errors="coerce")
    sma200 = pd.to_numeric(df["sma200"], errors="coerce")
    atr_14 = pd.to_numeric(df["atr_14"], errors="coerce")
    amihud = pd.to_numeric(df["amihud_illiquidity_proxy"], errors="coerce")

    features["log_prev_close"] = prev_close.map(log1p_safe)
    features["log_avg_volume_20d"] = avg_volume_20d.map(log1p_safe)
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


def compute_binary_metrics(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "brier_score": float(brier_score_loss(y_true, proba)),
    }


def evaluate_model(raw: pd.DataFrame, target_col: str) -> dict[str, Any]:
    X = engineer_model_frame(raw)
    y = raw[target_col].astype(int)
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
    spike_pipeline = build_pipeline().fit(X, raw["target_spike"].astype(int))
    toxic_pipeline = build_pipeline().fit(X, raw["target_toxic"].astype(int))

    spike_coef = spike_pipeline.named_steps["model"].coef_[0]
    toxic_coef = toxic_pipeline.named_steps["model"].coef_[0]
    feature_importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "spike_logit_coef": spike_coef,
            "toxic_logit_coef": toxic_coef,
        }
    ).sort_values("spike_logit_coef", ascending=False)

    return {
        "spike_pipeline": spike_pipeline,
        "toxic_pipeline": toxic_pipeline,
        "feature_names": MODEL_FEATURES,
        "positive_labels": sorted(POSITIVE_LABELS),
        "toxic_label": TOXIC_LABEL,
        "trained_at": datetime.now(tz=NY_TZ).isoformat(),
        "training_rows": int(len(raw)),
        "feature_importance": feature_importance.to_dict(orient="records"),
    }


def save_artifact(artifact: dict[str, Any], path: Path | str = MODEL_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: Path | str = MODEL_PATH) -> dict[str, Any]:
    return joblib.load(path)


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
    if last_close <= 0:
        return None

    avg_volume_20d = float(df["Volume"].tail(20).mean())
    major_high_idx = int(df["High"].idxmax())
    major_high = float(df["High"].iloc[major_high_idx])
    major_high_date = pd.Timestamp(df["Date"].iloc[major_high_idx]).tz_convert(NY_TZ)

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
        "distance_to_20d_high_pct": safe_ratio_pct(last_close, float(df["High"].tail(20).max())),
        "distance_to_50d_high_pct": safe_ratio_pct(last_close, float(df["High"].tail(50).max())),
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
        "premarket_volume": int(pre_volume),
        "premarket_dollar_volume": round(pre_dollar, 2),
        "premarket_vwap": round(pre_vwap, 6) if pre_vwap else None,
        "premarket_gap_pct": safe_ratio_pct(pre_open, prev_close),
        "premarket_hold_quality": hold_quality,
        "premarket_range_pct": safe_ratio_pct(pre_high, pre_low),
    }


def score_snapshot_rows(rows: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    features = engineer_model_frame(rows)
    spike_proba = artifact["spike_pipeline"].predict_proba(features)[:, 1]
    toxic_proba = artifact["toxic_pipeline"].predict_proba(features)[:, 1]

    scored = rows.copy()
    scored["spike_probability"] = spike_proba
    scored["toxic_probability"] = toxic_proba
    scored["clean_spike_probability"] = scored["spike_probability"] * (1.0 - scored["toxic_probability"])
    scored["scan_priority_score"] = scored.apply(apply_scan_priority_score, axis=1)
    scored["setup_grade"] = scored["scan_priority_score"].map(priority_to_grade)
    return scored.sort_values("scan_priority_score", ascending=False).reset_index(drop=True)


def run_scan(
    artifact: dict[str, Any] | None = None,
    artifact_path: Path | str = MODEL_PATH,
    tickers: list[str] | None = None,
    append_default_universe: bool = False,
    skip_premarket: bool = False,
    top_context_rows: int = 30,
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

    return scored


def apply_scan_priority_score(row: pd.Series) -> float:
    base = float(row["clean_spike_probability"])
    adjustment = 1.0

    gap = pd.to_numeric(row.get("premarket_gap_pct"), errors="coerce")
    pre_dollar = pd.to_numeric(row.get("premarket_dollar_volume"), errors="coerce")
    hold_quality = str(row.get("premarket_hold_quality") or "").upper()
    reverse_split_recent = str(row.get("reverse_split_in_last_24m") or "").upper() == "YES"

    if pd.notna(gap) and gap > 0:
        adjustment += min(0.25, float(gap) / 200.0)
    if pd.notna(pre_dollar):
        if pre_dollar >= 5_000_000:
            adjustment += 0.15
        elif pre_dollar >= 1_000_000:
            adjustment += 0.10
        elif pre_dollar >= 250_000:
            adjustment += 0.05
    if hold_quality == "GOOD":
        adjustment += 0.10
    elif hold_quality == "FAIR":
        adjustment += 0.04
    elif hold_quality == "POOR":
        adjustment -= 0.05

    if reverse_split_recent:
        adjustment -= 0.35

    score = min(max(base * adjustment * 100.0, 0.0), 100.0)
    return round(score, 2)


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
