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

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "clean_recovery_supernova_labeled_dataset.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "spike_model.joblib"
METRICS_PATH = MODELS_DIR / "spike_model_metrics.json"
DEFAULT_SCREEN_PATH = ROOT / "spike_scan_results.csv"

POSITIVE_LABELS = {"CLEAN_RECOVERY_2X", "NEAR_POSITIVE", "OPEN_DRIVE_SQUEEZE"}
TOXIC_LABEL = "TOXIC_EXCLUDE"


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
