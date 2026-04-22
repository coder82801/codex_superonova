from __future__ import annotations

import os
import traceback
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

from spike_model_core import MODEL_PATH, load_artifact, run_dual_scan


ROOT = Path(__file__).resolve().parent
DEFAULT_TICKERS = "MYSE,WSHP,CMPS,QBTS,RGTI,QUBT,ONFO,AGAE"
DEFAULT_CONTINUATION_PICKS = 2
DEFAULT_SUPERNOVA_ALERTS = 1

app = Flask(__name__)


def ensure_model() -> None:
    from train_spike_model import main as train_main

    if MODEL_PATH.exists():
        try:
            load_artifact(MODEL_PATH)
            return
        except Exception:
            pass

    train_main([])


def parse_tickers(raw: str) -> list[str]:
    if not raw:
        return []
    normalized = raw.replace("\n", ",").replace("\t", ",")
    return [ticker.strip().upper() for ticker in normalized.split(",") if ticker.strip()]


def clamp_int(value: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(maximum, parsed))


def prepare_table(df: pd.DataFrame, limit: int = 50) -> list[dict]:
    if df.empty:
        return []
    display = df.copy().head(limit)
    wanted = [
        "selection_rank",
        "ticker",
        "selection_bucket",
        "trade_profile",
        "scan_priority_score",
        "setup_grade",
        "supernova_probability",
        "continuation_probability",
        "failed_setup_probability",
        "clean_spike_probability",
        "toxic_probability",
        "avg_volume_20d",
        "avg_dollar_volume_20d",
        "entry_price",
        "stop_price",
        "target_price",
        "expected_move_pct",
        "risk_reward_ratio",
        "no_trade_below",
        "selection_reason",
        "prev_close",
        "premarket_gap_pct",
        "premarket_dollar_volume",
        "premarket_hold_quality",
    ]
    display = display[[col for col in wanted if col in display.columns]].copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].round(4)
    display = display.replace({pd.NA: None, float("nan"): None})
    return display.to_dict(orient="records")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def index():
    return render_template(
        "index.html",
        continuation_results=[],
        supernova_results=[],
        error=None,
        form_data={
            "tickers": DEFAULT_TICKERS,
