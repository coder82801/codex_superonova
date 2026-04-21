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
    if MODEL_PATH.exists():
        return
    from train_spike_model import main as train_main

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
        "clean_spike_probability",
        "toxic_probability",
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
            "append_default_universe": False,
            "skip_premarket": False,
            "table_limit": DEFAULT_CONTINUATION_PICKS,
        },
        summary=None,
    )


@app.post("/scan")
def scan():
    form_data = {
        "tickers": request.form.get("tickers", DEFAULT_TICKERS).strip(),
        "append_default_universe": request.form.get("append_default_universe") == "on",
        "skip_premarket": request.form.get("skip_premarket") == "on",
        "table_limit": clamp_int(
            request.form.get("table_limit", str(DEFAULT_CONTINUATION_PICKS)),
            minimum=1,
            maximum=DEFAULT_CONTINUATION_PICKS,
        ),
    }
    try:
        ensure_model()
        artifact = load_artifact(MODEL_PATH)
        tickers = parse_tickers(form_data["tickers"])
        scan_bundle = run_dual_scan(
            artifact=artifact,
            tickers=tickers or None,
            append_default_universe=form_data["append_default_universe"],
            skip_premarket=form_data["skip_premarket"],
            top_context_rows=max(form_data["table_limit"] * 20, 80),
            continuation_max_picks=form_data["table_limit"],
            supernova_max_alerts=DEFAULT_SUPERNOVA_ALERTS,
        )
        continuation_picks = scan_bundle["continuation_picks"]
        supernova_alerts = scan_bundle["supernova_alerts"]
        if continuation_picks.empty and supernova_alerts.empty:
            return render_template(
                "index.html",
                continuation_results=[],
                supernova_results=[],
                error="Tarama sonucu oluşturulamadı. Ticker listesini daraltıp tekrar dene.",
                form_data=form_data,
                summary=None,
            )

        continuation_results = prepare_table(continuation_picks, limit=form_data["table_limit"])
        supernova_results = prepare_table(supernova_alerts, limit=DEFAULT_SUPERNOVA_ALERTS)
        summary = {
            "total_candidates": int(scan_bundle["scanned_universe_size"]),
            "continuation_focus_candidates": int(
                continuation_picks.attrs.get("eligible_focus_count", len(continuation_picks))
            ),
            "supernova_focus_candidates": int(
                supernova_alerts.attrs.get("eligible_focus_count", len(supernova_alerts))
            ),
            "top_continuation_score": float(continuation_picks["scan_priority_score"].max())
            if not continuation_picks.empty
            else None,
            "top_supernova_score": float(supernova_alerts["scan_priority_score"].max())
            if not supernova_alerts.empty
            else None,
            "continuation_used_fallback": bool(continuation_picks.attrs.get("used_fallback", False)),
            "used_watchlist": bool(tickers),
        }
        return render_template(
            "index.html",
            continuation_results=continuation_results,
            supernova_results=supernova_results,
            error=None,
            form_data=form_data,
            summary=summary,
        )
    except Exception as exc:
        traceback.print_exc()
        return render_template(
            "index.html",
            continuation_results=[],
            supernova_results=[],
            error=f"Tarama sırasında hata oluştu: {exc}",
            form_data=form_data,
            summary=None,
        )


@app.post("/api/scan")
def api_scan():
    payload = request.get_json(silent=True) or {}
    tickers = payload.get("tickers", [])
    if isinstance(tickers, str):
        tickers = parse_tickers(tickers)

    append_default_universe = bool(payload.get("append_default_universe", False))
    skip_premarket = bool(payload.get("skip_premarket", False))
    limit = clamp_int(str(payload.get("limit", DEFAULT_CONTINUATION_PICKS)), minimum=1, maximum=DEFAULT_CONTINUATION_PICKS)

    try:
        ensure_model()
        artifact = load_artifact(MODEL_PATH)
        scan_bundle = run_dual_scan(
            artifact=artifact,
            tickers=tickers or None,
            append_default_universe=append_default_universe,
            skip_premarket=skip_premarket,
            top_context_rows=max(limit * 20, 80),
            continuation_max_picks=limit,
            supernova_max_alerts=DEFAULT_SUPERNOVA_ALERTS,
        )
        continuation_picks = scan_bundle["continuation_picks"]
        supernova_alerts = scan_bundle["supernova_alerts"]
        return jsonify(
            {
                "ok": True,
                "count": int(len(continuation_picks) + len(supernova_alerts)),
                "universe_size": int(scan_bundle["scanned_universe_size"]),
                "continuation_focus_count": int(
                    continuation_picks.attrs.get("eligible_focus_count", len(continuation_picks))
                ),
                "supernova_focus_count": int(
                    supernova_alerts.attrs.get("eligible_focus_count", len(supernova_alerts))
                ),
                "continuation_results": prepare_table(continuation_picks, limit=limit),
                "supernova_results": prepare_table(supernova_alerts, limit=DEFAULT_SUPERNOVA_ALERTS),
            }
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    ensure_model()
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
