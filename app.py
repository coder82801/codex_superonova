from __future__ import annotations

import os
import traceback
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

from spike_model_core import MODEL_PATH, load_artifact, run_scan


ROOT = Path(__file__).resolve().parent
DEFAULT_TICKERS = "MYSE,WSHP,CMPS,QBTS,RGTI,QUBT,ONFO,AGAE"
DEFAULT_MAX_PICKS = 2

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
        results=[],
        error=None,
        form_data={
            "tickers": DEFAULT_TICKERS,
            "append_default_universe": False,
            "skip_premarket": False,
            "table_limit": DEFAULT_MAX_PICKS,
        },
        summary=None,
    )


@app.post("/scan")
def scan():
    form_data = {
        "tickers": request.form.get("tickers", DEFAULT_TICKERS).strip(),
        "append_default_universe": request.form.get("append_default_universe") == "on",
        "skip_premarket": request.form.get("skip_premarket") == "on",
        "table_limit": clamp_int(request.form.get("table_limit", str(DEFAULT_MAX_PICKS)), minimum=1, maximum=DEFAULT_MAX_PICKS),
    }
    try:
        ensure_model()
        artifact = load_artifact(MODEL_PATH)
        tickers = parse_tickers(form_data["tickers"])
        scored = run_scan(
            artifact=artifact,
            tickers=tickers or None,
            append_default_universe=form_data["append_default_universe"],
            skip_premarket=form_data["skip_premarket"],
            top_context_rows=max(form_data["table_limit"] * 15, 60),
            max_picks=form_data["table_limit"],
        )
        if scored.empty:
            return render_template(
                "index.html",
                results=[],
                error="Tarama sonucu oluşturulamadı. Ticker listesini daraltıp tekrar dene.",
                form_data=form_data,
                summary=None,
            )

        results = prepare_table(scored, limit=form_data["table_limit"])
        summary = {
            "total_candidates": int(scored.attrs.get("universe_size", len(scored))),
            "focus_candidates": int(scored.attrs.get("eligible_focus_count", len(scored))),
            "top_score": float(scored["scan_priority_score"].max()),
            "a_grade_count": int((scored["setup_grade"].isin(["A+", "A"])).sum()),
            "used_fallback": bool(scored.attrs.get("used_fallback", False)),
            "used_watchlist": bool(tickers),
        }
        return render_template(
            "index.html",
            results=results,
            error=None,
            form_data=form_data,
            summary=summary,
        )
    except Exception as exc:
        traceback.print_exc()
        return render_template(
            "index.html",
            results=[],
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
    limit = clamp_int(str(payload.get("limit", DEFAULT_MAX_PICKS)), minimum=1, maximum=DEFAULT_MAX_PICKS)

    try:
        ensure_model()
        artifact = load_artifact(MODEL_PATH)
        scored = run_scan(
            artifact=artifact,
            tickers=tickers or None,
            append_default_universe=append_default_universe,
            skip_premarket=skip_premarket,
            top_context_rows=max(limit * 15, 60),
            max_picks=limit,
        )
        return jsonify(
            {
                "ok": True,
                "count": int(len(scored)),
                "universe_size": int(scored.attrs.get("universe_size", len(scored))),
                "eligible_focus_count": int(scored.attrs.get("eligible_focus_count", len(scored))),
                "results": prepare_table(scored, limit=limit),
            }
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    ensure_model()
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
