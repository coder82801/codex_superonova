from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from spike_model_core import (
    DEFAULT_SCREEN_PATH,
    MODEL_PATH,
    load_artifact,
    run_dual_scan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score US stock candidates for pre-spike potential.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Trained model artifact path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_SCREEN_PATH, help="CSV output path.")
    parser.add_argument("--top", type=int, default=2, help="How many continuation picks to print.")
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated custom tickers. If omitted, the script builds a default candidate universe.",
    )
    parser.add_argument(
        "--tickers-file",
        type=Path,
        default=None,
        help="Optional newline-delimited ticker file. Appended to the default universe.",
    )
    parser.add_argument(
        "--skip-premarket",
        action="store_true",
        help="Skip premarket context fetch and rank on the pure model score.",
    )
    parser.add_argument(
        "--append-default-universe",
        action="store_true",
        help="When custom tickers are supplied, append them to the default StockAnalysis universe instead of replacing it.",
    )
    return parser.parse_args()


def load_extra_tickers(args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    if args.tickers:
        extra.extend([ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()])
    if args.tickers_file and args.tickers_file.exists():
        extra.extend(
            [
                line.strip().upper()
                for line in args.tickers_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
    return extra


def main() -> None:
    args = parse_args()
    try:
        artifact = load_artifact(args.model)
    except Exception:
        from train_spike_model import main as train_main

        train_main([])
        artifact = load_artifact(args.model)

    extra_tickers = load_extra_tickers(args)
    scan_bundle = run_dual_scan(
        artifact=artifact,
        tickers=extra_tickers or None,
        append_default_universe=args.append_default_universe,
        skip_premarket=args.skip_premarket,
        top_context_rows=max(args.top * 15, 60),
        continuation_max_picks=args.top,
        supernova_max_alerts=1,
    )
    continuation = scan_bundle["continuation_picks"]
    supernova = scan_bundle["supernova_alerts"]
    if continuation.empty and supernova.empty:
        raise SystemExit("No scan snapshots could be built.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([continuation, supernova], ignore_index=True)
    combined.to_csv(args.output, index=False)
    print(f"[done] wrote scan results: {args.output}")
    display_cols = [
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
