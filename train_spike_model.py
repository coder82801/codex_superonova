from __future__ import annotations

import argparse
import json
from pathlib import Path

from spike_model_core import (
    DATASET_PATH,
    METRICS_PATH,
    MODEL_PATH,
    evaluate_model,
    fit_final_models,
    load_labeled_dataset,
    save_artifact,
    save_metrics,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the pre-spike screener model.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Labeled CSV dataset path.")
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH, help="Output joblib path.")
    parser.add_argument("--metrics-out", type=Path, default=METRICS_PATH, help="Output metrics JSON path.")
    args = parser.parse_args(argv)

    raw = load_labeled_dataset(args.dataset)
    spike_metrics = evaluate_model(raw, "target_spike")
    continuation_metrics = evaluate_model(raw, "target_continuation")
    supernova_metrics = evaluate_model(raw, "target_supernova")
    toxic_metrics = evaluate_model(raw, "target_toxic")
    failed_setup_metrics = evaluate_model(raw, "target_failed_setup")

    artifact = fit_final_models(raw)
    save_artifact(artifact, args.model_out)

    metrics = {
        "dataset": str(args.dataset),
        "training_rows": int(len(raw)),
        "positive_labels": artifact["positive_labels"],
        "spike_model_metrics": spike_metrics,
        "continuation_model_metrics": continuation_metrics,
        "supernova_model_metrics": supernova_metrics,
        "toxic_model_metrics": toxic_metrics,
        "failed_setup_model_metrics": failed_setup_metrics,
        "top_features_by_spike_coef": artifact["feature_importance"][:12],
    }
    save_metrics(metrics, args.metrics_out)

    print("[done] wrote model:", args.model_out)
    print("[done] wrote metrics:", args.metrics_out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
