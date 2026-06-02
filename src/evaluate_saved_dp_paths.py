from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_autoformalization import GenerationConfig
from evaluate_autoformalization import ModelSpec
from evaluate_autoformalization import build_report_summary_markdown
from evaluate_autoformalization import infer_trainer_kind
from evaluate_autoformalization import sanitize_filename
from evaluate_autoformalization import score_prediction
from evaluate_autoformalization import summarize_results
from evaluate_autoformalization import write_evaluation_reports
from evaluate_autoformalization import write_json
from evaluate_autoformalization import write_jsonl
from evaluate_autoformalization import write_summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute offline single-path metrics from saved Draft-and-Prune "
            "prediction JSONL files, without model generation."
        )
    )
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help=(
            "D&P prediction JSONL to evaluate. Use NAME=PATH for readable "
            "outputs. Repeat for multiple models."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for offline metrics, predictions, and reports.",
    )
    parser.add_argument(
        "--path-index",
        type=int,
        default=1,
        help="One-indexed D&P path to use as the single-path baseline.",
    )
    parser.add_argument(
        "--report-examples",
        type=int,
        default=3,
        help="Number of examples per section in Markdown reports.",
    )
    return parser.parse_args()


def parse_prediction_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
            name = name.strip()
            path = path.strip()
        else:
            path = value.strip()
            name = Path(path).stem.replace("_predictions", "")
        if not name or not path:
            raise ValueError(f"Invalid --predictions value: {value!r}")
        specs.append((name, Path(path).expanduser()))
    return specs


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_from_dp_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": result.get("example_id", ""),
        "story_id": result.get("story_id", ""),
        "premises": result.get("nl_premises", ""),
        "conclusion": result.get("nl_conclusion", ""),
        "premises-FOL": result.get("gold_premises", ""),
        "conclusion-FOL": result.get("gold_conclusion", ""),
        "label": result.get("label", ""),
    }


def select_path(result: dict[str, Any], path_index: int) -> dict[str, Any] | None:
    for path_result in result.get("dp_paths", []):
        if int(path_result.get("dp_path_index", -1)) == path_index:
            return path_result
    return None


def score_saved_path(
    result: dict[str, Any],
    path_result: dict[str, Any] | None,
    *,
    model_name: str,
    model_path: str,
    source_predictions: str,
    path_index: int,
) -> dict[str, Any]:
    record = record_from_dp_result(result)
    generation = str((path_result or {}).get("generation") or "")
    scored = score_prediction(
        record,
        generation,
        run_prover=False,
        apply_postprocessing=bool(result.get("postprocessed", False)),
    )

    if path_result is None:
        scored.update(
            {
                "prover_status": "not_parsed",
                "prover_prediction": None,
                "prover_label_correct": False,
                "prover_feedback": None,
                "offline_source_missing_path": True,
            }
        )
    else:
        scored.update(
            {
                "prover_status": path_result.get("prover_status", "not_parsed"),
                "prover_prediction": path_result.get("prover_prediction"),
                "prover_label_correct": bool(
                    path_result.get("dp_path_label_correct", False)
                ),
                "prover_feedback": path_result.get("prover_feedback"),
                "offline_source_missing_path": False,
                "offline_source_dp_path_executable": bool(
                    path_result.get("dp_path_executable", False)
                ),
                "offline_source_dp_path_label": path_result.get("dp_path_label"),
                "offline_source_repair_round": int(
                    path_result.get("repair_round", 0) or 0
                ),
            }
        )

    scored.update(
        {
            "model_name": model_name,
            "model_path": model_path,
            "include_gold_schema": bool(result.get("include_gold_schema", False)),
            "offline_from_draft_and_prune": True,
            "offline_source_predictions": source_predictions,
            "offline_path_index": path_index,
            "offline_selection": f"path_{path_index}",
            "offline_note": (
                "Single saved D&P path scored without majority vote or new model "
                "generation. This is not a direct-prompt non-D&P generation."
            ),
        }
    )
    return scored


def summarize_offline_model(
    name: str,
    source_path: Path,
    source_results: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    path_index: int,
) -> dict[str, Any]:
    first = source_results[0]
    spec = ModelSpec(
        name=name,
        path=str(first.get("model_path") or source_path),
        trainer_kind=infer_trainer_kind(name, str(first.get("model_path") or "")),
    )
    generation_config = GenerationConfig(
        batch_size=int(first.get("batch_size", 1) or 1),
        max_new_tokens=0,
        max_input_tokens=None,
        temperature=0.0,
        top_p=1.0,
    )
    metrics = summarize_results(
        spec,
        str(first.get("dataset") or "saved_draft_and_prune_predictions"),
        predictions,
        run_prover=True,
        generation_config=generation_config,
    )
    metrics.update(
        {
            "offline_from_draft_and_prune": True,
            "offline_source_predictions": str(source_path),
            "offline_path_index": path_index,
            "offline_selection": f"path_{path_index}",
            "offline_note": (
                "Single saved D&P path scored without majority vote or new model "
                "generation. This is not a direct-prompt non-D&P generation."
            ),
        }
    )
    return metrics


def main() -> None:
    args = parse_args()
    if args.path_index <= 0:
        raise ValueError("--path-index must be greater than 0.")

    output_dir = Path(args.output_dir).expanduser()
    model_outputs = []
    summaries = []
    for name, source_path in parse_prediction_specs(args.predictions):
        source_results = load_jsonl(source_path)
        if not source_results:
            raise ValueError(f"No predictions found in {source_path}")

        model_path = str(source_results[0].get("model_path") or source_path)
        predictions = [
            score_saved_path(
                result,
                select_path(result, args.path_index),
                model_name=name,
                model_path=model_path,
                source_predictions=str(source_path),
                path_index=args.path_index,
            )
            for result in source_results
        ]
        metrics = summarize_offline_model(
            name,
            source_path,
            source_results,
            predictions,
            args.path_index,
        )
        summaries.append(metrics)
        model_outputs.append((metrics, predictions))

        prefix = sanitize_filename(name)
        write_json(output_dir / f"{prefix}_metrics.json", metrics)
        write_jsonl(output_dir / f"{prefix}_predictions.jsonl", predictions)

    write_json(output_dir / "metrics_summary.json", summaries)
    write_summary_csv(output_dir / "metrics_summary.csv", summaries)
    write_evaluation_reports(
        output_dir / "eval_reports",
        model_outputs,
        examples_per_section=args.report_examples,
    )
    print(f"Wrote offline single-path metrics to {output_dir}")
    print(build_report_summary_markdown(
        [
            {
                "metrics": metrics,
                "predictions": predictions,
                "path": output_dir
                / "eval_reports"
                / sanitize_filename(str(metrics.get("trainer_kind", "unknown")))
                / f"{sanitize_filename(str(metrics.get('model_name', 'model')))}_report.md",
            }
            for metrics, predictions in model_outputs
        ],
        output_dir / "eval_reports",
    ))


if __name__ == "__main__":
    main()
