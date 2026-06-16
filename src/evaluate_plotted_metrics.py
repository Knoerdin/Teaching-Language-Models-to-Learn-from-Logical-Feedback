from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import time
from typing import Any

from evaluate_autoformalization import GenerationConfig
from evaluate_autoformalization import ModelSpec
from evaluate_autoformalization import build_prompt
from evaluate_autoformalization import format_duration
from evaluate_autoformalization import generate_batch
from evaluate_autoformalization import label_classification_metrics
from evaluate_autoformalization import load_jsonl_dataset
from evaluate_autoformalization import load_model
from evaluate_autoformalization import load_tokenizer
from evaluate_autoformalization import parse_model_specs
from evaluate_autoformalization import release_model
from evaluate_autoformalization import sanitize_filename
from evaluate_autoformalization import score_prediction
from evaluate_autoformalization import set_seed
from evaluate_autoformalization import validate_model_specs
from evaluate_autoformalization import write_json
from evaluate_autoformalization import write_jsonl


PLOTTED_METRIC_KEYS = (
    "label_accuracy",
    "parse_rate",
    "gold_fol_accuracy",
    "label_macro_f1",
    "label_true_f1",
    "label_false_f1",
    "label_uncertain_f1",
)

PREDICTION_OUTPUT_KEYS = (
    "example_id",
    "story_id",
    "label",
    "parsed",
    "joint_unordered_exact",
    "prover_status",
    "prover_prediction",
    "prover_label_correct",
    "generation",
    "predicted_premises",
    "predicted_conclusion",
    "gold_premises",
    "gold_conclusion",
    "model_name",
    "model_path",
    "include_gold_schema",
    "postprocessed",
    "evaluation_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast direct evaluator for the metrics used by plot_evaluation_metrics.py. "
            "This script never runs Draft-and-Prune."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Model checkpoint or adapter to evaluate. Use NAME=PATH for readable "
            "outputs. Repeat for multiple models."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="DATA/FOLIO/folio_test.jsonl",
        help="JSONL dataset containing FOLIO-style gold fields.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for compact metrics and prediction JSONL files.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="Optional prompt truncation length.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument(
        "--include-gold-schema",
        action="store_true",
        help="Include the per-example gold predicate/constant schema in prompts.",
    )
    parser.add_argument(
        "--postprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply the existing evaluation-only FOL cleanup before scoring.",
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Include the prompt in prediction JSONL outputs.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def rate(results: list[dict[str, Any]], key: str) -> float:
    return (
        sum(1 for result in results if result.get(key, False)) / len(results)
        if results
        else 0.0
    )


def compact_prediction(result: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: result.get(key)
        for key in PREDICTION_OUTPUT_KEYS
        if key in result
    }
    if "prompt" in result:
        compact["prompt"] = result["prompt"]
    return compact


def compute_plotted_metrics(
    spec: ModelSpec,
    dataset_path: str,
    results: list[dict[str, Any]],
    generation_config: GenerationConfig,
) -> dict[str, Any]:
    label_metrics = label_classification_metrics(results)
    label_metric_keys = (
        "label_accuracy",
        "label_macro_f1",
        "label_true_f1",
        "label_false_f1",
        "label_uncertain_f1",
    )
    missing = [key for key in label_metric_keys if key not in label_metrics]
    if missing:
        raise ValueError(
            "Label metrics could not be computed. This evaluator should always "
            "run the prover; missing keys: " + ", ".join(missing)
        )

    prover_status_counts = Counter(result["prover_status"] for result in results)
    return {
        "model_name": spec.name,
        "model_path": spec.path,
        "trainer_kind": spec.trainer_kind,
        "dataset": dataset_path,
        "n_examples": len(results),
        "batch_size": generation_config.batch_size,
        "max_new_tokens": generation_config.max_new_tokens,
        "max_input_tokens": generation_config.max_input_tokens,
        "temperature": generation_config.temperature,
        "top_p": generation_config.top_p,
        "direct_only": True,
        "draft_and_prune": False,
        "include_gold_schema": any(
            result.get("include_gold_schema", False) for result in results
        ),
        "postprocess": any(result.get("postprocessed", False) for result in results),
        "parse_rate": rate(results, "parsed"),
        "gold_fol_accuracy": rate(results, "joint_unordered_exact"),
        "label_accuracy": label_metrics["label_accuracy"],
        "label_macro_f1": label_metrics["label_macro_f1"],
        "label_true_f1": label_metrics["label_true_f1"],
        "label_false_f1": label_metrics["label_false_f1"],
        "label_uncertain_f1": label_metrics["label_uncertain_f1"],
        "prover_status_counts": dict(sorted(prover_status_counts.items())),
    }


def evaluate_model(
    spec: ModelSpec,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    generation_config: GenerationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(f"Loading {spec.name}: {spec.path}", flush=True)
    tokenizer = load_tokenizer(spec.path, trust_remote_code=args.trust_remote_code)
    model = load_model(
        spec.path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )

    results = []
    started_at = time.monotonic()
    for start in range(0, len(records), generation_config.batch_size):
        batch_started_at = time.monotonic()
        end = min(start + generation_config.batch_size, len(records))
        batch = records[start:end]
        prompts = [
            build_prompt(
                record,
                include_gold_schema=args.include_gold_schema,
            )
            for record in batch
        ]
        generations = generate_batch(model, tokenizer, prompts, generation_config)
        for record, prompt, generation in zip(
            batch, prompts, generations, strict=True
        ):
            result = score_prediction(
                record,
                generation,
                run_prover=True,
                apply_postprocessing=args.postprocess,
            )
            result["model_name"] = spec.name
            result["model_path"] = spec.path
            result["include_gold_schema"] = args.include_gold_schema
            result["evaluation_seconds"] = time.monotonic() - batch_started_at
            if args.include_prompt:
                result["prompt"] = prompt
            results.append(compact_prediction(result))

        if args.log_every > 0 and (
            end == len(records) or end % args.log_every == 0
        ):
            elapsed = time.monotonic() - started_at
            print(
                f"  {spec.name}: evaluated {end}/{len(records)} "
                f"elapsed={format_duration(elapsed)}",
                flush=True,
            )

    metrics = compute_plotted_metrics(
        spec,
        args.dataset,
        results,
        generation_config,
    )
    release_model(model)
    return metrics, results


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        key
        for key in summaries[0]
        if not isinstance(summaries[0][key], (dict, list))
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in fieldnames})


def print_metrics(metrics: dict[str, Any]) -> None:
    print(f"\n{metrics['model_name']}")
    for key in PLOTTED_METRIC_KEYS:
        print(f"  {key}: {metrics[key]:.4f}")


def main() -> None:
    args = parse_args()
    specs = parse_model_specs(args.model)
    validate_model_specs(specs)
    set_seed(args.seed)

    records = load_jsonl_dataset(args.dataset, args.limit)
    generation_config = GenerationConfig(
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    output_dir = Path(args.output_dir).expanduser()
    summaries = []
    for spec in specs:
        metrics, predictions = evaluate_model(spec, records, args, generation_config)
        summaries.append(metrics)
        print_metrics(metrics)

        prefix = sanitize_filename(spec.name)
        write_json(output_dir / f"{prefix}_metrics.json", metrics)
        write_jsonl(output_dir / f"{prefix}_predictions.jsonl", predictions)

    write_json(output_dir / "metrics_summary.json", summaries)
    write_summary_csv(output_dir / "metrics_summary.csv", summaries)
    print(f"\nWrote plotted-metric evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
