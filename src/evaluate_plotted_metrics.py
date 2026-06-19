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
from evaluate_autoformalization import loaded_model_metadata
from evaluate_autoformalization import parse_model_specs
from evaluate_autoformalization import release_model
from evaluate_autoformalization import sanitize_filename
from evaluate_autoformalization import score_prediction
from evaluate_autoformalization import set_seed
from evaluate_autoformalization import solver_parser_parse_rate
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
    "nl_premises",
    "nl_conclusion",
    "parsed",
    "joint_unordered_exact",
    "premise_f1",
    "schema_predicate_f1",
    "schema_constant_f1",
    "gold_fol_reward",
    "schema_clean",
    "prover_status",
    "prover_prediction",
    "prover_label_correct",
    "prover_feedback",
    "generation",
    "raw_predicted_premises",
    "raw_predicted_conclusion",
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
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help=(
            "Generation repetition penalty. Set to 1.1 to mirror the Qwen3.5 "
            "GRPO training config."
        ),
    )
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
    parser.add_argument(
        "--write-model-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write readable per-example model outputs next to the JSONL files.",
    )
    parser.add_argument(
        "--model-output-examples",
        type=int,
        default=0,
        help="Maximum examples per readable model-output file. Use 0 for all.",
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
    model_metadata: dict[str, Any],
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
    label_counts = Counter(result["label"] for result in results)
    prover_prediction_counts = Counter(
        str(result.get("prover_prediction")) for result in results
    )
    return {
        "model_name": spec.name,
        "model_path": spec.path,
        "trainer_kind": spec.trainer_kind,
        "model_load_type": model_metadata.get("model_load_type"),
        "model_class": model_metadata.get("model_class"),
        "is_peft_model": model_metadata.get("is_peft_model"),
        "peft_active_adapter": model_metadata.get("peft_active_adapter"),
        "peft_base_model_name_or_path": model_metadata.get(
            "peft_base_model_name_or_path"
        ),
        "model_metadata": model_metadata,
        "dataset": dataset_path,
        "n_examples": len(results),
        "batch_size": generation_config.batch_size,
        "max_new_tokens": generation_config.max_new_tokens,
        "max_input_tokens": generation_config.max_input_tokens,
        "temperature": generation_config.temperature,
        "top_p": generation_config.top_p,
        "repetition_penalty": generation_config.repetition_penalty,
        "direct_only": True,
        "draft_and_prune": False,
        "include_gold_schema": any(
            result.get("include_gold_schema", False) for result in results
        ),
        "postprocess": any(result.get("postprocessed", False) for result in results),
        "parse_rate": solver_parser_parse_rate(results),
        "solver_parse_rate": solver_parser_parse_rate(results),
        "format_extraction_rate": rate(results, "parsed"),
        "gold_fol_accuracy": rate(results, "joint_unordered_exact"),
        "label_accuracy": label_metrics["label_accuracy"],
        "label_macro_f1": label_metrics["label_macro_f1"],
        "label_true_f1": label_metrics["label_true_f1"],
        "label_false_f1": label_metrics["label_false_f1"],
        "label_uncertain_f1": label_metrics["label_uncertain_f1"],
        "label_counts": dict(sorted(label_counts.items())),
        "prover_prediction_counts": dict(sorted(prover_prediction_counts.items())),
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
    model_metadata = loaded_model_metadata(spec.path, model)
    print(
        "  loaded "
        f"{model_metadata.get('model_load_type')} "
        f"class={model_metadata.get('model_class')} "
        f"is_peft={model_metadata.get('is_peft_model')}",
        flush=True,
    )
    if model_metadata.get("peft_base_model_name_or_path"):
        print(
            "  peft base="
            f"{model_metadata.get('peft_base_model_name_or_path')} "
            f"active={model_metadata.get('peft_active_adapter')}",
            flush=True,
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
        model_metadata,
    )
    release_model(model)
    return metrics, results


def selected_predictions(
    predictions: list[dict[str, Any]],
    max_examples: int,
) -> list[dict[str, Any]]:
    if max_examples <= 0:
        return predictions
    return predictions[:max_examples]


def text_block(title: str, value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return f"{title}:\n{text or '<empty>'}\n"


def markdown_block(title: str, value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return f"**{title}**\n\n```text\n{text or '<empty>'}\n```\n"


def write_model_outputs(
    output_dir: Path,
    spec: ModelSpec,
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    max_examples: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = sanitize_filename(spec.name)
    selected = selected_predictions(predictions, max_examples)
    total = len(predictions)

    text_path = output_dir / f"{prefix}_outputs.txt"
    markdown_path = output_dir / f"{prefix}_outputs.md"

    text_parts = [
        f"Model: {metrics['model_name']}",
        f"Path: {metrics['model_path']}",
        f"Loaded as: {metrics.get('model_load_type')}",
        f"Examples shown: {len(selected)}/{total}",
        f"Label accuracy: {metrics['label_accuracy']:.4f}",
        f"Parser parse rate: {metrics['parse_rate']:.4f}",
        f"Format extraction rate: {metrics.get('format_extraction_rate', 0.0):.4f}",
        f"Gold FOL accuracy: {metrics['gold_fol_accuracy']:.4f}",
        "",
    ]
    markdown_parts = [
        f"# {metrics['model_name']} Outputs",
        "",
        f"- Path: `{metrics['model_path']}`",
        f"- Loaded as: `{metrics.get('model_load_type')}`",
        f"- Examples shown: {len(selected)}/{total}",
        f"- Label accuracy: {metrics['label_accuracy']:.4f}",
        f"- Parser parse rate: {metrics['parse_rate']:.4f}",
        f"- Format extraction rate: {metrics.get('format_extraction_rate', 0.0):.4f}",
        f"- Gold FOL accuracy: {metrics['gold_fol_accuracy']:.4f}",
        "",
    ]

    for index, result in enumerate(selected, start=1):
        label_correct = bool(result.get("prover_label_correct"))
        text_parts.extend(
            [
                "=" * 88,
                f"Example {index}/{total}",
                f"example_id={result.get('example_id')} story_id={result.get('story_id')}",
                (
                    f"gold_label={result.get('label')} "
                    f"prover_prediction={result.get('prover_prediction')} "
                    f"prover_status={result.get('prover_status')} "
                    f"label_correct={label_correct}"
                ),
                (
                    f"parsed={result.get('parsed')} "
                    f"gold_fol_exact={result.get('joint_unordered_exact')} "
                    f"schema_clean={result.get('schema_clean')} "
                    f"gold_fol_reward={result.get('gold_fol_reward')}"
                ),
                "",
                text_block("Natural-language premises", result.get("nl_premises")),
                text_block("Natural-language conclusion", result.get("nl_conclusion")),
                text_block("Raw generation", result.get("generation")),
                text_block("Predicted premises", result.get("predicted_premises")),
                text_block("Predicted conclusion", result.get("predicted_conclusion")),
                text_block("Gold premises", result.get("gold_premises")),
                text_block("Gold conclusion", result.get("gold_conclusion")),
            ]
        )
        if result.get("prover_feedback"):
            text_parts.append(text_block("Prover feedback", result.get("prover_feedback")))
        if result.get("prompt"):
            text_parts.append(text_block("Prompt", result.get("prompt")))

        markdown_parts.extend(
            [
                f"## Example {index}/{total}",
                "",
                f"- example_id: `{result.get('example_id')}`",
                f"- story_id: `{result.get('story_id')}`",
                (
                    f"- labels: gold `{result.get('label')}`, prover "
                    f"`{result.get('prover_prediction')}`, status "
                    f"`{result.get('prover_status')}`, correct `{label_correct}`"
                ),
                (
                    f"- FOL: parsed `{result.get('parsed')}`, exact "
                    f"`{result.get('joint_unordered_exact')}`, schema clean "
                    f"`{result.get('schema_clean')}`"
                ),
                "",
                markdown_block("Natural-language premises", result.get("nl_premises")),
                markdown_block("Natural-language conclusion", result.get("nl_conclusion")),
                markdown_block("Raw generation", result.get("generation")),
                markdown_block("Predicted premises", result.get("predicted_premises")),
                markdown_block("Predicted conclusion", result.get("predicted_conclusion")),
                markdown_block("Gold premises", result.get("gold_premises")),
                markdown_block("Gold conclusion", result.get("gold_conclusion")),
            ]
        )
        if result.get("prover_feedback"):
            markdown_parts.append(
                markdown_block("Prover feedback", result.get("prover_feedback"))
            )
        if result.get("prompt"):
            markdown_parts.append(markdown_block("Prompt", result.get("prompt")))

    text_path.write_text("\n".join(text_parts).rstrip() + "\n", encoding="utf-8")
    markdown_path.write_text(
        "\n".join(markdown_parts).rstrip() + "\n",
        encoding="utf-8",
    )
    return [text_path, markdown_path]


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
    print(f"  format_extraction_rate: {metrics['format_extraction_rate']:.4f}")


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
        repetition_penalty=args.repetition_penalty,
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
        if args.write_model_outputs:
            paths = write_model_outputs(
                output_dir / "model_outputs",
                spec,
                metrics,
                predictions,
                args.model_output_examples,
            )
            print(
                "  wrote model outputs: "
                + ", ".join(str(path) for path in paths),
                flush=True,
            )

    write_json(output_dir / "metrics_summary.json", summaries)
    write_summary_csv(output_dir / "metrics_summary.csv", summaries)
    print(f"\nWrote plotted-metric evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
