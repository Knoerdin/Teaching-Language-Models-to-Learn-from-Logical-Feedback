from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
from typing import Any

from REWARDS.formatting import extract_formalization
from autoformalization import build_prompt, normalize_label


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


REQUIRED_DATASET_COLUMNS = {
    "premises",
    "premises-FOL",
    "conclusion",
    "conclusion-FOL",
    "label",
}
ARROW_ALIASES = ("->", "=>", "⇒")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str
    trainer_kind: str


@dataclass(frozen=True)
class GenerationConfig:
    batch_size: int
    max_new_tokens: int
    max_input_tokens: int | None
    temperature: float
    top_p: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GRPO/SFT autoformalization checkpoints against FOL gold labels."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Model checkpoint or adapter to evaluate. Use NAME=PATH for readable "
            "outputs, e.g. --model grpo=outputs/grpo_qwen3.5-9b. Repeat for SFT."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="DATA/FOLIO/folio_test.jsonl",
        help="JSONL dataset containing premises-FOL and conclusion-FOL gold fields.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for metrics JSON, summary CSV, and predictions JSONL.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help=(
            "Optional directory for Markdown evaluation reports. Defaults to "
            "OUTPUT_DIR/eval_reports when --output-dir is set."
        ),
    )
    parser.add_argument(
        "--report-examples",
        type=int,
        default=3,
        help="Number of correct and wrong examples to include in each report.",
    )
    parser.add_argument(
        "--predictions-output",
        default=None,
        help="Optional JSONL path for predictions. Only valid with one --model.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="Optional prompt truncation length. By default prompts are not truncated.",
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
        "--run-prover",
        action="store_true",
        help=(
            "Also run the symbolic prover on generated FOL and report label accuracy. "
            "Exact gold-FOL accuracy is always reported."
        ),
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Include the full prompt in prediction JSONL outputs.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def parse_model_specs(values: list[str]) -> list[ModelSpec]:
    specs = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
            name = name.strip()
            path = path.strip()
        else:
            path = value.strip()
            name = Path(path).name or re.sub(r"[^A-Za-z0-9_.-]+", "_", path)
        if not name or not path:
            raise ValueError(f"Invalid --model value: {value!r}")
        specs.append(
            ModelSpec(name=name, path=path, trainer_kind=infer_trainer_kind(name, path))
        )
    return specs


def is_local_like_model_path(path: str) -> bool:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return True
    if path.startswith(("./", "../", "~/")):
        return True
    return expanded.parts[:1] in {("outputs",), ("checkpoints",), ("runs",)}


def validate_model_specs(specs: list[ModelSpec]) -> None:
    missing = [
        spec
        for spec in specs
        if is_local_like_model_path(spec.path) and not Path(spec.path).expanduser().exists()
    ]
    if not missing:
        return

    formatted = "\n".join(
        f"  --model {spec.name}={spec.path}" for spec in missing
    )
    raise FileNotFoundError(
        "Model path does not exist locally:\n"
        f"{formatted}\n"
        "If this checkpoint was trained on Snellius, run evaluation on Snellius "
        "or sync the checkpoint directory to this machine first."
    )


def infer_trainer_kind(name: str, path: str | None = None) -> str:
    name_lowered = name.lower()
    path_lowered = (path or "").lower()
    if "sft" in name_lowered:
        return "sft"
    if "grpo" in name_lowered:
        return "grpo"
    if "sft" in path_lowered:
        return "sft"
    if "grpo" in path_lowered:
        return "grpo"
    return "unknown"


def load_jsonl_dataset(path: str, limit: int | None) -> list[dict[str, Any]]:
    records = []
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {source_path}")

    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            missing = REQUIRED_DATASET_COLUMNS.difference(record)
            if missing:
                raise ValueError(
                    f"{source_path}:{line_number} is missing fields: {sorted(missing)}"
                )
            records.append(record)
            if limit is not None and len(records) >= limit:
                break

    return records


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_fol_line(line: str) -> str:
    line = line.strip()
    for alias in ARROW_ALIASES:
        line = line.replace(alias, "→")
    line = line.replace("~", "¬")
    return re.sub(r"\s+", "", line)


def normalized_lines(block: str | None) -> list[str]:
    if block is None:
        return []
    return [
        normalized
        for line in block.splitlines()
        if (normalized := normalize_fol_line(line))
    ]


def normalized_block(block: str | None) -> str:
    return "\n".join(normalized_lines(block))


def stripped_block(block: str | None) -> str:
    if block is None:
        return ""
    return "\n".join(line.strip() for line in block.splitlines() if line.strip())


def multiset_equal(left: list[str], right: list[str]) -> bool:
    return Counter(left) == Counter(right)


def premise_overlap(predicted: list[str], gold: list[str]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0

    overlap = sum((Counter(predicted) & Counter(gold)).values())
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    if precision + recall == 0.0:
        return precision, recall, 0.0
    return precision, recall, 2.0 * precision * recall / (precision + recall)


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "model"


def is_peft_adapter(path: str) -> bool:
    local_path = Path(path)
    return (local_path / "adapter_config.json").exists()


def torch_dtype_from_name(name: str):
    import torch

    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def resolve_device_loading(device: str) -> tuple[str | None, str | None]:
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "auto", None
        if torch.backends.mps.is_available():
            return None, "mps"
        return None, "cpu"

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "auto", None
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return None, "mps"
    return None, "cpu"


def first_model_device(model: Any):
    import torch

    device_map = getattr(model, "hf_device_map", None)
    if device_map is None and hasattr(model, "base_model"):
        device_map = getattr(model.base_model, "hf_device_map", None)
    if device_map:
        for device in device_map.values():
            if str(device) in {"cpu", "disk"}:
                continue
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            return torch.device(str(device))
    return next(model.parameters()).device


def load_tokenizer(model_path: str, *, trust_remote_code: bool):
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        if not is_peft_adapter(model_path):
            raise
        from peft import PeftConfig

        peft_config = PeftConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(
    model_path: str,
    *,
    device: str,
    torch_dtype: str,
    trust_remote_code: bool,
    attn_implementation: str | None,
):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch_dtype_from_name(torch_dtype)
    device_map, move_to_device = resolve_device_loading(device)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if is_peft_adapter(model_path):
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if move_to_device is not None:
        model = model.to(torch.device(move_to_device))
    model.eval()
    return model


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    config: GenerationConfig,
) -> list[str]:
    import torch

    tokenizer_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": config.max_input_tokens is not None,
    }
    if config.max_input_tokens is not None:
        tokenizer_kwargs["max_length"] = config.max_input_tokens

    tokenized = tokenizer(prompts, **tokenizer_kwargs)
    input_length = tokenized["input_ids"].shape[1]
    input_device = first_model_device(model)
    tokenized = {key: value.to(input_device) for key, value in tokenized.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": config.temperature > 0.0,
    }
    if config.temperature > 0.0:
        generation_kwargs["temperature"] = config.temperature
        generation_kwargs["top_p"] = config.top_p

    with torch.inference_mode():
        generated = model.generate(**tokenized, **generation_kwargs)

    new_tokens = generated[:, input_length:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def score_prediction(
    record: dict[str, Any],
    generation: str,
    *,
    run_prover: bool,
) -> dict[str, Any]:
    predicted_premises, predicted_conclusion = extract_formalization(generation)
    parsed = predicted_premises is not None and predicted_conclusion is not None

    gold_premises = str(record["premises-FOL"]).strip()
    gold_conclusion = str(record["conclusion-FOL"]).strip()
    predicted_premise_lines = normalized_lines(predicted_premises)
    gold_premise_lines = normalized_lines(gold_premises)
    predicted_conclusion_norm = normalized_block(predicted_conclusion)
    gold_conclusion_norm = normalized_block(gold_conclusion)

    premises_ordered_exact = normalized_block(predicted_premises) == normalized_block(
        gold_premises
    )
    premises_unordered_exact = multiset_equal(
        predicted_premise_lines,
        gold_premise_lines,
    )
    conclusion_exact = predicted_conclusion_norm == gold_conclusion_norm
    premise_precision, premise_recall, premise_f1 = premise_overlap(
        predicted_premise_lines,
        gold_premise_lines,
    )

    result = {
        "example_id": str(record.get("example_id", "")),
        "story_id": str(record.get("story_id", "")),
        "label": normalize_label(str(record["label"])),
        "nl_premises": str(record["premises"]),
        "nl_conclusion": str(record["conclusion"]),
        "parsed": parsed,
        "generation": generation,
        "predicted_premises": predicted_premises,
        "predicted_conclusion": predicted_conclusion,
        "gold_premises": gold_premises,
        "gold_conclusion": gold_conclusion,
        "premises_ordered_exact": premises_ordered_exact,
        "premises_unordered_exact": premises_unordered_exact,
        "conclusion_exact": conclusion_exact,
        "joint_ordered_exact": premises_ordered_exact and conclusion_exact,
        "joint_unordered_exact": premises_unordered_exact and conclusion_exact,
        "strict_joint_exact": stripped_block(predicted_premises)
        == stripped_block(gold_premises)
        and stripped_block(predicted_conclusion) == stripped_block(gold_conclusion),
        "premise_precision": premise_precision,
        "premise_recall": premise_recall,
        "premise_f1": premise_f1,
    }

    if run_prover:
        result.update(
            run_prover_score(record, predicted_premises, predicted_conclusion)
        )

    return result


def run_prover_score(
    record: dict[str, Any],
    predicted_premises: str | None,
    predicted_conclusion: str | None,
) -> dict[str, Any]:
    if predicted_premises is None or predicted_conclusion is None:
        return {
            "prover_status": "not_parsed",
            "prover_prediction": None,
            "prover_label_correct": False,
            "prover_feedback": None,
        }

    from REWARDS.proving import evaluate_correctness

    prover_result = evaluate_correctness(
        nl_premises=str(record["premises"]),
        nl_conclusion=str(record["conclusion"]),
        formal_premises=predicted_premises,
        formal_conclusion=predicted_conclusion,
        gold_label=normalize_label(str(record["label"])),
    )
    return {
        "prover_status": prover_result.status,
        "prover_prediction": prover_result.prediction,
        "prover_label_correct": prover_result.status == "correct",
        "prover_feedback": prover_result.feedback or prover_result.error_message,
    }


def summarize_results(
    spec: ModelSpec,
    dataset_path: str,
    results: list[dict[str, Any]],
    *,
    run_prover: bool,
    generation_config: GenerationConfig,
) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        raise ValueError("No examples were evaluated.")

    def rate(key: str) -> float:
        return sum(1 for result in results if result[key]) / total

    metrics: dict[str, Any] = {
        "model_name": spec.name,
        "model_path": spec.path,
        "trainer_kind": spec.trainer_kind,
        "dataset": dataset_path,
        "n_examples": total,
        "batch_size": generation_config.batch_size,
        "max_new_tokens": generation_config.max_new_tokens,
        "max_input_tokens": generation_config.max_input_tokens,
        "temperature": generation_config.temperature,
        "top_p": generation_config.top_p,
        "parse_rate": rate("parsed"),
        "autoformalization_accuracy": rate("joint_ordered_exact"),
        "autoformalization_accuracy_ignore_premise_order": rate(
            "joint_unordered_exact"
        ),
        "strict_autoformalization_accuracy": rate("strict_joint_exact"),
        "premises_accuracy": rate("premises_ordered_exact"),
        "premises_accuracy_ignore_order": rate("premises_unordered_exact"),
        "conclusion_accuracy": rate("conclusion_exact"),
        "premise_macro_precision": mean(
            result["premise_precision"] for result in results
        ),
        "premise_macro_recall": mean(result["premise_recall"] for result in results),
        "premise_macro_f1": mean(result["premise_f1"] for result in results),
    }

    if run_prover:
        prover_status_counts = Counter(result["prover_status"] for result in results)
        attempted = total - prover_status_counts.get("not_parsed", 0)
        correct = sum(1 for result in results if result["prover_label_correct"])
        metrics["prover_label_accuracy"] = correct / total
        metrics["prover_label_accuracy_on_parsed"] = (
            correct / attempted if attempted else 0.0
        )
        metrics["prover_attempted"] = attempted
        metrics["prover_status_counts"] = dict(sorted(prover_status_counts.items()))

    return metrics


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


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
    for start in range(0, len(records), generation_config.batch_size):
        end = min(start + generation_config.batch_size, len(records))
        batch = records[start:end]
        prompts = [build_prompt(record) for record in batch]
        generations = generate_batch(model, tokenizer, prompts, generation_config)
        for record, prompt, generation in zip(batch, prompts, generations):
            result = score_prediction(
                record,
                generation,
                run_prover=args.run_prover,
            )
            result["model_name"] = spec.name
            result["model_path"] = spec.path
            if args.include_prompt:
                result["prompt"] = prompt
            results.append(result)

        if args.log_every > 0 and (
            end == len(records) or end % args.log_every == 0
        ):
            print(f"  {spec.name}: evaluated {end}/{len(records)}", flush=True)

    metrics = summarize_results(
        spec,
        args.dataset,
        results,
        run_prover=args.run_prover,
        generation_config=generation_config,
    )

    release_model(model)
    return metrics, results


def release_model(model: Any) -> None:
    del model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_keys = [
        key
        for key in summaries[0]
        if not isinstance(summaries[0][key], (dict, list))
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in scalar_keys})


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def resolve_report_dir(args: argparse.Namespace, output_dir: Path | None) -> Path | None:
    if args.report_dir:
        return Path(args.report_dir).expanduser()
    if output_dir:
        return output_dir / "eval_reports"
    return None


def write_evaluation_reports(
    report_dir: Path,
    model_outputs: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    *,
    examples_per_section: int,
) -> list[Path]:
    written_paths = []
    report_entries = []
    for metrics, predictions in model_outputs:
        trainer_kind = sanitize_filename(str(metrics.get("trainer_kind", "unknown")))
        model_name = sanitize_filename(str(metrics.get("model_name", "model")))
        report_path = report_dir / trainer_kind / f"{model_name}_report.md"
        write_text(
            report_path,
            build_model_report_markdown(metrics, predictions, examples_per_section),
        )
        written_paths.append(report_path)
        report_entries.append(
            {
                "metrics": metrics,
                "predictions": predictions,
                "path": report_path,
            }
        )

    summary_path = report_dir / "summary.md"
    write_text(summary_path, build_report_summary_markdown(report_entries, report_dir))
    written_paths.append(summary_path)
    return written_paths


def build_model_report_markdown(
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    examples_per_section: int,
) -> str:
    examples_per_section = max(0, examples_per_section)
    correct_examples = [
        result for result in predictions if result["joint_unordered_exact"]
    ][:examples_per_section]
    wrong_examples = select_wrong_examples(predictions, examples_per_section)
    model_name = str(metrics["model_name"])
    trainer_kind = str(metrics["trainer_kind"]).upper()

    lines = [
        f"# Evaluation Report: {model_name}",
        "",
        "## Run",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Model | {markdown_cell(model_name)} |",
        f"| Trainer kind | {markdown_cell(trainer_kind)} |",
        f"| Model path | {markdown_cell(str(metrics['model_path']))} |",
        f"| Dataset | {markdown_cell(str(metrics['dataset']))} |",
        f"| Examples | {metrics['n_examples']} |",
        f"| Batch size | {metrics['batch_size']} |",
        f"| Max new tokens | {metrics['max_new_tokens']} |",
        f"| Max input tokens | {markdown_cell(metrics['max_input_tokens'])} |",
        f"| Temperature | {format_metric_value(metrics['temperature'])} |",
        f"| Top p | {format_metric_value(metrics['top_p'])} |",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Count |",
        "| --- | ---: | ---: |",
    ]

    for label, metric_key, result_key in report_metric_rows(metrics):
        count_text = ""
        if result_key is not None:
            count_text = f"{count_true(predictions, result_key)}/{len(predictions)}"
        lines.append(
            f"| {markdown_cell(label)} | "
            f"{format_metric_value(metrics[metric_key])} | "
            f"{count_text} |"
        )

    lines.extend(
        [
            "",
            "## Outcome Counts",
            "",
            "| Outcome | Count |",
            "| --- | ---: |",
            f"| Parsed | {count_true(predictions, 'parsed')} |",
            f"| Parse failures | {count_false(predictions, 'parsed')} |",
            (
                "| Full match, ordered premises | "
                f"{count_true(predictions, 'joint_ordered_exact')} |"
            ),
            (
                "| Full match, ignoring premise order | "
                f"{count_true(predictions, 'joint_unordered_exact')} |"
            ),
            f"| Wrong full formalization | {count_false(predictions, 'joint_unordered_exact')} |",
            f"| Premises match, ordered | {count_true(predictions, 'premises_ordered_exact')} |",
            (
                "| Premises match, ignoring order | "
                f"{count_true(predictions, 'premises_unordered_exact')} |"
            ),
            f"| Conclusion match | {count_true(predictions, 'conclusion_exact')} |",
            "",
            (
                "Correct examples below use the full autoformalization match while "
                "ignoring premise order."
            ),
            "",
        ]
    )

    append_examples_section(
        lines,
        "Correct Examples",
        correct_examples,
        empty_message="No fully correct examples were found for this model.",
    )
    append_examples_section(
        lines,
        "Wrong Examples",
        wrong_examples,
        empty_message="No wrong examples were found for this model.",
    )
    return "\n".join(lines).rstrip() + "\n"


def report_metric_rows(
    metrics: dict[str, Any],
) -> list[tuple[str, str, str | None]]:
    rows = [
        ("Parse rate", "parse_rate", "parsed"),
        (
            "Autoformalization accuracy",
            "autoformalization_accuracy",
            "joint_ordered_exact",
        ),
        (
            "Autoformalization accuracy, ignore premise order",
            "autoformalization_accuracy_ignore_premise_order",
            "joint_unordered_exact",
        ),
        (
            "Strict autoformalization accuracy",
            "strict_autoformalization_accuracy",
            "strict_joint_exact",
        ),
        ("Premises accuracy", "premises_accuracy", "premises_ordered_exact"),
        (
            "Premises accuracy, ignore order",
            "premises_accuracy_ignore_order",
            "premises_unordered_exact",
        ),
        ("Conclusion accuracy", "conclusion_accuracy", "conclusion_exact"),
        ("Premise macro precision", "premise_macro_precision", None),
        ("Premise macro recall", "premise_macro_recall", None),
        ("Premise macro F1", "premise_macro_f1", None),
    ]
    if "prover_label_accuracy" in metrics:
        rows.extend(
            [
                ("Prover label accuracy", "prover_label_accuracy", None),
                (
                    "Prover label accuracy on parsed",
                    "prover_label_accuracy_on_parsed",
                    None,
                ),
                ("Prover attempted", "prover_attempted", None),
            ]
        )
    return [row for row in rows if row[1] in metrics]


def build_report_summary_markdown(
    report_entries: list[dict[str, Any]],
    report_dir: Path,
) -> str:
    lines = [
        "# Evaluation Report Summary",
        "",
        "Reports are grouped by trainer kind so GRPO and SFT runs stay separate.",
        "",
    ]
    grouped_entries: dict[str, list[dict[str, Any]]] = {}
    for entry in report_entries:
        metrics = entry["metrics"]
        grouped_entries.setdefault(str(metrics["trainer_kind"]), []).append(entry)

    for trainer_kind in sorted(grouped_entries):
        lines.extend(
            [
                f"## {trainer_kind.upper()}",
                "",
                "| Model | Report | Examples | Parse rate | Full accuracy | Full accuracy, ignore order | Conclusion accuracy | Premise macro F1 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for entry in grouped_entries[trainer_kind]:
            metrics = entry["metrics"]
            predictions = entry["predictions"]
            report_path = entry["path"]
            relative_report_path = report_path.relative_to(report_dir).as_posix()
            lines.append(
                f"| {markdown_cell(metrics['model_name'])} | "
                f"[report]({relative_report_path}) | "
                f"{len(predictions)} | "
                f"{format_metric_value(metrics['parse_rate'])} | "
                f"{format_metric_value(metrics['autoformalization_accuracy'])} | "
                f"{format_metric_value(metrics['autoformalization_accuracy_ignore_premise_order'])} | "
                f"{format_metric_value(metrics['conclusion_accuracy'])} | "
                f"{format_metric_value(metrics['premise_macro_f1'])} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def select_wrong_examples(
    predictions: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    wrong_predictions = [
        result for result in predictions if not result["joint_unordered_exact"]
    ]
    selected = []
    selected_ids = set()
    selectors = [
        lambda result: not result["parsed"],
        lambda result: result["parsed"] and not result["conclusion_exact"],
        lambda result: (
            result["parsed"]
            and result["conclusion_exact"]
            and not result["premises_unordered_exact"]
        ),
    ]

    for selector in selectors:
        for result in wrong_predictions:
            result_id = id(result)
            if result_id not in selected_ids and selector(result):
                selected.append(result)
                selected_ids.add(result_id)
                break
        if len(selected) >= limit:
            return selected

    for result in wrong_predictions:
        result_id = id(result)
        if result_id in selected_ids:
            continue
        selected.append(result)
        selected_ids.add(result_id)
        if len(selected) >= limit:
            break

    return selected


def append_examples_section(
    lines: list[str],
    title: str,
    examples: list[dict[str, Any]],
    *,
    empty_message: str,
) -> None:
    lines.extend([f"## {title}", ""])
    if not examples:
        lines.extend([empty_message, ""])
        return

    for index, result in enumerate(examples, start=1):
        append_example(lines, result, index)


def append_example(lines: list[str], result: dict[str, Any], index: int) -> None:
    example_id = result.get("example_id") or "unknown"
    story_id = result.get("story_id") or "unknown"
    status = "correct" if result["joint_unordered_exact"] else "wrong"
    if not result["parsed"]:
        status = "parse failure"

    lines.extend(
        [
            f"### {index}. Example {example_id} (story {story_id})",
            "",
            f"- Status: `{status}`",
            f"- Gold label: `{result.get('label', '')}`",
            f"- Parsed: `{yes_no(result['parsed'])}`",
            f"- Premises exact: `{yes_no(result['premises_ordered_exact'])}`",
            (
                f"- Premises exact, ignore order: "
                f"`{yes_no(result['premises_unordered_exact'])}`"
            ),
            f"- Conclusion exact: `{yes_no(result['conclusion_exact'])}`",
            f"- Premise F1: `{result['premise_f1']:.4f}`",
            "",
            "Natural-language premises:",
            fenced_block(result.get("nl_premises")),
            "",
            "Natural-language conclusion:",
            fenced_block(result.get("nl_conclusion")),
            "",
            "Gold premises FOL:",
            fenced_block(result.get("gold_premises")),
            "",
            "Gold conclusion FOL:",
            fenced_block(result.get("gold_conclusion")),
            "",
            "Predicted premises FOL:",
            fenced_block(result.get("predicted_premises")),
            "",
            "Predicted conclusion FOL:",
            fenced_block(result.get("predicted_conclusion")),
            "",
        ]
    )
    if not result["parsed"]:
        lines.extend(["Raw generation:", fenced_block(result.get("generation")), ""])


def markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", "<br>").replace("|", "\\|")


def format_metric_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return markdown_cell(value)


def fenced_block(value: Any) -> str:
    if value is None:
        text = "(none)"
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        text = str(value).strip() or "(empty)"

    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}text\n{text}\n{fence}"


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def count_true(results: list[dict[str, Any]], key: str) -> int:
    return sum(1 for result in results if result[key])


def count_false(results: list[dict[str, Any]], key: str) -> int:
    return sum(1 for result in results if not result[key])


def print_metrics(metrics: dict[str, Any]) -> None:
    print(f"\n{metrics['model_name']}")
    print(f"  examples: {metrics['n_examples']}")
    print(f"  parse_rate: {metrics['parse_rate']:.4f}")
    print(
        "  autoformalization_accuracy: "
        f"{metrics['autoformalization_accuracy']:.4f}"
    )
    print(
        "  autoformalization_accuracy_ignore_premise_order: "
        f"{metrics['autoformalization_accuracy_ignore_premise_order']:.4f}"
    )
    print(f"  premises_accuracy: {metrics['premises_accuracy']:.4f}")
    print(
        "  premises_accuracy_ignore_order: "
        f"{metrics['premises_accuracy_ignore_order']:.4f}"
    )
    print(f"  conclusion_accuracy: {metrics['conclusion_accuracy']:.4f}")
    print(f"  premise_macro_f1: {metrics['premise_macro_f1']:.4f}")
    if "prover_label_accuracy" in metrics:
        print(f"  prover_label_accuracy: {metrics['prover_label_accuracy']:.4f}")
        print(
            "  prover_label_accuracy_on_parsed: "
            f"{metrics['prover_label_accuracy_on_parsed']:.4f}"
        )


def main() -> None:
    args = parse_args()
    specs = parse_model_specs(args.model)
    validate_model_specs(specs)
    if args.predictions_output and len(specs) != 1:
        raise ValueError("--predictions-output can only be used with one --model.")
    if args.report_examples < 0:
        raise ValueError("--report-examples must be greater than or equal to 0.")

    set_seed(args.seed)
    records = load_jsonl_dataset(args.dataset, args.limit)
    generation_config = GenerationConfig(
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"Loaded {len(records)} examples from {args.dataset}", flush=True)
    output_dir = Path(args.output_dir) if args.output_dir else None
    report_dir = resolve_report_dir(args, output_dir)
    summaries = []
    model_outputs = []

    for spec in specs:
        metrics, predictions = evaluate_model(spec, records, args, generation_config)
        summaries.append(metrics)
        model_outputs.append((metrics, predictions))
        print_metrics(metrics)

        if output_dir:
            prefix = sanitize_filename(spec.name)
            write_json(output_dir / f"{prefix}_metrics.json", metrics)
            write_jsonl(output_dir / f"{prefix}_predictions.jsonl", predictions)
        if args.predictions_output:
            write_jsonl(Path(args.predictions_output), predictions)

    if output_dir:
        write_json(output_dir / "metrics_summary.json", summaries)
        write_summary_csv(output_dir / "metrics_summary.csv", summaries)
        print(f"\nWrote evaluation outputs to {output_dir}", flush=True)
    if report_dir:
        write_evaluation_reports(
            report_dir,
            model_outputs,
            examples_per_section=args.report_examples,
        )
        print(f"Wrote evaluation reports to {report_dir}", flush=True)


if __name__ == "__main__":
    main()
