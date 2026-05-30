from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any

from REWARDS.fol_schema import gold_fol_reward as score_gold_fol_reward
from REWARDS.fol_schema import postprocess_formalization
from REWARDS.fol_schema import schema_violations
from REWARDS.formatting import extract_formalization
from autoformalization import build_plan_prompt, build_prompt, build_repair_prompt
from autoformalization import normalize_label


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
BICONDITIONAL_ALIASES = ("⇔", "<->")
DATASET_TO_SOLVER_LABEL = {
    "true": "TRUE",
    "false": "FALSE",
    "uncertain": "UNKNOWN",
}
SOLVER_TO_DATASET_LABEL = {
    solver_label: dataset_label
    for dataset_label, solver_label in DATASET_TO_SOLVER_LABEL.items()
}
DP_LABELS = tuple(DATASET_TO_SOLVER_LABEL)
UNEXECUTABLE_PROVER_STATUSES = {
    "not_parsed",
    "parse_error",
    "exception",
    "invalid_label",
    "all_paths_pruned",
}


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
        "--include-gold-schema",
        action="store_true",
        help=(
            "Include the per-example predicate/constant schema derived from gold FOL. "
            "This reproduces the older schema-conditioned setup and is intentionally "
            "off by default for autonomous autoformalization."
        ),
    )
    parser.add_argument(
        "--draft-and-prune",
        action="store_true",
        help=(
            "Use a Draft-and-Prune style inference loop: sample multiple natural-"
            "language plans, generate one FOL formalization per plan, prune paths "
            "that are not valid solver inputs, and majority-vote solver labels."
        ),
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=20,
        help="Number of draft/formalization paths for --draft-and-prune.",
    )
    parser.add_argument(
        "--draft-max-new-tokens",
        type=int,
        default=384,
        help="Maximum new tokens for each drafted plan.",
    )
    parser.add_argument(
        "--draft-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for drafted plans.",
    )
    parser.add_argument(
        "--formalization-temperature",
        type=float,
        default=None,
        help=(
            "Temperature for formalization generation. Defaults to --temperature "
            "for one-step evaluation and 0.0 for --draft-and-prune."
        ),
    )
    parser.add_argument(
        "--repair-rounds",
        type=int,
        default=2,
        help=(
            "Maximum solver-feedback repair rounds per Draft-and-Prune path. "
            "Only used with --draft-and-prune."
        ),
    )
    parser.add_argument(
        "--postprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply evaluation-only cleanup before scoring: trim trailing junk, "
            "canonicalize schema casing, and flag schema violations. This uses "
            "gold FOL symbols, so it is off by default."
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
    for alias in BICONDITIONAL_ALIASES:
        line = line.replace(alias, "↔")
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


def generate_in_chunks(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    config: GenerationConfig,
) -> list[str]:
    outputs = []
    batch_size = max(1, config.batch_size)
    for start in range(0, len(prompts), batch_size):
        outputs.extend(
            generate_batch(
                model,
                tokenizer,
                prompts[start : start + batch_size],
                config,
            )
        )
    return outputs


def score_prediction(
    record: dict[str, Any],
    generation: str,
    *,
    run_prover: bool,
    apply_postprocessing: bool,
) -> dict[str, Any]:
    raw_predicted_premises, raw_predicted_conclusion = extract_formalization(generation)

    gold_premises = str(record["premises-FOL"]).strip()
    gold_conclusion = str(record["conclusion-FOL"]).strip()
    if apply_postprocessing:
        postprocessed = postprocess_formalization(
            raw_predicted_premises,
            raw_predicted_conclusion,
            gold_premises,
            gold_conclusion,
        )
        predicted_premises = postprocessed.premises
        predicted_conclusion = postprocessed.conclusion
        postprocessing_changed = postprocessed.changed
        invalid_predicates = list(postprocessed.invalid_predicates)
        invalid_constants = list(postprocessed.invalid_constants)
    else:
        predicted_premises = raw_predicted_premises
        predicted_conclusion = raw_predicted_conclusion
        postprocessing_changed = False
        raw_invalid_predicates, raw_invalid_constants = schema_violations(
            raw_predicted_premises,
            raw_predicted_conclusion,
            gold_premises,
            gold_conclusion,
        )
        invalid_predicates = list(raw_invalid_predicates)
        invalid_constants = list(raw_invalid_constants)

    parsed = predicted_premises is not None and predicted_conclusion is not None
    predicted_premise_lines = normalized_lines(predicted_premises)
    gold_premise_lines = normalized_lines(gold_premises)
    predicted_conclusion_norm = normalized_block(predicted_conclusion)
    gold_conclusion_norm = normalized_block(gold_conclusion)
    gold_overlap = score_gold_fol_reward(
        predicted_premises,
        predicted_conclusion,
        gold_premises,
        gold_conclusion,
    )

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
        "raw_predicted_premises": raw_predicted_premises,
        "raw_predicted_conclusion": raw_predicted_conclusion,
        "predicted_premises": predicted_premises,
        "predicted_conclusion": predicted_conclusion,
        "gold_premises": gold_premises,
        "gold_conclusion": gold_conclusion,
        "postprocessed": apply_postprocessing,
        "postprocessing_changed": postprocessing_changed,
        "invalid_predicates": invalid_predicates,
        "invalid_constants": invalid_constants,
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
        "schema_predicate_precision": gold_overlap.predicate_score.precision,
        "schema_predicate_recall": gold_overlap.predicate_score.recall,
        "schema_predicate_f1": gold_overlap.predicate_score.f1,
        "schema_constant_precision": gold_overlap.constant_score.precision,
        "schema_constant_recall": gold_overlap.constant_score.recall,
        "schema_constant_f1": gold_overlap.constant_score.f1,
        "gold_fol_reward": gold_overlap.reward,
        "schema_clean": parsed and not invalid_predicates and not invalid_constants,
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


def solver_prediction_to_dataset_label(prediction: Any) -> str | None:
    return SOLVER_TO_DATASET_LABEL.get(str(prediction or "").strip().upper())


def dataset_label_to_solver_prediction(label: str | None) -> str | None:
    if label is None:
        return None
    return DATASET_TO_SOLVER_LABEL.get(label)


def annotate_path_result(result: dict[str, Any], record: dict[str, Any]) -> None:
    path_label = solver_prediction_to_dataset_label(result.get("prover_prediction"))
    status = str(result.get("prover_status") or "")
    executable = path_label is not None and status not in UNEXECUTABLE_PROVER_STATUSES
    result["dp_path_label"] = path_label
    result["dp_path_executable"] = executable
    result["dp_path_label_correct"] = (
        executable and path_label == normalize_label(str(record["label"]))
    )


def path_is_executable(result: dict[str, Any]) -> bool:
    return bool(result.get("dp_path_executable"))


def render_broken_formalization(result: dict[str, Any]) -> str:
    raw_premises = result.get("raw_predicted_premises")
    raw_conclusion = result.get("raw_predicted_conclusion")
    if raw_premises is not None and raw_conclusion is not None:
        return f"Premises:\n{raw_premises}\n\nConclusion:\n{raw_conclusion}"
    return str(result.get("generation") or "")


def score_draft_path(
    record: dict[str, Any],
    generation: str,
    *,
    path_index: int,
    draft_plan: str,
    apply_postprocessing: bool,
    repair_round: int = 0,
    prompt: str | None = None,
    include_prompt: bool = False,
) -> dict[str, Any]:
    result = score_prediction(
        record,
        generation,
        run_prover=True,
        apply_postprocessing=apply_postprocessing,
    )
    result["dp_path_index"] = path_index
    result["draft_plan"] = draft_plan
    result["repair_round"] = repair_round
    annotate_path_result(result, record)
    if include_prompt and prompt is not None:
        result["prompt"] = prompt
    return result


def repair_draft_path(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    path_result: dict[str, Any],
    *,
    draft_plan: str,
    config: GenerationConfig,
    max_rounds: int,
    apply_postprocessing: bool,
    include_prompt: bool,
) -> dict[str, Any]:
    result = path_result
    repair_attempts = []
    for repair_round in range(1, max(0, max_rounds) + 1):
        if path_is_executable(result):
            break

        repair_prompt = build_repair_prompt(
            record,
            broken_formalization=render_broken_formalization(result),
            solver_feedback=result.get("prover_feedback"),
            draft_plan=draft_plan,
        )
        repaired_generation = generate_batch(
            model,
            tokenizer,
            [repair_prompt],
            config,
        )[0]
        repair_attempts.append(
            {
                "round": repair_round,
                "previous_status": result.get("prover_status"),
                "generation": repaired_generation,
            }
        )
        result = score_draft_path(
            record,
            repaired_generation,
            path_index=int(path_result["dp_path_index"]),
            draft_plan=draft_plan,
            apply_postprocessing=apply_postprocessing,
            repair_round=repair_round,
            prompt=repair_prompt,
            include_prompt=include_prompt,
        )

    result["repair_attempts"] = repair_attempts
    return result


def select_majority_label(
    path_results: list[dict[str, Any]],
) -> tuple[str | None, bool, Counter[str]]:
    vote_counts = Counter(
        result["dp_path_label"]
        for result in path_results
        if path_is_executable(result) and result.get("dp_path_label") is not None
    )
    if not vote_counts:
        return None, False, vote_counts

    max_count = max(vote_counts.values())
    tied_labels = {
        label for label, count in vote_counts.items() if count == max_count
    }
    tie = len(tied_labels) > 1

    for result in path_results:
        label = result.get("dp_path_label")
        if path_is_executable(result) and label in tied_labels:
            return str(label), tie, vote_counts
    return None, tie, vote_counts


def normalized_vote_entropy(vote_counts: Counter[str]) -> float:
    total = sum(vote_counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in vote_counts.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(len(DP_LABELS))


def compact_path_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "dp_path_index",
        "draft_plan",
        "generation",
        "raw_predicted_premises",
        "raw_predicted_conclusion",
        "predicted_premises",
        "predicted_conclusion",
        "parsed",
        "dp_path_executable",
        "dp_path_label",
        "dp_path_label_correct",
        "prover_status",
        "prover_prediction",
        "prover_feedback",
        "repair_round",
        "repair_attempts",
        "postprocessing_changed",
        "invalid_predicates",
        "invalid_constants",
        "premise_f1",
        "gold_fol_reward",
    )
    return {key: result.get(key) for key in keys if key in result}


def aggregate_draft_and_prune_result(
    record: dict[str, Any],
    path_results: list[dict[str, Any]],
) -> dict[str, Any]:
    predicted_label, tie, vote_counts = select_majority_label(path_results)
    surviving_paths = [result for result in path_results if path_is_executable(result)]
    gold_label = normalize_label(str(record["label"]))
    selected_path = next(
        (
            result
            for result in path_results
            if path_is_executable(result)
            and result.get("dp_path_label") == predicted_label
        ),
        path_results[0],
    )

    result = dict(selected_path)
    correct_paths = sum(
        1 for path_result in path_results if path_result["dp_path_label_correct"]
    )
    repaired_paths = sum(
        1 for path_result in path_results if int(path_result.get("repair_round", 0)) > 0
    )
    pruned_status_counts = Counter(
        str(path_result.get("prover_status") or "unknown")
        for path_result in path_results
        if not path_is_executable(path_result)
    )
    path_status_counts = Counter(
        str(path_result.get("prover_status") or "unknown")
        for path_result in path_results
    )
    dp_label_correct = predicted_label == gold_label

    result.update(
        {
            "draft_and_prune": True,
            "dp_total_paths": len(path_results),
            "dp_surviving_paths": len(surviving_paths),
            "dp_pruned_paths": len(path_results) - len(surviving_paths),
            "dp_correct_paths": correct_paths,
            "dp_repaired_paths": repaired_paths,
            "dp_executable": bool(surviving_paths),
            "dp_prediction": predicted_label,
            "dp_solver_prediction": dataset_label_to_solver_prediction(predicted_label),
            "dp_label_correct": dp_label_correct,
            "dp_hit": correct_paths > 0,
            "dp_tie": tie,
            "dp_abstained": predicted_label is None,
            "dp_vote_counts": dict(sorted(vote_counts.items())),
            "dp_vote_entropy": normalized_vote_entropy(vote_counts),
            "dp_selected_path_index": result.get("dp_path_index"),
            "dp_path_status_counts": dict(sorted(path_status_counts.items())),
            "dp_pruned_status_counts": dict(sorted(pruned_status_counts.items())),
            "dp_paths": [compact_path_result(path_result) for path_result in path_results],
            "prover_status": (
                "correct"
                if dp_label_correct
                else "incorrect"
                if predicted_label is not None
                else "all_paths_pruned"
            ),
            "prover_prediction": dataset_label_to_solver_prediction(predicted_label),
            "prover_label_correct": dp_label_correct,
            "prover_feedback": None,
        }
    )
    return result


def evaluate_record_draft_and_prune(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    args: argparse.Namespace,
    draft_config: GenerationConfig,
    formalization_config: GenerationConfig,
) -> dict[str, Any]:
    plan_prompt = build_plan_prompt(record)
    plan_prompts = [plan_prompt] * args.paths
    draft_plans = generate_in_chunks(model, tokenizer, plan_prompts, draft_config)
    formalization_prompts = [
        build_prompt(
            record,
            include_gold_schema=args.include_gold_schema,
            draft_plan=draft_plan,
        )
        for draft_plan in draft_plans
    ]
    generations = generate_in_chunks(
        model,
        tokenizer,
        formalization_prompts,
        formalization_config,
    )

    path_results = []
    for path_index, (draft_plan, formalization_prompt, generation) in enumerate(
        zip(draft_plans, formalization_prompts, generations, strict=True),
        start=1,
    ):
        path_result = score_draft_path(
            record,
            generation,
            path_index=path_index,
            draft_plan=draft_plan,
            apply_postprocessing=args.postprocess,
            prompt=formalization_prompt,
            include_prompt=args.include_prompt,
        )
        if not path_is_executable(path_result) and args.repair_rounds > 0:
            path_result = repair_draft_path(
                model,
                tokenizer,
                record,
                path_result,
                draft_plan=draft_plan,
                config=formalization_config,
                max_rounds=args.repair_rounds,
                apply_postprocessing=args.postprocess,
                include_prompt=args.include_prompt,
            )
        path_results.append(path_result)

    result = aggregate_draft_and_prune_result(record, path_results)
    if args.include_prompt:
        result["plan_prompt"] = plan_prompt
    return result


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
        "include_gold_schema": any(
            result.get("include_gold_schema", False) for result in results
        ),
        "postprocess": any(result.get("postprocessed", False) for result in results),
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
        "schema_predicate_macro_f1": mean(
            result["schema_predicate_f1"] for result in results
        ),
        "schema_constant_macro_f1": mean(
            result["schema_constant_f1"] for result in results
        ),
        "gold_fol_reward_mean": mean(result["gold_fol_reward"] for result in results),
        "schema_clean_rate": rate("schema_clean"),
        "postprocessing_changed_rate": mean(
            float(result.get("postprocessing_changed", False)) for result in results
        ),
    }

    if run_prover:
        prover_status_counts = Counter(result["prover_status"] for result in results)
        attempted = total - sum(
            prover_status_counts.get(status, 0)
            for status in UNEXECUTABLE_PROVER_STATUSES
        )
        correct = sum(1 for result in results if result["prover_label_correct"])
        metrics["prover_label_accuracy"] = correct / total
        metrics["prover_label_accuracy_on_parsed"] = (
            correct / attempted if attempted else 0.0
        )
        metrics["prover_attempted"] = attempted
        metrics["prover_status_counts"] = dict(sorted(prover_status_counts.items()))

    if any(result.get("draft_and_prune", False) for result in results):
        total_paths = sum(int(result.get("dp_total_paths", 0)) for result in results)
        surviving_paths = sum(
            int(result.get("dp_surviving_paths", 0)) for result in results
        )
        correct_paths = sum(
            int(result.get("dp_correct_paths", 0)) for result in results
        )
        executable_results = [
            result for result in results if result.get("dp_executable", False)
        ]
        dp_correct = sum(1 for result in results if result.get("dp_label_correct"))
        dp_path_status_counts = Counter()
        dp_pruned_status_counts = Counter()
        for result in results:
            dp_path_status_counts.update(result.get("dp_path_status_counts", {}))
            dp_pruned_status_counts.update(result.get("dp_pruned_status_counts", {}))

        metrics.update(
            {
                "draft_and_prune": True,
                "dp_paths": results[0].get("dp_total_paths", 0),
                "dp_total_paths": total_paths,
                "dp_surviving_paths": surviving_paths,
                "dp_path_execution_rate": (
                    surviving_paths / total_paths if total_paths else 0.0
                ),
                "dp_execution_rate": rate("dp_executable"),
                "dp_label_accuracy": dp_correct / total,
                "dp_label_accuracy_on_executable": (
                    dp_correct / len(executable_results)
                    if executable_results
                    else 0.0
                ),
                "dp_path_accuracy": (
                    correct_paths / total_paths if total_paths else 0.0
                ),
                "dp_hit_rate": rate("dp_hit"),
                "dp_abstention_rate": rate("dp_abstained"),
                "dp_tie_rate": rate("dp_tie"),
                "dp_avg_surviving_paths": surviving_paths / total,
                "dp_avg_vote_entropy": mean(
                    result.get("dp_vote_entropy", 0.0) for result in results
                ),
                "dp_path_status_counts": dict(sorted(dp_path_status_counts.items())),
                "dp_pruned_status_counts": dict(sorted(dp_pruned_status_counts.items())),
            }
        )

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
    if args.draft_and_prune:
        draft_config = GenerationConfig(
            batch_size=generation_config.batch_size,
            max_new_tokens=args.draft_max_new_tokens,
            max_input_tokens=generation_config.max_input_tokens,
            temperature=args.draft_temperature,
            top_p=generation_config.top_p,
        )
        for index, record in enumerate(records, start=1):
            result = evaluate_record_draft_and_prune(
                model,
                tokenizer,
                record,
                args,
                draft_config,
                generation_config,
            )
            result["model_name"] = spec.name
            result["model_path"] = spec.path
            result["include_gold_schema"] = args.include_gold_schema
            results.append(result)

            if args.log_every > 0 and (
                index == len(records) or index % args.log_every == 0
            ):
                print(
                    f"  {spec.name}: evaluated {index}/{len(records)}",
                    flush=True,
                )
    else:
        for start in range(0, len(records), generation_config.batch_size):
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
            for record, prompt, generation in zip(batch, prompts, generations):
                result = score_prediction(
                    record,
                    generation,
                    run_prover=args.run_prover,
                    apply_postprocessing=args.postprocess,
                )
                result["model_name"] = spec.name
                result["model_path"] = spec.path
                result["include_gold_schema"] = args.include_gold_schema
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
        run_prover=args.run_prover or args.draft_and_prune,
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
    if metrics.get("draft_and_prune", False):
        correct_examples = [
            result for result in predictions if result.get("dp_label_correct", False)
        ][:examples_per_section]
        wrong_examples = select_wrong_dp_examples(predictions, examples_per_section)
    else:
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
        f"| Gold schema in prompt | {yes_no(metrics.get('include_gold_schema', False))} |",
        f"| Evaluation postprocessing | {yes_no(metrics.get('postprocess', False))} |",
        f"| Draft-and-Prune | {yes_no(metrics.get('draft_and_prune', False))} |",
        f"| D&P paths | {markdown_cell(metrics.get('dp_paths'))} |",
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
            f"| Schema clean | {count_true(predictions, 'schema_clean')} |",
            (
                "| Postprocessing changed output | "
                f"{count_true(predictions, 'postprocessing_changed')} |"
            ),
        ]
    )
    if metrics.get("draft_and_prune", False):
        lines.extend(
            [
                (
                    "| D&P executable examples | "
                    f"{count_true(predictions, 'dp_executable')} |"
                ),
                (
                    "| D&P correct labels | "
                    f"{count_true(predictions, 'dp_label_correct')} |"
                ),
            ]
        )
    lines.extend(
        [
            "",
            (
                "Correct examples below use the D&P solver-label vote when "
                "Draft-and-Prune is enabled; otherwise they use the full "
                "autoformalization match while ignoring premise order."
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
        ("Schema predicate macro F1", "schema_predicate_macro_f1", None),
        ("Schema constant macro F1", "schema_constant_macro_f1", None),
        ("Gold-FOL reward mean", "gold_fol_reward_mean", None),
        ("Schema clean rate", "schema_clean_rate", "schema_clean"),
        (
            "Postprocessing changed rate",
            "postprocessing_changed_rate",
            "postprocessing_changed",
        ),
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
    if metrics.get("draft_and_prune", False):
        rows.extend(
            [
                ("D&P label accuracy", "dp_label_accuracy", "dp_label_correct"),
                (
                    "D&P label accuracy on executable",
                    "dp_label_accuracy_on_executable",
                    None,
                ),
                ("D&P execution rate", "dp_execution_rate", "dp_executable"),
                (
                    "D&P path execution rate",
                    "dp_path_execution_rate",
                    None,
                ),
                ("D&P path accuracy", "dp_path_accuracy", None),
                ("D&P hit rate", "dp_hit_rate", "dp_hit"),
                ("D&P abstention rate", "dp_abstention_rate", "dp_abstained"),
                ("D&P tie rate", "dp_tie_rate", "dp_tie"),
                ("D&P avg surviving paths", "dp_avg_surviving_paths", None),
                ("D&P avg vote entropy", "dp_avg_vote_entropy", None),
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
                (
                    "| Model | Report | Examples | Parse rate | Full accuracy | "
                    "Full accuracy, ignore order | Conclusion accuracy | "
                    "Premise macro F1 | Schema pred F1 | Schema const F1 | "
                    "D&P acc | D&P exec |"
                ),
                (
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
                    "---: | ---: | ---: | ---: |"
                ),
            ]
        )
        for entry in grouped_entries[trainer_kind]:
            metrics = entry["metrics"]
            predictions = entry["predictions"]
            report_path = entry["path"]
            relative_report_path = report_path.relative_to(report_dir).as_posix()
            full_ignore_order = format_metric_value(
                metrics["autoformalization_accuracy_ignore_premise_order"]
            )
            lines.append(
                f"| {markdown_cell(metrics['model_name'])} | "
                f"[report]({relative_report_path}) | "
                f"{len(predictions)} | "
                f"{format_metric_value(metrics['parse_rate'])} | "
                f"{format_metric_value(metrics['autoformalization_accuracy'])} | "
                f"{full_ignore_order} | "
                f"{format_metric_value(metrics['conclusion_accuracy'])} | "
                f"{format_metric_value(metrics['premise_macro_f1'])} | "
                f"{format_metric_value(metrics.get('schema_predicate_macro_f1'))} | "
                f"{format_metric_value(metrics.get('schema_constant_macro_f1'))} | "
                f"{format_metric_value(metrics.get('dp_label_accuracy'))} | "
                f"{format_metric_value(metrics.get('dp_execution_rate'))} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def select_wrong_dp_examples(
    predictions: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    wrong_predictions = [
        result for result in predictions if not result.get("dp_label_correct", False)
    ]
    selected = []
    selected_ids = set()
    selectors = [
        lambda result: result.get("dp_abstained", False),
        lambda result: result.get("dp_hit", False)
        and not result.get("dp_label_correct", False),
        lambda result: not result.get("dp_hit", False),
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
    if result.get("draft_and_prune", False):
        status = "correct" if result.get("dp_label_correct", False) else "wrong"
        if result.get("dp_abstained", False):
            status = "all paths pruned"
    else:
        status = "correct" if result["joint_unordered_exact"] else "wrong"
    if not result["parsed"] and not result.get("draft_and_prune", False):
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
            f"- Schema predicate F1: `{result['schema_predicate_f1']:.4f}`",
            f"- Schema constant F1: `{result['schema_constant_f1']:.4f}`",
            f"- Schema clean: `{yes_no(result['schema_clean'])}`",
            f"- Postprocessing changed output: `{yes_no(result['postprocessing_changed'])}`",
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
    if result.get("draft_and_prune", False):
        lines.extend(
            [
                f"- D&P prediction: `{result.get('dp_prediction')}`",
                f"- D&P label correct: `{yes_no(result.get('dp_label_correct'))}`",
                f"- D&P surviving paths: `{result.get('dp_surviving_paths')}/{result.get('dp_total_paths')}`",
                f"- D&P correct paths: `{result.get('dp_correct_paths')}/{result.get('dp_total_paths')}`",
                f"- D&P vote counts: `{markdown_cell(result.get('dp_vote_counts'))}`",
                f"- D&P selected path: `{result.get('dp_selected_path_index')}`",
                "",
            ]
        )
    invalid_predicates = result.get("invalid_predicates") or []
    invalid_constants = result.get("invalid_constants") or []
    if invalid_predicates or invalid_constants:
        lines.extend(
            [
                "Schema violations after postprocessing:",
                fenced_block(
                    {
                        "invalid_predicates": invalid_predicates,
                        "invalid_constants": invalid_constants,
                    }
                ),
                "",
            ]
        )
    if result.get("postprocessing_changed"):
        lines.extend(
            [
                "Raw parsed premises FOL:",
                fenced_block(result.get("raw_predicted_premises")),
                "",
                "Raw parsed conclusion FOL:",
                fenced_block(result.get("raw_predicted_conclusion")),
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
    print(f"  schema_predicate_macro_f1: {metrics['schema_predicate_macro_f1']:.4f}")
    print(f"  schema_constant_macro_f1: {metrics['schema_constant_macro_f1']:.4f}")
    print(f"  gold_fol_reward_mean: {metrics['gold_fol_reward_mean']:.4f}")
    print(f"  schema_clean_rate: {metrics['schema_clean_rate']:.4f}")
    print(
        "  postprocessing_changed_rate: "
        f"{metrics['postprocessing_changed_rate']:.4f}"
    )
    if "prover_label_accuracy" in metrics:
        print(f"  prover_label_accuracy: {metrics['prover_label_accuracy']:.4f}")
        print(
            "  prover_label_accuracy_on_parsed: "
            f"{metrics['prover_label_accuracy_on_parsed']:.4f}"
        )
    if metrics.get("draft_and_prune", False):
        print(f"  dp_label_accuracy: {metrics['dp_label_accuracy']:.4f}")
        print(f"  dp_execution_rate: {metrics['dp_execution_rate']:.4f}")
        print(f"  dp_path_execution_rate: {metrics['dp_path_execution_rate']:.4f}")
        print(f"  dp_path_accuracy: {metrics['dp_path_accuracy']:.4f}")
        print(f"  dp_hit_rate: {metrics['dp_hit_rate']:.4f}")
        print(f"  dp_abstention_rate: {metrics['dp_abstention_rate']:.4f}")


def main() -> None:
    args = parse_args()
    specs = parse_model_specs(args.model)
    validate_model_specs(specs)
    if args.predictions_output and len(specs) != 1:
        raise ValueError("--predictions-output can only be used with one --model.")
    if args.report_examples < 0:
        raise ValueError("--report-examples must be greater than or equal to 0.")
    if args.paths <= 0:
        raise ValueError("--paths must be greater than 0.")
    if args.repair_rounds < 0:
        raise ValueError("--repair-rounds must be greater than or equal to 0.")

    set_seed(args.seed)
    records = load_jsonl_dataset(args.dataset, args.limit)
    formalization_temperature = args.formalization_temperature
    if formalization_temperature is None:
        formalization_temperature = 0.0 if args.draft_and_prune else args.temperature
    generation_config = GenerationConfig(
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        temperature=formalization_temperature,
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
