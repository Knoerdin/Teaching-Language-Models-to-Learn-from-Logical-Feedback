from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import hydra
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer


VALID_LABELS = {"true", "false", "uncertain"}


def _normalize_label(value: str) -> str:
    return value.strip().lower()


def _extract_predicted_label(text: str) -> str | None:
    match = re.search(r"\b(true|false|uncertain)\b", text.lower())
    if not match:
        return None
    return match.group(1)


def _build_prompt(example: dict[str, Any]) -> str:
    return (
        "You are solving a logical entailment task.\n"
        "Given premises and a conclusion, classify whether the conclusion follows from the premises.\n"
        "Reply with exactly one label: True, False, or Uncertain.\n\n"
        f"Premises:\n{example['premises']}\n\n"
        f"Conclusion:\n{example['conclusion']}\n\n"
        "Answer:"
    )


def _prepare_dataset(path: str) -> Dataset:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    raw = load_dataset("json", data_files=str(source_path), split="train")

    def _map_row(row: dict[str, Any]) -> dict[str, str]:
        label = _normalize_label(str(row["label"]))
        if label not in VALID_LABELS:
            raise ValueError(
                f"Unsupported label '{row['label']}'. Expected one of: {sorted(VALID_LABELS)}"
            )
        return {
            "prompt": _build_prompt(row),
            "solution": label,
            "example_id": str(row.get("example_id", "")),
        }

    return raw.map(_map_row, remove_columns=raw.column_names)


def logical_feedback_reward(
    prompts: list[str],
    completions: list[Any],
    solution: list[str],
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for _, completion_item, gold in zip(prompts, completions, solution, strict=True):
        if isinstance(completion_item, str):
            completion_text = completion_item
        elif isinstance(completion_item, list) and completion_item:
            first_item = completion_item[0]
            if isinstance(first_item, dict):
                completion_text = str(first_item.get("content", ""))
            else:
                completion_text = str(first_item)
        else:
            completion_text = ""

        predicted = _extract_predicted_label(completion_text)
        if predicted is None:
            rewards.append(0.0)
            continue

        rewards.append(1.0 if predicted == _normalize_label(gold) else 0.0)
    return rewards


def _resolve_runtime_device(cfg: DictConfig) -> str:
    requested = str(cfg.trainer.get("device", "auto")).lower()

    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError(
            "trainer.device is set to 'mps', but MPS is not available in this environment."
        )

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError(
            "trainer.device is set to 'cuda', but CUDA is not available in this environment."
        )

    if requested == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_grpo(cfg: DictConfig) -> None:
    set_seed(int(cfg.trainer.seed))
    runtime_device = _resolve_runtime_device(cfg)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print(f"Using runtime device: {runtime_device}")

    train_dataset = _prepare_dataset(str(cfg.dataset.train_path))
    eval_dataset = _prepare_dataset(str(cfg.dataset.validation_path))

    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_args = GRPOConfig(
        output_dir=str(cfg.trainer.output_dir),
        learning_rate=float(cfg.model.training.learning_rate),
        per_device_train_batch_size=int(cfg.model.training.batch_size),
        generation_batch_size=int(cfg.model.training.batch_size),
        num_generations=int(cfg.model.training.num_generations),
        logging_steps=int(cfg.model.training.logging_steps),
        max_steps=int(cfg.model.training.max_steps),
        report_to="none",
        run_name=str(cfg.experiment.name),
        use_cpu=runtime_device == "cpu",
        dataloader_pin_memory=runtime_device == "cuda",
    )

    trainer = GRPOTrainer(
        model=str(cfg.model.model_name_or_path),
        reward_funcs=logical_feedback_reward,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(cfg.trainer.output_dir))


MODEL_TRAINERS = {
    "grpo": train_grpo,
}


@hydra.main(version_base=None, config_path="../CONFIGS", config_name="config")
def main(cfg: DictConfig) -> None:
    model_name = str(cfg.model.name).lower()

    if model_name not in MODEL_TRAINERS:
        available = ", ".join(sorted(MODEL_TRAINERS.keys()))
        raise ValueError(
            f"Unknown model '{cfg.model.name}'. Available model trainers: {available}."
        )

    MODEL_TRAINERS[model_name](cfg)


if __name__ == "__main__":
    main()
