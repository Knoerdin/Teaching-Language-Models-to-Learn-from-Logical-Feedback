from __future__ import annotations

from REWARDS.formatting import completion_to_text, format_reward
from REWARDS.formatting import (
    completion_to_text,
    extract_formalization,
    format_reward,
)
from REWARDS.proving import correctness_reward

import os
import re
from pathlib import Path
from typing import Any

import hydra
import torch
from datasets import Dataset, load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, set_seed


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
        "You are an autoformalization model.\n"
        "Translate the natural-language reasoning problem into first-order logic.\n"
        "Return exactly this format:\n\n"
        "Premises:\n"
        "<one formal premise per line>\n\n"
        "Conclusion:\n"
        "<one formal conclusion>\n\n"
        "Do not answer True, False, or Uncertain.\n\n"
        f"Natural-language premises:\n{example['premises']}\n\n"
        f"Natural-language conclusion:\n{example['conclusion']}\n\n"
        "Formalization:\n"
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
                f"Unsupported label '{row['label']}'. "
                f"Expected one of: {sorted(VALID_LABELS)}"
            )

        return {
            "prompt": _build_prompt(row),
            "solution": label,
            "premises": str(row["premises"]),
            "conclusion": str(row["conclusion"]),
            "example_id": str(row.get("example_id", "")),
        }

    return raw.map(_map_row, remove_columns=raw.column_names)


def logical_feedback_reward(
    prompts: list[str],
    completions: list[Any],
    solution: list[str],
    premises: list[str],
    conclusion: list[str],
    **_: Any,
) -> list[float]:
    rewards = []

    for completion_item, gold, nl_premises, nl_conclusion in zip(
        completions, solution, premises, conclusion, strict=True
    ):
        text = completion_to_text(completion_item)

        reward = format_reward(text)
        formal_premises, formal_conclusion = extract_formalization(text)

        if formal_premises is None or formal_conclusion is None:
            rewards.append(reward)
            continue

        reward += correctness_reward(
            nl_premises=nl_premises,
            nl_conclusion=nl_conclusion,
            formal_premises=formal_premises,
            formal_conclusion=formal_conclusion,
            gold_label=gold,
        )

        rewards.append(reward)

    return rewards


def _resolve_runtime_device(cfg: DictConfig) -> str:
    requested = str(cfg.trainer.get("device", "auto")).lower()

    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError("MPS requested but not available.")

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA requested but not available.")

    if requested == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@hydra.main(version_base=None, config_path="../CONFIGS", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.trainer.seed))
    runtime_device = _resolve_runtime_device(cfg)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    train_dataset = _prepare_dataset(str(cfg.dataset.train_path))
    eval_dataset = _prepare_dataset(str(cfg.dataset.validation_path))

    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer_args = instantiate(
        cfg.trainer.args,
        output_dir=str(cfg.trainer.output_dir),
        run_name=str(cfg.experiment.name),
        use_cpu=(runtime_device == "cpu"),
        dataloader_pin_memory=(runtime_device == "cuda"),
    )

    trainer = instantiate(
        cfg.trainer.trainer_cls,
        model=str(cfg.model.model_name_or_path),
        reward_funcs=logical_feedback_reward,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print(f"Starting training {OmegaConf.to_yaml(cfg)} on device: {runtime_device}")

    trainer.train()
    trainer.save_model(str(cfg.trainer.output_dir))


if __name__ == "__main__":
    main()