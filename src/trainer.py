from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hydra
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

from REWARDS.logical_feedback import logical_feedback_reward
from REWARDS.mlflow_logging import configure_reward_logging, flush_reward_logging


VALID_LABELS = {"true", "false", "uncertain"}


def _normalize_label(value: str) -> str:
    return value.strip().lower()


def _build_prompt(example):
    return (
        "Translate the following Natural-language reasoning problem into first-order logic.\n"
        "Use the same notation as these operators: ∀, ∃, ¬, →, ∧, ∨, ⊕.\n"
        "Output exactly in this format:\n\n"
        "Premises:\n"
        "<one FOL premise per line>\n\n"
        "Conclusion:\n"
        "<one FOL conclusion>\n\n"
        "Do not explain.\n"
        "Do not restate the problem.\n"
        "Do not answer True, False, or Uncertain.\n\n"
        f"Natural-language premises:\n{example['premises']}\n\n"
        f"Natural-language conclusion:\n{example['conclusion']}\n"

        "This is an EXAMPLE of the expected input output format:\n\n"
        "INPUT:\n"
        "Premises NL\":\"All eels are fish. \n"
        "No fish are plants. \n"
        "Everything displayed in the collection is either a plant or an animal.\n"
        "All multicellular animals are not bacteria.\n"
        "All animals displayed in the collection are multicellular.\n"
        "A sea eel is displayed in the collection.\n"
        "The sea eel is an eel or an animal or not a plant.\n"
        "Conclusion NL:\n"
        "The sea eel is an eel.\n\n"
        "OUTPUT:\n"
        "Premises FOL:\n"
        "∀x (Eel(x) → Fish(x))\n"
        "∀x (Fish(x) → ¬Plant(x))\n"
        "∀x (DisplayedIn(x, collection) → Plant(x) ⊕ Animal(x))\n"
        "∀x (Multicellular(x) → ¬Bacteria(x))\n"
        "∀x (DisplayedIn(x, collection) ∧ Animal(x) → Multicellular(x))\n"
        "DisplayedIn(seaEel, collection)\n"
        "Eel(seaEel) ∨ Animal(seaEel) ∨ ¬Plant(seaEel)\n"
        "Conclusion FOL:\n"
        "Eel(seaEel)"
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
            "solution": _normalize_label(str(row["label"])),
            "premises": row["premises"],
            "conclusion": row["conclusion"],
            "premises_fol_gold": row["premises-FOL"],
            "conclusion_fol_gold": row["conclusion-FOL"],
            "example_id": str(row.get("example_id", "")),
        }

    return raw.map(_map_row, remove_columns=raw.column_names)


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


def _configure_reward_logging(cfg: DictConfig):
    mlflow_cfg = cfg.trainer.get("mlflow", {})
    enabled = bool(mlflow_cfg.get("enabled", True))
    tracking_uri = mlflow_cfg.get("tracking_uri", None)
    artifact_subdir = str(mlflow_cfg.get("artifact_subdir", "reward_plots"))
    plot_every_n_steps = int(mlflow_cfg.get("plot_every_n_steps", 10))
    experiment_name = mlflow_cfg.get("experiment_name", None) or cfg.experiment.name
    run_name = mlflow_cfg.get("run_name", None) or cfg.experiment.name

    logger = configure_reward_logging(
        enabled=enabled,
        tracking_uri=tracking_uri,
        experiment_name=str(experiment_name),
        run_name=str(run_name),
        artifact_subdir=artifact_subdir,
        plot_every_n_steps=plot_every_n_steps,
    )
    logger.log_params(
        {
            "model_name_or_path": cfg.model.model_name_or_path,
            "trainer_output_dir": cfg.trainer.output_dir,
            "trainer_device": cfg.trainer.get("device", "auto"),
            "dataset_train_path": cfg.dataset.train_path,
            "dataset_validation_path": cfg.dataset.validation_path,
        }
    )

    return logger


@hydra.main(version_base=None, config_path="../CONFIGS", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.trainer.seed))
    runtime_device = _resolve_runtime_device(cfg)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    _configure_reward_logging(cfg)

    train_dataset = _prepare_dataset(str(cfg.dataset.train_path))
    eval_dataset = _prepare_dataset(str(cfg.dataset.validation_path))

    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer_args_dict = OmegaConf.to_container(cfg.trainer.args, resolve=True)

    trainer_args_dict.pop("_target_", None)

    trainer_args_dict["output_dir"] = str(cfg.trainer.output_dir)
    trainer_args_dict["run_name"] = str(cfg.experiment.name)
    trainer_args_dict["use_cpu"] = (runtime_device == "cpu")
    trainer_args_dict["dataloader_pin_memory"] = (runtime_device == "cuda")

    trainer_args = GRPOConfig(**trainer_args_dict)

    trainer = GRPOTrainer(
        model=str(cfg.model.model_name_or_path),
        reward_funcs=logical_feedback_reward,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print(f"Starting training {OmegaConf.to_yaml(cfg)} on device: {runtime_device}")

    try:
        trainer.train()
        trainer.save_model(str(cfg.trainer.output_dir))
    finally:
        flush_reward_logging()


if __name__ == "__main__":
    main()
