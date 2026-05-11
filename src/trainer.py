from __future__ import annotations

import os
import random
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _bootstrap_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


_bootstrap_log("Starting trainer process")
_bootstrap_log("Importing hydra")
import hydra
_bootstrap_log("Importing torch")
import torch
_bootstrap_log("Importing datasets")
from datasets import Dataset, disable_progress_bar, enable_progress_bar, load_dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
_bootstrap_log("Importing transformers")
from transformers import AutoTokenizer, logging as transformers_logging
_bootstrap_log("Importing TRL")
from trl import GRPOConfig, GRPOTrainer

_bootstrap_log("Importing reward modules")
from REWARDS.logical_feedback import configure_reward_console
from REWARDS.logical_feedback import logical_feedback_reward
from REWARDS.mlflow_logging import configure_reward_logging, flush_reward_logging
_bootstrap_log("Finished trainer imports")


VALID_LABELS = {"true", "false", "uncertain"}


def _log_stage(message: str) -> None:
    _bootstrap_log(message)


def _normalize_label(value: str) -> str:
    return value.strip().lower()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_prompt(example):
    return (
        "Translate natural-language premises and conclusion into first-order logic.\n"
        "Return only the formalization. No explanation. No truth label.\n\n"
        "Completion format:\n"
        "Premises:\n"
        "<one complete FOL premise per line>\n\n"
        "Conclusion:\n"
        "<one complete FOL conclusion>\n\n"
        "Formula rules:\n"
        "Use only these logical operators: ∀, ∃, ¬, →, ∧, ∨, ⊕.\n"
        "Operator meanings: ∀ means for all; ∃ means there exists; ¬ means not; "
        "→ means implies; ∧ means and; ∨ means inclusive or; ⊕ means exclusive or.\n"
        "Do not use ↔, ⇔, ⇒, ∴, bullets, markdown, quotes, or extra labels.\n"
        "Do not write chat-role words such as user, assistant, or system.\n"
        "Predicate, variable, and constant names must use English letters, digits, or underscores.\n"
        "Every line must be a complete formula with balanced parentheses.\n"
        "Use commas only inside predicate arguments, never between formulas; use ∧ for and.\n"
        "Stop immediately after the conclusion formula.\n\n"
        "Example natural-language premises:\n"
        "All eels are fish.\n"
        "No fish are plants.\n"
        "Everything displayed in the collection is either a plant or an animal.\n"
        "All multicellular animals are not bacteria.\n"
        "All animals displayed in the collection are multicellular.\n"
        "A sea eel is displayed in the collection.\n"
        "The sea eel is an eel or an animal or not a plant.\n\n"
        "Example natural-language conclusion:\n"
        "The sea eel is an eel.\n\n"
        "Correct formalization for the example:\n"
        "Premises:\n"
        "∀x (Eel(x) → Fish(x))\n"
        "∀x (Fish(x) → ¬Plant(x))\n"
        "∀x (DisplayedIn(x, collection) → Plant(x) ⊕ Animal(x))\n"
        "∀x (Multicellular(x) ∧ Animal(x) → ¬Bacteria(x))\n"
        "∀x (DisplayedIn(x, collection) ∧ Animal(x) → Multicellular(x))\n"
        "DisplayedIn(seaEel, collection)\n"
        "Eel(seaEel) ∨ Animal(seaEel) ∨ ¬Plant(seaEel)\n\n"
        "Conclusion:\n"
        "Eel(seaEel)\n\n"
        "Problem natural-language premises:\n"
        f"{example['premises']}\n\n"
        "Problem natural-language conclusion:\n"
        f"{example['conclusion']}\n\n"
        "Formalization:\n"
        "Premises:\n"
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


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def _repo_tracking_uri() -> str:
    return (Path(get_original_cwd()) / "mlruns").resolve().as_uri()


def _resolve_mlflow_tracking_uri(mlflow_cfg: DictConfig) -> str:
    configured_uri = _optional_str(mlflow_cfg.get("tracking_uri", None))
    if configured_uri:
        return configured_uri

    env_uri = _optional_str(os.environ.get("MLFLOW_TRACKING_URI"))
    if env_uri:
        return env_uri

    return _repo_tracking_uri()


def _disable_mlflow_tracing() -> None:
    os.environ.setdefault("MLFLOW_TRACE_SAMPLING_RATIO", "0.0")
    os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")

    try:
        import mlflow

        mlflow.tracing.disable()
    except Exception:
        pass


def _default_mlflow_run_name(cfg: DictConfig) -> str:
    configured_name = _optional_str(cfg.trainer.get("mlflow", {}).get("run_name", None))
    if configured_name:
        return configured_name

    parts = [
        str(cfg.model.get("name", cfg.model.model_name_or_path)),
        Path(str(cfg.trainer.output_dir)).name,
    ]
    slurm_job_id = _optional_str(os.environ.get("SLURM_JOB_ID"))
    if slurm_job_id:
        parts.append(f"slurm-{slurm_job_id}")
    else:
        parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))

    return "-".join(part for part in parts if part)


def _configure_reward_logging(cfg: DictConfig):
    mlflow_cfg = cfg.trainer.get("mlflow", {})
    enabled = bool(mlflow_cfg.get("enabled", True))
    disable_traces = bool(mlflow_cfg.get("disable_traces", True))
    tracking_uri = _resolve_mlflow_tracking_uri(mlflow_cfg)
    artifact_subdir = str(mlflow_cfg.get("artifact_subdir", "reward_plots"))
    plot_every_n_steps = int(mlflow_cfg.get("plot_every_n_steps", 10))
    experiment_name = mlflow_cfg.get("experiment_name", None) or cfg.experiment.name
    run_name = _default_mlflow_run_name(cfg)

    if disable_traces:
        _disable_mlflow_tracing()

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
            "reward_primary_metric": "total_reward",
            "reward_total_metric": "total_reward",
            "reward_format_metric": "format_reward",
            "reward_parsability_metric": "parsability_reward",
            "reward_correctness_metric": "correctness_reward",
            "mlflow_disable_traces": disable_traces,
        }
    )
    logger.log_tags(
        {
            "primary_metric": "total_reward",
            "mlflow_tracing": "disabled" if disable_traces else "enabled",
            "model": cfg.model.get("name", cfg.model.model_name_or_path),
            "trainer_output_dir": cfg.trainer.output_dir,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        }
    )

    return logger


def _configure_terminal_output(cfg: DictConfig) -> None:
    terminal_cfg = cfg.trainer.get("terminal", {})
    if bool(terminal_cfg.get("show_dataset_progress", False)):
        enable_progress_bar()
    else:
        disable_progress_bar()
    transformers_logging.set_verbosity_warning()

    configure_reward_console(
        print_autoformalizations=bool(
            terminal_cfg.get("print_autoformalizations", True)
        ),
        print_every_n_steps=int(
            terminal_cfg.get("print_autoformalizations_every_n_steps", 10)
        ),
        max_examples=int(terminal_cfg.get("max_autoformalizations", 1)),
        max_chars=int(terminal_cfg.get("max_autoformalization_chars", 2000)),
    )


def _print_training_summary(cfg: DictConfig, runtime_device: str) -> None:
    args = cfg.trainer.args
    mlflow_cfg = cfg.trainer.get("mlflow", {})
    terminal_cfg = cfg.trainer.get("terminal", {})

    print("Starting training")
    print(f"  model: {cfg.model.model_name_or_path}")
    print(f"  device: {runtime_device}")
    print(f"  output_dir: {cfg.trainer.output_dir}")
    print(
        "  steps/batch/gen: "
        f"{args.max_steps}/{args.per_device_train_batch_size}/"
        f"{args.get('num_generations', 'n/a')}"
    )
    print(
        "  mlflow: "
        f"enabled={mlflow_cfg.get('enabled', True)} "
        f"traces={'off' if mlflow_cfg.get('disable_traces', True) else 'on'} "
        f"experiment={mlflow_cfg.get('experiment_name', None) or cfg.experiment.name} "
        f"tracking_uri={_resolve_mlflow_tracking_uri(mlflow_cfg)}"
    )
    if terminal_cfg.get("print_autoformalizations", True):
        print(
            "  autoformalizations: "
            "printing "
            f"{terminal_cfg.get('max_autoformalizations', 1)} sample(s) every "
            f"{terminal_cfg.get('print_autoformalizations_every_n_steps', 10)} "
            "reward batch(es)"
        )


def _bad_word_ids(tokenizer: Any, bad_words: list[str]) -> list[list[int]]:
    token_ids = []
    seen = set()

    for word in bad_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            continue
        key = tuple(ids)
        if key in seen:
            continue
        token_ids.append(ids)
        seen.add(key)

    return token_ids


def _add_generation_guards(
    trainer_args_dict: dict[str, Any],
    tokenizer: Any,
) -> None:
    generation_kwargs = dict(trainer_args_dict.get("generation_kwargs") or {})
    existing_bad_words = list(generation_kwargs.get("bad_words_ids") or [])
    role_marker_bad_words = _bad_word_ids(
        tokenizer,
        [
            "user",
            " user",
            "\nuser",
            "assistant",
            " assistant",
            "\nassistant",
            "system",
            " system",
            "\nsystem",
            "Your",
            " Your",
            "\nYour",
            "your",
            " your",
            "\nyour",
        ],
    )

    if role_marker_bad_words:
        generation_kwargs["bad_words_ids"] = existing_bad_words + role_marker_bad_words

    trainer_args_dict["generation_kwargs"] = generation_kwargs


def _torch_dtype_name(value: Any) -> str | None:
    text = _optional_str(value)
    if text is None or text == "auto":
        return text
    return text


def _add_model_init_kwargs(
    trainer_args_dict: dict[str, Any],
    cfg: DictConfig,
) -> None:
    model_init_kwargs = dict(trainer_args_dict.get("model_init_kwargs") or {})

    torch_dtype = _torch_dtype_name(cfg.model.get("torch_dtype", None))
    if torch_dtype:
        model_init_kwargs.setdefault("torch_dtype", torch_dtype)

    trust_remote_code = cfg.model.get("trust_remote_code", None)
    if trust_remote_code is not None:
        model_init_kwargs.setdefault("trust_remote_code", bool(trust_remote_code))

    if model_init_kwargs:
        trainer_args_dict["model_init_kwargs"] = model_init_kwargs


def _build_peft_config(cfg: DictConfig):
    peft_cfg = cfg.trainer.get("peft", None)
    if not peft_cfg or not bool(peft_cfg.get("enabled", False)):
        return None

    from peft import LoraConfig

    peft_kwargs = OmegaConf.to_container(peft_cfg, resolve=True)
    peft_kwargs.pop("enabled", None)
    peft_kwargs.setdefault("task_type", "CAUSAL_LM")

    return LoraConfig(**peft_kwargs)


@hydra.main(version_base=None, config_path="../CONFIGS", config_name="config")
def main(cfg: DictConfig) -> None:
    _log_stage("Configuring runtime")
    _set_seed(int(cfg.trainer.seed))
    runtime_device = _resolve_runtime_device(cfg)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    _configure_terminal_output(cfg)
    _configure_reward_logging(cfg)

    _print_training_summary(cfg, runtime_device)

    _log_stage(f"Loading train dataset: {cfg.dataset.train_path}")
    train_dataset = _prepare_dataset(str(cfg.dataset.train_path))
    _log_stage(f"Loading validation dataset: {cfg.dataset.validation_path}")
    eval_dataset = _prepare_dataset(str(cfg.dataset.validation_path))

    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.model_name_or_path
    _log_stage(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _log_stage("Building GRPO config")
    trainer_args_dict = OmegaConf.to_container(cfg.trainer.args, resolve=True)

    trainer_args_dict.pop("_target_", None)
    _add_generation_guards(trainer_args_dict, tokenizer)
    _add_model_init_kwargs(trainer_args_dict, cfg)

    trainer_args_dict["output_dir"] = str(cfg.trainer.output_dir)
    trainer_args_dict["run_name"] = str(cfg.experiment.name)
    trainer_args_dict["use_cpu"] = (runtime_device == "cpu")
    trainer_args_dict["dataloader_pin_memory"] = (runtime_device == "cuda")

    trainer_args = GRPOConfig(**trainer_args_dict)
    peft_config = _build_peft_config(cfg)

    _log_stage(f"Initializing GRPOTrainer and loading model: {cfg.model.model_name_or_path}")
    trainer = GRPOTrainer(
        model=str(cfg.model.model_name_or_path),
        reward_funcs=logical_feedback_reward,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    try:
        _log_stage("Starting trainer.train()")
        trainer.train()
        _log_stage(f"Saving model to: {cfg.trainer.output_dir}")
        trainer.save_model(str(cfg.trainer.output_dir))
    finally:
        flush_reward_logging()


if __name__ == "__main__":
    main()
