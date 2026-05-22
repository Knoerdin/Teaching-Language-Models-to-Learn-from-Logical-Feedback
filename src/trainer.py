from __future__ import annotations

import json
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
from datasets import disable_progress_bar, enable_progress_bar
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
_bootstrap_log("Importing transformers")
from transformers import AutoTokenizer, logging as transformers_logging
_bootstrap_log("Importing TRL")
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

_bootstrap_log("Importing reward modules")
from REWARDS.logical_feedback import configure_reward_console
from REWARDS.logical_feedback import logical_feedback_reward
from REWARDS.mlflow_logging import configure_reward_logging, flush_reward_logging
from autoformalization import TRAINER_KIND_GRPO
from autoformalization import TRAINER_KIND_SFT
from autoformalization import VALID_TRAINER_KINDS
from autoformalization import prepare_dataset as _prepare_dataset
_bootstrap_log("Finished trainer imports")


def _log_stage(message: str) -> None:
    _bootstrap_log(message)


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


def _trainer_kind(cfg: DictConfig) -> str:
    configured_kind = _optional_str(cfg.trainer.get("kind", None))
    if configured_kind:
        trainer_kind = configured_kind.lower()
    else:
        target_hints = [
            str(cfg.trainer.get("trainer_cls", {}).get("_target_", "")),
            str(cfg.trainer.get("args", {}).get("_target_", "")),
        ]
        if any("SFT" in target_hint for target_hint in target_hints):
            trainer_kind = TRAINER_KIND_SFT
        else:
            trainer_kind = TRAINER_KIND_GRPO

    if trainer_kind not in VALID_TRAINER_KINDS:
        raise ValueError(
            f"Unsupported trainer kind '{trainer_kind}'. "
            f"Expected one of: {sorted(VALID_TRAINER_KINDS)}"
        )

    return trainer_kind


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


def _mlflow_enabled(cfg: DictConfig) -> bool:
    mlflow_cfg = cfg.trainer.get("mlflow", None)
    return bool(mlflow_cfg) and bool(mlflow_cfg.get("enabled", True))


def _configure_trainer_mlflow(cfg: DictConfig, trainer_kind: str) -> None:
    if trainer_kind != TRAINER_KIND_SFT or not _mlflow_enabled(cfg):
        return

    mlflow_cfg = cfg.trainer.get("mlflow", {})
    disable_traces = bool(mlflow_cfg.get("disable_traces", True))
    if disable_traces:
        _disable_mlflow_tracing()

    os.environ["MLFLOW_TRACKING_URI"] = _resolve_mlflow_tracking_uri(mlflow_cfg)
    os.environ["MLFLOW_EXPERIMENT_NAME"] = str(
        mlflow_cfg.get("experiment_name", None) or cfg.experiment.name
    )
    os.environ.setdefault("MLFLOW_FLATTEN_PARAMS", "true")
    os.environ.setdefault("HF_MLFLOW_LOG_ARTIFACTS", "false")

    tags = _mlflow_tags(cfg, trainer_kind, disable_traces=disable_traces)
    os.environ["MLFLOW_TAGS"] = json.dumps(tags)


def _mlflow_tags(
    cfg: DictConfig,
    trainer_kind: str,
    *,
    disable_traces: bool,
) -> dict[str, str]:
    return {
        "primary_metric": (
            "eval_loss" if trainer_kind == TRAINER_KIND_SFT else "total_reward"
        ),
        "trainer_kind": trainer_kind,
        "mlflow_tracing": "disabled" if disable_traces else "enabled",
        "model": str(cfg.model.get("name", cfg.model.model_name_or_path)),
        "trainer_output_dir": str(cfg.trainer.output_dir),
        "hostname": socket.gethostname(),
        "slurm_job_id": str(os.environ.get("SLURM_JOB_ID", "")),
        "slurm_job_name": str(os.environ.get("SLURM_JOB_NAME", "")),
        "slurm_partition": str(os.environ.get("SLURM_JOB_PARTITION", "")),
    }


def _configure_terminal_output(cfg: DictConfig, trainer_kind: str) -> None:
    terminal_cfg = cfg.trainer.get("terminal", {})
    if bool(terminal_cfg.get("show_dataset_progress", False)):
        enable_progress_bar()
    else:
        disable_progress_bar()
    transformers_logging.set_verbosity_warning()

    if trainer_kind != TRAINER_KIND_GRPO:
        return

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


def _print_training_summary(
    cfg: DictConfig,
    runtime_device: str,
    trainer_kind: str,
) -> None:
    args = cfg.trainer.args
    mlflow_cfg = cfg.trainer.get("mlflow", {})
    terminal_cfg = cfg.trainer.get("terminal", {})

    print("Starting training")
    print(f"  trainer: {trainer_kind}")
    print(f"  model: {cfg.model.model_name_or_path}")
    print(f"  device: {runtime_device}")
    print(f"  output_dir: {cfg.trainer.output_dir}")
    batch_summary = (
        f"{args.get('max_steps', 'n/a')}/"
        f"{args.get('per_device_train_batch_size', 'n/a')}"
    )
    if "num_generations" in args:
        batch_summary = f"{batch_summary}/{args.num_generations}"
        print(f"  steps/batch/gen: {batch_summary}")
    else:
        print(f"  steps/batch: {batch_summary}")
    if trainer_kind == TRAINER_KIND_SFT:
        print(
            "  completion_only_loss: "
            f"{args.get('completion_only_loss', 'auto')}"
        )
    if trainer_kind == TRAINER_KIND_GRPO or cfg.trainer.get("mlflow", None):
        print(
            "  mlflow: "
            f"enabled={mlflow_cfg.get('enabled', True)} "
            f"traces={'off' if mlflow_cfg.get('disable_traces', True) else 'on'} "
            f"experiment={mlflow_cfg.get('experiment_name', None) or cfg.experiment.name} "
            f"tracking_uri={_resolve_mlflow_tracking_uri(mlflow_cfg)}"
        )
    if trainer_kind == TRAINER_KIND_GRPO and terminal_cfg.get(
        "print_autoformalizations", True
    ):
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


def _build_trainer_args_dict(
    cfg: DictConfig,
    *,
    runtime_device: str,
    tokenizer: Any,
    trainer_kind: str,
) -> dict[str, Any]:
    trainer_args_dict = OmegaConf.to_container(cfg.trainer.args, resolve=True)
    trainer_args_dict.pop("_target_", None)

    if trainer_kind == TRAINER_KIND_GRPO:
        _add_generation_guards(trainer_args_dict, tokenizer)
    elif _mlflow_enabled(cfg):
        _add_mlflow_report_to(trainer_args_dict)
    _add_model_init_kwargs(trainer_args_dict, cfg)

    trainer_args_dict["output_dir"] = str(cfg.trainer.output_dir)
    trainer_args_dict["run_name"] = (
        _default_mlflow_run_name(cfg)
        if _mlflow_enabled(cfg)
        else str(cfg.experiment.name)
    )
    trainer_args_dict["use_cpu"] = runtime_device == "cpu"
    trainer_args_dict["dataloader_pin_memory"] = runtime_device == "cuda"

    return trainer_args_dict


def _add_mlflow_report_to(trainer_args_dict: dict[str, Any]) -> None:
    report_to = trainer_args_dict.get("report_to", None)

    if report_to is None:
        trainer_args_dict["report_to"] = ["mlflow"]
        return

    if isinstance(report_to, str):
        normalized = report_to.strip().lower()
        if normalized in {"", "none", "null"}:
            trainer_args_dict["report_to"] = ["mlflow"]
        elif normalized != "mlflow":
            trainer_args_dict["report_to"] = [report_to, "mlflow"]
        return

    if isinstance(report_to, list) and "mlflow" not in report_to:
        trainer_args_dict["report_to"] = [*report_to, "mlflow"]


@hydra.main(version_base=None, config_path="../CONFIGS", config_name="config")
def main(cfg: DictConfig) -> None:
    _log_stage("Configuring runtime")
    trainer_kind = _trainer_kind(cfg)
    _set_seed(int(cfg.trainer.seed))
    runtime_device = _resolve_runtime_device(cfg)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    _configure_terminal_output(cfg, trainer_kind)
    if trainer_kind == TRAINER_KIND_GRPO:
        _configure_reward_logging(cfg)
    else:
        _configure_trainer_mlflow(cfg, trainer_kind)
        if bool(cfg.trainer.get("mlflow", {}).get("disable_traces", True)):
            _disable_mlflow_tracing()

    _print_training_summary(cfg, runtime_device, trainer_kind)

    _log_stage(f"Loading train dataset: {cfg.dataset.train_path}")
    train_dataset = _prepare_dataset(str(cfg.dataset.train_path), trainer_kind)
    _log_stage(f"Loading validation dataset: {cfg.dataset.validation_path}")
    eval_dataset = _prepare_dataset(str(cfg.dataset.validation_path), trainer_kind)

    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.model_name_or_path
    _log_stage(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _log_stage(f"Building {trainer_kind.upper()} config")
    trainer_args_dict = _build_trainer_args_dict(
        cfg,
        runtime_device=runtime_device,
        tokenizer=tokenizer,
        trainer_kind=trainer_kind,
    )
    peft_config = _build_peft_config(cfg)

    if trainer_kind == TRAINER_KIND_GRPO:
        trainer_args = GRPOConfig(**trainer_args_dict)
        _log_stage(
            f"Initializing GRPOTrainer and loading model: {cfg.model.model_name_or_path}"
        )
        trainer = GRPOTrainer(
            model=str(cfg.model.model_name_or_path),
            reward_funcs=logical_feedback_reward,
            args=trainer_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    else:
        trainer_args = SFTConfig(**trainer_args_dict)
        _log_stage(
            f"Initializing SFTTrainer and loading model: {cfg.model.model_name_or_path}"
        )
        trainer = SFTTrainer(
            model=str(cfg.model.model_name_or_path),
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
        if trainer_kind == TRAINER_KIND_GRPO:
            flush_reward_logging()


if __name__ == "__main__":
    main()
