#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any


START_PATTERN = re.compile(
    r"^\[(?P<started>[^\]]+)\] Starting (?P<job_name>\S+) "
    r"(?P<job_id>\d+) on (?P<host>\S+)"
)
FINISH_PATTERN = re.compile(r"^\[(?P<finished>[^\]]+)\] Training finished")
DICT_PATTERN = re.compile(r"\{[^{}]*\}")
SAMPLE_SCORE_PATTERN = re.compile(
    r"(?:best sample|sample)\s+\d+.*?total=(?P<total>[-+0-9.eE]+)\s+"
    r"format=(?P<format_reward>[-+0-9.eE]+)\s+"
    r"parsability=(?P<parsability_reward>[-+0-9.eE]+)\s+"
    r"correctness=(?P<correctness_reward>[-+0-9.eE]+)\s+"
    r"status=(?P<status>[A-Za-z0-9_]+)"
)
REWARD_RANGE_PATTERN = re.compile(
    r"reward range:\s*best=(?P<best>[-+0-9.eE]+)\s+"
    r"mean=(?P<mean>[-+0-9.eE]+)\s+"
    r"worst=(?P<worst>[-+0-9.eE]+)"
)
REWARD_BATCH_PATTERN = re.compile(r"\[reward batch (?P<batch>\d+)\]")
PROVER_STATE_PATTERN = re.compile(r"^prover state:\s*(?P<state>.*)$", re.MULTILINE)
PROVER_FEEDBACK_PATTERN = re.compile(
    r"^prover feedback:\s*(?P<feedback>.*)$",
    re.MULTILINE,
)
PROGRESS_PATTERN = re.compile(r"^\s*\d+%\|")
ERROR_MARKERS = (
    "Traceback",
    "OutOfMemoryError",
    "DistNetworkError",
    "RuntimeError",
    "ModuleNotFoundError",
    "No virtual environment",
    "Project directory does not exist",
    "FAILED",
)


@dataclass
class SampleBlock:
    batch: int | None
    range_best: float | None
    range_mean: float | None
    range_worst: float | None
    total: float | None
    format_reward: float | None
    parsability_reward: float | None
    correctness_reward: float | None
    status: str
    prover_state: str = ""
    prover_feedback: str = ""
    premises: str = ""
    conclusion: str = ""
    text: str = ""


@dataclass
class LogSummary:
    path: Path
    job_id: str = ""
    job_name: str = ""
    status: str = "unknown"
    started: str = ""
    finished: str = ""
    host: str = ""
    model: str = ""
    output_dir: str = ""
    max_steps: str = ""
    batch_size: str = ""
    num_generations: str = ""
    torch_processes: str = ""
    master_port: str = ""
    cuda_visible_devices: str = ""
    saved_model: str = ""
    failure_reason: str = ""
    reward_log_count: int = 0
    first_reward_mean: float | None = None
    final_reward_mean: float | None = None
    best_reward_mean: float | None = None
    avg_reward_mean: float | None = None
    final_reward_std: float | None = None
    final_frac_reward_zero_std: float | None = None
    final_entropy: float | None = None
    final_mean_length: float | None = None
    final_clipped_ratio: float | None = None
    train_runtime: float | None = None
    train_loss: float | None = None
    train_samples_per_second: float | None = None
    train_steps_per_second: float | None = None
    epoch: float | None = None
    last_sample: SampleBlock | None = None
    best_sample: SampleBlock | None = None
    samples: list[SampleBlock] = field(default_factory=list)
    metric_tail: dict[str, Any] = field(default_factory=dict)
    reward_metrics: list[dict[str, Any]] = field(default_factory=list)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Slurm training logs into CSV and Markdown.",
    )
    parser.add_argument(
        "--logs",
        default=None,
        help="Log root to scan. Defaults to SLURM/logs next to this script.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to SLURM/log_summaries.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    logs_root = Path(args.logs).expanduser() if args.logs else script_dir / "logs"
    out_dir = Path(args.out).expanduser() if args.out else script_dir / "log_summaries"
    if not logs_root.is_absolute():
        logs_root = (Path.cwd() / logs_root).resolve()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()

    logs = sorted(logs_root.rglob("job_*.out"))
    if not logs:
        print(f"No job_*.out logs found under {logs_root}")
        return

    summaries = [summarize_log(path) for path in logs]
    summaries.sort(key=lambda summary: (summary.started, str(summary.path)))

    write_csv(out_dir / "log_summary.csv", summaries, repo_root)
    write_markdown(out_dir / "log_summary.md", summaries, repo_root, out_dir, logs_root)
    for summary in summaries:
        run_path = run_markdown_path(out_dir, logs_root, summary)
        run_path.parent.mkdir(parents=True, exist_ok=True)
        write_reward_batches_csv(run_reward_batches_path(run_path), [summary], repo_root)
        write_metric_history_csv(run_trainer_metrics_path(run_path), [summary], repo_root)
        write_run_markdown(run_path, summary, repo_root)

    print(f"Wrote {len(summaries)} log summaries to {out_dir}")
    print(f"- {out_dir / 'log_summary.md'}")
    print(f"- {out_dir / 'log_summary.csv'}")


def summarize_log(path: Path) -> LogSummary:
    text = read_log(path)
    lines = text.splitlines()
    summary = LogSummary(path=path)
    summary.job_id = job_id_from_path(path)

    for line in lines:
        if start_match := START_PATTERN.match(line):
            summary.started = start_match.group("started")
            summary.job_name = start_match.group("job_name")
            summary.job_id = start_match.group("job_id")
            summary.host = start_match.group("host")
        elif finish_match := FINISH_PATTERN.match(line):
            summary.finished = finish_match.group("finished")
        elif line.startswith("Torch processes:"):
            summary.torch_processes = value_after_colon(line)
        elif line.startswith("Torch master port:"):
            summary.master_port = value_after_colon(line)
        elif line.startswith("CUDA_VISIBLE_DEVICES="):
            summary.cuda_visible_devices = line.split("=", 1)[1].strip()
        elif line.strip().startswith("model:"):
            summary.model = value_after_colon(line)
        elif line.strip().startswith("output_dir:"):
            summary.output_dir = value_after_colon(line)
        elif line.strip().startswith("steps/batch/gen:"):
            steps = value_after_colon(line)
            parts = steps.split("/")
            if len(parts) == 3:
                summary.max_steps, summary.batch_size, summary.num_generations = parts
        elif "Saving model to:" in line:
            summary.saved_model = line.split("Saving model to:", 1)[1].strip()

    metric_dicts = parse_metric_dicts(text)
    reward_metrics = [metric for metric in metric_dicts if metric_float(metric, "reward") is not None]
    train_metrics = [metric for metric in metric_dicts if metric_float(metric, "train_runtime") is not None]
    fill_reward_metrics(summary, reward_metrics)
    fill_train_metrics(summary, train_metrics)

    samples = extract_samples(lines)
    summary.samples = samples
    if samples:
        summary.last_sample = samples[-1]
        samples_with_reward = [sample for sample in samples if sample.total is not None]
        if samples_with_reward:
            summary.best_sample = max(
                samples_with_reward,
                key=lambda sample: sample.total if sample.total is not None else float("-inf"),
            )

    summary.failure_reason = first_failure_line(lines)
    if summary.finished:
        summary.status = "finished"
    elif summary.failure_reason:
        summary.status = "failed"
    else:
        summary.status = "incomplete"

    return summary


def read_log(path: Path) -> str:
    text = path.read_text(errors="replace")
    text = text.replace("\r", "\n")
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def job_id_from_path(path: Path) -> str:
    match = re.search(r"job_(\d+)", path.name)
    return match.group(1) if match else path.stem


def value_after_colon(line: str) -> str:
    return line.split(":", 1)[1].strip()


def parse_metric_dicts(text: str) -> list[dict[str, Any]]:
    metric_dicts = []
    for match in DICT_PATTERN.finditer(text):
        try:
            value = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            metric_dicts.append(value)
    return metric_dicts


def fill_reward_metrics(summary: LogSummary, reward_metrics: list[dict[str, Any]]) -> None:
    if not reward_metrics:
        return

    summary.reward_metrics = reward_metrics
    rewards = [metric_float(metric, "reward") for metric in reward_metrics]
    rewards = [reward for reward in rewards if reward is not None]
    if rewards:
        summary.reward_log_count = len(rewards)
        summary.first_reward_mean = rewards[0]
        summary.final_reward_mean = rewards[-1]
        summary.best_reward_mean = max(rewards)
        summary.avg_reward_mean = fmean(rewards)

    tail = reward_metrics[-1]
    summary.metric_tail = tail
    summary.final_reward_std = metric_float(tail, "reward_std")
    summary.final_frac_reward_zero_std = metric_float(tail, "frac_reward_zero_std")
    summary.final_entropy = metric_float(tail, "entropy")
    summary.final_mean_length = metric_float(tail, "completions/mean_length")
    summary.final_clipped_ratio = metric_float(tail, "completions/clipped_ratio")


def fill_train_metrics(summary: LogSummary, train_metrics: list[dict[str, Any]]) -> None:
    if not train_metrics:
        return

    final_train = train_metrics[-1]
    summary.train_runtime = metric_float(final_train, "train_runtime")
    summary.train_loss = metric_float(final_train, "train_loss")
    summary.train_samples_per_second = metric_float(final_train, "train_samples_per_second")
    summary.train_steps_per_second = metric_float(final_train, "train_steps_per_second")
    summary.epoch = metric_float(final_train, "epoch")


def metric_float(metric: dict[str, Any], key: str) -> float | None:
    if key not in metric:
        return None
    try:
        return float(metric[key])
    except (TypeError, ValueError):
        return None


def extract_samples(lines: list[str]) -> list[SampleBlock]:
    samples = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if "[reward batch " not in line:
            index += 1
            continue

        block_lines = [line]
        index += 1
        while index < len(lines):
            next_line = lines[index]
            stripped = next_line.lstrip()
            if "[reward batch " in next_line:
                break
            if stripped.startswith("{'loss'") or stripped.startswith("{'train_runtime'"):
                break
            if PROGRESS_PATTERN.match(stripped):
                break
            block_lines.append(next_line)
            index += 1

        block = "\n".join(block_lines).strip()
        batch_match = REWARD_BATCH_PATTERN.search(block)
        range_match = REWARD_RANGE_PATTERN.search(block)
        score_match = SAMPLE_SCORE_PATTERN.search(block)
        prover_state_match = PROVER_STATE_PATTERN.search(block)
        prover_feedback_match = PROVER_FEEDBACK_PATTERN.search(block)
        premises, conclusion = extract_model_result(block)
        samples.append(
            SampleBlock(
                batch=int(batch_match.group("batch")) if batch_match else None,
                range_best=float(range_match.group("best")) if range_match else None,
                range_mean=float(range_match.group("mean")) if range_match else None,
                range_worst=float(range_match.group("worst")) if range_match else None,
                total=float(score_match.group("total")) if score_match else None,
                format_reward=float(score_match.group("format_reward")) if score_match else None,
                parsability_reward=float(score_match.group("parsability_reward")) if score_match else None,
                correctness_reward=float(score_match.group("correctness_reward")) if score_match else None,
                status=score_match.group("status") if score_match else "",
                prover_state=prover_state_match.group("state").strip()
                if prover_state_match
                else "",
                prover_feedback=prover_feedback_match.group("feedback").strip()
                if prover_feedback_match
                else "",
                premises=premises,
                conclusion=conclusion,
                text=block,
            )
        )

    return samples


def extract_model_result(block: str) -> tuple[str, str]:
    premises_marker = "\nPremises:\n"
    conclusion_marker = "\nConclusion:\n"
    premises_marker_start = block.find(premises_marker)
    if premises_marker_start < 0:
        return "", ""

    premises_start = premises_marker_start + len(premises_marker)
    conclusion_start = block.find(conclusion_marker, premises_start)
    if conclusion_start < 0:
        return "", ""

    conclusion_text_start = conclusion_start + len(conclusion_marker)
    premises = block[premises_start:conclusion_start].strip()
    conclusion = block[conclusion_text_start:].strip()
    return premises, conclusion


def first_failure_line(lines: list[str]) -> str:
    for line in lines:
        if any(marker in line for marker in ERROR_MARKERS):
            return line.strip()
    return ""


def write_csv(path: Path, summaries: list[LogSummary], repo_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "log_path",
        "job_id",
        "job_name",
        "status",
        "started",
        "finished",
        "host",
        "model",
        "output_dir",
        "max_steps",
        "batch_size",
        "num_generations",
        "torch_processes",
        "gpu_count",
        "cuda_visible_devices",
        "saved_model",
        "train_runtime",
        "train_loss",
        "final_reward_mean",
        "best_reward_mean",
        "avg_reward_mean",
        "final_reward_std",
        "final_entropy",
        "final_mean_length",
        "final_clipped_ratio",
        "last_sample_total",
        "last_sample_status",
        "best_sample_total",
        "best_sample_status",
        "failure_reason",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(row_for_summary(summary, repo_root))


def write_reward_batches_csv(
    path: Path,
    summaries: list[LogSummary],
    repo_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "log_path",
        "job_id",
        "job_name",
        "status",
        "model",
        "output_dir",
        "max_steps",
        "batch_size",
        "num_generations",
        "gpu_count",
        "reward_batch",
        "range_best",
        "range_mean",
        "range_worst",
        "sample_total",
        "sample_format",
        "sample_parsability",
        "sample_correctness",
        "sample_status",
        "prover_state",
        "prover_feedback",
        "premises",
        "conclusion",
        "sample_text",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            for sample in samples_for_summary(summary):
                writer.writerow(
                    {
                        "log_path": relative_path(summary.path, repo_root),
                        "job_id": summary.job_id,
                        "job_name": summary.job_name,
                        "status": summary.status,
                        "model": summary.model,
                        "output_dir": summary.output_dir,
                        "max_steps": summary.max_steps,
                        "batch_size": summary.batch_size,
                        "num_generations": summary.num_generations,
                        "gpu_count": gpu_count(summary),
                        "reward_batch": sample.batch or "",
                        "range_best": format_optional(sample.range_best),
                        "range_mean": format_optional(sample.range_mean),
                        "range_worst": format_optional(sample.range_worst),
                        "sample_total": format_optional(sample.total),
                        "sample_format": format_optional(sample.format_reward),
                        "sample_parsability": format_optional(sample.parsability_reward),
                        "sample_correctness": format_optional(sample.correctness_reward),
                        "sample_status": sample.status,
                        "prover_state": sample.prover_state,
                        "prover_feedback": sample.prover_feedback,
                        "premises": sample.premises,
                        "conclusion": sample.conclusion,
                        "sample_text": sample.text,
                    }
                )


def write_metric_history_csv(
    path: Path,
    summaries: list[LogSummary],
    repo_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted(
        {
            key
            for summary in summaries
            for metric in summary.reward_metrics
            for key in metric
        }
    )
    fieldnames = [
        "log_path",
        "job_id",
        "job_name",
        "status",
        "model",
        "metric_index",
        *metric_keys,
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            for index, metric in enumerate(summary.reward_metrics, start=1):
                row = {
                    "log_path": relative_path(summary.path, repo_root),
                    "job_id": summary.job_id,
                    "job_name": summary.job_name,
                    "status": summary.status,
                    "model": summary.model,
                    "metric_index": index,
                }
                row.update(metric)
                writer.writerow(row)


def samples_for_summary(summary: LogSummary) -> list[SampleBlock]:
    return summary.samples


def row_for_summary(summary: LogSummary, repo_root: Path) -> dict[str, Any]:
    return {
        "log_path": relative_path(summary.path, repo_root),
        "job_id": summary.job_id,
        "job_name": summary.job_name,
        "status": summary.status,
        "started": summary.started,
        "finished": summary.finished,
        "host": summary.host,
        "model": summary.model,
        "output_dir": summary.output_dir,
        "max_steps": summary.max_steps,
        "batch_size": summary.batch_size,
        "num_generations": summary.num_generations,
        "torch_processes": summary.torch_processes,
        "gpu_count": gpu_count(summary),
        "cuda_visible_devices": summary.cuda_visible_devices,
        "saved_model": summary.saved_model,
        "train_runtime": format_optional(summary.train_runtime),
        "train_loss": format_optional(summary.train_loss),
        "final_reward_mean": format_optional(summary.final_reward_mean),
        "best_reward_mean": format_optional(summary.best_reward_mean),
        "avg_reward_mean": format_optional(summary.avg_reward_mean),
        "final_reward_std": format_optional(summary.final_reward_std),
        "final_entropy": format_optional(summary.final_entropy),
        "final_mean_length": format_optional(summary.final_mean_length),
        "final_clipped_ratio": format_optional(summary.final_clipped_ratio),
        "last_sample_total": format_optional(summary.last_sample.total if summary.last_sample else None),
        "last_sample_status": summary.last_sample.status if summary.last_sample else "",
        "best_sample_total": format_optional(summary.best_sample.total if summary.best_sample else None),
        "best_sample_status": summary.best_sample.status if summary.best_sample else "",
        "failure_reason": summary.failure_reason,
    }


def write_markdown(
    path: Path,
    summaries: list[LogSummary],
    repo_root: Path,
    out_dir: Path,
    logs_root: Path,
) -> None:
    lines = [
        "# Slurm Log Summary",
        "",
        f"Summarized {len(summaries)} log files.",
        "",
        "- Run pages and per-run CSVs mirror the log directory layout.",
        "",
        "| Job | Status | Model | Steps | GPUs | Final Reward | Best Reward | Train Loss | Saved Model | Log |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for summary in summaries:
        run_link = relative_path(run_markdown_path(out_dir, logs_root, summary), out_dir)
        log_path = relative_path(summary.path, repo_root)
        lines.append(
            "| "
            f"[{summary.job_id or summary.path.stem}]({run_link}) | "
            f"{summary.status} | "
            f"{summary.model or summary.job_name or '-'} | "
            f"{summary.max_steps or '-'} | "
            f"{gpu_count(summary) or '-'} | "
            f"{format_optional(summary.final_reward_mean)} | "
            f"{format_optional(summary.best_reward_mean)} | "
            f"{format_optional(summary.train_loss)} | "
            f"{summary.saved_model or '-'} | "
            f"{log_path} |"
        )

    path.write_text("\n".join(lines) + "\n")


def write_run_markdown(
    path: Path,
    summary: LogSummary,
    repo_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reward_csv = run_reward_batches_path(path).name
    trainer_csv = run_trainer_metrics_path(path).name
    lines = [
        f"# Job {summary.job_id or summary.path.stem}",
        "",
        "## Overview",
        "",
        f"- Status: {summary.status}",
        f"- Job name: {summary.job_name or '-'}",
        f"- Host: {summary.host or '-'}",
        f"- Started: {summary.started or '-'}",
        f"- Finished: {summary.finished or '-'}",
        f"- Log: {relative_path(summary.path, repo_root)}",
        f"- Reward batch CSV: {reward_csv}",
        f"- Trainer metrics CSV: {trainer_csv}",
        "",
        "## Configuration",
        "",
        f"- Model: {summary.model or '-'}",
        f"- Output dir: {summary.output_dir or '-'}",
        f"- Saved model: {summary.saved_model or '-'}",
        f"- Steps/batch/generations: {summary.max_steps or '-'}/{summary.batch_size or '-'}/{summary.num_generations or '-'}",
        f"- Torch processes: {summary.torch_processes or '-'}",
        f"- GPU count: {gpu_count(summary) or '-'}",
        f"- CUDA visible devices: {summary.cuda_visible_devices or '-'}",
        "",
        "## Results",
        "",
        f"- Train runtime: {format_optional(summary.train_runtime)} seconds",
        f"- Train loss: {format_optional(summary.train_loss)}",
        f"- Final reward mean: {format_optional(summary.final_reward_mean)}",
        f"- Best logged reward mean: {format_optional(summary.best_reward_mean)}",
        f"- Average logged reward mean: {format_optional(summary.avg_reward_mean)}",
        f"- Final reward std: {format_optional(summary.final_reward_std)}",
        f"- Final entropy: {format_optional(summary.final_entropy)}",
        f"- Final completion mean length: {format_optional(summary.final_mean_length)}",
        f"- Final clipped ratio: {format_optional(summary.final_clipped_ratio)}",
    ]
    if summary.failure_reason:
        lines.extend(["", "## Failure", "", code_block(summary.failure_reason)])

    if summary.last_sample:
        lines.extend(["", "## Last Model Sample", "", code_block(summary.last_sample.text, max_chars=5000)])
    if summary.best_sample and summary.best_sample is not summary.last_sample:
        lines.extend(["", "## Best Printed Model Sample", "", code_block(summary.best_sample.text, max_chars=5000)])

    path.write_text("\n".join(lines) + "\n")


def relative_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def summary_group(summary: LogSummary, logs_root: Path) -> Path:
    try:
        relative_log_path = summary.path.resolve().relative_to(logs_root.resolve())
    except ValueError:
        return Path("ungrouped")

    if relative_log_path.parent == Path("."):
        return Path("ungrouped")
    return relative_log_path.parent


def run_markdown_path(out_dir: Path, logs_root: Path, summary: LogSummary) -> Path:
    run_name = summary.job_id or summary.path.stem
    return out_dir / summary_group(summary, logs_root) / f"{run_name}.md"


def run_reward_batches_path(run_path: Path) -> Path:
    return run_path.with_name(f"{run_path.stem}_reward_batches.csv")


def run_trainer_metrics_path(run_path: Path) -> Path:
    return run_path.with_name(f"{run_path.stem}_trainer_metrics.csv")


def gpu_count(summary: LogSummary) -> str:
    if summary.torch_processes:
        return summary.torch_processes
    devices = summary.cuda_visible_devices
    if not devices or devices in {"unset", "NoDevFiles"}:
        return ""
    return str(len([device for device in devices.split(",") if device.strip()]))


def format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def code_block(text: str, *, max_chars: int = 2000) -> str:
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n... [truncated]"
    return f"```text\n{text}\n```"


if __name__ == "__main__":
    main()
