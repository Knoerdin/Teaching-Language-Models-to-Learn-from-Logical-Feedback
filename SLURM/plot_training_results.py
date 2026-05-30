#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_COLUMNS = (
    "final_reward_mean",
    "best_reward_mean",
    "avg_reward_mean",
    "train_loss",
    "train_runtime",
)
REWARD_COMPONENT_COLUMNS = (
    "sample_total",
    "sample_format",
    "sample_parsability",
    "sample_correctness",
)
REWARD_RANGE_COLUMNS = (
    "range_best",
    "range_mean",
    "range_worst",
)
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training results from SLURM log summary CSV files.",
    )
    parser.add_argument(
        "--summary-dir",
        default="SLURM/log_summaries",
        help="Directory containing log_summary.csv and per-run CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Plot output directory. Defaults to <summary-dir>/plots.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=("png", "pdf", "svg"),
        help="Image format for plots.",
    )
    parser.add_argument(
        "--dpi",
        default=160,
        type=int,
        help="DPI for raster plot output.",
    )
    parser.add_argument(
        "--skip-per-run",
        action="store_true",
        help="Only make the overview plot from log_summary.csv.",
    )
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        help="Only make per-run plots.",
    )
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir).expanduser()
    if not summary_dir.is_absolute():
        summary_dir = (Path.cwd() / summary_dir).resolve()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else summary_dir / "plots"
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if not args.skip_overview:
        written.extend(plot_overview(summary_dir, out_dir, args.format, args.dpi))

    if not args.skip_per_run:
        written.extend(plot_runs(summary_dir, out_dir, args.format, args.dpi))

    if not written:
        print(f"No plots written from {summary_dir}")
        return

    print(f"Wrote {len(written)} plot(s) to {out_dir}")
    for path in written:
        print(f"- {path}")


def plot_overview(
    summary_dir: Path,
    out_dir: Path,
    image_format: str,
    dpi: int,
) -> list[Path]:
    summary_path = summary_dir / "log_summary.csv"
    if not summary_path.exists():
        return []

    rows = [
        row
        for row in read_csv(summary_path)
        if row.get("status") == "finished"
        and any(to_float(row.get(column)) is not None for column in SUMMARY_COLUMNS)
    ]
    if not rows:
        return []

    labels = [run_label(row) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(max(11, len(rows) * 0.55), 8), constrained_layout=True)
    axes_list = axes.ravel()

    bar_plot(
        axes_list[0],
        labels,
        [to_float(row.get("final_reward_mean")) for row in rows],
        "Final Reward Mean",
    )
    bar_plot(
        axes_list[1],
        labels,
        [to_float(row.get("best_reward_mean")) for row in rows],
        "Best Logged Reward Mean",
    )
    bar_plot(
        axes_list[2],
        labels,
        [to_float(row.get("train_loss")) for row in rows],
        "Train Loss",
    )
    bar_plot(
        axes_list[3],
        labels,
        seconds_to_minutes([to_float(row.get("train_runtime")) for row in rows]),
        "Train Runtime (min)",
    )

    fig.suptitle("Finished Training Runs", fontsize=14)
    path = out_dir / f"training_overview.{image_format}"
    savefig(fig, path, dpi)
    return [path]


def plot_runs(
    summary_dir: Path,
    out_dir: Path,
    image_format: str,
    dpi: int,
) -> list[Path]:
    written: list[Path] = []
    reward_paths = sorted(summary_dir.rglob("*_reward_batches.csv"))

    for reward_path in reward_paths:
        stem = reward_path.name.removesuffix("_reward_batches.csv")
        trainer_path = reward_path.with_name(f"{stem}_trainer_metrics.csv")
        relative_parent = reward_path.parent.relative_to(summary_dir)
        run_out_dir = out_dir / relative_parent
        run_out_dir.mkdir(parents=True, exist_ok=True)

        reward_rows = read_csv(reward_path)
        trainer_rows = read_csv(trainer_path) if trainer_path.exists() else []
        if reward_rows:
            reward_plot = run_out_dir / f"{stem}_rewards.{image_format}"
            plot_reward_batches(reward_rows, reward_plot, dpi)
            written.append(reward_plot)
        if trainer_rows:
            trainer_plot = run_out_dir / f"{stem}_trainer_metrics.{image_format}"
            plot_trainer_metrics(trainer_rows, trainer_plot, dpi)
            written.append(trainer_plot)

    return written


def plot_reward_batches(rows: list[dict[str, str]], path: Path, dpi: int) -> None:
    x = series(rows, "reward_batch", fallback_index=True)
    title = title_from_rows(rows)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    plot_lines(axes[0], x, rows, REWARD_RANGE_COLUMNS, "Reward Range")
    plot_lines(axes[1], x, rows, REWARD_COMPONENT_COLUMNS, "Printed Sample Reward Components")
    plot_status_rates(axes[2], x, rows)

    axes[2].set_xlabel("Reward Batch")
    fig.suptitle(title, fontsize=14)
    savefig(fig, path, dpi)


def plot_trainer_metrics(rows: list[dict[str, str]], path: Path, dpi: int) -> None:
    x = series(rows, "metric_index", fallback_index=True)
    title = title_from_rows(rows)
    column_groups = [
        ("Reward and Loss", ("reward", "loss")),
        ("Reward Std and Entropy", ("reward_std", "entropy")),
        (
            "Completion Length",
            ("completions/mean_length", "completions/clipped_ratio"),
        ),
        ("Optimization", ("learning_rate", "grad_norm")),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    for axis, (axis_title, columns) in zip(axes.ravel(), column_groups):
        plot_lines(axis, x, rows, columns, axis_title)
    for axis in axes[-1, :]:
        axis.set_xlabel("Trainer Metric Index")

    fig.suptitle(title, fontsize=14)
    savefig(fig, path, dpi)


def plot_lines(
    axis: plt.Axes,
    x: list[float],
    rows: list[dict[str, str]],
    columns: Iterable[str],
    title: str,
) -> None:
    plotted = False
    for column in columns:
        y = [to_float(row.get(column)) for row in rows]
        pairs = [(x_value, y_value) for x_value, y_value in zip(x, y) if y_value is not None]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        axis.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.4, label=clean_label(column))
        plotted = True

    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    if plotted:
        axis.legend(loc="best", fontsize=8)
    else:
        axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)


def plot_status_rates(axis: plt.Axes, x: list[float], rows: list[dict[str, str]]) -> None:
    statuses = [row.get("sample_status", "") for row in rows]
    if not statuses:
        axis.text(0.5, 0.5, "No status data", ha="center", va="center", transform=axis.transAxes)
        return

    preferred = ["correct", "parse_error", "not_parsed", "exception"]
    unique_statuses = {status or "unknown" for status in statuses}
    ordered_statuses = [status for status in preferred if status in unique_statuses]
    ordered_statuses.extend(sorted(unique_statuses - set(ordered_statuses)))

    cumulative: dict[str, int] = defaultdict(int)
    status_values: dict[str, list[float]] = defaultdict(list)
    for index, status in enumerate(statuses, start=1):
        normalized = status or "unknown"
        cumulative[normalized] += 1
        for known_status in ordered_statuses:
            status_values[known_status].append(cumulative[known_status] / index)

    for status in ordered_statuses:
        axis.plot(x, status_values[status], marker="o", markersize=2.5, linewidth=1.4, label=status)

    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Cumulative Printed Sample Status Rate")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=8)


def bar_plot(
    axis: plt.Axes,
    labels: list[str],
    values: list[float | None],
    title: str,
) -> None:
    xs = list(range(len(labels)))
    plotted_values = [math.nan if value is None else value for value in values]
    axis.bar(xs, plotted_values)
    axis.set_title(title)
    axis.set_xticks(xs)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.grid(True, axis="y", alpha=0.3)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def series(
    rows: list[dict[str, str]],
    column: str,
    *,
    fallback_index: bool = False,
) -> list[float]:
    values: list[float] = []
    for index, row in enumerate(rows, start=1):
        value = to_float(row.get(column))
        if value is None and fallback_index:
            value = float(index)
        values.append(float(value) if value is not None else math.nan)
    return values


def seconds_to_minutes(values: list[float | None]) -> list[float | None]:
    return [None if value is None else value / 60.0 for value in values]


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def title_from_rows(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "Training Run"
    row = rows[0]
    pieces = [
        row.get("job_id") or "",
        row.get("job_name") or "",
        row.get("model") or "",
    ]
    return " | ".join(piece for piece in pieces if piece) or "Training Run"


def run_label(row: dict[str, str]) -> str:
    job_id = row.get("job_id") or "unknown"
    name = row.get("job_name") or Path(row.get("output_dir") or job_id).name
    generations = row.get("num_generations")
    gpus = row.get("gpu_count")
    details = []
    if generations:
        details.append(f"gen{generations}")
    if gpus:
        details.append(f"gpu{gpus}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"{job_id} {name}{suffix}"


def clean_label(column: str) -> str:
    label = column.replace("rewards/logical_feedback_reward/", "reward/")
    label = label.replace("completions/", "completion/")
    return label.replace("_", " ")


def savefig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
