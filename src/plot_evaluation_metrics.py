from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from evaluate_autoformalization import label_classification_metrics
from evaluate_autoformalization import sanitize_filename
from evaluate_autoformalization import solver_parser_parse_rate


PLOT_METRICS: tuple[tuple[str, str, str], ...] = (
    ("Label\nacc.", "label_accuracy", "score"),
    ("Parser\nparse", "parse_rate", "score"),
    ("Gold FOL\nacc.", "gold_fol_accuracy", "score"),
    ("Macro\nF1", "label_macro_f1", "label_f1"),
    ("True\nF1", "label_true_f1", "label_f1"),
    ("False\nF1", "label_false_f1", "label_f1"),
    ("Uncertain\nF1", "label_uncertain_f1", "label_f1"),
)

BAR_STYLES = {
    "score": {
        "label": "Score",
        "color": "#2f62df",
    },
    "label_f1": {
        "label": "Label F1",
        "color": "#19a34a",
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot compact paper-style evaluation summaries, one figure per model."
        )
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help=(
            "Prediction JSONL to plot. Use NAME=PATH for readable plot names. "
            "Repeat for multiple models."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluations/comparison_plots",
        help="Directory where plot files are written.",
    )
    parser.add_argument(
        "--format",
        action="append",
        default=None,
        choices=("png", "pdf", "svg"),
        help="Output format. Repeat to write multiple formats. Defaults to png and svg.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def parse_prediction_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
            name = name.strip()
            path = path.strip()
        else:
            path = value.strip()
            name = Path(path).stem.replace("_predictions", "")
        if not name or not path:
            raise ValueError(f"Invalid --prediction value: {value!r}")
        specs.append((name, Path(path).expanduser()))
    return specs


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return (
        sum(1 for row in rows if row.get(key, False)) / len(rows)
        if rows
        else 0.0
    )


def compute_plot_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = label_classification_metrics(rows)
    metrics.update(
        {
            "parse_rate": solver_parser_parse_rate(rows),
            "solver_parse_rate": solver_parser_parse_rate(rows),
            "format_extraction_rate": rate(rows, "parsed"),
            "gold_fol_accuracy": rate(rows, "joint_unordered_exact"),
        }
    )
    return metrics


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def require_plot_metrics(model_name: str, metrics: dict[str, float]) -> None:
    missing = [
        label
        for label, key, _series in PLOT_METRICS
        if key not in metrics
    ]
    if missing:
        raise ValueError(
            f"{model_name} is missing metric(s): {', '.join(missing)}. "
            "Use predictions with label outputs, e.g. --run-prover, D&P, or "
            "offline single-path predictions."
        )


def plot_model(
    model_name: str,
    metrics: dict[str, float],
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    require_plot_metrics(model_name, metrics)

    labels = [label for label, _key, _series in PLOT_METRICS]
    x_positions = list(range(len(labels)))
    values = [metrics[key] * 100.0 for _label, key, _series in PLOT_METRICS]
    colors = [BAR_STYLES[series]["color"] for _label, _key, series in PLOT_METRICS]

    fig, axis = plt.subplots(figsize=(7.1, 4.0), constrained_layout=True)
    axis.set_title(model_name)
    axis.set_xlabel("Metric")
    axis.set_ylabel("Score (%)")
    axis.set_ylim(0.0, 105.0)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)
    axis.set_yticks([0, 20, 40, 60, 80, 100])
    axis.grid(axis="y", linestyle="--", color="0.78", linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)

    bars = axis.bar(
        x_positions,
        values,
        width=0.62,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 1.4,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axis.legend(
        handles=[
            Patch(
                facecolor=style["color"],
                edgecolor="white",
                label=style["label"],
            )
            for style in BAR_STYLES.values()
        ],
        loc="upper left",
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    stem = f"{sanitize_filename(model_name)}_evaluation_metrics"
    for image_format in formats:
        path = output_dir / f"{stem}.{image_format}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    output_dir = Path(args.output_dir).expanduser()
    formats = args.format or ["png", "svg"]

    written = []
    for model_name, prediction_path in parse_prediction_specs(args.prediction):
        rows = load_jsonl(prediction_path)
        metrics = compute_plot_metrics(rows)
        written.extend(
            plot_model(model_name, metrics, output_dir, formats, args.dpi)
        )

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
