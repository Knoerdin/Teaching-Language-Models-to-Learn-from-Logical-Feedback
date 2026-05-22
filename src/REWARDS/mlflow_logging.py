from __future__ import annotations

import csv
import logging
import os
import tempfile
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Protocol


class RewardBreakdownLike(Protocol):
    total_reward: float
    format_reward: float
    parsability_reward: float
    correctness_reward: float
    gold_fol_reward: float
    parsed: bool
    prover_attempted: bool
    prover_status: str


def _is_main_process() -> bool:
    rank = os.environ.get("RANK")
    return rank in (None, "", "0")


def _mean(values: list[float]) -> float:
    return float(fmean(values)) if values else 0.0


def _rate(values: list[bool]) -> float:
    return _mean([float(value) for value in values])


def _status_rate(
    breakdowns: list[RewardBreakdownLike],
    status: str,
    *,
    attempted_only: bool = False,
) -> float:
    selected = [
        breakdown
        for breakdown in breakdowns
        if not attempted_only or breakdown.prover_attempted
    ]
    if not selected:
        return 0.0
    return _rate([breakdown.prover_status == status for breakdown in selected])


class MLflowRewardLogger:
    def __init__(
        self,
        *,
        enabled: bool = True,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
        artifact_subdir: str = "reward_plots",
        plot_every_n_steps: int = 10,
    ) -> None:
        self.enabled = enabled and _is_main_process()
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.artifact_subdir = artifact_subdir
        self.plot_every_n_steps = max(1, plot_every_n_steps)
        self.step = 0
        self.history: list[dict[str, float]] = []
        self._mlflow: Any | None = None
        self._managed_run = False

    def _ensure_run(self):
        if not self.enabled:
            return None

        if self._mlflow is None:
            import mlflow

            logging.getLogger("mlflow").setLevel(logging.WARNING)
            self._mlflow = mlflow
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            if self.experiment_name:
                mlflow.set_experiment(self.experiment_name)

        active_run = self._mlflow.active_run()
        if active_run is None:
            active_run = self._mlflow.start_run(run_name=self.run_name)
            self._managed_run = True

        return active_run

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled:
            return

        self._ensure_run()
        clean_params = {
            key: str(value)
            for key, value in params.items()
            if value is not None
        }
        self._mlflow.log_params(clean_params)

    def log_tags(self, tags: dict[str, Any]) -> None:
        if not self.enabled:
            return

        self._ensure_run()
        clean_tags = {
            key: str(value)
            for key, value in tags.items()
            if value is not None
        }
        if clean_tags:
            self._mlflow.set_tags(clean_tags)

    def log_batch(self, breakdowns: list[RewardBreakdownLike]) -> None:
        if not self.enabled or not breakdowns:
            return

        self._ensure_run()
        self.step += 1

        totals = [breakdown.total_reward for breakdown in breakdowns]
        formats = [breakdown.format_reward for breakdown in breakdowns]
        parsability = [breakdown.parsability_reward for breakdown in breakdowns]
        correctness = [breakdown.correctness_reward for breakdown in breakdowns]
        gold_fol = [
            getattr(breakdown, "gold_fol_reward", 0.0)
            for breakdown in breakdowns
        ]

        reward_total_mean = _mean(totals)
        reward_format_mean = _mean(formats)
        reward_parsability_mean = _mean(parsability)
        reward_correctness_mean = _mean(correctness)
        reward_gold_fol_mean = _mean(gold_fol)
        parse_success_rate = _rate([breakdown.parsed for breakdown in breakdowns])
        prover_attempt_rate = _rate(
            [breakdown.prover_attempted for breakdown in breakdowns]
        )
        prover_correct_rate = _status_rate(
            breakdowns,
            "correct",
            attempted_only=True,
        )
        prover_incorrect_rate = _status_rate(
            breakdowns,
            "incorrect",
            attempted_only=True,
        )
        prover_unknown_rate = _status_rate(
            breakdowns,
            "unknown",
            attempted_only=True,
        )
        prover_parse_error_rate = _status_rate(
            breakdowns,
            "parse_error",
            attempted_only=True,
        )
        prover_exception_rate = _status_rate(
            breakdowns,
            "exception",
            attempted_only=True,
        )
        not_parsed_rate = _status_rate(breakdowns, "not_parsed")

        metrics = {
            # Top-level aliases make MLflow's run tables easier to scan.
            "total_reward": reward_total_mean,
            "format_reward": reward_format_mean,
            "parsability_reward": reward_parsability_mean,
            "correctness_reward": reward_correctness_mean,
            "gold_fol_reward": reward_gold_fol_mean,
            "parse_success_rate": parse_success_rate,
            "prover_correct_rate": prover_correct_rate,
            "prover_unknown_rate": prover_unknown_rate,
            "prover_parse_error_rate": prover_parse_error_rate,
            "prover_exception_rate": prover_exception_rate,
            "not_parsed_rate": not_parsed_rate,
            # Namespaced metrics preserve the existing reward dashboard.
            "reward/total_mean": _mean(totals),
            "reward/total_min": min(totals),
            "reward/total_max": max(totals),
            "reward/total_std": float(pstdev(totals)) if len(totals) > 1 else 0.0,
            "reward/format_mean": reward_format_mean,
            "reward/parsability_mean": reward_parsability_mean,
            "reward/correctness_mean": reward_correctness_mean,
            "reward/gold_fol_mean": reward_gold_fol_mean,
            "reward/parse_success_rate": parse_success_rate,
            "reward/prover_attempt_rate": prover_attempt_rate,
            "reward/prover_correct_rate": prover_correct_rate,
            "reward/prover_incorrect_rate": prover_incorrect_rate,
            "reward/prover_unknown_rate": prover_unknown_rate,
            "reward/prover_parse_error_rate": prover_parse_error_rate,
            "reward/prover_exception_rate": prover_exception_rate,
            "reward/not_parsed_rate": not_parsed_rate,
            "reward/batch_size": float(len(breakdowns)),
        }

        self.history.append({"step": float(self.step), **metrics})
        self._mlflow.log_metrics(metrics, step=self.step)

        if self.step % self.plot_every_n_steps == 0:
            self.log_artifacts()

    def log_artifacts(self) -> None:
        if not self.enabled or not self.history:
            return

        self._ensure_run()
        self._log_history_csv()
        self._log_reward_plot()

    def _log_history_csv(self) -> None:
        fieldnames = list(self.history[0].keys())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reward_history.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.history)

            self._mlflow.log_artifact(str(path), artifact_path=self.artifact_subdir)

    def _log_reward_plot(self) -> None:
        import matplotlib.pyplot as plt

        steps = [row["step"] for row in self.history]
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(steps, [row["reward/total_mean"] for row in self.history], label="total")
        axes[0].plot(steps, [row["reward/format_mean"] for row in self.history], label="format")
        axes[0].plot(
            steps,
            [row["reward/parsability_mean"] for row in self.history],
            label="parsability",
        )
        axes[0].plot(
            steps,
            [row["reward/correctness_mean"] for row in self.history],
            label="correctness",
        )
        axes[0].plot(
            steps,
            [row["reward/gold_fol_mean"] for row in self.history],
            label="gold FOL",
        )
        axes[0].set_ylabel("mean reward")
        axes[0].legend(loc="best")
        axes[0].grid(alpha=0.25)

        axes[1].plot(
            steps,
            [row["reward/parse_success_rate"] for row in self.history],
            label="parse success",
        )
        axes[1].plot(
            steps,
            [row["reward/prover_correct_rate"] for row in self.history],
            label="prover correct",
        )
        axes[1].plot(
            steps,
            [row["reward/prover_parse_error_rate"] for row in self.history],
            label="prover parse error",
        )
        axes[1].plot(
            steps,
            [row["reward/prover_exception_rate"] for row in self.history],
            label="prover exception",
        )
        axes[1].set_xlabel("reward batch")
        axes[1].set_ylabel("rate")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend(loc="best")
        axes[1].grid(alpha=0.25)

        fig.tight_layout()
        self._mlflow.log_figure(
            fig,
            f"{self.artifact_subdir}/reward_timeseries_step_{self.step:06d}.png",
        )
        plt.close(fig)

    def end_run(self) -> None:
        if not self.enabled:
            return

        self.log_artifacts()
        if self._mlflow is not None and self._managed_run:
            self._mlflow.end_run()
            self._managed_run = False


_LOGGER = MLflowRewardLogger(enabled=False)


def configure_reward_logging(
    *,
    enabled: bool = True,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    artifact_subdir: str = "reward_plots",
    plot_every_n_steps: int = 10,
) -> MLflowRewardLogger:
    global _LOGGER

    _LOGGER = MLflowRewardLogger(
        enabled=enabled,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        artifact_subdir=artifact_subdir,
        plot_every_n_steps=plot_every_n_steps,
    )
    return _LOGGER


def log_reward_batch(breakdowns: list[RewardBreakdownLike]) -> None:
    _LOGGER.log_batch(breakdowns)


def flush_reward_logging() -> None:
    _LOGGER.end_run()
