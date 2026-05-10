from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from .formatting import completion_to_text, extract_formalization, format_reward
from .mlflow_logging import log_reward_batch
from .parsing import parsability_reward

UNPARSEABLE_STATUSES = {"parse_error"}


@dataclass(frozen=True)
class RewardBreakdown:
    total_reward: float
    format_reward: float
    parsability_reward: float
    correctness_reward: float
    parsed: bool
    prover_attempted: bool
    prover_status: str
    formal_premises: str | None = None
    formal_conclusion: str | None = None
    completion_text: str | None = None
    prover_prediction: str | None = None
    prover_state_status: str | None = None
    prover_feedback: str | None = None
    prover_error_message: str | None = None


_PRINT_AUTOFORMALIZATIONS = True
_PRINT_EVERY_N_STEPS = 10
_PRINT_MAX_EXAMPLES = 1
_PRINT_MAX_CHARS = 2000
_PRINT_STEP = 0


def configure_reward_console(
    *,
    print_autoformalizations: bool = True,
    print_every_n_steps: int = 10,
    max_examples: int = 1,
    max_chars: int = 2000,
) -> None:
    global _PRINT_AUTOFORMALIZATIONS
    global _PRINT_EVERY_N_STEPS
    global _PRINT_MAX_EXAMPLES
    global _PRINT_MAX_CHARS
    global _PRINT_STEP

    _PRINT_AUTOFORMALIZATIONS = print_autoformalizations
    _PRINT_EVERY_N_STEPS = max(1, print_every_n_steps)
    _PRINT_MAX_EXAMPLES = max(1, max_examples)
    _PRINT_MAX_CHARS = max(200, max_chars)
    _PRINT_STEP = 0


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}\n... [truncated]"


def _is_main_process() -> bool:
    rank = os.environ.get("RANK")
    return rank in (None, "", "0")


def _mean_reward(breakdowns: list[RewardBreakdown]) -> float:
    return sum(breakdown.total_reward for breakdown in breakdowns) / len(breakdowns)


def _print_autoformalizations(breakdowns: list[RewardBreakdown]) -> None:
    global _PRINT_STEP

    if not _PRINT_AUTOFORMALIZATIONS or not breakdowns or not _is_main_process():
        return

    _PRINT_STEP += 1
    if _PRINT_STEP % _PRINT_EVERY_N_STEPS != 0:
        return

    sorted_breakdowns = sorted(
        enumerate(breakdowns, start=1),
        key=lambda indexed: indexed[1].total_reward,
        reverse=True,
    )
    best_reward = sorted_breakdowns[0][1].total_reward
    worst_reward = sorted_breakdowns[-1][1].total_reward
    print(f"\n[reward batch {_PRINT_STEP}] best local autoformalization sample")
    print(
        "reward range: "
        f"best={best_reward:.3f} "
        f"mean={_mean_reward(breakdowns):.3f} "
        f"worst={worst_reward:.3f}"
    )

    for display_index, (batch_index, breakdown) in enumerate(
        sorted_breakdowns[:_PRINT_MAX_EXAMPLES],
        start=1,
    ):
        _print_breakdown(display_index, batch_index, breakdown)


def _print_breakdown(
    display_index: int,
    batch_index: int,
    breakdown: RewardBreakdown,
) -> None:
    print(
        f"best sample {display_index} (batch item {batch_index}): "
        f"total={breakdown.total_reward:.3f} "
        f"format={breakdown.format_reward:.3f} "
        f"parsability={breakdown.parsability_reward:.3f} "
        f"correctness={breakdown.correctness_reward:.3f} "
        f"status={breakdown.prover_status}"
    )
    if breakdown.prover_state_status:
        print(f"prover state: {breakdown.prover_state_status}")
    if breakdown.prover_feedback and breakdown.prover_status in {
        "parse_error",
        "exception",
    }:
        print(f"prover feedback: {_truncate(breakdown.prover_feedback, 400)}")
    if breakdown.prover_error_message:
        print(f"prover error: {_truncate(breakdown.prover_error_message, 400)}")

    if breakdown.formal_premises is None or breakdown.formal_conclusion is None:
        print("unparsed completion:")
        print(_truncate(breakdown.completion_text or "", _PRINT_MAX_CHARS))
        return

    print("Premises:")
    print(_truncate(breakdown.formal_premises, _PRINT_MAX_CHARS))
    print("Conclusion:")
    print(_truncate(breakdown.formal_conclusion, _PRINT_MAX_CHARS))


def _has_parser_warning(prover_result: Any) -> bool:
    diagnostic_text = "\n".join(
        str(value or "")
        for value in (
            getattr(prover_result, "feedback", None),
            getattr(prover_result, "error_message", None),
        )
    ).lower()
    warning_markers = (
        "token recognition error",
        "extraneous input",
        "mismatched input",
        "no viable alternative",
    )
    return any(marker in diagnostic_text for marker in warning_markers)


def score_logical_feedback_breakdown(
    text: str,
    *,
    nl_premises: str,
    nl_conclusion: str,
    gold_label: str,
) -> RewardBreakdown:
    formatting_score = format_reward(text)
    formal_premises, formal_conclusion = extract_formalization(text)

    if formal_premises is None or formal_conclusion is None:
        parsability_score = parsability_reward(
            text,
            formal_premises=formal_premises,
            formal_conclusion=formal_conclusion,
        )
        return RewardBreakdown(
            total_reward=formatting_score + parsability_score,
            format_reward=formatting_score,
            parsability_reward=parsability_score,
            correctness_reward=0.0,
            parsed=False,
            prover_attempted=False,
            prover_status="not_parsed",
            completion_text=text,
        )

    from .proving import evaluate_correctness

    prover_result = evaluate_correctness(
        nl_premises=nl_premises,
        nl_conclusion=nl_conclusion,
        formal_premises=formal_premises,
        formal_conclusion=formal_conclusion,
        gold_label=gold_label,
    )
    correctness_score = prover_result.reward
    solver_parsed = (
        prover_result.status not in UNPARSEABLE_STATUSES
        and prover_result.status != "exception"
    )
    cleanly_parsed = solver_parsed and not _has_parser_warning(prover_result)
    parsability_score = parsability_reward(
        text,
        formal_premises=formal_premises,
        formal_conclusion=formal_conclusion,
        solver_parsed=cleanly_parsed,
        solver_exception=prover_result.status == "exception",
    )
    if prover_result.status in UNPARSEABLE_STATUSES:
        correctness_score = 0.0
    elif prover_result.status == "exception":
        correctness_score = 0.0
    total_reward = formatting_score + parsability_score + correctness_score

    return RewardBreakdown(
        total_reward=total_reward,
        format_reward=formatting_score,
        parsability_reward=parsability_score,
        correctness_reward=correctness_score,
        parsed=cleanly_parsed,
        prover_attempted=True,
        prover_status=prover_result.status,
        formal_premises=formal_premises,
        formal_conclusion=formal_conclusion,
        completion_text=text,
        prover_prediction=prover_result.prediction,
        prover_state_status=prover_result.state_status,
        prover_feedback=prover_result.feedback,
        prover_error_message=prover_result.error_message,
    )


def score_logical_feedback(
    text: str,
    *,
    nl_premises: str,
    nl_conclusion: str,
    gold_label: str,
) -> float:
    return score_logical_feedback_breakdown(
        text,
        nl_premises=nl_premises,
        nl_conclusion=nl_conclusion,
        gold_label=gold_label,
    ).total_reward


def logical_feedback_reward(
    prompts: list[str],
    completions: list[Any],
    solution: list[str],
    premises: list[str],
    conclusion: list[str],
    **_: Any,
) -> list[float]:
    del prompts

    breakdowns = [
        score_logical_feedback_breakdown(
            completion_to_text(completion_item),
            nl_premises=nl_premises,
            nl_conclusion=nl_conclusion,
            gold_label=gold_label,
        )
        for completion_item, gold_label, nl_premises, nl_conclusion in zip(
            completions, solution, premises, conclusion, strict=True
        )
    ]

    log_reward_batch(breakdowns)
    _print_autoformalizations(breakdowns)

    return [breakdown.total_reward for breakdown in breakdowns]
