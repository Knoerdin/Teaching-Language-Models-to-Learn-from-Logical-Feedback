from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .formatting import completion_to_text, extract_formalization, format_reward
from .mlflow_logging import log_reward_batch

UNPARSEABLE_PARSABILITY_REWARD = -2.0
UNPARSEABLE_STATUSES = {"parse_error", "exception"}


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


def _print_autoformalizations(breakdowns: list[RewardBreakdown]) -> None:
    global _PRINT_STEP

    if not _PRINT_AUTOFORMALIZATIONS or not breakdowns:
        return

    _PRINT_STEP += 1
    if _PRINT_STEP % _PRINT_EVERY_N_STEPS != 0:
        return

    print(f"\n[reward batch {_PRINT_STEP}] model autoformalization sample")
    for index, breakdown in enumerate(breakdowns[:_PRINT_MAX_EXAMPLES], start=1):
        print(
            f"sample {index}: total={breakdown.total_reward:.3f} "
            f"format={breakdown.format_reward:.3f} "
            f"parsability={breakdown.parsability_reward:.3f} "
            f"correctness={breakdown.correctness_reward:.3f} "
            f"status={breakdown.prover_status}"
        )

        if breakdown.formal_premises is None or breakdown.formal_conclusion is None:
            print("unparsed completion:")
            print(_truncate(breakdown.completion_text or "", _PRINT_MAX_CHARS))
            continue

        print("Premises:")
        print(_truncate(breakdown.formal_premises, _PRINT_MAX_CHARS))
        print("Conclusion:")
        print(_truncate(breakdown.formal_conclusion, _PRINT_MAX_CHARS))


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
        parsability_score = UNPARSEABLE_PARSABILITY_REWARD
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
    parsability_score = 0.0
    if prover_result.status in UNPARSEABLE_STATUSES:
        parsability_score = UNPARSEABLE_PARSABILITY_REWARD
        correctness_score = 0.0
    total_reward = formatting_score + parsability_score + correctness_score

    return RewardBreakdown(
        total_reward=total_reward,
        format_reward=formatting_score,
        parsability_reward=parsability_score,
        correctness_reward=correctness_score,
        parsed=True,
        prover_attempted=True,
        prover_status=prover_result.status,
        formal_premises=formal_premises,
        formal_conclusion=formal_conclusion,
        completion_text=text,
        prover_prediction=prover_result.prediction,
        prover_state_status=prover_result.state_status,
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
