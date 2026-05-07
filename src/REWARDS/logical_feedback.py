from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .formatting import completion_to_text, extract_formalization, format_reward
from .mlflow_logging import log_reward_batch


@dataclass(frozen=True)
class RewardBreakdown:
    total_reward: float
    format_reward: float
    correctness_reward: float
    parsed: bool
    prover_attempted: bool
    prover_status: str
    prover_prediction: str | None = None
    prover_state_status: str | None = None


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
        return RewardBreakdown(
            total_reward=formatting_score,
            format_reward=formatting_score,
            correctness_reward=0.0,
            parsed=False,
            prover_attempted=False,
            prover_status="not_parsed",
        )

    from .proving import evaluate_correctness

    prover_result = evaluate_correctness(
        nl_premises=nl_premises,
        nl_conclusion=nl_conclusion,
        formal_premises=formal_premises,
        formal_conclusion=formal_conclusion,
        gold_label=gold_label,
    )

    return RewardBreakdown(
        total_reward=formatting_score + prover_result.reward,
        format_reward=formatting_score,
        correctness_reward=prover_result.reward,
        parsed=True,
        prover_attempted=True,
        prover_status=prover_result.status,
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

    return [breakdown.total_reward for breakdown in breakdowns]
