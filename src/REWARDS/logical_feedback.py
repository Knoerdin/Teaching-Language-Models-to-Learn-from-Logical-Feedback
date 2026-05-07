from __future__ import annotations

from typing import Any

from .formatting import completion_to_text, extract_formalization, format_reward


def score_logical_feedback(
    text: str,
    *,
    nl_premises: str,
    nl_conclusion: str,
    gold_label: str,
) -> float:
    reward = format_reward(text)
    formal_premises, formal_conclusion = extract_formalization(text)

    if formal_premises is None or formal_conclusion is None:
        return reward

    from .proving import correctness_reward

    return reward + correctness_reward(
        nl_premises=nl_premises,
        nl_conclusion=nl_conclusion,
        formal_premises=formal_premises,
        formal_conclusion=formal_conclusion,
        gold_label=gold_label,
    )


def logical_feedback_reward(
    prompts: list[str],
    completions: list[Any],
    solution: list[str],
    premises: list[str],
    conclusion: list[str],
    **_: Any,
) -> list[float]:
    del prompts

    return [
        score_logical_feedback(
            completion_to_text(completion_item),
            nl_premises=nl_premises,
            nl_conclusion=nl_conclusion,
            gold_label=gold_label,
        )
        for completion_item, gold_label, nl_premises, nl_conclusion in zip(
            completions, solution, premises, conclusion, strict=True
        )
    ]
