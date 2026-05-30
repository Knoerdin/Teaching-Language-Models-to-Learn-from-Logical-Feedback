from src.REWARDS.formatting import extract_formalization, format_reward
from src.REWARDS.logical_feedback import score_logical_feedback
from src.REWARDS.proving import SolverUnavailableError


def main() -> None:
    text = """Premises:
∀x(cat(x) → mammal(x))
cat(luna)

Conclusion:
mammal(luna)
"""

    print("format reward:", format_reward(text))
    formal_premises, formal_conclusion = extract_formalization(text)
    print("formal premises:", formal_premises)
    print("formal conclusion:", formal_conclusion)

    try:
        reward = score_logical_feedback(
            text,
            nl_premises="All cats are mammals. Luna is a cat.",
            nl_conclusion="Luna is a mammal.",
            gold_label="true",
        )
    except SolverUnavailableError as exc:
        raise SystemExit(f"combined reward unavailable: {exc}") from exc

    print("combined reward:", reward)


if __name__ == "__main__":
    main()
