import re


def completion_to_text(completion_item) -> str:
    if isinstance(completion_item, str):
        return completion_item

    if isinstance(completion_item, list) and completion_item:
        first_item = completion_item[0]
        if isinstance(first_item, dict):
            return str(first_item.get("content", ""))
        return str(first_item)

    return ""


def extract_formalization(text: str) -> tuple[str | None, str | None]:
    match = re.search(
        r"Premises:\s*(.*?)\s*Conclusion:\s*(.*)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not match:
        return None, None

    formal_premises = match.group(1).strip()
    formal_conclusion = match.group(2).strip()

    if not formal_premises or not formal_conclusion:
        return None, None

    return formal_premises, formal_conclusion


def format_reward(text: str) -> float:
    reward = 0.0
    lower = text.lower()

    if "premises:" in lower:
        reward += 0.1
    if "conclusion:" in lower:
        reward += 0.1

    logic_markers = ["forall", "exists", "->", "(", ")", "¬", "∧", "∨"]
    if any(marker in lower for marker in logic_markers):
        reward += 0.2

    formal_premises, formal_conclusion = extract_formalization(text)
    if formal_premises is not None and formal_conclusion is not None:
        reward += 0.2

    return reward