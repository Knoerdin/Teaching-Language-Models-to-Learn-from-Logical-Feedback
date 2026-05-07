import re

LOGIC_OPERATOR_PATTERN = re.compile(
    r"(∀|∃|¬|→|∧|∨|⊕|->|\bforall\b|\bexists\b)",
    flags=re.IGNORECASE,
)
PREDICATE_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\s*\([^()\n]*\)")
PREMISES_HEADER_PATTERN = re.compile(r"\bPremises(?:\s+FOL)?:", flags=re.IGNORECASE)
CONCLUSION_HEADER_PATTERN = re.compile(r"\bConclusion(?:\s+FOL)?:", flags=re.IGNORECASE)
FORMALIZATION_PATTERN = re.compile(
    r"Premises(?:\s+FOL)?:\s*(.*?)\s*Conclusion(?:\s+FOL)?:\s*(.*)",
    flags=re.DOTALL | re.IGNORECASE,
)


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
    match = FORMALIZATION_PATTERN.search(text)

    if not match:
        return None, None

    formal_premises = match.group(1).strip()
    formal_conclusion = match.group(2).strip()

    if not formal_premises or not formal_conclusion:
        return None, None

    return formal_premises, formal_conclusion


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _has_balanced_parentheses(text: str) -> bool:
    depth = 0

    for character in text:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return False

    return depth == 0


def _is_fol_like(line: str) -> bool:
    has_predicate = PREDICATE_PATTERN.search(line) is not None
    if not has_predicate:
        return False

    has_logic_operator = LOGIC_OPERATOR_PATTERN.search(line) is not None
    has_atomic_formula = re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\s*\([^()\n]*\)", line) is not None

    return (has_logic_operator or has_atomic_formula) and _has_balanced_parentheses(line)


def format_reward(text: str) -> float:
    reward = 0.0

    if PREMISES_HEADER_PATTERN.search(text):
        reward += 0.02
    if CONCLUSION_HEADER_PATTERN.search(text):
        reward += 0.02

    formal_premises, formal_conclusion = extract_formalization(text)
    if formal_premises is None or formal_conclusion is None:
        return reward

    premise_lines = _nonempty_lines(formal_premises)
    conclusion_lines = _nonempty_lines(formal_conclusion)
    if not premise_lines or not conclusion_lines:
        return reward

    reward += 0.03

    fol_premise_count = sum(_is_fol_like(line) for line in premise_lines)
    fol_conclusion_count = sum(_is_fol_like(line) for line in conclusion_lines)

    reward += min(fol_premise_count, 3) * 0.05
    if fol_premise_count == len(premise_lines):
        reward += 0.05

    if fol_conclusion_count > 0:
        reward += 0.15

    if (
        fol_premise_count > 0
        and fol_conclusion_count > 0
        and _has_balanced_parentheses(formal_premises)
        and _has_balanced_parentheses(formal_conclusion)
    ):
        reward += 0.05

    return reward
