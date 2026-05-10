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
ROLE_MARKER_LINE_PATTERN = re.compile(
    r"^\s*(?:user|assistant|system)\s*:?\s*$",
    flags=re.IGNORECASE,
)
ROLE_MARKER_SUFFIX_PATTERN = re.compile(
    r"^(?:user|assistant|system)(?:\s*(?:user|assistant|system))*\s*:?\s*$",
    flags=re.IGNORECASE,
)
INLINE_ROLE_MARKER_PATTERN = re.compile(
    r"\)\s*(?:user|assistant|system)\b",
    flags=re.IGNORECASE,
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

    formal_premises = _clean_premises_section(match.group(1))
    formal_conclusion = _clean_conclusion_section(match.group(2))

    if not formal_premises or not formal_conclusion:
        return None, None

    return formal_premises, formal_conclusion


def _is_role_marker_line(line: str) -> bool:
    return ROLE_MARKER_LINE_PATTERN.fullmatch(line) is not None


def _is_role_marker_suffix(text: str) -> bool:
    return ROLE_MARKER_SUFFIX_PATTERN.fullmatch(text.strip()) is not None


def _trim_role_suffix_from_formula_line(line: str) -> str:
    depth = 0

    for index, character in enumerate(line):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return line.strip()
            if depth == 0 and _is_role_marker_suffix(line[index + 1 :]):
                return line[: index + 1].strip()

    return line.strip()


def _clean_premises_section(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _is_role_marker_line(stripped):
            break
        lines.append(_trim_role_suffix_from_formula_line(stripped))

    return "\n".join(lines).strip()


def _clean_conclusion_section(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _is_role_marker_line(stripped):
            break
        return _trim_role_suffix_from_formula_line(stripped)

    return ""


def _has_role_marker_junk(text: str) -> bool:
    if INLINE_ROLE_MARKER_PATTERN.search(text):
        return True

    return any(_is_role_marker_line(line) for line in text.splitlines())


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

    if _has_role_marker_junk(text):
        reward -= 0.35

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
