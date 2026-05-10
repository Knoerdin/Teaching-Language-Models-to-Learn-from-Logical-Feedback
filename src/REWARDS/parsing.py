from __future__ import annotations

import re

from .formatting import extract_formalization


MIN_PARSABILITY_REWARD = -2.0
MAX_UNPARSEABLE_REWARD = -0.05
PARSEABLE_PARSABILITY_REWARD = 0.5
EXCEPTION_PARSABILITY_REWARD = -0.5

HEADER_PATTERN = re.compile(
    r"\b(?:Premises|Conclusion)(?:\s+FOL)?:",
    flags=re.IGNORECASE,
)
LOGIC_OPERATOR_PATTERN = re.compile(r"(∀|∃|¬|→|∧|∨|⊕)")
PREDICATE_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\s*\(([^()\n]*)\)")
VALID_PREDICATE_PATTERN = re.compile(
    r"\b[A-Za-z][A-Za-z0-9_]*\s*"
    r"\(\s*[A-Za-z][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z][A-Za-z0-9_]*)*\s*\)"
)
BAD_SYMBOL_PATTERN = re.compile(
    r"(\.|:|;|\"|'|`|\\|\[|\]|\{|\}|<|>|~|\^|&|\||=|\+|-|"
    r"↔|⇔|⇒|↦|∈|∉|،)"
)
BINARY_OPERATOR_PATTERN = re.compile(r"(→|∧|∨|⊕)")
REPEATED_OPERATOR_PATTERN = re.compile(r"(→|∧|∨|⊕)\s*(→|∧|∨|⊕)")
QUANTIFIER_PATTERN = re.compile(r"(∀|∃)\s*([A-Za-z][A-Za-z0-9_]*)")


def parsability_reward(
    text: str,
    *,
    formal_premises: str | None = None,
    formal_conclusion: str | None = None,
    solver_parsed: bool = False,
    solver_exception: bool = False,
) -> float:
    """Reward syntactic progress toward solver-parseable FOL.

    GRPO compares generations within the same prompt. A flat parse-failure
    penalty makes all malformed generations equally bad, so this heuristic gives
    partial credit for structure that is closer to what the solver accepts.
    """

    if solver_parsed:
        return PARSEABLE_PARSABILITY_REWARD

    if formal_premises is None or formal_conclusion is None:
        formal_premises, formal_conclusion = extract_formalization(text)

    if formal_premises is None or formal_conclusion is None:
        return _missing_sections_reward(text)

    estimated_reward = _formula_sections_reward(formal_premises, formal_conclusion)
    if solver_exception:
        return min(estimated_reward, EXCEPTION_PARSABILITY_REWARD)

    return estimated_reward


def _missing_sections_reward(text: str) -> float:
    score = 0.0

    header_count = len(HEADER_PATTERN.findall(text))
    score += min(header_count, 2) * 0.10

    if LOGIC_OPERATOR_PATTERN.search(text):
        score += 0.10
    if PREDICATE_PATTERN.search(text):
        score += 0.20
    if _has_balanced_parentheses(text):
        score += 0.10
    if not BAD_SYMBOL_PATTERN.search(text):
        score += 0.10

    # Keep header-free or section-broken completions clearly worse than
    # extracted-but-invalid formulas.
    return _clamp(MIN_PARSABILITY_REWARD + score, MIN_PARSABILITY_REWARD, -1.20)


def _formula_sections_reward(formal_premises: str, formal_conclusion: str) -> float:
    lines = _formula_lines(formal_premises) + _formula_lines(formal_conclusion)
    if not lines:
        return MIN_PARSABILITY_REWARD

    line_quality = sum(_formula_line_quality(line) for line in lines) / len(lines)
    section_quality = 0.0
    if _formula_lines(formal_premises):
        section_quality += 0.25
    if _formula_lines(formal_conclusion):
        section_quality += 0.25
    if _has_balanced_parentheses(formal_premises):
        section_quality += 0.25
    if _has_balanced_parentheses(formal_conclusion):
        section_quality += 0.25

    quality = (0.80 * line_quality) + (0.20 * section_quality)
    reward = MIN_PARSABILITY_REWARD + (2.0 * quality)
    return _clamp(reward, MIN_PARSABILITY_REWARD, MAX_UNPARSEABLE_REWARD)


def _formula_line_quality(line: str) -> float:
    stripped = line.strip()
    if not stripped:
        return 0.0

    score = 0.05

    if not BAD_SYMBOL_PATTERN.search(stripped):
        score += 0.20
    if _has_balanced_parentheses(stripped):
        score += 0.20
    if PREDICATE_PATTERN.search(stripped):
        score += 0.20
    if _all_predicates_have_valid_arguments(stripped):
        score += 0.15
    if _operators_are_well_placed(stripped):
        score += 0.15
    if _quantifiers_are_well_formed(stripped):
        score += 0.05

    return _clamp(score, 0.0, 1.0)


def _formula_lines(text: str) -> list[str]:
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


def _all_predicates_have_valid_arguments(text: str) -> bool:
    predicates = list(PREDICATE_PATTERN.finditer(text))
    if not predicates:
        return False

    valid_predicates = list(VALID_PREDICATE_PATTERN.finditer(text))
    return len(predicates) == len(valid_predicates)


def _operators_are_well_placed(text: str) -> bool:
    if REPEATED_OPERATOR_PATTERN.search(text):
        return False

    stripped = text.strip()
    if BINARY_OPERATOR_PATTERN.match(stripped):
        return False
    if BINARY_OPERATOR_PATTERN.search(stripped[-1:]):
        return False
    if re.search(r"\(\s*(→|∧|∨|⊕)", stripped):
        return False
    if re.search(r"(→|∧|∨|⊕)\s*\)", stripped):
        return False

    return True


def _quantifiers_are_well_formed(text: str) -> bool:
    quantifier_symbols = text.count("∀") + text.count("∃")
    if quantifier_symbols == 0:
        return True
    return len(QUANTIFIER_PATTERN.findall(text)) == quantifier_symbols


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
