from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re


ARROW_ALIASES = ("->", "=>", "⇒")
BICONDITIONAL_ALIASES = ("⇔", "<->")
PREDICATE_CALL_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]*)\s*\(([^()\n]*)\)"
)
QUANTIFIED_VARIABLE_PATTERN = re.compile(r"(?:∀|∃)\s*([A-Za-z][A-Za-z0-9_]*)")
TOKEN_PATTERN = re.compile(r"\b[A-Za-z0-9_]+\b")
SINGLE_LETTER_VARIABLE_PATTERN = re.compile(r"^[a-z]$")


@dataclass(frozen=True)
class FormulaSchema:
    predicates: tuple[tuple[str, int], ...]
    constants: tuple[str, ...]


@dataclass(frozen=True)
class OverlapScore:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class GoldFolReward:
    reward: float
    predicate_score: OverlapScore
    constant_score: OverlapScore
    premise_score: OverlapScore
    conclusion_exact: bool


@dataclass(frozen=True)
class PostprocessResult:
    premises: str | None
    conclusion: str | None
    changed: bool
    invalid_predicates: tuple[str, ...]
    invalid_constants: tuple[str, ...]


def combined_fol(premises: str | None, conclusion: str | None) -> str:
    parts = [part.strip() for part in (premises, conclusion) if part and part.strip()]
    return "\n".join(parts)


def formula_schema(text: str | None) -> FormulaSchema:
    if not text:
        return FormulaSchema(predicates=(), constants=())

    predicate_order: list[tuple[str, int]] = []
    predicate_seen: set[tuple[str, int]] = set()
    constant_order: list[str] = []
    constant_seen: set[str] = set()
    variable_names = quantified_variables(text)

    for name, raw_args in PREDICATE_CALL_PATTERN.findall(text):
        args = split_arguments(raw_args)
        signature = (name, len(args))
        if signature not in predicate_seen:
            predicate_seen.add(signature)
            predicate_order.append(signature)

        for arg in args:
            if not _is_constant_token(arg, variable_names):
                continue
            if arg not in constant_seen:
                constant_seen.add(arg)
                constant_order.append(arg)

    return FormulaSchema(
        predicates=tuple(predicate_order),
        constants=tuple(constant_order),
    )


def quantified_variables(text: str) -> set[str]:
    variables = set(QUANTIFIED_VARIABLE_PATTERN.findall(text))
    variables.update(TOKEN_PATTERN.findall(text))
    return {
        token
        for token in variables
        if SINGLE_LETTER_VARIABLE_PATTERN.fullmatch(token)
    }


def split_arguments(raw_args: str) -> list[str]:
    return [arg.strip() for arg in raw_args.split(",") if arg.strip()]


def _is_constant_token(token: str, variable_names: set[str]) -> bool:
    if token in variable_names:
        return False
    if not re.fullmatch(r"[A-Za-z0-9_]+", token):
        return False
    return True


def predicate_signatures(text: str | None) -> Counter[str]:
    return Counter(
        format_predicate_signature(name, arity)
        for name, arity in formula_schema(text).predicates
    )


def constants(text: str | None) -> Counter[str]:
    return Counter(formula_schema(text).constants)


def format_predicate_signature(name: str, arity: int) -> str:
    return f"{name}/{arity}"


def format_schema_section(
    premises: str | None,
    conclusion: str | None,
) -> str:
    schema = formula_schema(combined_fol(premises, conclusion))
    if not schema.predicates and not schema.constants:
        return ""

    predicate_text = (
        ", ".join(
            format_predicate_signature(name, arity)
            for name, arity in schema.predicates
        )
        if schema.predicates
        else "(none)"
    )
    constant_text = ", ".join(schema.constants) if schema.constants else "(none)"
    return (
        "Allowed predicates:\n"
        f"{predicate_text}\n\n"
        "Allowed constants/literals:\n"
        f"{constant_text}"
    )


def normalize_formula_line(line: str) -> str:
    line = line.strip()
    for alias in ARROW_ALIASES:
        line = line.replace(alias, "→")
    for alias in BICONDITIONAL_ALIASES:
        line = line.replace(alias, "↔")
    line = line.replace("~", "¬")
    return re.sub(r"\s+", "", line)


def normalized_lines(block: str | None) -> list[str]:
    if block is None:
        return []
    return [
        normalized
        for line in block.splitlines()
        if (normalized := normalize_formula_line(line))
    ]


def normalized_block(block: str | None) -> str:
    return "\n".join(normalized_lines(block))


def multiset_overlap(left: Counter[str], right: Counter[str]) -> OverlapScore:
    if not left and not right:
        return OverlapScore(precision=1.0, recall=1.0, f1=1.0)
    if not left or not right:
        return OverlapScore(precision=0.0, recall=0.0, f1=0.0)

    overlap = sum((left & right).values())
    precision = overlap / sum(left.values())
    recall = overlap / sum(right.values())
    if precision + recall == 0.0:
        return OverlapScore(precision=precision, recall=recall, f1=0.0)
    return OverlapScore(
        precision=precision,
        recall=recall,
        f1=2.0 * precision * recall / (precision + recall),
    )


def line_overlap(predicted: str | None, gold: str | None) -> OverlapScore:
    return multiset_overlap(
        Counter(normalized_lines(predicted)),
        Counter(normalized_lines(gold)),
    )


def schema_overlap(
    predicted: str | None,
    gold: str | None,
) -> tuple[OverlapScore, OverlapScore]:
    return (
        multiset_overlap(predicate_signatures(predicted), predicate_signatures(gold)),
        multiset_overlap(constants(predicted), constants(gold)),
    )


def gold_fol_reward(
    predicted_premises: str | None,
    predicted_conclusion: str | None,
    gold_premises: str | None,
    gold_conclusion: str | None,
) -> GoldFolReward:
    predicted_fol = combined_fol(predicted_premises, predicted_conclusion)
    gold_fol = combined_fol(gold_premises, gold_conclusion)
    predicate_score, constant_score = schema_overlap(predicted_fol, gold_fol)
    premise_score = line_overlap(predicted_premises, gold_premises)
    conclusion_exact = normalized_block(predicted_conclusion) == normalized_block(
        gold_conclusion
    )
    reward = (
        0.30 * predicate_score.f1
        + 0.20 * constant_score.f1
        + 0.30 * premise_score.f1
        + (0.20 if conclusion_exact else 0.0)
    )
    return GoldFolReward(
        reward=reward,
        predicate_score=predicate_score,
        constant_score=constant_score,
        premise_score=premise_score,
        conclusion_exact=conclusion_exact,
    )


def postprocess_formalization(
    premises: str | None,
    conclusion: str | None,
    gold_premises: str | None,
    gold_conclusion: str | None,
) -> PostprocessResult:
    if premises is None or conclusion is None:
        return PostprocessResult(
            premises=premises,
            conclusion=conclusion,
            changed=False,
            invalid_predicates=(),
            invalid_constants=(),
        )

    gold = formula_schema(combined_fol(gold_premises, gold_conclusion))
    predicate_name_map = _predicate_name_map(gold)
    constant_map = {constant.lower(): constant for constant in gold.constants}

    processed_premises = _postprocess_block(
        premises,
        predicate_name_map=predicate_name_map,
        constant_map=constant_map,
    )
    processed_conclusion = _postprocess_block(
        conclusion,
        predicate_name_map=predicate_name_map,
        constant_map=constant_map,
    )
    processed_fol = combined_fol(processed_premises, processed_conclusion)
    invalid_predicates = _invalid_predicates(processed_fol, gold)
    invalid_constants = _invalid_constants(processed_fol, gold)

    return PostprocessResult(
        premises=processed_premises,
        conclusion=processed_conclusion,
        changed=(processed_premises != premises or processed_conclusion != conclusion),
        invalid_predicates=tuple(invalid_predicates),
        invalid_constants=tuple(invalid_constants),
    )


def schema_violations(
    premises: str | None,
    conclusion: str | None,
    gold_premises: str | None,
    gold_conclusion: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if premises is None or conclusion is None:
        return (), ()

    gold = formula_schema(combined_fol(gold_premises, gold_conclusion))
    text = combined_fol(premises, conclusion)
    return (
        tuple(_invalid_predicates(text, gold)),
        tuple(_invalid_constants(text, gold)),
    )


def _predicate_name_map(schema: FormulaSchema) -> dict[tuple[str, int], str]:
    mapping: dict[tuple[str, int], str] = {}
    for name, arity in schema.predicates:
        mapping.setdefault((name.lower(), arity), name)
    return mapping


def _postprocess_block(
    block: str,
    *,
    predicate_name_map: dict[tuple[str, int], str],
    constant_map: dict[str, str],
) -> str:
    lines = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        repaired = _trim_trailing_junk(stripped)
        repaired = _canonicalize_predicate_names(repaired, predicate_name_map)
        repaired = _canonicalize_constants(repaired, constant_map)
        lines.append(repaired)
    return "\n".join(lines).strip()


def _trim_trailing_junk(line: str) -> str:
    depth = 0
    last_balanced_close = -1
    for index, character in enumerate(line):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                last_balanced_close = index
            elif depth < 0:
                return line.strip()

    if depth != 0 or last_balanced_close < 0:
        return line.strip()

    suffix = line[last_balanced_close + 1 :].strip()
    if not suffix:
        return line.strip()
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", suffix):
        return line[: last_balanced_close + 1].strip()
    return line.strip()


def _canonicalize_predicate_names(
    line: str,
    predicate_name_map: dict[tuple[str, int], str],
) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        raw_args = match.group(2)
        canonical = predicate_name_map.get((name.lower(), len(split_arguments(raw_args))))
        if canonical is None:
            return match.group(0)
        return f"{canonical}({raw_args})"

    return PREDICATE_CALL_PATTERN.sub(replace, line)


def _canonicalize_constants(line: str, constant_map: dict[str, str]) -> str:
    variable_names = quantified_variables(line)

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in variable_names:
            return token
        next_index = match.end()
        if next_index < len(line) and line[next_index:].lstrip().startswith("("):
            return token
        return constant_map.get(token.lower(), token)

    return TOKEN_PATTERN.sub(replace, line)


def _invalid_predicates(text: str, gold: FormulaSchema) -> list[str]:
    allowed = {
        format_predicate_signature(name, arity).lower()
        for name, arity in gold.predicates
    }
    invalid = []
    seen = set()
    for name, arity in formula_schema(text).predicates:
        signature = format_predicate_signature(name, arity)
        key = signature.lower()
        if key in allowed or key in seen:
            continue
        seen.add(key)
        invalid.append(signature)
    return invalid


def _invalid_constants(text: str, gold: FormulaSchema) -> list[str]:
    allowed = {constant.lower() for constant in gold.constants}
    invalid = []
    seen = set()
    for constant in formula_schema(text).constants:
        key = constant.lower()
        if key in allowed or key in seen:
            continue
        seen.add(key)
        invalid.append(constant)
    return invalid
