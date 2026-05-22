from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from REWARDS.fol_schema import format_schema_section
except ModuleNotFoundError:
    from src.REWARDS.fol_schema import format_schema_section


VALID_LABELS = {"true", "false", "uncertain"}
TRAINER_KIND_GRPO = "grpo"
TRAINER_KIND_SFT = "sft"
VALID_TRAINER_KINDS = {TRAINER_KIND_GRPO, TRAINER_KIND_SFT}


def normalize_label(value: str) -> str:
    return value.strip().lower()


def _gold_value(example: dict[str, Any], hyphen_key: str, snake_key: str) -> str | None:
    value = example.get(hyphen_key)
    if value is None:
        value = example.get(snake_key)
    if value is None:
        return None
    return str(value).strip()


def build_schema_section(example: dict[str, Any]) -> str:
    return format_schema_section(
        _gold_value(example, "premises-FOL", "premises_fol_gold"),
        _gold_value(example, "conclusion-FOL", "conclusion_fol_gold"),
    )


def build_prompt(example: dict[str, Any]) -> str:
    problem_schema = build_schema_section(example)
    schema_prompt = (
        "Problem schema:\n"
        f"{problem_schema}\n\n"
        if problem_schema
        else ""
    )
    return (
        "Translate natural-language premises and conclusion into first-order logic.\n"
        "Return only the formalization. No explanation. No truth label.\n\n"
        "Completion format:\n"
        "Premises:\n"
        "<one complete FOL premise per line>\n\n"
        "Conclusion:\n"
        "<one complete FOL conclusion>\n\n"
        "Formula rules:\n"
        "Use only these logical operators: ∀, ∃, ¬, →, ∧, ∨, ⊕, ↔.\n"
        "Operator meanings: ∀ means for all; ∃ means there exists; ¬ means not; "
        "→ means implies; ∧ means and; ∨ means inclusive or; ⊕ means exclusive or.\n"
        "↔ means if and only if.\n"
        "Do not use ⇔, ⇒, ∴, bullets, markdown, quotes, or extra labels.\n"
        "Do not write chat-role words such as user, assistant, or system.\n"
        "Predicate, variable, and constant names must use English letters, "
        "digits, or underscores.\n"
        "Every line must be a complete formula with balanced parentheses.\n"
        "Use commas only inside predicate arguments, never between formulas; use ∧ for and.\n"
        "Equality and inequality may be used as = and ≠ when needed.\n"
        "When a problem schema is provided, use only its predicate names, arities, "
        "and constants/literals; introduce variables as needed.\n"
        "If a phrase combines concepts that appear separately in the schema, "
        "write a conjunction of those predicates instead of inventing a compound predicate.\n"
        "Stop immediately after the conclusion formula.\n\n"
        "Example schema:\n"
        "Allowed predicates:\n"
        "Eel/1, Fish/1, Plant/1, DisplayedIn/2, Animal/1, Multicellular/1, Bacteria/1\n\n"
        "Allowed constants/literals:\n"
        "collection, seaEel\n\n"
        "Example natural-language premises:\n"
        "All eels are fish.\n"
        "No fish are plants.\n"
        "Everything displayed in the collection is either a plant or an animal.\n"
        "All multicellular animals are not bacteria.\n"
        "All animals displayed in the collection are multicellular.\n"
        "A sea eel is displayed in the collection.\n"
        "The sea eel is an eel or an animal or not a plant.\n\n"
        "Example natural-language conclusion:\n"
        "The sea eel is an eel.\n\n"
        "Correct formalization for the example:\n"
        "Premises:\n"
        "∀x (Eel(x) → Fish(x))\n"
        "∀x (Fish(x) → ¬Plant(x))\n"
        "∀x (DisplayedIn(x, collection) → Plant(x) ⊕ Animal(x))\n"
        "∀x (Multicellular(x) ∧ Animal(x) → ¬Bacteria(x))\n"
        "∀x (DisplayedIn(x, collection) ∧ Animal(x) → Multicellular(x))\n"
        "DisplayedIn(seaEel, collection)\n"
        "Eel(seaEel) ∨ Animal(seaEel) ∨ ¬Plant(seaEel)\n\n"
        "Conclusion:\n"
        "Eel(seaEel)\n\n"
        f"{schema_prompt}"
        "Problem natural-language premises:\n"
        f"{example['premises']}\n\n"
        "Problem natural-language conclusion:\n"
        f"{example['conclusion']}\n\n"
        "Formalization:\n"
        "Premises:\n"
    )


def build_sft_completion(example: dict[str, Any]) -> str:
    return (
        f"{str(example['premises-FOL']).strip()}\n\n"
        "Conclusion:\n"
        f"{str(example['conclusion-FOL']).strip()}"
    )


def prepare_dataset(path: str, trainer_kind: str):
    from datasets import load_dataset

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    raw = load_dataset("json", data_files=str(source_path), split="train")

    def _map_row(row: dict[str, Any]) -> dict[str, str]:
        label = normalize_label(str(row["label"]))
        if label not in VALID_LABELS:
            raise ValueError(
                f"Unsupported label '{row['label']}'. "
                f"Expected one of: {sorted(VALID_LABELS)}"
            )
        mapped_row = {
            "prompt": build_prompt(row),
            "solution": normalize_label(str(row["label"])),
            "premises": row["premises"],
            "conclusion": row["conclusion"],
            "premises_fol_gold": row["premises-FOL"],
            "conclusion_fol_gold": row["conclusion-FOL"],
            "example_id": str(row.get("example_id", "")),
        }
        if trainer_kind == TRAINER_KIND_SFT:
            mapped_row["completion"] = build_sft_completion(row)
        return mapped_row

    return raw.map(_map_row, remove_columns=raw.column_names)
