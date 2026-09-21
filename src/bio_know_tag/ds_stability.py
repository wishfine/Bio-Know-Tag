"""Utilities for measuring repeated DeepSeek adjudication stability."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from bio_know_tag.adjudication import (
    build_adjudication_prompt,
    validate_adjudication_result,
)
from bio_know_tag.ds import parse_json_content


@dataclass(frozen=True)
class StabilityCondition:
    """One API condition in ``name:temperature:n:workers[:seed]`` form."""

    name: str
    temperature: float
    n: int
    workers: int
    seed: int | None = None


def parse_condition(value: str) -> StabilityCondition:
    """Parse a condition specification used by the CLI."""
    parts = value.split(":")
    if len(parts) not in (4, 5):
        raise ValueError(
            "condition must be name:temperature:n:workers[:seed]"
        )
    name = parts[0].strip()
    if not name:
        raise ValueError("condition name must not be empty")
    try:
        temperature = float(parts[1])
        n = int(parts[2])
        workers = int(parts[3])
        seed = int(parts[4]) if len(parts) == 5 and parts[4] else None
    except ValueError as exc:
        raise ValueError(f"invalid condition: {value}") from exc
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if n < 1 or workers < 1:
        raise ValueError("n and workers must be positive")
    return StabilityCondition(name, temperature, n, workers, seed)


def default_conditions() -> tuple[StabilityCondition, ...]:
    """The four service-level checks proposed for the DS volatility audit."""
    return (
        StabilityCondition("temp0-workers1", 0.0, 1, 1),
        StabilityCondition("temp0-workers10", 0.0, 1, 10),
        StabilityCondition("temp0-workers30", 0.0, 1, 30),
        StabilityCondition("temp0-seed42", 0.0, 1, 1, 42),
    )


def selected_ids_from_parsed(
    parsed: dict[str, Any], code_map: dict[str, str]
) -> set[str]:
    """Map validated short codes to Label IDs."""
    return {
        str(code_map[code])
        for code in parsed.get("selected", [])
        if code in code_map
    }


def parse_choice(
    content: str,
    code_map: dict[str, str],
) -> tuple[dict[str, Any] | None, set[str], str | None]:
    """Parse and validate one DS choice without hiding malformed output."""
    try:
        parsed = validate_adjudication_result(
            parse_json_content(content), set(code_map)
        )
    except Exception as exc:  # keep the raw response for later audit
        return None, set(), f"{type(exc).__name__}: {exc}"
    return parsed, selected_ids_from_parsed(parsed, code_map), None


def strict_majority_ids(choice_ids: Iterable[set[str]]) -> set[str]:
    """Return Labels selected by more than half of the parsed choices."""
    values = list(choice_ids)
    if not values:
        return set()
    threshold = len(values) // 2 + 1
    counts: dict[str, int] = {}
    for selected in values:
        for label_id in selected:
            counts[label_id] = counts.get(label_id, 0) + 1
    return {label_id for label_id, count in counts.items() if count >= threshold}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def stable_row_key(seed: int, question_id: str) -> str:
    """Stable sampling key independent of Python's randomized hash function."""
    return hashlib.sha256(f"{seed}\0{question_id}".encode()).hexdigest()


def choose_unstable_rows(
    group_paths: Iterable[str | Path],
    *,
    limit: int | None = None,
    seed: int = 20260921,
) -> list[dict[str, Any]]:
    """Load previous A/B/C rows whose output changed and choose a stable subset."""
    rows: dict[str, dict[str, Any]] = {}
    for path in group_paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                question_id = str(row.get("question_id") or "")
                if not question_id or row.get("same_output") is not False:
                    continue
                if question_id in rows:
                    raise ValueError(
                        f"duplicate unstable question {question_id} at {path}:{line_number}"
                    )
                rows[question_id] = row
    values = list(rows.values())
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if limit is not None and len(values) > limit:
        values.sort(key=lambda row: stable_row_key(seed, str(row["question_id"])))
        values = values[:limit]
    values.sort(key=lambda row: str(row["question_id"]))
    return values


def build_task_payload(
    unit: dict[str, Any],
    candidate_row: dict[str, Any],
    labels_by_id: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, str]]:
    """Build exactly the same adjudication prompt used in production."""
    candidates = candidate_row.get("candidates") or []
    return build_adjudication_prompt(unit, candidates, labels_by_id)
