"""Build paired question/Label samples for definition ablation experiments."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            yield value


def stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def load_labels(path: str | Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["label_id"]): row
        for row in read_jsonl(path)
        if row.get("label_id")
    }


def load_positive_results(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        question_id = str(row.get("question_id") or "")
        label_id = str(row.get("label_id") or "")
        if question_id and label_id:
            result[(question_id, label_id)] = row
    return result


def compute_match_rates(
    positive_results: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, dict[str, int]], dict[str, float]]:
    counts: dict[str, Counter[str]] = {}
    for (_question_id, label_id), row in positive_results.items():
        counter = counts.setdefault(label_id, Counter())
        counter["total"] += 1
        counter["match"] += int(bool(row.get("match")))
    rates = {
        label_id: counter["match"] / counter["total"]
        for label_id, counter in counts.items()
        if counter["total"]
    }
    return {label_id: dict(counter) for label_id, counter in counts.items()}, rates


def load_volatile_question_ids(group_paths: Iterable[str | Path]) -> set[str]:
    question_ids: set[str] = set()
    for path in group_paths:
        for row in read_jsonl(path):
            if row.get("same_output") is False and row.get("question_id"):
                question_ids.add(str(row["question_id"]))
    return question_ids


def build_definition_ablation_sample(
    *,
    positive_samples_path: str | Path,
    positive_results_path: str | Path,
    labels_path: str | Path,
    volatile_group_paths: Iterable[str | Path],
    low_threshold: float = 0.70,
    low_count: int = 10,
    high_count: int = 5,
    seed: str = "definition-ablation-v1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 <= low_threshold <= 1.0:
        raise ValueError("low_threshold must be between 0 and 1")
    if low_count < 1 or high_count < 1:
        raise ValueError("sample counts must be positive")

    labels = load_labels(labels_path)
    positive_results = load_positive_results(positive_results_path)
    result_counts, match_rates = compute_match_rates(positive_results)
    volatile_ids = load_volatile_question_ids(volatile_group_paths)

    all_by_label: dict[str, list[dict[str, Any]]] = {}
    volatile_by_label: dict[str, list[dict[str, Any]]] = {}
    scanned = 0
    for row in read_jsonl(positive_samples_path):
        scanned += 1
        label_id = str(row.get("label_id") or "")
        question_id = str(row.get("question_id") or "")
        if label_id not in labels or not question_id:
            continue
        item = {
            "question_id": question_id,
            "label_id": label_id,
            "question": {
                "unit_type": row.get("unit_type", ""),
                "parent_stem": row.get("parent_stem", ""),
                "stem": row.get("stem", ""),
                "options": row.get("options", ""),
                "answer_text": row.get("answer_text", ""),
                "analysis": row.get("analysis", ""),
            },
            "historical_result": positive_results.get((question_id, label_id)),
            "image_context_missing": bool(
                (row.get("flags") or {}).get("image_context_missing")
            ),
        }
        all_by_label.setdefault(label_id, []).append(item)
        if question_id in volatile_ids:
            volatile_by_label.setdefault(label_id, []).append(item)

    selected: list[dict[str, Any]] = []
    per_label: list[dict[str, Any]] = []
    for label_id in sorted(labels):
        rate = match_rates.get(label_id)
        stratum = "low" if rate is not None and rate < low_threshold else "high"
        requested = low_count if stratum == "low" else high_count
        volatile = sorted(
            volatile_by_label.get(label_id, []),
            key=lambda item: stable_key(seed, f"volatile:{item['question_id']}"),
        )
        fallback = sorted(
            all_by_label.get(label_id, []),
            key=lambda item: stable_key(seed, f"all:{item['question_id']}"),
        )
        chosen: list[dict[str, Any]] = []
        chosen_ids: set[str] = set()
        for pool_name, pool in (("volatile_5391", volatile), ("positive_sample_fallback", fallback)):
            for item in pool:
                if len(chosen) >= requested:
                    break
                question_id = item["question_id"]
                if question_id in chosen_ids:
                    continue
                copied = dict(item)
                copied["sample_source"] = pool_name
                chosen.append(copied)
                chosen_ids.add(question_id)
        for item in chosen:
            label = labels[label_id]
            pair = {
                "pair_id": f"{item['question_id']}::{label_id}",
                "question_id": item["question_id"],
                "label_id": label_id,
                "label_name": str(label.get("label_name") or ""),
                "label_path": str(label.get("label_path") or ""),
                "definition": str(label.get("definition") or ""),
                "core_concepts": str(label.get("core_concepts") or ""),
                "common_assessments": str(label.get("common_assessments") or ""),
                "distinctions": str(label.get("distinctions") or ""),
                "match_rate": rate,
                "match_rate_count": result_counts.get(label_id, {}),
                "stratum": stratum,
                "requested_count": requested,
                "sample_source": item["sample_source"],
                "question": item["question"],
                "historical_result": item.get("historical_result"),
                "image_context_missing": item["image_context_missing"],
            }
            selected.append(pair)
        per_label.append(
            {
                "label_id": label_id,
                "label_name": str(labels[label_id].get("label_name") or ""),
                "match_rate": rate,
                "stratum": stratum,
                "requested": requested,
                "selected": len(chosen),
                "volatile_available": len(volatile),
                "fallback_available": len(fallback),
                "source_counts": dict(Counter(item["sample_source"] for item in chosen)),
            }
        )

    report = {
        "experiment": "definition-ablation-sample-v1",
        "labels": len(labels),
        "positive_samples_scanned": scanned,
        "positive_result_pairs": len(positive_results),
        "volatile_question_ids": len(volatile_ids),
        "low_threshold": low_threshold,
        "low_count": low_count,
        "high_count": high_count,
        "selected_pairs": len(selected),
        "labels_with_selected_pairs": sum(item["selected"] > 0 for item in per_label),
        "labels_without_selected_pairs": sum(item["selected"] == 0 for item in per_label),
        "selected_by_stratum": dict(Counter(item["stratum"] for item in per_label for _ in range(item["selected"]))),
        "selected_by_source": dict(Counter(item["sample_source"] for item in selected)),
        "seed": seed,
        "per_label": per_label,
    }
    return selected, report
