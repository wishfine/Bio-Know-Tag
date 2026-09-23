#!/usr/bin/env python3
"""Compare running Qwen adjudication with Luna's reviewed question–Label pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


POSITIVE_DECISIONS = {"DIRECT_MATCH", "REASONABLE_CO_LABEL"}
NEGATIVE_DECISION = "WRONG_LABEL"
UNRESOLVED_DECISION = "INSUFFICIENT_CONTEXT"


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"expected object in {path}:{line_number}")
                yield value


def load_luna(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    reviews: dict[tuple[str, str], dict[str, Any]] = {}
    valid = POSITIVE_DECISIONS | {NEGATIVE_DECISION, UNRESOLVED_DECISION}
    for row in read_jsonl(path):
        key = (str(row.get("question_id") or ""), str(row.get("label_id") or ""))
        if not all(key) or row.get("decision") not in valid:
            raise ValueError(f"invalid Luna review: {key}")
        if key in reviews:
            raise ValueError(f"duplicate Luna review: {key}")
        reviews[key] = row
    return reviews


def load_latest_success(
    path: Path, question_ids: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    latest: dict[str, dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            stats["evidence_rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed_lines"] += 1
                continue
            question_id = str(row.get("question_id") or "")
            if question_id not in question_ids:
                continue
            if row.get("error") or not isinstance(row.get("parsed_response"), dict):
                stats["failed_review_question_rows"] += 1
                continue
            if not isinstance(row.get("candidate_code_map"), dict):
                raise ValueError(f"missing candidate_code_map at {path}:{line_number}")
            latest[question_id] = row
            stats["successful_review_question_rows"] += 1
    stats["unique_review_questions_with_success"] = len(latest)
    return latest, dict(stats)


def candidate_and_selected(record: dict[str, Any]) -> tuple[set[str], set[str]]:
    code_map = record["candidate_code_map"]
    candidate_ids = {str(value) for value in code_map.values()}
    selected_ids = {
        str(code_map[code])
        for code in record["parsed_response"].get("selected") or []
        if code in code_map
    }
    return candidate_ids, selected_ids


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def summarize_model(counts: Counter[str]) -> dict[str, Any]:
    return {
        **dict(counts),
        "reviewed_selection_precision": rate(
            counts["positive_selected"],
            counts["positive_selected"] + counts["wrong_selected"],
        ),
        "positive_selection_rate_given_candidate": rate(
            counts["positive_selected"], counts["positive_candidate_present"]
        ),
        "wrong_selection_rate_given_candidate": rate(
            counts["wrong_selected"], counts["wrong_candidate_present"]
        ),
    }


def compare(
    luna_path: Path,
    top25_evidence_path: Path,
    legacy_evidence_path: Path,
    labels_path: Path,
    units_path: Path,
    output_dir: Path,
    *,
    sample_per_category: int = 4,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reviews = load_luna(luna_path)
    question_ids = {question_id for question_id, _ in reviews}
    top_rows, top_input_stats = load_latest_success(top25_evidence_path, question_ids)
    legacy_rows, legacy_input_stats = load_latest_success(legacy_evidence_path, question_ids)
    labels = {
        str(row["label_id"]): str(row.get("label_name") or "")
        for row in read_jsonl(labels_path)
    }
    decision_counts = Counter(row["decision"] for row in reviews.values())
    pair_counts: Counter[str] = Counter()
    model_counts = {"top25": Counter(), "top25_plus_legacy": Counter()}
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    sample_candidates: dict[str, list[tuple[Any, ...]]] = defaultdict(list)

    for (question_id, label_id), review in reviews.items():
        top = top_rows.get(question_id)
        legacy = legacy_rows.get(question_id)
        if top is None or legacy is None:
            pair_counts["not_paired"] += 1
            continue
        pair_counts["paired_reviews"] += 1
        decision = str(review["decision"])
        if decision == UNRESOLVED_DECISION:
            pair_counts["paired_insufficient_context"] += 1
            continue

        top_candidates, top_selected = candidate_and_selected(top)
        legacy_candidates, legacy_selected = candidate_and_selected(legacy)
        if not top_candidates <= legacy_candidates:
            raise ValueError(f"legacy candidates removed a Top25 Label for {question_id}")
        positive = decision in POSITIVE_DECISIONS
        top_available = label_id in top_candidates
        legacy_available = label_id in legacy_candidates
        top_hit = label_id in top_selected
        legacy_hit = label_id in legacy_selected
        added_target = not top_available and legacy_available

        pair_counts["paired_positive" if positive else "paired_wrong"] += 1
        pair_counts["same_candidate_set" if top_candidates == legacy_candidates else "expanded_candidate_set"] += 1
        if top_hit and legacy_hit:
            outcome = "both_selected"
        elif top_hit:
            outcome = "top25_only_selected"
        elif legacy_hit:
            outcome = "legacy_only_selected"
        else:
            outcome = "neither_selected"
        pair_counts[f"{decision}:{outcome}"] += 1
        per_label[label_id][f"{decision}:{outcome}"] += 1

        for name, available, hit in (
            ("top25", top_available, top_hit),
            ("top25_plus_legacy", legacy_available, legacy_hit),
        ):
            counts = model_counts[name]
            counts["reviewed_pairs"] += 1
            if positive:
                counts["positive_pairs"] += 1
                counts["positive_candidate_present"] += int(available)
                counts["positive_missing_from_candidates"] += int(not available)
                counts["positive_selected"] += int(hit)
                counts["positive_missed_with_candidate"] += int(available and not hit)
            else:
                counts["wrong_pairs"] += 1
                counts["wrong_candidate_present"] += int(available)
                counts["wrong_selected"] += int(hit)
                counts["wrong_rejected_with_candidate"] += int(available and not hit)
            if positive:
                status = "selected_valid" if hit else (
                    "missed_valid" if available else "unavailable_valid"
                )
            else:
                status = "selected_wrong" if hit else (
                    "rejected_wrong" if available else "unavailable_wrong"
                )
            per_label[label_id][f"{name}:{status}"] += 1

        if added_target and legacy_hit:
            pair_counts["added_legacy_target_selected_positive" if positive else
                        "added_legacy_target_selected_wrong"] += 1
        if positive:
            if top_hit and not legacy_hit:
                category = "valid_lost_after_legacy"
            elif legacy_hit and not top_hit:
                category = "valid_gained_after_legacy"
            elif not top_hit and not legacy_hit and top_available:
                category = "valid_missed_by_both"
            else:
                category = "valid_selected_by_both" if top_hit and legacy_hit else "other_valid"
        else:
            if top_hit and legacy_hit:
                category = "wrong_selected_by_both"
            elif top_hit:
                category = "wrong_selected_only_top25"
            elif legacy_hit:
                category = "wrong_selected_only_legacy"
            else:
                category = "wrong_rejected_by_both"
        if added_target and legacy_hit:
            sample_candidates["added_legacy_valid" if positive else "added_legacy_wrong"].append(
                (question_id, label_id, review, top, legacy, category)
            )
        sample_candidates[category].append((question_id, label_id, review, top, legacy, category))

    selected_samples = []
    for category, candidates in sorted(sample_candidates.items()):
        if category in ("valid_selected_by_both", "wrong_rejected_by_both", "other_valid"):
            continue
        ordered = sorted(
            candidates,
            key=lambda item: hashlib.sha256(
                f"{category}:{item[0]}:{item[1]}".encode("utf-8")
            ).digest(),
        )
        for question_id, label_id, review, top, legacy, _ in ordered[:sample_per_category]:
            top_candidates, top_selected = candidate_and_selected(top)
            legacy_candidates, legacy_selected = candidate_and_selected(legacy)
            selected_samples.append({
                "category": category,
                "question_id": question_id,
                "label_id": label_id,
                "label_name": labels.get(label_id, ""),
                "luna_decision": review["decision"],
                "luna_reason": review.get("reason"),
                "top25_candidate_present": label_id in top_candidates,
                "top25_selected": label_id in top_selected,
                "legacy_candidate_present": label_id in legacy_candidates,
                "legacy_selected": label_id in legacy_selected,
                "top25_reason": (top.get("parsed_response") or {}).get("reason"),
                "legacy_reason": (legacy.get("parsed_response") or {}).get("reason"),
            })
    samples_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in selected_samples:
        samples_by_question[sample["question_id"]].append(sample)
    for unit in read_jsonl(units_path):
        for sample in samples_by_question.get(str(unit.get("question_id") or ""), []):
            for field, limit in (
                ("parent_stem", 800), ("stem", 1200), ("options", 800),
                ("answer_text", 300), ("analysis", 1600),
            ):
                sample[field] = str(unit.get(field) or "")[:limit]

    report = {
        "luna_audit_pairs": len(reviews),
        "luna_unique_questions": len(question_ids),
        "luna_decision_counts": dict(decision_counts),
        "top25_evidence": top_input_stats,
        "legacy_evidence": legacy_input_stats,
        "paired_question_count": len(set(top_rows) & set(legacy_rows)),
        "pair_counts": dict(pair_counts),
        "models": {
            name: summarize_model(counts)
            for name, counts in model_counts.items()
        },
        "sample_category_counts": {
            category: len(candidates)
            for category, candidates in sample_candidates.items()
        },
        "interpretation_limit": (
            "Luna rows are a DS-risk-stratified subset and not exhaustive gold labels. "
            "Rates describe only reviewed question–Label pairs with both Qwen results. "
            "DIRECT_MATCH and REASONABLE_CO_LABEL count as acceptable; "
            "INSUFFICIENT_CONTEXT is excluded from quality metrics."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_label.jsonl").open("w", encoding="utf-8") as handle:
        for label_id, counts in sorted(per_label.items()):
            handle.write(json.dumps({
                "label_id": label_id,
                "label_name": labels.get(label_id, ""),
                "counts": dict(counts),
            }, ensure_ascii=False, sort_keys=True) + "\n")
    with (output_dir / "review_samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in selected_samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--luna-reviews", type=Path, required=True)
    parser.add_argument("--top25-evidence", type=Path, required=True)
    parser.add_argument("--legacy-evidence", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-per-category", type=int, default=4)
    args = parser.parse_args()
    if args.sample_per_category < 0:
        parser.error("--sample-per-category must be non-negative")
    report = compare(
        args.luna_reviews,
        args.top25_evidence,
        args.legacy_evidence,
        args.labels,
        args.units,
        args.run_dir,
        sample_per_category=args.sample_per_category,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
