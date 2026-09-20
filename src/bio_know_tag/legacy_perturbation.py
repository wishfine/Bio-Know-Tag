"""Partition paired adjudication results by legacy-candidate perturbation type."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


GROUP_A = "A_same_candidate_set"
GROUP_B = "B_candidate_set_expanded_added_not_selected"
GROUP_C = "C_added_legacy_selected"
GROUPS = (GROUP_A, GROUP_B, GROUP_C)


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield value


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_latest_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load one latest successful prediction per question from common formats."""
    rows: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        question_id = str(row.get("question_id") or "")
        if not question_id:
            continue
        if row.get("error"):
            continue
        if "parsed_response" in row and not isinstance(
            row.get("parsed_response"), dict
        ):
            continue
        rows[question_id] = row
    return rows


def _selected_ids(row: dict[str, Any]) -> set[str]:
    selected_labels = row.get("selected_labels")
    if isinstance(selected_labels, list):
        return {
            str(label.get("label_id") or "")
            for label in selected_labels
            if isinstance(label, dict) and label.get("label_id")
        }

    prediction = row.get("prediction")
    if isinstance(prediction, dict) and isinstance(
        prediction.get("selected_labels"), list
    ):
        return _selected_ids(prediction)

    parsed = row.get("parsed_response")
    code_map = row.get("candidate_code_map")
    if isinstance(parsed, dict) and isinstance(code_map, dict):
        return {
            str(code_map[code])
            for code in (parsed.get("selected") or [])
            if code in code_map
        }
    raise ValueError(
        f"prediction for {row.get('question_id')} has no supported selected-label format"
    )


def _candidate_ids(row: dict[str, Any]) -> list[str]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"candidate row {row.get('question_id')} has no candidates")
    ids = [
        str(candidate.get("label_id") or "")
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("label_id")
    ]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate candidate IDs for {row.get('question_id')}")
    return ids


def _load_labels(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {
        str(row["label_id"]): row
        for row in _read_jsonl(path)
        if row.get("label_id")
    }


def _load_luna_reviews(
    path: str | Path | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    return {
        (str(row.get("question_id") or ""), str(row.get("label_id") or "")): row
        for row in _read_jsonl(path)
        if row.get("question_id") and row.get("label_id")
    }


def _label_card(label_id: str, labels: dict[str, dict[str, Any]]) -> dict[str, str]:
    label = labels.get(label_id, {})
    return {
        "label_id": label_id,
        "label_name": str(label.get("label_name") or ""),
        "label_path": str(label.get("label_path") or ""),
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _new_group_stats() -> dict[str, Any]:
    return {
        "questions": 0,
        "same_output": 0,
        "different_output": 0,
        "both_empty": 0,
        "top25_empty_legacy_nonempty": 0,
        "top25_nonempty_legacy_empty": 0,
        "top25_assignments": 0,
        "legacy_assignments": 0,
        "shared_assignments": 0,
        "legacy_only_assignments": 0,
        "top25_only_assignments": 0,
        "added_candidate_count": 0,
        "selected_added_legacy_count": 0,
        "jaccard_sum": 0.0,
    }


def analyze_legacy_candidate_perturbation(
    *,
    top25_candidates_path: str | Path,
    legacy_candidates_path: str | Path,
    top25_predictions_path: str | Path,
    legacy_predictions_path: str | Path,
    run_dir: str | Path,
    labels_path: str | Path | None = None,
    luna_reviews_path: str | Path | None = None,
    progress_every: int = 10_000,
) -> dict[str, Any]:
    """Split paired questions into identical, perturbed, and selected-legacy groups."""
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = output_dir / "groups"
    groups_dir.mkdir(exist_ok=True)

    top25_predictions = _load_latest_rows(top25_predictions_path)
    legacy_predictions = _load_latest_rows(legacy_predictions_path)
    common_ids = set(top25_predictions) & set(legacy_predictions)
    labels = _load_labels(labels_path)
    luna_reviews = _load_luna_reviews(luna_reviews_path)

    group_stats = {group: _new_group_stats() for group in GROUPS}
    group_outputs = {
        group: (groups_dir / f"{group}.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        )
        for group in GROUPS
    }
    per_question_path = output_dir / "per_question.jsonl"
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    added_label_counts: Counter[str] = Counter()
    selected_added_label_counts: Counter[str] = Counter()
    luna_by_group: dict[str, Counter[str]] = {
        group: Counter() for group in GROUPS
    }
    scanned = paired = 0

    try:
        with per_question_path.open("w", encoding="utf-8", newline="\n") as output:
            top25_candidates = _read_jsonl(top25_candidates_path)
            legacy_candidates = _read_jsonl(legacy_candidates_path)
            for base_row, augmented_row in itertools.zip_longest(
                top25_candidates, legacy_candidates
            ):
                if base_row is None or augmented_row is None:
                    raise ValueError("candidate files must contain the same number of rows")
                scanned += 1
                question_id = str(base_row.get("question_id") or "")
                if question_id != str(augmented_row.get("question_id") or ""):
                    raise ValueError(
                        f"candidate order mismatch at row {scanned}: {question_id}"
                    )
                if question_id not in common_ids:
                    continue

                paired += 1
                base_ids = set(_candidate_ids(base_row))
                augmented_ids = set(_candidate_ids(augmented_row))
                removed_candidate_ids = base_ids - augmented_ids
                if removed_candidate_ids:
                    raise ValueError(
                        f"legacy candidates removed base labels for {question_id}: "
                        f"{sorted(removed_candidate_ids)}"
                    )
                added_ids = augmented_ids - base_ids
                top25_selected = _selected_ids(top25_predictions[question_id])
                legacy_selected = _selected_ids(legacy_predictions[question_id])
                selected_added = legacy_selected & added_ids

                if not added_ids:
                    group = GROUP_A
                elif selected_added:
                    group = GROUP_C
                else:
                    group = GROUP_B

                legacy_only = legacy_selected - top25_selected
                top25_only = top25_selected - legacy_selected
                shared = top25_selected & legacy_selected
                luna_selected_added = []
                for label_id in sorted(selected_added):
                    review = luna_reviews.get((question_id, label_id))
                    if review:
                        luna_selected_added.append(review)
                        luna_by_group[group][str(review.get("decision") or "missing")] += 1

                row = {
                    "question_id": question_id,
                    "perturbation_group": group,
                    "base_candidate_count": len(base_ids),
                    "augmented_candidate_count": len(augmented_ids),
                    "added_candidate_ids": sorted(added_ids),
                    "added_candidates": [
                        _label_card(label_id, labels) for label_id in sorted(added_ids)
                    ],
                    "top25_selected_label_ids": sorted(top25_selected),
                    "legacy_selected_label_ids": sorted(legacy_selected),
                    "shared_selected_label_ids": sorted(shared),
                    "legacy_only_selected_label_ids": sorted(legacy_only),
                    "top25_only_selected_label_ids": sorted(top25_only),
                    "selected_added_legacy_ids": sorted(selected_added),
                    "same_output": top25_selected == legacy_selected,
                    "selection_jaccard": round(
                        _jaccard(top25_selected, legacy_selected), 6
                    ),
                    "luna_selected_added_reviews": luna_selected_added,
                }
                serialized = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                output.write(serialized)
                group_outputs[group].write(serialized)

                stats = group_stats[group]
                stats["questions"] += 1
                stats["same_output"] += int(top25_selected == legacy_selected)
                stats["different_output"] += int(top25_selected != legacy_selected)
                stats["both_empty"] += int(not top25_selected and not legacy_selected)
                stats["top25_empty_legacy_nonempty"] += int(
                    not top25_selected and bool(legacy_selected)
                )
                stats["top25_nonempty_legacy_empty"] += int(
                    bool(top25_selected) and not legacy_selected
                )
                stats["top25_assignments"] += len(top25_selected)
                stats["legacy_assignments"] += len(legacy_selected)
                stats["shared_assignments"] += len(shared)
                stats["legacy_only_assignments"] += len(legacy_only)
                stats["top25_only_assignments"] += len(top25_only)
                stats["added_candidate_count"] += len(added_ids)
                stats["selected_added_legacy_count"] += len(selected_added)
                stats["jaccard_sum"] += _jaccard(top25_selected, legacy_selected)

                for label_id in shared:
                    per_label[label_id][f"{group}:both_selected"] += 1
                for label_id in legacy_only:
                    per_label[label_id][f"{group}:legacy_only_selected"] += 1
                for label_id in top25_only:
                    per_label[label_id][f"{group}:top25_only_selected"] += 1
                for label_id in added_ids:
                    added_label_counts[label_id] += 1
                for label_id in selected_added:
                    selected_added_label_counts[label_id] += 1

                if progress_every and paired % progress_every == 0:
                    print(
                        f"legacy perturbation: candidate_rows={scanned}, "
                        f"paired_questions={paired}"
                    )
    finally:
        for handle in group_outputs.values():
            handle.close()

    if paired != len(common_ids):
        missing = common_ids - {
            json.loads(line)["question_id"]
            for line in per_question_path.open(encoding="utf-8")
            if line.strip()
        }
        raise ValueError(
            f"candidate files did not cover {len(missing)} common prediction questions"
        )

    finalized_groups: dict[str, dict[str, Any]] = {}
    for group, raw_stats in group_stats.items():
        stats = dict(raw_stats)
        questions = int(stats["questions"])
        stats["output_difference_rate"] = round(
            stats["different_output"] / questions, 6
        ) if questions else None
        stats["mean_selection_jaccard"] = round(
            stats.pop("jaccard_sum") / questions, 6
        ) if questions else None
        stats["mean_added_candidates"] = round(
            stats["added_candidate_count"] / questions, 6
        ) if questions else None
        stats["luna_selected_added_decisions"] = dict(luna_by_group[group])
        finalized_groups[group] = stats

    with (output_dir / "per_label.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        all_label_ids = set(per_label) | set(added_label_counts)
        for label_id in sorted(all_label_ids):
            row = _label_card(label_id, labels)
            row["counts"] = dict(sorted(per_label[label_id].items()))
            row["added_as_candidate"] = added_label_counts[label_id]
            row["selected_after_legacy_addition"] = selected_added_label_counts[
                label_id
            ]
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "candidate_rows_scanned": scanned,
        "top25_prediction_questions": len(top25_predictions),
        "legacy_prediction_questions": len(legacy_predictions),
        "common_prediction_questions": len(common_ids),
        "paired_questions": paired,
        "groups": finalized_groups,
        "input_sha256": {
            "top25_candidates": _file_sha256(top25_candidates_path),
            "legacy_candidates": _file_sha256(legacy_candidates_path),
            "top25_predictions": _file_sha256(top25_predictions_path),
            "legacy_predictions": _file_sha256(legacy_predictions_path),
            **(
                {"labels": _file_sha256(labels_path)} if labels_path else {}
            ),
            **(
                {"luna_reviews": _file_sha256(luna_reviews_path)}
                if luna_reviews_path
                else {}
            ),
        },
        "interpretation": {
            GROUP_A: "Candidate IDs are identical; output differences measure same-prompt model/service instability.",
            GROUP_B: "Legacy IDs expanded the prompt but no added legacy candidate was selected; output differences measure candidate-set perturbation plus baseline instability.",
            GROUP_C: "At least one newly added legacy candidate was selected; audit these labels for useful recall versus historical noise.",
        },
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report

