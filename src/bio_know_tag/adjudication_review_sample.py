"""Build a risk-stratified human review sample from two adjudication runs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HIGH_POSITIVE_LEVELS = {
    "P0_图谱冲突",
    "P0_明显异常",
    "P1_重点核验",
    "L0_极端长尾",
    "L1_长尾异常",
}
MEDIUM_POSITIVE_LEVELS = {"P2_边界观察", "L2_长尾待核"}
HIGH_BOUNDARY_SCREENS = {
    "C_NARROW_OR_LEGACY_NOISE_REVIEW",
    "D_BROAD_BOUNDARY_REVIEW",
    "E_BOUNDARY_CONFLICT_REVIEW",
    "U_LONG_TAIL_REVIEW",
}
MEDIUM_BOUNDARY_SCREENS = {
    "B_MINOR_BOUNDARY_REVIEW",
    "U_INSUFFICIENT_VALID_NEGATIVES",
    "U_INCOMPLETE_POSITIVE_RESULTS",
}


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


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(row: dict[str, Any] | None, key: str) -> float | None:
    if not row or not isinstance(row.get(key), (int, float)):
        return None
    return float(row[key])


def _positive_risk_level(
    positive: dict[str, Any] | None,
    strategy: dict[str, Any] | None = None,
) -> str:
    final_strategy = (strategy or {}).get("final_strategy") or {}
    if final_strategy.get("status") == "taxonomy_hold":
        return "P0_图谱冲突"
    if positive is None:
        return "U_正样本缺失"
    planned = int(positive.get("planned") or positive.get("completed") or 0)
    match_rate = _number(positive, "match_rate")
    zero_rate = _number(positive, "zero_rate")
    preliminary = str(positive.get("preliminary_grade") or "")
    if planned < 30:
        return "L0_极端长尾"
    if planned < 300:
        if (match_rate is not None and match_rate < 0.40) or (
            zero_rate is not None and zero_rate >= 0.50
        ):
            return "L1_长尾异常"
        return "L2_长尾待核"
    if (match_rate is not None and match_rate < 0.20) or (
        zero_rate is not None and zero_rate >= 0.60
    ):
        return "P0_明显异常"
    if (match_rate is not None and match_rate < 0.55) or (
        zero_rate is not None and zero_rate >= 0.30
    ):
        return "P1_重点核验"
    if (
        match_rate is None
        or match_rate < 0.70
        or preliminary not in {"", "A_STABLE_CANDIDATE"}
        or bool(final_strategy.get("manual_followup_required"))
    ):
        return "P2_边界观察"
    return "S_正样本稳定"


def classify_audit_tier(
    positive: dict[str, Any] | None,
    boundary: dict[str, Any] | None,
    strategy: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Return HIGH/MEDIUM/STABLE plus auditable reasons."""
    positive_level = _positive_risk_level(positive, strategy)
    boundary_screen = str((boundary or {}).get("final_screen") or "U_边界证据缺失")
    reasons = [f"positive={positive_level}", f"boundary={boundary_screen}"]
    if positive_level in HIGH_POSITIVE_LEVELS or boundary_screen in HIGH_BOUNDARY_SCREENS:
        return "HIGH", reasons
    if (
        positive_level in MEDIUM_POSITIVE_LEVELS
        or boundary_screen in MEDIUM_BOUNDARY_SCREENS
        or positive_level.startswith("U_")
        or boundary_screen.startswith("U_")
    ):
        return "MEDIUM", reasons
    if positive_level == "S_正样本稳定" and boundary_screen == "A_STABLE_CANDIDATE":
        return "STABLE", reasons
    return "MEDIUM", reasons


def _selected_by_id(prediction: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(label["label_id"]): dict(label)
        for label in prediction.get("selected_labels") or []
        if label.get("label_id")
    }


def _stable_rank(seed: str, label_id: str, question_id: str, bucket: str) -> bytes:
    return hashlib.sha256(
        f"{seed}\0{label_id}\0{question_id}\0{bucket}".encode("utf-8")
    ).digest()


def _bucket(row: dict[str, Any]) -> str:
    if row["duplicate_inconsistency"]:
        return "duplicate_inconsistency"
    if row["selection_status"] == "legacy_only_selected":
        return "legacy_only_selected"
    if row["selection_status"] == "top25_only_selected":
        return "top25_only_selected"
    if row["label_set_changed"]:
        return "both_selected_set_changed"
    ranks = [
        selection.get("candidate_rank")
        for selection in (row.get("top25_selection"), row.get("legacy_selection"))
        if isinstance(selection, dict)
        and isinstance(selection.get("candidate_rank"), int)
    ]
    if (
        any(rank >= 21 for rank in ranks)
        or row["context_insufficient"]
        or row["need_expand_recall"]
    ):
        return "tail_or_context_risk"
    return "both_selected"


QUOTAS = {
    "HIGH": {
        "duplicate_inconsistency": 2,
        "legacy_only_selected": 8,
        "top25_only_selected": 3,
        "both_selected_set_changed": 2,
        "tail_or_context_risk": 2,
        "both_selected": 3,
    },
    "MEDIUM": {
        "duplicate_inconsistency": 1,
        "legacy_only_selected": 4,
        "top25_only_selected": 2,
        "both_selected_set_changed": 1,
        "tail_or_context_risk": 1,
        "both_selected": 1,
    },
    "STABLE": {
        "duplicate_inconsistency": 1,
        "legacy_only_selected": 1,
        "top25_only_selected": 1,
        "both_selected_set_changed": 0,
        "tail_or_context_risk": 1,
        "both_selected": 1,
    },
}


def _sample_rows(
    rows: list[dict[str, Any]],
    *,
    tier: str,
    target: int,
    seed: str,
    label_id: str,
) -> list[dict[str, Any]]:
    if len(rows) <= target or len(rows) < 30:
        return sorted(rows, key=lambda row: str(row["question_id"]))
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = _bucket(row)
        row["review_bucket"] = bucket
        by_bucket[bucket].append(row)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for bucket, quota in QUOTAS[tier].items():
        ordered = sorted(
            by_bucket.get(bucket, []),
            key=lambda row: _stable_rank(seed, label_id, str(row["question_id"]), bucket),
        )
        for row in ordered[:quota]:
            question_id = str(row["question_id"])
            if question_id not in selected_ids:
                selected.append(row)
                selected_ids.add(question_id)
    if len(selected) < target:
        remaining = sorted(
            (row for row in rows if str(row["question_id"]) not in selected_ids),
            key=lambda row: _stable_rank(
                seed, label_id, str(row["question_id"]), "fill"
            ),
        )
        selected.extend(remaining[: target - len(selected)])
    return selected[:target]


def build_adjudication_review_sample(
    units_path: str | Path,
    top25_predictions_path: str | Path,
    legacy_predictions_path: str | Path,
    labels_path: str | Path,
    positive_per_label_path: str | Path,
    boundary_assessments_path: str | Path,
    output_dir: str | Path,
    *,
    strategies_path: str | Path | None = None,
    image_context_path: str | Path | None = None,
    high_count: int = 20,
    medium_count: int = 10,
    stable_count: int = 5,
    seed: str = "adjudication-review-v1",
) -> dict[str, Any]:
    """Build deterministic per-Label review packages from paired predictions."""
    if min(high_count, medium_count, stable_count) < 1:
        raise ValueError("review counts must be positive")
    labels = {str(row["label_id"]): row for row in _read_jsonl(labels_path)}
    positive = {
        str(row["label_id"]): row for row in _read_jsonl(positive_per_label_path)
    }
    boundaries = {
        str(row["label_id"]): row for row in _read_jsonl(boundary_assessments_path)
    }
    strategies = (
        {str(row["label_id"]): row for row in _read_jsonl(strategies_path)}
        if strategies_path is not None
        else {}
    )
    image_context = (
        {
            str(row["question_id"]): row
            for row in _read_jsonl(image_context_path)
        }
        if image_context_path is not None
        else {}
    )
    units = {str(row["question_id"]): row for row in _read_jsonl(units_path)}
    top25 = {
        str(row["question_id"]): row for row in _read_jsonl(top25_predictions_path)
    }
    legacy = {
        str(row["question_id"]): row for row in _read_jsonl(legacy_predictions_path)
    }
    if set(units) != set(top25) or set(units) != set(legacy):
        raise ValueError("units and both prediction files must contain identical question IDs")

    top_selected = {question_id: _selected_by_id(row) for question_id, row in top25.items()}
    legacy_selected = {
        question_id: _selected_by_id(row) for question_id, row in legacy.items()
    }
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for question_id, unit in units.items():
        dedupe_hash = str(unit.get("dedupe_hash") or "")
        if dedupe_hash:
            duplicate_groups[dedupe_hash].append(question_id)
    inconsistent_pairs: set[tuple[str, str]] = set()
    for question_ids in duplicate_groups.values():
        if len(question_ids) < 2:
            continue
        label_ids = set().union(
            *(set(top_selected[qid]) | set(legacy_selected[qid]) for qid in question_ids)
        )
        for label_id in label_ids:
            top_membership = {label_id in top_selected[qid] for qid in question_ids}
            legacy_membership = {label_id in legacy_selected[qid] for qid in question_ids}
            if len(top_membership) > 1 or len(legacy_membership) > 1:
                inconsistent_pairs.update((qid, label_id) for qid in question_ids)

    examples_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question_id, unit in units.items():
        top_ids = set(top_selected[question_id])
        legacy_ids = set(legacy_selected[question_id])
        label_set_changed = top_ids != legacy_ids
        for label_id in sorted(top_ids | legacy_ids):
            if label_id not in labels:
                raise ValueError(f"prediction uses unknown label_id: {label_id}")
            if label_id in top_ids and label_id in legacy_ids:
                status = "both_selected"
            elif label_id in legacy_ids:
                status = "legacy_only_selected"
            else:
                status = "top25_only_selected"
            top_prediction = top25[question_id]
            legacy_prediction = legacy[question_id]
            flags = unit.get("flags") or {}
            images = image_context.get(question_id, {})
            examples_by_label[label_id].append(
                {
                    "label_id": label_id,
                    "label_name": str(labels[label_id].get("label_name") or ""),
                    "question_id": question_id,
                    "parent_id": str(unit.get("parent_id") or question_id),
                    "unit_type": str(unit.get("unit_type") or ""),
                    "dedupe_hash": str(unit.get("dedupe_hash") or ""),
                    "stem": unit.get("stem", ""),
                    "options": unit.get("options", ""),
                    "answer_text": unit.get("answer_text", ""),
                    "analysis": unit.get("analysis", ""),
                    "flags": flags,
                    "stem_image_url": str(images.get("stem_image_url") or ""),
                    "analysis_image_url": str(
                        images.get("analysis_image_url") or ""
                    ),
                    "parent_stem_image_url": str(
                        images.get("parent_stem_image_url") or ""
                    ),
                    "parent_analysis_image_url": str(
                        images.get("parent_analysis_image_url") or ""
                    ),
                    "image_content_status": str(
                        images.get("content_status") or ""
                    ),
                    "image_needs_content_review": bool(
                        images.get("needs_content_review")
                    ),
                    "selection_status": status,
                    "label_set_changed": label_set_changed,
                    "duplicate_inconsistency": (question_id, label_id)
                    in inconsistent_pairs,
                    "context_insufficient": bool(
                        top_prediction.get("context_insufficient")
                        or legacy_prediction.get("context_insufficient")
                    ),
                    "need_expand_recall": bool(
                        top_prediction.get("need_expand_recall")
                        or legacy_prediction.get("need_expand_recall")
                    ),
                    "top25_selection": top_selected[question_id].get(label_id),
                    "legacy_selection": legacy_selected[question_id].get(label_id),
                    "top25_selected_label_ids": sorted(top_ids),
                    "legacy_selected_label_ids": sorted(legacy_ids),
                }
            )

    output = Path(output_dir)
    labels_output = output / "labels"
    labels_output.mkdir(parents=True, exist_ok=True)
    target_by_tier = {"HIGH": high_count, "MEDIUM": medium_count, "STABLE": stable_count}
    per_label_rows = []
    review_tasks = []
    tier_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for label_id, label in sorted(labels.items()):
        tier, risk_reasons = classify_audit_tier(
            positive.get(label_id), boundaries.get(label_id), strategies.get(label_id)
        )
        tier_counts[tier] += 1
        examples = examples_by_label.get(label_id, [])
        production = Counter(row["selection_status"] for row in examples)
        production["selected_union"] = len(examples)
        production["label_set_changed"] = sum(
            bool(row["label_set_changed"]) for row in examples
        )
        production["duplicate_inconsistency"] = sum(
            bool(row["duplicate_inconsistency"]) for row in examples
        )
        sampled = _sample_rows(
            examples,
            tier=tier,
            target=target_by_tier[tier],
            seed=seed,
            label_id=label_id,
        )
        for row in sampled:
            row.setdefault("review_bucket", _bucket(row))
            row["audit_tier"] = tier
            row["risk_reasons"] = risk_reasons
            review_tasks.append(row)
            status_counts[row["selection_status"]] += 1
        summary = {
            "label_id": label_id,
            "label_name": str(label.get("label_name") or ""),
            "label_path": str(label.get("label_path") or "").replace("->", "@"),
            "audit_tier": tier,
            "risk_reasons": risk_reasons,
            "recommended_review_count": target_by_tier[tier],
            "actual_review_count": len(sampled),
            "historical_positive": positive.get(label_id),
            "historical_boundary": boundaries.get(label_id),
            "production": dict(sorted(production.items())),
        }
        per_label_rows.append(summary)
        _write_json_atomic(
            labels_output / f"{label_id}.json",
            {**summary, "review_examples": sampled},
        )

    _write_jsonl_atomic(output / "per_label.jsonl", per_label_rows)
    _write_jsonl_atomic(output / "review_tasks.jsonl", review_tasks)
    report = {
        "questions": len(units),
        "labels": len(labels),
        "labels_with_predictions": sum(bool(examples_by_label.get(label_id)) for label_id in labels),
        "review_tasks": len(review_tasks),
        "tier_counts": dict(sorted(tier_counts.items())),
        "sampled_selection_status_counts": dict(sorted(status_counts.items())),
        "review_count_by_tier": target_by_tier,
        "seed": seed,
        "sampling_version": "label-risk-paired-adjudication-v1",
        "input_sha256": {
            "units": _file_sha256(units_path),
            "top25_predictions": _file_sha256(top25_predictions_path),
            "legacy_predictions": _file_sha256(legacy_predictions_path),
            "labels": _file_sha256(labels_path),
            "positive_per_label": _file_sha256(positive_per_label_path),
            "boundary_assessments": _file_sha256(boundary_assessments_path),
        },
    }
    if image_context_path is not None:
        report["input_sha256"]["image_context"] = _file_sha256(
            image_context_path
        )
        report["image_context_path"] = str(image_context_path)
    _write_json_atomic(output / "report.json", report)
    print(
        "review sample tests whether historical risk and legacy candidate augmentation "
        "predict production labeling errors",
        flush=True,
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report
