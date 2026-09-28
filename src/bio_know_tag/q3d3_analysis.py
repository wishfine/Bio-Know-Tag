"""Analyze labels selected by all three Qwen and all three DS votes."""

from __future__ import annotations

import json
import hashlib
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bio_know_tag.live_six_vote_analysis import VOTES, _check_manifests, _load_vote


def _labels(path: str | Path) -> dict[str, dict[str, Any]]:
    result = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[str(row["label_id"])] = row
    return result


def _reservoir_add(
    bucket: list[dict[str, Any]], item: dict[str, Any],
    seen: int, limit: int, rng: random.Random,
) -> None:
    if len(bucket) < limit:
        bucket.append(item)
    elif limit:
        candidate = rng.randrange(seen)
        if candidate < limit:
            bucket[candidate] = item


def analyze_q3d3(
    votes_root: str | Path,
    *,
    labels_path: str | Path,
    units_path: str | Path,
    top_k: int = 20,
    sample_per_label: int = 10,
    max_success_per_vote: int = 500000,
) -> dict[str, Any]:
    if min(top_k, max_success_per_vote) < 1 or sample_per_label < 0:
        raise ValueError("top_k and max_success_per_vote must be positive; sample_per_label nonnegative")
    root = Path(votes_root)
    manifest_alignment = _check_manifests(root)
    parent_manifest_path = root / "run_manifest.json"
    expected_hashes = (
        json.loads(parent_manifest_path.read_text(encoding="utf-8")).get("input_sha256") or {}
        if parent_manifest_path.exists() else {}
    )
    labels = _labels(labels_path)
    if expected_hashes.get("labels"):
        digest = hashlib.sha256(Path(labels_path).read_bytes()).hexdigest()
        if digest != expected_hashes["labels"]:
            raise ValueError("labels SHA256 differs from vote controller manifest")
    data: dict[str, dict[str, frozenset[str]]] = {}
    vote_reports = {}
    for vote in VOTES:
        data[vote], vote_reports[vote] = _load_vote(
            root / "votes" / vote / "evidence.jsonl",
            max_success=max_success_per_vote,
        )
    common = set.intersection(*(set(rows) for rows in data.values()))
    counts: Counter[str] = Counter()
    other_vote_pairs: Counter[str] = Counter()
    other_bands: Counter[str] = Counter()
    q3d3_count_distribution: Counter[str] = Counter()
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    co_q3d3: dict[str, Counter[str]] = defaultdict(Counter)
    unit_types: dict[str, Counter[str]] = defaultdict(Counter)
    sampled: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sampled_seen: Counter[str] = Counter()
    anchor_by_question: dict[str, frozenset[str]] = {}
    rng = random.Random(20260928)

    for qid in sorted(common):
        vote_sets = [data[vote][qid] for vote in VOTES]
        qsets, dsets = vote_sets[:3], vote_sets[3:]
        anchor = frozenset.intersection(*vote_sets)
        ever_selected = frozenset.union(*vote_sets)
        counts["all_six_common_questions"] += 1
        for label_id in ever_selected:
            per_label[label_id]["ever_selected_questions"] += 1
        if not anchor:
            continue
        anchor_by_question[qid] = anchor
        counts["questions_with_q3d3"] += 1
        counts["q3d3_label_assignments"] += len(anchor)
        q3d3_count_distribution[str(len(anchor))] += 1
        other = ever_selected - anchor
        counts["questions_with_only_q3d3_among_ever_selected"] += int(not other)
        counts["questions_with_other_ever_selected_labels"] += int(bool(other))
        near = conflict = weak = False
        other_details = []
        for label_id in sorted(other):
            qvotes = sum(label_id in selected for selected in qsets)
            dvotes = sum(label_id in selected for selected in dsets)
            pair = f"Q{qvotes}/D{dvotes}"
            other_vote_pairs[pair] += 1
            if qvotes >= 2 and dvotes >= 2:
                band = "near_consensus_both_majority"
                near = True
            elif (qvotes >= 2 and dvotes <= 1) or (dvotes >= 2 and qvotes <= 1):
                band = "cross_model_conflict"
                conflict = True
            else:
                band = "weak_both_minority"
                weak = True
            other_bands[band] += 1
            other_details.append({
                "label_id": label_id,
                "label_name": str(labels.get(label_id, {}).get("label_name") or ""),
                "qwen_votes": qvotes,
                "ds_votes": dvotes,
                "state": pair,
                "band": band,
            })
        counts["questions_with_other_near_consensus"] += int(near)
        counts["questions_with_other_cross_model_conflict"] += int(conflict)
        counts["questions_with_other_weak"] += int(weak)
        for label_id in sorted(anchor):
            stats = per_label[label_id]
            stats["q3d3_count"] += 1
            stats["questions_with_other_q3d3"] += int(len(anchor) > 1)
            stats["questions_with_other_near_consensus"] += int(near)
            stats["questions_with_other_cross_model_conflict"] += int(conflict)
            stats["questions_with_other_weak"] += int(weak)
            for co_id in anchor - {label_id}:
                co_q3d3[label_id][co_id] += 1
            sampled_seen[label_id] += 1
            _reservoir_add(sampled[label_id], {
                "question_id": qid,
                "anchor_label_id": label_id,
                "q3d3_label_ids": sorted(anchor),
                "other_ever_selected_labels": other_details,
            }, sampled_seen[label_id], sample_per_label, rng)

    label_rows = []
    assignments = counts["q3d3_label_assignments"]
    for label_id, stats in per_label.items():
        if not stats["q3d3_count"]:
            continue
        row = {
            "label_id": label_id,
            "label_name": str(labels.get(label_id, {}).get("label_name") or ""),
            "label_path": str(labels.get(label_id, {}).get("label_path") or ""),
            **dict(stats),
            "q3d3_share": round(stats["q3d3_count"] / assignments, 6) if assignments else None,
            "q3d3_given_ever_selected_rate": round(
                stats["q3d3_count"] / stats["ever_selected_questions"], 6
            ),
        }
        label_rows.append(row)
    label_rows.sort(key=lambda row: (-row["q3d3_count"], row["label_id"]))
    top_labels = label_rows[:top_k]
    top_ids = {row["label_id"] for row in top_labels}
    sample_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label_id in sorted(top_ids):
        for item in sampled[label_id]:
            sample_by_qid[item["question_id"]].append(item)

    units_scanned = 0
    matched_anchors = 0
    units_digest = hashlib.sha256()
    with Path(units_path).open("rb") as handle:
        for line in handle:
            units_digest.update(line)
            if not line.strip():
                continue
            unit = json.loads(line)
            units_scanned += 1
            qid = str(unit.get("question_id") or "")
            anchor = anchor_by_question.get(qid)
            if anchor is not None:
                matched_anchors += 1
                for label_id in anchor:
                    unit_types[label_id][str(unit.get("unit_type") or "unknown")] += 1
            for item in sample_by_qid.get(qid, []):
                for field in (
                    "parent_id", "unit_type", "stem", "parent_stem",
                    "options", "answer_text", "analysis",
                ):
                    item[field] = unit.get(field)
    if expected_hashes.get("units") and units_digest.hexdigest() != expected_hashes["units"]:
        raise ValueError("units SHA256 differs from vote controller manifest")

    for row in top_labels:
        label_id = row["label_id"]
        row["unit_type_counts"] = dict(unit_types[label_id])
        row["top_other_q3d3_labels"] = [
            {
                "label_id": other_id,
                "label_name": str(labels.get(other_id, {}).get("label_name") or ""),
                "questions": count,
            }
            for other_id, count in sorted(
                co_q3d3[label_id].items(), key=lambda item: (-item[1], item[0])
            )[:5]
        ]
    top_share = sum(row["q3d3_count"] for row in label_rows[:10]) / assignments if assignments else None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "votes_root": str(root),
        "manifest_alignment": manifest_alignment,
        "snapshot_warning": "Only all-six successful questions are analyzed; running files are read sequentially. Q3/D3 measures agreement, not correctness.",
        "votes": vote_reports,
        "all_six_common_questions": len(common),
        "q3d3_label_assignments": assignments,
        "questions_with_q3d3": counts["questions_with_q3d3"],
        "questions_with_only_q3d3_among_ever_selected": counts["questions_with_only_q3d3_among_ever_selected"],
        "questions_with_other_ever_selected_labels": counts["questions_with_other_ever_selected_labels"],
        "questions_with_other_near_consensus": counts["questions_with_other_near_consensus"],
        "questions_with_other_cross_model_conflict": counts["questions_with_other_cross_model_conflict"],
        "questions_with_other_weak": counts["questions_with_other_weak"],
        "other_label_vote_pairs": dict(other_vote_pairs.most_common()),
        "other_label_bands": dict(other_bands),
        "q3d3_count_per_question_distribution": dict(sorted(q3d3_count_distribution.items(), key=lambda item: int(item[0]))),
        "top_10_label_assignment_share": round(top_share, 6) if top_share is not None else None,
        "labels": label_rows,
        "samples": {label_id: sampled[label_id] for label_id in sorted(top_ids)},
        "units_scanned": units_scanned,
        "q3d3_questions_found_in_units": matched_anchors,
        "sample_questions_found_in_units": sum(
            bool(items[0].get("stem") is not None or items[0].get("unit_type") is not None)
            for items in sample_by_qid.values()
        ),
    }
