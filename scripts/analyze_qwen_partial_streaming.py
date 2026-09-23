#!/usr/bin/env python3
"""Compare two live adjudication snapshots without loading full records into RAM."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


GROUP_A = "A_same_candidates"
GROUP_B = "B_added_not_selected"
GROUP_C = "C_added_legacy_selected"


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc


def chosen(row: dict[str, Any]) -> set[str]:
    return {
        str(item["label_id"])
        for item in row.get("selected_labels") or []
        if item.get("label_id")
    }


def manifest(path: Path) -> dict[str, Any]:
    candidate = path.parent / "run_manifest.json"
    return json.loads(candidate.read_text(encoding="utf-8")) if candidate.exists() else {}


def compare(
    top25_path: Path,
    legacy_path: Path,
    units_path: Path,
    labels_path: Path,
    sample_per_group: int,
    top_manifest_path: Path | None = None,
    legacy_manifest_path: Path | None = None,
) -> dict[str, Any]:
    top_manifest = (
        json.loads(top_manifest_path.read_text(encoding="utf-8"))
        if top_manifest_path else manifest(top25_path)
    )
    legacy_manifest = (
        json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
        if legacy_manifest_path else manifest(legacy_path)
    )
    compared_settings = {}
    if top_manifest and legacy_manifest:
        for key in ("model", "prompt_version", "max_tokens"):
            values = (top_manifest.get(key), legacy_manifest.get(key))
            if values[0] != values[1]:
                raise ValueError(f"run manifests disagree on {key}: {values}")
            compared_settings[key] = values[0]
        for key in ("units", "labels"):
            values = (
                (top_manifest.get("input_sha256") or {}).get(key),
                (legacy_manifest.get("input_sha256") or {}).get(key),
            )
            if values[0] != values[1]:
                raise ValueError(f"run manifests disagree on {key} SHA256: {values}")
            compared_settings[f"{key}_sha256"] = values[0]

    labels = {
        str(row["label_id"]): str(row.get("label_name") or "")
        for row in rows(labels_path)
    }
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE top25 (question_id TEXT PRIMARY KEY, selected TEXT, candidate_count INTEGER, reason TEXT)"
    )
    top_rows = 0
    for row in rows(top25_path):
        if row.get("error"):
            continue
        question_id = str(row.get("question_id") or "")
        if not question_id:
            continue
        count = int(row.get("candidate_count") or 0)
        if count < 1:
            raise ValueError(f"missing candidate_count for {question_id} in Top25")
        connection.execute(
            "INSERT OR REPLACE INTO top25 VALUES (?, ?, ?, ?)",
            (question_id, json.dumps(sorted(chosen(row))), count, str(row.get("reason") or "")),
        )
        top_rows += 1
    connection.commit()

    counts: Counter[str] = Counter()
    by_group: dict[str, Counter[str]] = {
        group: Counter() for group in (GROUP_A, GROUP_B, GROUP_C)
    }
    samples: dict[str, list[dict[str, Any]]] = {
        group: [] for group in by_group
    }
    sample_seen: Counter[str] = Counter()
    rng = random.Random(20260923)

    for row in rows(legacy_path):
        if row.get("error"):
            continue
        question_id = str(row.get("question_id") or "")
        if not question_id:
            continue
        counts["legacy_rows"] += 1
        fetched = connection.execute(
            "SELECT selected, candidate_count, reason FROM top25 WHERE question_id = ?",
            (question_id,),
        ).fetchone()
        if fetched is None:
            counts["legacy_only_questions"] += 1
            continue

        top_ids = set(json.loads(fetched[0]))
        top_count = int(fetched[1])
        legacy_count = int(row.get("candidate_count") or 0)
        if legacy_count < top_count:
            raise ValueError(f"augmented candidates shrank for {question_id}")
        legacy_ids = chosen(row)
        added_selected = {
            str(item["label_id"])
            for item in row.get("selected_labels") or []
            if int(item.get("candidate_rank") or 0) > top_count
        }
        if added_selected and legacy_count == top_count:
            raise ValueError(f"selected label rank exceeds candidate count for {question_id}")
        for item in row.get("selected_labels") or []:
            if str(item.get("label_id")) in added_selected and item.get("sources") != ["legacy"]:
                raise ValueError(f"unexpected added candidate source for {question_id}")

        group = (
            GROUP_A if legacy_count == top_count else
            GROUP_C if added_selected else GROUP_B
        )
        shared = top_ids & legacy_ids
        top_only = top_ids - legacy_ids
        legacy_only = legacy_ids - top_ids
        changed = bool(top_only or legacy_only)
        counts["common_questions"] += 1
        counts["different_outputs"] += int(changed)
        counts["top25_assignments"] += len(top_ids)
        counts["legacy_assignments"] += len(legacy_ids)
        counts["shared_assignments"] += len(shared)
        counts["top25_only_assignments"] += len(top_only)
        counts["legacy_only_assignments"] += len(legacy_only)
        counts["new_legacy_labels_selected"] += len(added_selected)
        stats = by_group[group]
        stats["questions"] += 1
        stats["different_outputs"] += int(changed)
        stats["top25_empty_legacy_nonempty"] += int(not top_ids and bool(legacy_ids))
        stats["top25_nonempty_legacy_empty"] += int(bool(top_ids) and not legacy_ids)
        if changed and sample_per_group:
            sample_seen[group] += 1
            sample = {
                "question_id": question_id,
                "group": group,
                "top25_labels": sorted(top_ids),
                "legacy_labels": sorted(legacy_ids),
                "added_legacy_selected": sorted(added_selected),
                "top25_reason": fetched[2][:240],
                "legacy_reason": str(row.get("reason") or "")[:240],
            }
            bucket = samples[group]
            if len(bucket) < sample_per_group:
                bucket.append(sample)
            else:
                replacement = rng.randrange(sample_seen[group])
                if replacement < sample_per_group:
                    bucket[replacement] = sample

    sample_by_id = {
        item["question_id"]: item
        for bucket in samples.values()
        for item in bucket
    }
    for unit in rows(units_path):
        sample = sample_by_id.get(str(unit.get("question_id") or ""))
        if sample is None:
            continue
        sample["stem"] = str(unit.get("stem") or "")[:350]
        sample["parent_stem"] = str(unit.get("parent_stem") or "")[:200]
        sample["answer_text"] = str(unit.get("answer_text") or "")[:100]
        sample["analysis"] = str(unit.get("analysis") or "")[:500]
    for sample in sample_by_id.values():
        for field in ("top25_labels", "legacy_labels", "added_legacy_selected"):
            sample[field] = [
                {"label_id": label_id, "label_name": labels.get(label_id, "")}
                for label_id in sample[field]
            ]
    connection.close()
    return {
        "settings": compared_settings,
        "top25_rows": top_rows,
        "counts": dict(counts),
        "different_output_rate": round(
            counts["different_outputs"] / counts["common_questions"], 6
        ) if counts["common_questions"] else None,
        "groups": {
            group: {
                **dict(stats),
                "share_among_changed": round(
                    stats["different_outputs"] / counts["different_outputs"], 6
                ) if counts["different_outputs"] else None,
            }
            for group, stats in by_group.items()
        },
        "samples": samples,
        "warning": "Only common successful questions are compared. Group A changes include model/service instability. Luna risk samples are not population-level gold labels.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top25-predictions", type=Path, required=True)
    parser.add_argument("--legacy-predictions", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--sample-per-group", type=int, default=3)
    parser.add_argument("--top25-manifest", type=Path)
    parser.add_argument("--legacy-manifest", type=Path)
    args = parser.parse_args()
    if args.sample_per_group < 0:
        parser.error("--sample-per-group must be nonnegative")
    report = compare(
        args.top25_predictions,
        args.legacy_predictions,
        args.units,
        args.labels,
        args.sample_per_group,
        args.top25_manifest,
        args.legacy_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
