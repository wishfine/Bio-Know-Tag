#!/usr/bin/env python3
"""Select complete compound-question groups for a legacy Label audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def legacy_ids(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for field in ("legacy_knw_ids", "legacy_label_ids", "knw_ids"):
        for value in row.get(field) or []:
            if isinstance(value, dict):
                value = value.get("label_id") or value.get("id")
            if value:
                values.add(str(value))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--label-id", required=True)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.groups < 1:
        raise SystemExit("--groups must be positive")

    target = str(args.label_id)
    target_parents: Counter[str] = Counter()
    target_question_ids: defaultdict[str, list[str]] = defaultdict(list)

    for row in read_jsonl(args.units):
        if target not in legacy_ids(row):
            continue
        qid = str(row.get("question_id") or "")
        parent_id = str(row.get("parent_id") or qid)
        if qid and parent_id and parent_id != qid and row.get("unit_type") in {
            "sub_question",
            "orphan_sub_question",
        }:
            target_parents[parent_id] += 1
            target_question_ids[parent_id].append(qid)

    parent_rows = {
        str(row.get("question_id") or ""): row
        for row in read_jsonl(args.parents)
        if str(row.get("question_id") or "") in target_parents
    }
    children: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.units):
        parent_id = str(row.get("parent_id") or "")
        if parent_id in target_parents and row.get("unit_type") == "sub_question":
            children[parent_id].append(row)

    groups = []
    for parent_id, hit_count in target_parents.items():
        parent = parent_rows.get(parent_id)
        child_rows = children.get(parent_id, [])
        if not parent or not child_rows:
            continue
        child_rows.sort(key=lambda row: int(str(row.get("question_id") or "0")))
        for index, child in enumerate(child_rows, 1):
            child["compound_group_id"] = parent_id
            child["compound_role"] = "sub_question"
            child["sub_question_number"] = index
            child["sub_question_count"] = len(child_rows)
        parent = dict(parent)
        parent["compound_group_id"] = parent_id
        parent["compound_role"] = "parent"
        parent["sub_question_count"] = len(child_rows)
        groups.append(
            {
                "parent_id": parent_id,
                "target_hit_count": hit_count,
                "child_count": len(child_rows),
                "target_question_ids": sorted(target_question_ids[parent_id]),
                "parent": parent,
                "children": child_rows,
            }
        )

    groups.sort(key=lambda group: (-group["target_hit_count"], -group["child_count"], group["parent_id"]))
    selected = groups[: args.groups]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    units_path = args.run_dir / "compound_units.jsonl"
    with units_path.open("w", encoding="utf-8") as output:
        for group in selected:
            output.write(json.dumps(group["parent"], ensure_ascii=False) + "\n")
            for child in group["children"]:
                output.write(json.dumps(child, ensure_ascii=False) + "\n")
    report = {
        "target_label_id": target,
        "candidate_parent_groups": len(groups),
        "selected_parent_groups": len(selected),
        "selected_question_count": sum(1 + group["child_count"] for group in selected),
        "selected_groups": [
            {
                "parent_id": group["parent_id"],
                "child_count": group["child_count"],
                "target_hit_count": group["target_hit_count"],
                "target_question_ids": group["target_question_ids"],
                "child_question_ids": [row["question_id"] for row in group["children"]],
            }
            for group in selected
        ],
        "output_units": str(units_path),
    }
    (args.run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if len(selected) == args.groups else 2


if __name__ == "__main__":
    raise SystemExit(main())
