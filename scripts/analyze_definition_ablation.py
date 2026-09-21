#!/usr/bin/env python3
"""Compare name-only and name-plus-definition DS coverage judgments."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["task_id"])] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--name-only-results", type=Path, required=True)
    parser.add_argument("--definition-results", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    sample = read(args.sample)
    name_only = read(args.name_only_results)
    definition = read(args.definition_results)
    common = sorted(set(name_only) & set(definition))
    changes = Counter()
    diffs = []
    per_label: dict[str, dict] = defaultdict(
        lambda: {
            "label_id": "",
            "total": 0,
            "name_only_match": 0,
            "definition_match": 0,
            "false_to_true": 0,
            "true_to_false": 0,
            "same": 0,
            "mean_score_delta": 0.0,
        }
    )
    for task_id in common:
        left = name_only[task_id]
        right = definition[task_id]
        label_id = str(left.get("label_id") or right.get("label_id") or "")
        left_match = bool(left.get("match"))
        right_match = bool(right.get("match"))
        delta = float(right.get("relevance_score", 0)) - float(
            left.get("relevance_score", 0)
        )
        if left_match == right_match:
            changes["same"] += 1
            change = "same_match" if left_match else "same_nonmatch"
        elif not left_match and right_match:
            changes["false_to_true"] += 1
            change = "false_to_true"
        else:
            changes["true_to_false"] += 1
            change = "true_to_false"
        changes[change] += 1
        stats = per_label[label_id]
        stats["label_id"] = label_id
        stats["total"] += 1
        stats["name_only_match"] += int(left_match)
        stats["definition_match"] += int(right_match)
        stats["false_to_true"] += int(not left_match and right_match)
        stats["true_to_false"] += int(left_match and not right_match)
        stats["same"] += int(left_match == right_match)
        stats["mean_score_delta"] += delta
        if left_match != right_match or abs(delta) >= 0.20:
            source = sample.get(task_id, {})
            diffs.append(
                {
                    "task_id": task_id,
                    "question_id": left.get("question_id"),
                    "label_id": label_id,
                    "label_name": source.get("label_name", ""),
                    "match_change": change,
                    "name_only_score": left.get("relevance_score"),
                    "definition_score": right.get("relevance_score"),
                    "score_delta": round(delta, 6),
                    "name_only_match": left_match,
                    "definition_match": right_match,
                }
            )
    for stats in per_label.values():
        if stats["total"]:
            stats["name_only_match_rate"] = round(
                stats["name_only_match"] / stats["total"], 6
            )
            stats["definition_match_rate"] = round(
                stats["definition_match"] / stats["total"], 6
            )
            stats["mean_score_delta"] = round(
                stats["mean_score_delta"] / stats["total"], 6
            )
    report = {
        "sample_pairs": len(sample),
        "name_only_results": len(name_only),
        "definition_results": len(definition),
        "paired_results": len(common),
        "pending_name_only": len(set(sample) - set(name_only)),
        "pending_definition": len(set(sample) - set(definition)),
        "change_counts": dict(changes),
        "score_delta_changed_pairs": len(diffs),
        "interpretation": {
            "false_to_true": "释义帮助DS发现题目与Label的匹配，可能说明名称不足或释义补足覆盖范围。",
            "true_to_false": "释义使DS拒绝原先仅凭名称的匹配，可能说明名称过宽、原Label错挂或释义边界冲突。",
            "same_nonmatch": "题目从Label角度看可能确实不匹配，或两种输入都无法理解。",
            "same_match": "释义对该题判断没有造成明显改变。",
        },
    }
    (args.run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.run_dir / "per_label.jsonl").open("w", encoding="utf-8") as handle:
        for label_id in sorted(per_label):
            handle.write(json.dumps(per_label[label_id], ensure_ascii=False, sort_keys=True) + "\n")
    with (args.run_dir / "changed_pairs.jsonl").open("w", encoding="utf-8") as handle:
        for row in diffs:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"run_dir": str(args.run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
