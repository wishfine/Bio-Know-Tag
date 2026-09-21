#!/usr/bin/env python3
"""Build paired name-only and name-plus-definition DS tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bio_know_tag.definition_ablation import (
    build_definition_ablation_sample,
    load_labels,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-samples", type=Path, required=True)
    parser.add_argument("--positive-results", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--volatile-group", type=Path, action="append", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--low-threshold", type=float, default=0.70)
    parser.add_argument("--low-count", type=int, default=10)
    parser.add_argument("--high-count", type=int, default=5)
    parser.add_argument("--seed", default="definition-ablation-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    selected, report = build_definition_ablation_sample(
        positive_samples_path=args.positive_samples,
        positive_results_path=args.positive_results,
        labels_path=args.labels,
        volatile_group_paths=args.volatile_group,
        low_threshold=args.low_threshold,
        low_count=args.low_count,
        high_count=args.high_count,
        seed=args.seed,
    )

    paired = []
    for row in selected:
        question_fields = row["question"]
        base = {
            "pair_id": row["pair_id"],
            "question_id": row["question_id"],
            "label_id": row["label_id"],
            "label_name": row["label_name"],
            "label_path": row["label_path"],
            "match_rate": row["match_rate"],
            "match_rate_count": row["match_rate_count"],
            "stratum": row["stratum"],
            "requested_count": row["requested_count"],
            "sample_source": row["sample_source"],
            "question": question_fields,
            "historical_result": row["historical_result"],
            "image_context_missing": row["image_context_missing"],
        }
        base.update(question_fields)
        paired.append(
            {
                **base,
                "condition": "name_only",
                "label": {"label_name": row["label_name"]},
            }
        )
        paired.append(
            {
                **base,
                "condition": "name_plus_definition",
                "label": {
                    "label_name": row["label_name"],
                    "label_path": row["label_path"],
                    "definition": row["definition"],
                    "core_concepts": row["core_concepts"],
                    "common_assessments": row["common_assessments"],
                    "distinctions": row["distinctions"],
                },
            }
        )

    write_jsonl(args.run_dir / "paired_tasks.jsonl", paired)
    write_jsonl(
        args.run_dir / "name_only_tasks.jsonl",
        [row for row in paired if row["condition"] == "name_only"],
    )
    write_jsonl(
        args.run_dir / "name_plus_definition_tasks.jsonl",
        [row for row in paired if row["condition"] == "name_plus_definition"],
    )
    labels = load_labels(args.labels)
    write_jsonl(
        args.run_dir / "name_only_labels.jsonl",
        [
            {
                "label_id": label_id,
                "label_name": str(label.get("label_name") or ""),
                "label_path": "",
                "definition": "",
                "core_concepts": "",
                "common_assessments": "",
                "distinctions": "",
            }
            for label_id, label in sorted(labels.items())
        ],
    )
    write_jsonl(
        args.run_dir / "name_plus_definition_labels.jsonl",
        [labels[label_id] for label_id in sorted(labels)],
    )
    (args.run_dir / "per_label.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in report.pop("per_label")),
        encoding="utf-8",
    )
    report["input_sha256"] = {
        "positive_samples": sha256(args.positive_samples),
        "positive_results": sha256(args.positive_results),
        "labels": sha256(args.labels),
    }
    report["paths"] = {
        "positive_samples": str(args.positive_samples),
        "positive_results": str(args.positive_results),
        "labels": str(args.labels),
        "volatile_groups": [str(path) for path in args.volatile_group],
    }
    (args.run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(args.run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
