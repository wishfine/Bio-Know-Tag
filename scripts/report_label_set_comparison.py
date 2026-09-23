#!/usr/bin/env python3
"""Render one or more historical-vs-predicted 100k comparisons as Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bio_know_tag.label_set_comparison import CATEGORIES, read_jsonl


NAMES = {
    "exact": "完全一致",
    "contains_legacy": "预测包含旧集",
    "subset_of_legacy": "预测为旧集子集",
    "mixed_add_remove": "有增有减",
    "disjoint": "完全不同",
}


def _count_pct(count: int, total: int) -> str:
    return f"{count}（{count / total:.2%}）" if total else f"{count}（—）"


def _pct(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "—"


def _cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for raw in args.run:
        if "=" not in raw:
            parser.error("--run must be NAME=DIR")
        name, directory = raw.split("=", 1)
        root = Path(directory)
        report = json.loads((root / "report.json").read_text(encoding="utf-8"))
        labels = {str(row["label_id"]): row for row in read_jsonl(root / "per_label.jsonl")}
        runs.append((name, root, report, labels))
    first_ids = set(runs[0][3])
    for name, _root, report, labels in runs:
        if set(labels) != first_ids:
            raise ValueError(f"{name} has a different Label catalog")
        if report["counts"].get("units") != 100000:
            raise ValueError(f"{name} did not scan 100000 units")
        if report["input_sha256"]["units"] != runs[0][2]["input_sha256"]["units"]:
            raise ValueError(f"{name} used different units")
        if report["input_sha256"]["labels"] != runs[0][2]["input_sha256"]["labels"]:
            raise ValueError(f"{name} used different Label catalog")

    lines = [
        "# 10 万题：预测 Label 与原 `knw_ids` 对照",
        "",
        "原 `knw_ids` 仅作弱监督对照，不是人工金标；一致率不等于准确率，新选不等于错标，漏掉旧 ID 也不必然是漏标。",
        "",
        "## 口径",
        "",
        "仅将旧 ID 中属于当前 Label 目录的项组成旧集合，忽略过时 ID。仅统计有预测结果的题；没有当前有效旧 Label 的题单列，不进入五类关系的分母。空预测且旧集非空归入‘预测为旧集子集’，不归入‘完全不同’。每个 Label 的五类比例以‘原本挂该 Label 且预测成功的题’为分母，同一道多标题会计入多个 Label 行，不能跨行直接求和。",
        "",
        "## 总体",
        "",
        "| 指标 | " + " | ".join(_cell(name) for name, *_ in runs) + " |",
        "|---|" + "---:|" * len(runs),
    ]
    metrics = [
        ("已成功对比题", lambda r: str(r["counts"].get("compared", 0))),
        ("缺预测题", lambda r: str(r["counts"].get("missing_prediction", 0))),
        ("有效旧集非空题", lambda r: str(r["counts"].get("eligible", 0))),
        ("无当前有效旧 Label", lambda r: str(r["counts"].get("no_current_legacy", 0))),
    ]
    for category in CATEGORIES:
        metrics.append((NAMES[category], lambda r, category=category: _count_pct(
            r["counts"].get(category, 0), r["counts"].get("eligible", 0)
        )))
    metrics += [
        ("旧 Label 项保留率", lambda r: _count_pct(
            r["counts"].get("retained_assignments", 0),
            r["counts"].get("legacy_assignments", 0),
        )),
        ("被移除旧 Label 项", lambda r: str(r["counts"].get("missing_assignments", 0))),
        ("新增 Label 项", lambda r: str(r["counts"].get("added_assignments", 0))),
    ]
    for title, get in metrics:
        lines.append("| " + title + " | " + " | ".join(get(report) for _, _, report, _ in runs) + " |")

    for name, root, report, labels in runs:
        lines += [
            "", f"## {name}：逐 Label（{len(labels)} 个）", "",
            f"输入预测：`{root}`；单位 SHA256：`{report['input_sha256']['units']}`。", "",
            "| Label | 旧题数 | 预测题数 | 保留/旧题 | 漏掉/旧题 | 新增题数 | 一致 | 包含 | 子集 | 增减 | 不同 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        sorted_labels = sorted(labels.values(), key=lambda x: (
            -(x["legacy_questions"] - x["retained"]), x["label_name"], x["label_id"]
        ))
        for row in sorted_labels:
            counts = row["relationship_counts_among_legacy_questions"]
            rates = row["relationship_rates_among_legacy_questions"]
            parts = [
                f"{_cell(row['label_name'])} (`{row['label_id']}`)",
                str(row["legacy_questions"]), str(row["predicted_questions"]),
                f"{row['retained']} / {_pct(row['retention_rate'])}",
                f"{row['missed']} / {_pct(row['miss_rate'])}",
                str(row["new_vs_legacy"]),
            ]
            parts.extend(f"{counts[category]} / {_pct(rates[category])}" for category in CATEGORIES)
            lines.append("| " + " | ".join(parts) + " |")

    lines += [
        "", "## 解读限制", "",
        "- 预测与旧集完全一致可能共同错误；完全不同也可能是模型纠正了错误旧标。须结合题目和当前释义抽样人工复核。",
        "- 复合题小题的原 ID 可能是父题知识点并集，因此小题与旧集的差异天然偏高；应优先看 `report.json` 的题型分层。",
        "- 只用成功预测题作关系比例；失败/未完成题不默认为空选。若两组成功题集合不同，整体比例不能直接归因于策略差异。",
        "- 同内容重复题可能有不同题号；题目级比例按记录数统计，并非按去重内容统计。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runs": [name for name, *_ in runs], "labels": len(first_ids)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
