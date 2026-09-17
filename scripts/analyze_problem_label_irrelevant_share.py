#!/usr/bin/env python3
"""Report score<0.10 shares for every positive-stage problem Label."""

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


def risk_level(metric: dict[str, Any], strategy: dict[str, Any] | None) -> str:
    planned = int(metric["planned"])
    match_rate = float(metric["match_rate"])
    zero_rate = float(metric["zero_rate"])
    final = (strategy or {}).get("final_strategy") or {}
    if final.get("status") == "taxonomy_hold":
        return "P0_图谱冲突"
    if planned < 30:
        return "L0_极端长尾"
    if planned < 300:
        if match_rate < 0.40 or zero_rate >= 0.50:
            return "L1_长尾异常"
        return "L2_长尾待核"
    if match_rate < 0.20 or zero_rate >= 0.60:
        return "P0_明显异常"
    if match_rate < 0.55 or zero_rate >= 0.30:
        return "P1_重点核验"
    if match_rate < 0.70 or metric["preliminary_grade"] != "A_STABLE_CANDIDATE":
        return "P2_边界观察"
    if bool(final.get("manual_followup_required")):
        return "P2_边界观察"
    return "S_正样本稳定"


def share_band(rate: float) -> str:
    if rate >= 0.80:
        return ">=80%"
    if rate >= 0.60:
        return "60%-79.9%"
    if rate >= 0.40:
        return "40%-59.9%"
    if rate >= 0.20:
        return "20%-39.9%"
    return "<20%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--per-label", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--strategies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = {str(row["label_id"]): row for row in read_jsonl(args.per_label)}
    labels = {str(row["label_id"]): row for row in read_jsonl(args.labels)}
    strategies = {str(row["label_id"]): row for row in read_jsonl(args.strategies)}
    scores_by_label: dict[str, list[float]] = defaultdict(list)
    for row in read_jsonl(args.results):
        scores_by_label[str(row["label_id"])].append(float(row["relevance_score"]))

    primary = {
        "P0_图谱冲突",
        "P0_明显异常",
        "P1_重点核验",
        "L0_极端长尾",
        "L1_长尾异常",
    }
    rows = []
    for label_id, label in labels.items():
        metric = metrics[label_id]
        strategy = strategies.get(label_id)
        risk = risk_level(metric, strategy)
        manual = bool(
            ((strategy or {}).get("final_strategy") or {}).get(
                "manual_followup_required"
            )
        )
        if risk not in primary and not manual:
            continue
        scores = scores_by_label[label_id]
        zero = sum(score == 0 for score in scores)
        tiny = sum(0 < score < 0.10 for score in scores)
        irrelevant = zero + tiny
        rows.append(
            {
                "label_id": label_id,
                "label_name": label["label_name"],
                "risk": risk,
                "total": len(scores),
                "zero": zero,
                "tiny": tiny,
                "irrelevant": irrelevant,
                "irrelevant_rate": irrelevant / len(scores),
                "match_rate": float(metric["match_rate"]),
            }
        )
    rows.sort(
        key=lambda row: (-row["irrelevant_rate"], -row["total"], row["label_name"])
    )
    total = sum(row["total"] for row in rows)
    irrelevant = sum(row["irrelevant"] for row in rows)
    band_counts = Counter(share_band(row["irrelevant_rate"]) for row in rows)
    lines = [
        "# 问题Label中“按当前释义基本无关”题目占比",
        "",
        "## 口径",
        "",
        "- 范围：正样本阶段重点文档中的135个问题Label。",
        "- `relevance_score < 0.10`定义为“按当前释义基本无关”。",
        "- 其中`0.00`为完全无关，`0.01–0.09`为基本无关；两者合并计算。",
        "- 历史Label是弱监督；高占比优先指向旧ID误挂/语义漂移，不代表必须扩写老师释义。",
        "",
        "## 总体",
        "",
        f"- 问题Label：{len(rows)}个。",
        f"- 题目—Label对：{total:,}。",
        f"- 基本无关：{irrelevant:,}，占{irrelevant / total:.2%}。",
        "",
        "| 每Label基本无关占比 | Label数 |",
        "|---|---:|",
    ]
    for band in (">=80%", "60%-79.9%", "40%-59.9%", "20%-39.9%", "<20%"):
        lines.append(f"| {band} | {band_counts[band]} |")
    lines.extend(
        [
            "",
            "## 逐Label结果（按基本无关占比降序）",
            "",
            "| Label | ID | 风险层 | 总题数 | 0.00 | 0.01–0.09 | 基本无关 | 占比 | match占比 | 解读 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        if row["total"] < 30:
            note = "极端长尾，比例不稳定，应逐题复核。"
        elif row["irrelevant_rate"] >= 0.60:
            note = "旧ID误挂/语义漂移强信号；优先清理旧标，不直接改释义。"
        elif row["irrelevant_rate"] >= 0.40:
            note = "旧标噪声明显；抽查后区分旧标错与释义漏项。"
        elif row["irrelevant_rate"] >= 0.20:
            note = "存在一定旧标噪声，需与弱相关题分开处理。"
        else:
            note = "基本无关不是主要问题；更应关注0.10–0.69边界题。"
        lines.append(
            f"| {row['label_name']} | `{row['label_id']}` | {row['risk']} | "
            f"{row['total']} | {row['zero']} | {row['tiny']} | {row['irrelevant']} | "
            f"{row['irrelevant_rate']:.2%} | {row['match_rate']:.2%} | {note} |"
        )
    lines.extend(
        [
            "",
            "## 处理建议",
            "",
            "1. 样本数≥300且基本无关占比≥60%：作为旧ID语义/映射漂移的最高优先级核验项。",
            "2. 基本无关占比40%–60%：抽查具体题，判定是历史误标还是释义缺少稳定题型。",
            "3. 基本无关占比<20%但match仍低：主要问题不是完全误挂，而是上位/综合Label口径或边界阈值。",
            "4. 样本数<30：不使用百分比修改释义，直接逐题人工复核。",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "problem_labels": len(rows),
                "pairs": total,
                "irrelevant": irrelevant,
                "irrelevant_rate": round(irrelevant / total, 6),
                "label_band_counts": dict(sorted(band_counts.items())),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
