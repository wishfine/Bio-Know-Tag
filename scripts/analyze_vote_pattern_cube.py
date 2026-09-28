#!/usr/bin/env python3
"""Build the reusable 16-pattern Label risk cube from six live vote files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bio_know_tag.vote_pattern_cube import PATTERNS, analyze_vote_pattern_cube


def _percent(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.2%}" if denominator else "-"


def _markdown(report: dict) -> str:
    total = report["candidate_label_exposures"]
    lines = [
        "# 生物全量六票：逐 Label 十六格票型快照", "",
        f"生成时间：{report['generated_at']}", "",
        f"六票共同成功题 {report['common_successful_questions']:,}；候选曝光的题–Label 对 "
        f"{total:,}。错误/未完成请求不计入这两个分母。", "",
        "此表统计模型投票行为，**不是准确率**；旧 knw_ids 与以往 DS 释义覆盖只作弱监督线索。", "",
        "## 16 格分布（分母：候选曝光的题–Label 对）", "",
        "| 票型 | 数量 | 占候选曝光 |", "|---|---:|---:|",
    ]
    for pattern in PATTERNS:
        count = report["pattern_counts"][pattern]
        lines.append(f"| {pattern} | {count:,} | {_percent(count, total)} |")
    lines += ["", "## 强稳定跨模型冲突集中的 Label", "",
              "| Label | 候选曝光 | Q3/D0 | Q0/D3 | 强冲突占曝光 | Q3/D3 |", "|---|---:|---:|---:|---:|---:|"]
    for row in sorted(report["labels"], key=lambda item: (-item["stable_cross_model_conflict_count"], item["label_id"]))[:20]:
        p = row["patterns"]
        lines.append(
            f"| {row['label_name'] or row['label_id']} | {row['candidate_exposure']:,} | "
            f"{p['Q3/D0']:,} | {p['Q0/D3']:,} | "
            f"{_percent(row['stable_cross_model_conflict_count'], row['candidate_exposure'])} | {p['Q3/D3']:,} |"
        )
    lines += ["", "## Q3/D3 数量最高的 Label", "",
              "| Label | 候选曝光 | 至少一票选中 | Q3/D3 | 占曝光 | 占至少一票选中 |", "|---|---:|---:|---:|---:|---:|"]
    for row in sorted(report["labels"], key=lambda item: (-item["patterns"]["Q3/D3"], item["label_id"]))[:20]:
        count = row["patterns"]["Q3/D3"]
        lines.append(
            f"| {row['label_name'] or row['label_id']} | {row['candidate_exposure']:,} | "
            f"{row['any_vote_selected']:,} | {count:,} | "
            f"{_percent(count, row['candidate_exposure'])} | "
            f"{_percent(count, row['any_vote_selected'])} |"
        )
    lines += ["", "## 题型分层（仅展示非零格）", ""]
    for unit_type, patterns in report["by_unit_type"].items():
        subtotal = sum(patterns.values())
        highlights = ", ".join(
            f"{name}={patterns[name]:,}" for name in ("Q3/D3", "Q3/D0", "Q0/D3", "Q2/D1", "Q1/D2", "Q0/D0")
        )
        lines.append(f"- {unit_type}：{subtotal:,} 次候选曝光；{highlights}。")
    lines += ["", "## 教师审核解释限制", "",
              "- 只有候选曝光过，Q0/D0 才表示六票都未选；候选外漏召要单独评估。",
              "- 票型不能判定哪一方对；每个 Label 的 `teacher_calibration_status` 当前均为 `NO_TEACHER_GOLD`。",
              "- 自动通过目标错标率 ≤5%，教师预算约 1,000 个题–Label 判断。必须通过概率抽样和独立留出验证估计错误率区间后才能自动放行。",
              "- 运行中快照可能偏向题库前段；详细 458 Label、题型/质量/候选 rank/旧 ID 分层和每格样本见 `report.json`。",
              ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--positive-per-label", type=Path)
    parser.add_argument("--boundary-assessments", type=Path)
    parser.add_argument("--sample-per-pattern", type=int, default=10)
    parser.add_argument("--max-success-per-vote", type=int, default=500000)
    args = parser.parse_args()
    source = args.votes_root.resolve()
    output = args.run_dir.resolve()
    if output == source or source in output.parents:
        parser.error("--run-dir must be outside --votes-root to protect live vote files")
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            parser.error("--run-dir must be a new or empty directory")
    else:
        output.mkdir(parents=True)
    print("分析目标：六票共同成功题的16格票型、候选曝光、逐Label/题型分层；不自动判断对错。", flush=True)
    report = analyze_vote_pattern_cube(
        source,
        candidates_path=args.candidates,
        units_path=args.units,
        labels_path=args.labels,
        sample_per_pattern=args.sample_per_pattern,
        max_success_per_vote=args.max_success_per_vote,
        positive_per_label_path=args.positive_per_label,
        boundary_assessments_path=args.boundary_assessments,
    )
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = _markdown(report)
    (output / "report.md").write_text(markdown, encoding="utf-8")
    with (output / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for pattern in PATTERNS:
            for sample in report["samples"][pattern]:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(markdown, flush=True)
    print(f"报告：{output / 'report.md'}；完整逐Label数据：{output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
