#!/usr/bin/env python3
"""Summarize live six-vote volatility and Qwen/DS differences on matched successes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bio_know_tag.live_six_vote_analysis import analyze_live_six_vote


def _pct(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.2%}" if denominator else "-"


def _markdown(report: dict) -> str:
    denominator = report["all_six_common_questions"]
    cross = report["cross_model"]
    lines = [
        "# 全量六票运行中快照：波动与模型差异",
        "",
        f"生成时间：{report['generated_at']}",
        "",
        f"**仅比较六票均成功的 {denominator:,} 道题。** 运行中的未完成题和错误请求不进入差异率分母；本报告不判断哪个模型正确。",
        "",
        "## 六票覆盖与错误",
        "",
        "| 票次 | 证据行 | 成功题 | 错误行 | 未完成尾行 |",
        "|---|---:|---:|---:|---:|",
    ]
    for vote, row in report["votes"].items():
        lines.append(
            f"| {vote} | {row.get('evidence_rows', 0):,} | "
            f"{row['successful_questions']:,} | {row['errors']:,} | "
            f"{row.get('incomplete_tail_ignored', 0)} |"
        )
    lines += [
        "", "## 模型内部三票", "",
        "| 模型 | 整题集合三票全同 | 2+1 | 三票各不同 | 有任一Label波动 | 全空 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, row in report["within_model"].items():
        lines.append(
            f"| {model} | {row.get('all_same', 0):,} ({_pct(row.get('all_same', 0), denominator)}) | "
            f"{row.get('two_same_one_different', 0):,} | "
            f"{row.get('all_three_different', 0):,} | "
            f"{row.get('any_label_instability', 0):,} ({_pct(row.get('any_label_instability', 0), denominator)}) | "
            f"{row.get('all_empty', 0):,} |"
        )
    lines += [
        "", "### 若仅保留前两票", "",
        "| 模型 | 第1/2票同集合 | 前两票同但第3票不同 | 第1/2票分歧 | 分歧时第3票支持第1票 | 支持第2票 | 与两者均不同 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, row in report["within_model"].items():
        lines.append(
            f"| {model} | {row.get('first_two_same', 0):,} "
            f"({_pct(row.get('first_two_same', 0), denominator)}) | "
            f"{row.get('first_two_same_but_third_differs', 0):,} | "
            f"{row.get('first_two_different', 0):,} | "
            f"{row.get('third_agrees_first_when_split', 0):,} | "
            f"{row.get('third_agrees_second_when_split', 0):,} | "
            f"{row.get('third_differs_both_when_split', 0):,} |"
        )
    lines += [
        "", "## Qwen 与 DS：逐Label按至少2/3票构成多数集合", "",
        "| 指标 | 题数或Label赋值数 | 占六票共同成功题 |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("多数集合完全相同", "majority_same"),
        ("多数集合不同", "majority_different"),
        ("双方三票各自完全一致、但两模型不同", "both_unanimous_different"),
        ("仅Qwen多数集合增加Label", "qwen_only_additions"),
        ("仅DS多数集合增加Label", "ds_only_additions"),
        ("双向增减", "bidirectional_difference"),
        ("Qwen独有的多数Label赋值", "qwen_only_assignments"),
        ("DS独有的多数Label赋值", "ds_only_assignments"),
    ):
        value = cross.get(key, 0)
        proportion = _pct(value, denominator) if "赋值" not in label else "-"
        lines.append(f"| {label} | {value:,} | {proportion} |")
    lines += [
        "",
        f"多数集合平均 Jaccard：{cross.get('majority_mean_jaccard')}",
        "",
        "## 波动最多的Label（按不稳定题数，不等于错误率）",
        "",
        "| Label | Qwen任一票出现 | Qwen波动 | DS任一票出现 | DS波动 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["labels"][:20]:
        lines.append(
            f"| {row['label_name'] or row['label_id']} | {row.get('qwen_any', 0):,} | "
            f"{row.get('qwen_unstable', 0):,} | {row.get('ds_any', 0):,} | "
            f"{row.get('ds_unstable', 0):,} |"
        )
    lines += [
        "", "## 两模型分歧最多的Label（按多数票独有赋值数）", "",
        "| Label | Qwen独有 | DS独有 | 六票至少一次出现 |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(
        report["labels"],
        key=lambda x: -(x.get("qwen_majority_only", 0) + x.get("ds_majority_only", 0)),
    )[:20]:
        lines.append(
            f"| {row['label_name'] or row['label_id']} | "
            f"{row.get('qwen_majority_only', 0):,} | "
            f"{row.get('ds_majority_only', 0):,} | "
            f"{row['seen_in_six_votes']:,} |"
        )
    lines += ["", "## 稳定但跨模型不一致的题号样例", ""]
    for row in report["samples"]["stable_cross_model_difference"]:
        qwen = ", ".join(item["label_name"] or item["label_id"] for item in row["qwen_label_ids"]) or "空"
        ds = ", ".join(item["label_name"] or item["label_id"] for item in row["ds_label_ids"]) or "空"
        lines.append(f"- {row['question_id']}：Qwen [{qwen}]；DS [{ds}]")
    lines += [
        "", "## 解读限制", "",
        "- 这是运行中快照，六个文件依次读取；使用六票成功交集确保比较分母一致，但不能外推到全量题库。",
        "- 三票完全一致只是稳定，不代表正确；跨模型差异需看具体题目、释义和教师判断。",
        "- 多数集合按每个Label至少2/3票组成；它不是整题三票完全一致，也不是六票合并多数。",
        "- 取样题号是可重复的随机抽样；详细逐Label统计与样例在 report.json。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes-root", type=Path, required=True, help="adjudication-no-thinking directory containing votes/")
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path, required=True, help="new analysis output directory")
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--max-success-per-vote", type=int, default=500000,
                        help="fail before retaining an entire full-run evidence set in memory")
    args = parser.parse_args()
    source_root = args.votes_root.resolve()
    output_root = args.run_dir.resolve()
    if output_root == source_root or source_root in output_root.parents:
        parser.error("--run-dir must be outside --votes-root to protect live vote files")
    print("分析目标：六票共同成功题的模型内波动、模型间多数集合差异；不评判正确性。", flush=True)
    report = analyze_live_six_vote(
        source_root, labels_path=args.labels, sample_limit=args.sample_limit,
        max_success_per_vote=args.max_success_per_vote,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (args.run_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report), flush=True)
    print(f"报告路径：{args.run_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
