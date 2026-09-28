#!/usr/bin/env python3
"""Analyze all-six-agreed Q3/D3 labels and their question/co-label patterns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bio_know_tag.q3d3_analysis import analyze_q3d3


def _pct(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.2%}" if denominator else "-"


def _markdown(report: dict) -> str:
    total = report["all_six_common_questions"]
    anchor_questions = report["questions_with_q3d3"]
    other_question_count = report["questions_with_other_ever_selected_labels"]
    lines = [
        "# 六票全选 Q3/D3 专项分析（运行中快照）", "",
        f"生成时间：{report['generated_at']}", "",
        f"六票共同成功题：**{total:,}**；至少有一个 Q3/D3 Label 的题：**{anchor_questions:,}** "
        f"（{_pct(anchor_questions, total)}）；Q3/D3 题–Label 赋值数：**{report['q3d3_label_assignments']:,}**。",
        "", "这里的 Q3/D3 指同一道题的同一个 Label 被 Qwen 三票和 DS 三票全部选中；不等于教师确认正确。",
        "", "## 同题其他 Label 的强度", "",
        f"- 在含 Q3/D3 的题中，六票曾选过的 Label 全部也是 Q3/D3："
        f"{report['questions_with_only_q3d3_among_ever_selected']:,} 题 "
        f"（{_pct(report['questions_with_only_q3d3_among_ever_selected'], anchor_questions)}）。",
        f"- 还存在其他至少被一票选中的 Label：{other_question_count:,} 题 "
        f"（{_pct(other_question_count, anchor_questions)}）。",
        f"- 其他 Label 中存在双方多数票支持但未达 Q3/D3（Q3/D2、Q2/D3、Q2/D2）："
        f"{report['questions_with_other_near_consensus']:,} 题 "
        f"（{_pct(report['questions_with_other_near_consensus'], anchor_questions)}）。",
        f"- 其他 Label 中存在跨模型多数票冲突：{report['questions_with_other_cross_model_conflict']:,} 题；"
        f"存在双方都不过半的弱支持：{report['questions_with_other_weak']:,} 题。",
        "", "其他 Label 的逐题–Label 状态分布（仅含至少被一票选中的其他 Label，不把大量 Q0/D0 候选算入）：", "",
        "| 票型 | 题–Label 对数 | 占其他被选过的Label |", "|---|---:|---:|",
    ]
    other_total = sum(report["other_label_vote_pairs"].values())
    for state, count in report["other_label_vote_pairs"].items():
        lines.append(f"| {state} | {count:,} | {_pct(count, other_total)} |")
    lines += [
        "", "## Q3/D3 集中的 Label", "",
        f"前 10 个 Label 占全部 Q3/D3 赋值的 {report['top_10_label_assignment_share']:.2%}。"
        if report["top_10_label_assignment_share"] is not None else "暂无 Q3/D3 赋值。",
        "", "| Label | Q3/D3题数 | 占所有Q3/D3赋值 | 该Label曾被任一票选中的题数 | 条件一致率 | 伴随其他近一致Label的题数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["labels"][:20]:
        lines.append(
            f"| {row['label_name'] or row['label_id']} | {row['q3d3_count']:,} | "
            f"{row['q3d3_share']:.2%} | {row['ever_selected_questions']:,} | "
            f"{row['q3d3_given_ever_selected_rate']:.2%} | "
            f"{row.get('questions_with_other_near_consensus', 0):,} |"
        )
    lines += [
        "", "## 高频 Label 的题目结构与共标线索", "",
        "这部分只列事实线索，不能单凭 Label 名称推断题目内容共性；完整题干、选项、答案和解析在 `sample_questions.jsonl`。", "",
    ]
    for row in report["labels"][:10]:
        label_id = row["label_id"]
        line = f"- **{row['label_name'] or label_id}**：题型 {row.get('unit_type_counts') or {}}；"
        co = row.get("top_other_q3d3_labels") or []
        line += "常见同步 Q3/D3 Label " + (
            "、".join(f"{item['label_name'] or item['label_id']}（{item['questions']}）" for item in co)
            if co else "无"
        ) + "。"
        lines.append(line)
        for example in report["samples"].get(label_id, [])[:2]:
            stem = str(example.get("stem") or "").replace("\n", " ")[:160]
            lines.append(f"  - 题号 {example['question_id']}：{stem or '[题干文本缺失]'}")
    lines += [
        "", "## 限制", "",
        "- 运行中快照只覆盖六票共同成功题；顺序可能偏向题库前段，不代表全量分布。",
        "- `Q3/D3` 代表稳定一致，不代表金标正确；尤其是广义或综合 Label，仍需审题和释义。",
        "- 条件一致率的分母是该 Label 至少被六票中任一票选中的题，不是全题库。",
        "- 同题其他 Label 仅统计至少一票选过的候选；`Q0/D0` 不纳入“其他 Label”强度表。",
        "- 样本是确定性随机抽样，逐题原文见 `sample_questions.jsonl`，需人工复核是否有语义共性。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes-root", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sample-per-label", type=int, default=10)
    parser.add_argument("--max-success-per-vote", type=int, default=500000)
    args = parser.parse_args()
    source = args.votes_root.resolve()
    output = args.run_dir.resolve()
    if output == source or source in output.parents:
        parser.error("--run-dir must be outside --votes-root to protect live vote files")
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            parser.error("--run-dir must be a new or empty directory; existing files will not be overwritten")
    else:
        output.mkdir(parents=True)
    print("分析目标：Q3/D3 赋值、集中 Label、同题其他 Label 票型与题目样本。", flush=True)
    report = analyze_q3d3(
        source,
        labels_path=args.labels,
        units_path=args.units,
        top_k=args.top_k,
        sample_per_label=args.sample_per_label,
        max_success_per_vote=args.max_success_per_vote,
    )
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = _markdown(report)
    (output / "report.md").write_text(markdown, encoding="utf-8")
    with (output / "sample_questions.jsonl").open("w", encoding="utf-8") as handle:
        for label_id, items in report["samples"].items():
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(markdown)
    print(f"报告：{output / 'report.md'}；完整样本：{output / 'sample_questions.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
