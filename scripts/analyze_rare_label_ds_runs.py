#!/usr/bin/env python3
"""Compare two completed DS adjudication runs for a small set of legacy Labels.

The target questions are selected from ``legacy_knw_ids`` in the units file.
For each target question/Label the report distinguishes retrieval failure from
adjudication rejection and records the complete selected-Label set on both
sides.  The input can be the full 100k run; only the target questions are
materialized in the detailed output.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TARGETS = {
    "2276103144365903872": "光合作用综合",
    "2276103347605098496": "减数分裂模型制作",
    "2276104097546653696": "青少年常见的免疫异常的疾病",
    "2276104219433127936": "实验活动-探究果蝇种群的增长",
    "2276104589790171136": "测定土壤中的微生物数量",
    "2276105005043044352": "生物武器的传播途径与防护措施",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if isinstance(row, dict):
                yield row


def label_id(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("label_id") or value.get("id") or "")
    return str(value or "")


def unit_legacy_ids(row: dict[str, Any]) -> set[str]:
    """Read historical IDs while tolerating the older unit exports."""
    for field in ("legacy_knw_ids", "legacy_label_ids", "knw_ids"):
        values = row.get(field)
        if isinstance(values, list):
            return {label_id(value) for value in values if label_id(value)}
    return set()


def selected_ids(row: dict[str, Any]) -> set[str]:
    """Extract selected Label IDs from evidence/prediction variants."""
    for value in row.get("selected_label_ids") or []:
        if label_id(value):
            return {label_id(value) for value in row.get("selected_label_ids") or [] if label_id(value)}
    selected_labels = row.get("selected_labels")
    if isinstance(selected_labels, list):
        return {label_id(value) for value in selected_labels if label_id(value)}
    choices = row.get("choices") or []
    choice = next((value for value in reversed(choices) if isinstance(value, dict) and not value.get("parse_error")), None)
    parsed = (choice or {}).get("parsed_response") or row.get("parsed_response") or {}
    code_map = {str(key): label_id(value) for key, value in (row.get("candidate_code_map") or {}).items()}
    return {code_map[code] for code in parsed.get("selected") or [] if code in code_map}


def candidate_ids(row: dict[str, Any]) -> set[str]:
    return {label_id(value) for value in (row.get("candidate_code_map") or {}).values() if label_id(value)}


def load_latest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        qid = str(row.get("question_id") or "")
        if qid:
            previous = rows.get(qid)
            # A resumed run can append an error after an earlier successful
            # response (or vice versa). Prefer a parseable success so a
            # transient endpoint failure does not look like missing recall.
            current_success = bool(
                not row.get("error")
                and (
                    isinstance(row.get("parsed_response"), dict)
                    or any(isinstance(choice, dict) and isinstance(choice.get("parsed_response"), dict) for choice in row.get("choices") or [])
                    or isinstance(row.get("selected_label_ids"), list)
                )
            )
            previous_success = bool(
                previous
                and not previous.get("error")
                and (
                    isinstance(previous.get("parsed_response"), dict)
                    or any(isinstance(choice, dict) and isinstance(choice.get("parsed_response"), dict) for choice in previous.get("choices") or [])
                    or isinstance(previous.get("selected_label_ids"), list)
                )
            )
            if previous is None or current_success or not previous_success:
                rows[qid] = row
    return rows


def label_name(label_id_value: str, labels: dict[str, dict[str, Any]]) -> str:
    return str(labels.get(label_id_value, {}).get("label_name") or label_id_value)


def names(ids: Iterable[str], labels: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"label_id": value, "label_name": label_name(value, labels)}
        for value in sorted(set(ids), key=lambda item: (label_name(item, labels), item))
    ]


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    return {label_id(row): row for row in read_jsonl(path) if label_id(row)}


def build_report(
    units_path: Path,
    top25_path: Path,
    legacy_path: Path,
    labels_path: Path,
    target_map: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = load_labels(labels_path)
    top25 = load_latest(top25_path)
    legacy = load_latest(legacy_path)
    target_questions: dict[str, dict[str, Any]] = {}
    target_hits: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(units_path):
        qid = str(row.get("question_id") or "")
        hit_ids = unit_legacy_ids(row) & set(target_map)
        if not qid or not hit_ids:
            continue
        target_questions[qid] = row
        for target in sorted(hit_ids):
            target_hits[target].append(qid)

    detail_rows: list[dict[str, Any]] = []
    for qid, unit in target_questions.items():
        top_row = top25.get(qid)
        legacy_row = legacy.get(qid)
        top_candidates = candidate_ids(top_row or {})
        legacy_candidates = candidate_ids(legacy_row or {})
        top_selected = selected_ids(top_row or {})
        legacy_selected = selected_ids(legacy_row or {})
        source_legacy = unit_legacy_ids(unit)
        target_ids = sorted(source_legacy & set(target_map))
        selected_diff = top_selected != legacy_selected
        detail_rows.append(
            {
                "question_id": qid,
                "target_label_ids": target_ids,
                "target_labels": [target_map[value] for value in target_ids],
                "legacy_knw_ids": sorted(source_legacy),
                "unit_type": unit.get("unit_type"),
                "stem": unit.get("stem") or "",
                "options": unit.get("options") or "",
                "answer_text": unit.get("answer_text") or "",
                "analysis": unit.get("analysis") or "",
                "top25": {
                    "evidence_found": top_row is not None,
                    "candidate_count": len(top_candidates),
                    "candidate_contains_target": sorted(top_candidates & set(target_ids)),
                    "selected_label_ids": sorted(top_selected),
                    "selected_labels": names(top_selected, labels),
                    "selected_target_ids": sorted(top_selected & set(target_ids)),
                    "error": (top_row or {}).get("error"),
                },
                "top25_plus_legacy": {
                    "evidence_found": legacy_row is not None,
                    "candidate_count": len(legacy_candidates),
                    "candidate_contains_target": sorted(legacy_candidates & set(target_ids)),
                    "selected_label_ids": sorted(legacy_selected),
                    "selected_labels": names(legacy_selected, labels),
                    "selected_target_ids": sorted(legacy_selected & set(target_ids)),
                    "error": (legacy_row or {}).get("error"),
                },
                "comparison": {
                    "selected_label_sets_equal": not selected_diff,
                    "selection_jaccard": round(jaccard(top_selected, legacy_selected), 6),
                    "top25_only_selected_ids": sorted(top_selected - legacy_selected),
                    "legacy_only_selected_ids": sorted(legacy_selected - top_selected),
                    "top25_only_selected": names(top_selected - legacy_selected, labels),
                    "legacy_only_selected": names(legacy_selected - top_selected, labels),
                    "target_added_by_legacy_candidate": sorted((legacy_candidates - top_candidates) & set(target_ids)),
                    "target_selected_only_after_legacy_added": sorted((legacy_selected - top_selected) & set(target_ids)),
                },
            }
        )

    detail_rows.sort(key=lambda row: (row["target_label_ids"], row["question_id"]))
    summary: dict[str, Any] = {
        "units_scanned": sum(1 for _ in read_jsonl(units_path)),
        "target_label_count": len(target_map),
        "target_question_count": len(target_questions),
        "top25_evidence_questions": len(top25),
        "top25_plus_legacy_evidence_questions": len(legacy),
        "labels": {},
    }
    for target_id, target_name in target_map.items():
        rows = [row for row in detail_rows if target_id in row["target_label_ids"]]
        top_target_candidates = sum(bool(row["top25"]["candidate_contains_target"]) for row in rows)
        legacy_target_candidates = sum(bool(row["top25_plus_legacy"]["candidate_contains_target"]) for row in rows)
        top_target_selected = sum(bool(row["top25"]["selected_target_ids"]) for row in rows)
        legacy_target_selected = sum(bool(row["top25_plus_legacy"]["selected_target_ids"]) for row in rows)
        summary["labels"][target_id] = {
            "label_name": target_name,
            "historical_question_count": len(rows),
            "top25_candidate_recall": top_target_candidates,
            "top25_plus_legacy_candidate_recall": legacy_target_candidates,
            "top25_selected": top_target_selected,
            "top25_plus_legacy_selected": legacy_target_selected,
            "selected_in_both": sum(bool(row["top25"]["selected_target_ids"]) and bool(row["top25_plus_legacy"]["selected_target_ids"]) for row in rows),
            "selected_only_top25": sum(bool(row["top25"]["selected_target_ids"]) and not bool(row["top25_plus_legacy"]["selected_target_ids"]) for row in rows),
            "selected_only_top25_plus_legacy": sum(not bool(row["top25"]["selected_target_ids"]) and bool(row["top25_plus_legacy"]["selected_target_ids"]) for row in rows),
            "different_full_output": sum(not row["comparison"]["selected_label_sets_equal"] for row in rows),
            "missing_top25_evidence": sum(not row["top25"]["evidence_found"] for row in rows),
            "missing_top25_plus_legacy_evidence": sum(not row["top25_plus_legacy"]["evidence_found"] for row in rows),
        }
    return summary, detail_rows


def write_markdown(path: Path, summary: dict[str, Any], details: list[dict[str, Any]]) -> None:
    lines = [
        "# 六个低覆盖 Label 的 DS 双运行对照",
        "",
        "目标题目按 `legacy_knw_ids` 从 units 文件定位；候选召回和最终选择分别来自纯 Top25 与 Top25+旧 knw_ids 两次 DS 结果。",
        "",
        "## 汇总",
        "",
        "| Label | 历史题数 | Top25候选召回 | Top25+旧ID候选召回 | Top25选中 | Top25+旧ID选中 | 完整Label集合不同 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target_id, item in summary["labels"].items():
        lines.append(
            f"| {item['label_name']} (`{target_id}`) | {item['historical_question_count']} | "
            f"{item['top25_candidate_recall']} | {item['top25_plus_legacy_candidate_recall']} | "
            f"{item['top25_selected']} | {item['top25_plus_legacy_selected']} | {item['different_full_output']} |"
        )
    lines.extend(["", "## 逐题结果", ""])
    for row in details:
        top = row["top25"]
        legacy = row["top25_plus_legacy"]
        diff = row["comparison"]
        lines.extend(
            [
                f"### {row['question_id']} · {'、'.join(row['target_labels'])}",
                "",
                f"- 题型：{row.get('unit_type') or '-'}",
                f"- 题干：{row.get('stem') or '-'}",
                f"- Top25：候选含目标 `{','.join(top['candidate_contains_target']) or '否'}`；目标是否选中：`{'是' if top['selected_target_ids'] else '否'}`",
                f"- Top25+旧ID：候选含目标 `{','.join(legacy['candidate_contains_target']) or '否'}`；目标是否选中：`{'是' if legacy['selected_target_ids'] else '否'}`",
                f"- 两次完整输出：`{'相同' if diff['selected_label_sets_equal'] else '不同'}`；Jaccard={diff['selection_jaccard']}",
                f"- Top25 独有选中：{', '.join(x['label_name'] for x in diff['top25_only_selected']) or '无'}",
                f"- Top25+旧ID 独有选中：{', '.join(x['label_name'] for x in diff['legacy_only_selected']) or '无'}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--top25-evidence", type=Path, required=True)
    parser.add_argument("--legacy-evidence", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target", action="append", metavar="ID=NAME", help="override/add target Label; repeatable")
    args = parser.parse_args()
    target_map = dict(DEFAULT_TARGETS)
    for value in args.target or []:
        if "=" not in value:
            raise SystemExit(f"--target expects ID=NAME: {value}")
        key, name = value.split("=", 1)
        target_map[key] = name
    summary, details = build_report(args.units, args.top25_evidence, args.legacy_evidence, args.labels, target_map)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.run_dir / "question_comparisons.jsonl").open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_markdown(args.run_dir / "summary.md", summary, details)
    print(json.dumps({"run_dir": str(args.run_dir), "target_questions": len(details), "report": str(args.run_dir / 'report.json')}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
