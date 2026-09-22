#!/usr/bin/env python3
"""Build a prioritized question pool for combined teacher gold-label review."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_volatility(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(re.search(r'<script id="review-data" type="application/json">(.*?)</script>', text, re.S).group(1))
    return {str(row["question_id"]): row for row in payload}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volatility-html", type=Path, required=True)
    parser.add_argument("--stability-json", type=Path, required=True)
    parser.add_argument("--ablation-changed-pairs", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--recommended-count", type=int, default=3000)
    args = parser.parse_args()

    volatility = load_volatility(args.volatility_html)
    stability = json.loads(args.stability_json.read_text(encoding="utf-8"))
    question_rank = {
        str(row["question_id"]): row
        for row in stability["question_ranking"]
    }
    stability_rank = {
        str(row["question_id"]): index + 1
        for index, row in enumerate(stability["question_ranking"])
    }
    changed_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in jsonl(args.ablation_changed_pairs):
        changed_by_question[str(row["question_id"])].append(row)

    rows: list[dict[str, Any]] = []
    all_ids = set(volatility) | set(question_rank) | set(changed_by_question)
    for qid in all_ids:
        v = volatility.get(qid, {})
        s = question_rank.get(qid, {})
        pairs = changed_by_question.get(qid, [])
        status_pairs = [
            pair for pair in pairs
            if pair.get("status") in {"false_to_true", "true_to_false"}
        ]
        involved = set()
        original = set(str(x) for x in v.get("original_knw_label_ids") or [])
        for key in ("top25_labels", "legacy_labels", "added_candidate_labels", "selected_added_legacy_labels"):
            involved.update(str(x.get("label_id")) for x in v.get(key) or [])
        involved.update(str(pair.get("label_id")) for pair in pairs if pair.get("label_id"))
        original.update(str(pair.get("label_id")) for pair in pairs if pair.get("label_id"))
        has_volatility = qid in volatility
        has_ablation = bool(pairs)
        volatility_score = float(s.get("instability_score") or 0.0)
        max_abs_delta = max((abs(float(pair.get("score_delta") or 0.0)) for pair in pairs), default=0.0)
        status_change_count = len(status_pairs)
        if has_volatility and has_ablation:
            tier = "S_INTERSECTION"
        elif status_change_count:
            tier = "A_ABLATION_STATUS_CHANGE"
        elif has_ablation:
            tier = "B_ABLATION_SCORE_SHIFT"
        elif volatility_score >= 0.50:
            tier = "B_HIGH_VOLATILITY"
        else:
            tier = "C_VOLATILITY"
        priority = (
            (1000 if has_volatility and has_ablation else 0)
            + (500 if status_change_count else 0)
            + min(300, round(max_abs_delta * 300))
            + round(volatility_score * 100)
        )
        row = {
            "question_id": qid,
            "priority_tier": tier,
            "priority_score": priority,
            "recommended": False,
            "review_reasons": [
                *( ["ds_volatility"] if has_volatility else [] ),
                *( ["definition_ablation"] if has_ablation else [] ),
                *( ["definition_status_changed"] if status_change_count else [] ),
            ],
            "volatility": {
                "group": v.get("perturbation_group"),
                "selection_jaccard": v.get("selection_jaccard"),
                "stability_rank": stability_rank.get(qid),
                "instability_score": volatility_score,
            } if has_volatility else None,
            "definition_ablation": {
                "changed_pair_count": len(pairs),
                "status_change_count": status_change_count,
                "max_abs_score_delta": max_abs_delta,
                "changes": pairs,
            } if has_ablation else None,
            "stem": v.get("stem", "") or (pairs[0].get("stem", "") if pairs else "") or s.get("stem", ""),
            "options": v.get("options", "") or (pairs[0].get("options", "") if pairs else "") or s.get("options", ""),
            "answer_text": v.get("answer_text", "") or (pairs[0].get("answer_text", "") if pairs else ""),
            "analysis": v.get("analysis", "") or (pairs[0].get("analysis", "") if pairs else "") or s.get("analysis", ""),
            "unit_type": v.get("unit_type") or "standalone",
            "stem_image_url": v.get("stem_image_url", ""),
            "analysis_image_url": v.get("analysis_image_url", ""),
            "original_knw_label_ids": sorted(original),
            "involved_label_ids": sorted(involved),
            "historical_label_note": "释义消融配对中的测试Label及波动结果中的可识别旧Label；最终需教师确认。",
        }
        rows.append(row)

    rows.sort(key=lambda row: (-row["priority_score"], row["priority_tier"], row["question_id"]))
    for row in rows[: max(0, args.recommended_count)]:
        row["recommended"] = True
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "total_unique_questions": len(rows),
        "recommended_count": sum(row["recommended"] for row in rows),
        "tier_counts": dict(Counter(row["priority_tier"] for row in rows)),
        "reason_counts": dict(Counter(reason for row in rows for reason in row["review_reasons"])),
        "recommended_tier_counts": dict(Counter(row["priority_tier"] for row in rows if row["recommended"])),
        "output_jsonl": str(args.output_jsonl),
        "note": "This is a selection manifest only; it does not generate or modify the review HTML.",
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
