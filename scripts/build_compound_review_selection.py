#!/usr/bin/env python3
"""Build HTML gold-review rows from a complete compound-group adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def by_question(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("question_id") or ""): row for row in read_jsonl(path)}


def label_id(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("label_id") or value.get("id") or "")
    return str(value or "")


def legacy_ids(row: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for field in ("legacy_knw_ids", "legacy_label_ids", "knw_ids"):
        for value in row.get(field) or []:
            value = label_id(value)
            if value:
                result.add(value)
    return result


def card(value: dict[str, Any], labels: dict[str, dict[str, Any]], source: str) -> dict[str, Any]:
    lid = label_id(value)
    catalog = labels.get(lid, {})
    return {
        "label_id": lid,
        "label_name": value.get("label_name") or catalog.get("label_name") or lid,
        "label_path": (value.get("label_path") or catalog.get("label_path") or "").replace("->", "@"),
        "source": source,
    }


def cards(ids: set[str], catalog: dict[str, dict[str, Any]], source: str) -> list[dict[str, Any]]:
    return [card({"label_id": lid}, catalog, source) for lid in sorted(ids)]


def selected_ids(row: dict[str, Any]) -> set[str]:
    return {label_id(value) for value in row.get("selected_labels") or [] if label_id(value)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--top25-candidates", type=Path, required=True)
    parser.add_argument("--legacy-candidates", type=Path, required=True)
    parser.add_argument("--top25-predictions", type=Path, required=True)
    parser.add_argument("--legacy-predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--image-context", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    catalog = {str(row["label_id"]): row for row in read_jsonl(args.labels) if row.get("label_id")}
    units = read_jsonl(args.units)
    top_candidates = by_question(args.top25_candidates)
    legacy_candidates = by_question(args.legacy_candidates)
    top_predictions = by_question(args.top25_predictions)
    legacy_predictions = by_question(args.legacy_predictions)
    image_context = by_question(args.image_context) if args.image_context else {}

    output_rows: list[dict[str, Any]] = []
    for unit in units:
        qid = str(unit.get("question_id") or "")
        top_candidate_items = top_candidates.get(qid, {}).get("candidates") or []
        legacy_candidate_items = legacy_candidates.get(qid, {}).get("candidates") or []
        top_candidate_ids = {label_id(value) for value in top_candidate_items if label_id(value)}
        legacy_candidate_ids = {label_id(value) for value in legacy_candidate_items if label_id(value)}
        top_selected = selected_ids(top_predictions.get(qid, {}))
        legacy_selected = selected_ids(legacy_predictions.get(qid, {}))
        original_ids = legacy_ids(unit)
        image = image_context.get(qid, {})
        source = "复合题组DS精判"

        output_rows.append(
            {
                "question_id": qid,
                "priority_tier": "COMPOUND_LABEL_AUDIT",
                "priority_score": 0,
                "recommended": True,
                "review_reasons": ["compound_label_audit", "ds_volatility"],
                "volatility": {
                    "group": "C_added_legacy_selected" if legacy_selected - top_selected else "A_same_candidate_set",
                    "selection_jaccard": None,
                },
                "stem": unit.get("stem") or "",
                "options": unit.get("options") or "",
                "answer_text": unit.get("answer_text") or unit.get("answer") or "",
                "analysis": unit.get("analysis") or "",
                "unit_type": unit.get("unit_type") or "",
                "parent_id": unit.get("parent_id") or "",
                "sibling_question_ids": unit.get("sibling_question_ids") or [],
                "sub_question_number": unit.get("sub_question_number"),
                "sub_question_count": unit.get("sub_question_count"),
                "stem_image_url": image.get("stem_image_url") or unit.get("stem_image_url") or "",
                "analysis_image_url": image.get("analysis_image_url") or unit.get("analysis_image_url") or "",
                "parent_stem_image_url": image.get("parent_stem_image_url") or unit.get("parent_stem_image_url") or "",
                "parent_analysis_image_url": image.get("parent_analysis_image_url") or unit.get("parent_analysis_image_url") or "",
                "original_knw_label_ids": sorted(original_ids),
                "involved_label_ids": sorted(top_candidate_ids | legacy_candidate_ids | original_ids),
                "top25_labels": [card(value, catalog, source) for value in top_candidate_items],
                "legacy_labels": [card(value, catalog, source) for value in legacy_candidate_items],
                "shared_labels": cards(top_candidate_ids & legacy_candidate_ids, catalog, source),
                "top25_only_labels": cards(top_candidate_ids - legacy_candidate_ids, catalog, source),
                "legacy_only_labels": cards(legacy_candidate_ids - top_candidate_ids, catalog, source),
                "added_candidate_labels": cards(legacy_candidate_ids - top_candidate_ids, catalog, "新增旧knw_ids候选"),
                "selected_added_legacy_labels": cards(legacy_selected - top_selected, catalog, "新增旧Label被选中"),
                "historical_label_note": "复合题组 DS 精判实验题；请直接按当前设问确认 Label。",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(output_rows), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
