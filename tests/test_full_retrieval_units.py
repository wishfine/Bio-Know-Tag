import json
from pathlib import Path

import pytest

from bio_know_tag.full_retrieval_units import build_unified_retrieval_units


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_unified_units_include_parent_material_and_audit_missing_stems(tmp_path: Path):
    units = tmp_path / "label_units.jsonl"
    plans = tmp_path / "parent_aggregation.jsonl"
    out = tmp_path / "out"
    _write(units, [
        {"question_id": "solo", "parent_id": "solo", "unit_type": "standalone", "stem": "独立题", "options": "", "analysis": ""},
        {"question_id": "c1", "parent_id": "p1", "unit_type": "sub_question", "parent_stem": "共同材料", "stem": "", "options": "A. 正确", "analysis": "解析", "flags": {"image_context_missing": True}},
        {"question_id": "c2", "parent_id": "p1", "unit_type": "sub_question", "parent_stem": "共同材料", "stem": "小题二", "options": "", "analysis": ""},
        {"question_id": "empty", "parent_id": "empty", "unit_type": "standalone", "stem": "", "options": "", "analysis": "", "answer_text": "A"},
        {"question_id": "drop", "parent_id": "drop", "unit_type": "standalone", "stem": "重复题", "options": "", "analysis": ""},
    ])
    _write(plans, [
        {"question_id": "p1", "parent_stem": "父题<图>", "options": "", "analysis": "", "child_question_ids": ["c1", "c2"], "legacy_knw_ids": ["L1"]},
        {"question_id": "p2", "parent_stem": "另一个父题", "options": "", "analysis": "", "child_question_ids": ["drop"]},
    ])

    report = build_unified_retrieval_units(
        units, plans, out, keep_question_ids={"solo", "c1", "c2", "empty"}
    )
    rows = [json.loads(line) for line in (out / "retrieval_units.jsonl").read_text().splitlines()]
    parents = [json.loads(line) for line in (out / "parent_aggregation.jsonl").read_text().splitlines()]
    audit = [json.loads(line) for line in (out / "content_review.jsonl").read_text().splitlines()]
    excluded = [json.loads(line) for line in (out / "text_ineligible.jsonl").read_text().splitlines()]
    assert [row["question_id"] for row in rows] == ["solo", "c1", "c2", "p1"]
    assert rows[-1]["unit_type"] == "composite_parent_extra"
    assert rows[-1]["legacy_knw_ids"] == ["L1"]
    assert parents[0]["child_question_ids"] == ["c1", "c2"]
    assert len(parents) == 1
    assert {row["question_id"] for row in audit} == {"c1", "empty"}
    assert [row["question_id"] for row in excluded] == ["empty"]
    assert report["deduplicated_units_removed"] == 1
    assert report["retrieval_units"] == 4
    assert report["text_ineligible_label_units"] == 1
    assert report["parents_without_kept_children"] == 1


def test_parent_with_no_text_stays_in_aggregation_but_not_retrieval(tmp_path: Path):
    units = tmp_path / "units.jsonl"
    plans = tmp_path / "plans.jsonl"
    _write(units, [{"question_id": "c1", "parent_id": "p1", "unit_type": "sub_question", "stem": "子题"}])
    _write(plans, [{"question_id": "p1", "parent_stem": "", "options": "", "answer_text": "", "analysis": "", "child_question_ids": ["c1"]}])
    report = build_unified_retrieval_units(units, plans, tmp_path / "out")
    assert report["parents_without_text_material"] == 1
    assert report["retained_parent_plans"] == 1
    assert report["retrieval_parent_extra_units"] == 0


def test_unknown_keep_id_is_rejected(tmp_path: Path):
    units = tmp_path / "units.jsonl"
    plans = tmp_path / "plans.jsonl"
    _write(units, [{"question_id": "q1", "parent_id": "q1", "unit_type": "standalone", "stem": "题干"}])
    _write(plans, [])
    with pytest.raises(ValueError, match="absent from label units"):
        build_unified_retrieval_units(
            units, plans, tmp_path / "out", keep_question_ids={"q1", "unknown"}
        )
