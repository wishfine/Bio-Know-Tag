import json
from pathlib import Path

import pytest

from bio_know_tag.repair_dedup_parent_context import repair_dedup_parent_context
from bio_know_tag.label_units import build_labeling_derivatives
from bio_know_tag.full_retrieval_units import build_unified_retrieval_units


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_reconnects_recoverable_parent_without_adding_question_and_drops_missing_group(tmp_path: Path):
    processed = tmp_path / "questions.jsonl"
    source = tmp_path / "per_parent.jsonl"
    dedup_raw = tmp_path / "s85.jsonl"
    _write(dedup_raw, [
        {"question_id": value, "parent_id": parent_id}
        for value, parent_id in (
            ("c1", "p1"), ("c2", "p1"), ("c3", "p2"),
            ("s1", "s1"), ("p3", "p3"), ("c4", "p3"),
        )
    ])
    _write(processed, [
        {"question_id": "p1", "parent_id": "p1", "stem": "", "knw_ids": [], "sub_questions": [
            {"question_id": "c1", "parent_id": "p1", "stem": "子题一", "knw_ids": ["L2"]},
            {"question_id": "c2", "parent_id": "p1", "stem": "子题二", "knw_ids": ["L3"]},
        ]},
        {"question_id": "p2", "parent_id": "p2", "stem": "", "sub_questions": [
            {"question_id": "c3", "parent_id": "p2", "stem": "失去背景的子题"},
        ]},
        {"question_id": "s1", "parent_id": "s1", "stem": "独立题", "sub_questions": []},
        {"question_id": "p3", "parent_id": "p3", "stem": "原本保留的父题", "sub_questions": [
            {"question_id": "c4", "parent_id": "p3", "stem": "正常小题"},
        ]},
    ])
    _write(source, [
        {"parent_id": "p1", "child_count": 2, "status": "found_with_text_stem", "parent_stem": "原始父题材料", "options": "A. 甲", "answer": "A", "analysis": "父题解析", "knw_ids": ["L1"], "raw_stem_has_image": True},
        {"parent_id": "p2", "child_count": 1, "status": "missing_from_original", "parent_stem": "", "options": "", "answer": "", "analysis": "", "knw_ids": []},
    ])

    report = repair_dedup_parent_context(processed, source, dedup_raw, tmp_path / "repaired")
    rows = [json.loads(line) for line in (tmp_path / "repaired" / "questions.jsonl").read_text().splitlines()]
    dropped = [json.loads(line) for line in (tmp_path / "repaired" / "dropped_orphan_groups.jsonl").read_text().splitlines()]
    recovered = [json.loads(line) for line in (tmp_path / "repaired" / "recovered_context_parents.jsonl").read_text().splitlines()]

    assert [row["question_id"] for row in rows] == ["p1", "s1", "p3"]
    assert rows[0]["stem"] == "原始父题材料"
    assert rows[0]["options"] == "A. 甲"
    assert rows[0]["knw_ids"] == ["L1"]
    assert rows[0]["context_only_recovered_parent"] is True
    assert [child["question_id"] for child in rows[0]["sub_questions"]] == ["c1", "c2"]
    assert rows[1] == {"question_id": "s1", "parent_id": "s1", "stem": "独立题", "sub_questions": []}
    assert dropped == [{"parent_id": "p2", "child_question_ids": ["c3"], "reason": "missing_from_original"}]
    assert recovered == [{"parent_id": "p1", "child_question_ids": ["c1", "c2"], "context_only": True, "text_available": True}]
    assert (tmp_path / "repaired" / "orphan_parents.jsonl").read_text() == ""
    assert report["recovered_context_parents"] == 1
    assert report["children_with_recovered_context"] == 2
    assert report["dropped_orphan_children"] == 1
    assert report["content_deduplication_applied"] is False
    assert report["s85_question_ids"] == 6
    assert report["output_s85_question_ids"] == 5

    labels = tmp_path / "labels.jsonl"
    _write(labels, [
        {"label_id": value, "label_name": value, "reference_strategy": {"关键词策略代码": "K1"}}
        for value in ("L1", "L2", "L3")
    ])
    units_dir = tmp_path / "units"
    build_labeling_derivatives(
        tmp_path / "repaired" / "questions.jsonl", labels,
        tmp_path / "repaired" / "orphan_parents.jsonl", units_dir,
        progress_every=0,
    )
    unit_rows = [json.loads(line) for line in (units_dir / "label_units.jsonl").read_text().splitlines()]
    plans = [json.loads(line) for line in (units_dir / "parent_aggregation.jsonl").read_text().splitlines()]
    by_id = {row["question_id"]: row for row in unit_rows}
    assert "c3" not in by_id
    assert by_id["c1"]["unit_type"] == "sub_question"
    assert by_id["c1"]["parent_stem"] == "原始父题材料"
    assert by_id["c1"]["flags"]["image_context_missing"] is True
    assert plans[0]["legacy_knw_ids"] == ["L1"]
    assert plans[0]["flags"]["context_only_recovered_parent"] is True
    assert plans[0]["flags"]["recovered_parent_has_image"] is True
    unified_dir = tmp_path / "unified"
    build_unified_retrieval_units(
        units_dir / "label_units.jsonl", units_dir / "parent_aggregation.jsonl", unified_dir
    )
    retrieval = [json.loads(line) for line in (unified_dir / "retrieval_units.jsonl").read_text().splitlines()]
    recovered_parent_units = [row for row in retrieval if row["unit_type"] == "composite_parent_extra" and row["question_id"] == "p1"]
    assert len(recovered_parent_units) == 1
    assert recovered_parent_units[0]["flags"]["context_only_recovered_parent"] is True
    assert recovered_parent_units[0]["flags"]["image_context_missing"] is True


def test_repair_refuses_mismatched_audit_and_processed_groups(tmp_path: Path):
    processed = tmp_path / "questions.jsonl"
    source = tmp_path / "per_parent.jsonl"
    dedup_raw = tmp_path / "s85.jsonl"
    _write(dedup_raw, [{"question_id": "c1", "parent_id": "p1"}])
    _write(processed, [{"question_id": "p1", "parent_id": "p1", "stem": "", "sub_questions": [
        {"question_id": "c1", "parent_id": "p1", "stem": "子题"},
    ]}])
    _write(source, [{"parent_id": "p1", "child_count": 2, "status": "found_with_text_stem", "parent_stem": "材料"}])
    with pytest.raises(ValueError, match="child count mismatch"):
        repair_dedup_parent_context(processed, source, dedup_raw, tmp_path / "repaired")
    assert not (tmp_path / "repaired" / "report.json").exists()


def test_repair_rejects_processed_questions_not_in_s85(tmp_path: Path):
    processed = tmp_path / "questions.jsonl"
    source = tmp_path / "per_parent.jsonl"
    dedup_raw = tmp_path / "s85.jsonl"
    _write(dedup_raw, [{"question_id": "q1", "parent_id": "q1"}])
    _write(processed, [
        {"question_id": "q1", "parent_id": "q1", "stem": "原题", "sub_questions": []},
        {"question_id": "extra", "parent_id": "extra", "stem": "合并进来的新题", "sub_questions": []},
    ])
    _write(source, [{"parent_id": "p1", "child_count": 1, "status": "missing_from_original"}])
    with pytest.raises(ValueError, match="processed question ID not in s85"):
        repair_dedup_parent_context(processed, source, dedup_raw, tmp_path / "repaired")


def test_repair_rejects_duplicate_ids_in_s85(tmp_path: Path):
    processed = tmp_path / "questions.jsonl"
    source = tmp_path / "per_parent.jsonl"
    dedup_raw = tmp_path / "s85.jsonl"
    _write(dedup_raw, [
        {"question_id": "q1", "parent_id": "q1"},
        {"question_id": "q1", "parent_id": "q1"},
    ])
    _write(processed, [{"question_id": "q1", "parent_id": "q1", "stem": "题干", "sub_questions": []}])
    _write(source, [{"parent_id": "p1", "child_count": 1, "status": "missing_from_original"}])
    with pytest.raises(ValueError, match="duplicate question_id in s85"):
        repair_dedup_parent_context(processed, source, dedup_raw, tmp_path / "repaired")
    assert not (tmp_path / "repaired").exists()


def test_image_only_source_parent_keeps_children_for_review(tmp_path: Path):
    processed = tmp_path / "questions.jsonl"
    source = tmp_path / "per_parent.jsonl"
    dedup_raw = tmp_path / "s85.jsonl"
    _write(dedup_raw, [{"question_id": "c1", "parent_id": "p1"}])
    _write(processed, [{"question_id": "p1", "parent_id": "p1", "stem": "", "sub_questions": [
        {"question_id": "c1", "parent_id": "p1", "stem": "图中哪个结构"},
    ]}])
    _write(source, [{
        "parent_id": "p1", "child_count": 1, "status": "found_image_only_stem",
        "parent_stem": "", "options": "", "answer": "", "analysis": "",
        "knw_ids": [], "raw_material_has_image": True,
    }])
    report = repair_dedup_parent_context(processed, source, dedup_raw, tmp_path / "repaired")
    row = json.loads((tmp_path / "repaired" / "questions.jsonl").read_text())
    assert row["sub_questions"][0]["question_id"] == "c1"
    assert row["context_only_recovered_parent"] is True
    assert row["recovered_parent_text_missing"] is True
    assert row["recovered_parent_has_image"] is True
    assert report["dropped_orphan_children"] == 0
    assert report["retained_source_parents_without_text"] == 1

    labels = tmp_path / "labels.jsonl"
    _write(labels, [{"label_id": "L1", "label_name": "一", "reference_strategy": {}}])
    units_dir = tmp_path / "units"
    build_labeling_derivatives(
        tmp_path / "repaired" / "questions.jsonl", labels,
        tmp_path / "repaired" / "orphan_parents.jsonl", units_dir,
        progress_every=0,
    )
    unit = json.loads((units_dir / "label_units.jsonl").read_text())
    assert unit["flags"]["parent_context_missing"] is True
    assert unit["flags"]["image_context_missing"] is True
    unified = tmp_path / "unified"
    combined = build_unified_retrieval_units(
        units_dir / "label_units.jsonl", units_dir / "parent_aggregation.jsonl", unified
    )
    assert combined["retrieval_label_units"] == 1
    assert combined["retrieval_parent_extra_units"] == 0


def test_repair_rejects_wrong_parent_relation_even_when_ids_match(tmp_path: Path):
    processed = tmp_path / "questions.jsonl"
    source = tmp_path / "per_parent.jsonl"
    dedup_raw = tmp_path / "s85.jsonl"
    _write(dedup_raw, [{"question_id": "c1", "parent_id": "other"}])
    _write(processed, [{"question_id": "p1", "parent_id": "p1", "stem": "", "sub_questions": [
        {"question_id": "c1", "parent_id": "p1", "stem": "子题"},
    ]}])
    _write(source, [{
        "parent_id": "p1", "child_count": 1, "status": "found_with_text_stem",
        "parent_stem": "材料", "options": "", "answer": "", "analysis": "", "knw_ids": [],
    }])
    with pytest.raises(ValueError, match="parent relation mismatch"):
        repair_dedup_parent_context(processed, source, dedup_raw, tmp_path / "repaired")
