import json
from pathlib import Path

from bio_know_tag.orphan_parent_source import audit_orphan_parent_source


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_audit_distinguishes_removed_parent_missing_parent_and_image_only(tmp_path: Path):
    orphans = tmp_path / "orphans.jsonl"
    source = tmp_path / "original.raw.jsonl"
    _write(orphans, [
        {"orphan_parent_id": "p1", "quality": {"sub_question_count": 2}},
        {"orphan_parent_id": "p2", "quality": {"sub_question_count": 1}},
        {"orphan_parent_id": "p3", "quality": {"sub_question_count": 3}},
    ])
    _write(source, [
        {"question_id": "p1", "parent_id": "p1", "knw_ids": ["L1"], "question_info": {"stem": "<p>植物细胞失水</p>", "options": [], "answer": "A", "analysis": "<img src='figure.png'>"}},
        {"question_id": "p2", "parent_id": "p2", "question_info": {"stem": "<img src='figure.png'>", "options": [], "analysis": ""}},
        {"question_id": "c1", "parent_id": "p1", "question_info": {"stem": "子题"}},
    ])

    report = audit_orphan_parent_source(orphans, source, tmp_path / "audit")
    rows = [json.loads(line) for line in (tmp_path / "audit" / "per_parent.jsonl").read_text().splitlines()]
    by_id = {row["parent_id"]: row for row in rows}

    assert report["orphan_parent_ids"] == 3
    assert report["orphan_child_count"] == 6
    assert report["found_in_original"] == 2
    assert report["missing_from_original"] == 1
    assert report["children_with_original_parent"] == 3
    assert report["children_missing_original_parent"] == 3
    assert report["found_with_text_stem"] == 1
    assert report["found_image_only_stem"] == 1
    assert by_id["p1"]["status"] == "found_with_text_stem"
    assert by_id["p1"]["parent_stem"] == "植物细胞失水"
    assert by_id["p1"]["knw_ids"] == ["L1"]
    assert by_id["p1"]["answer"] == "A"
    assert by_id["p1"]["raw_material_has_image"] is True
    assert by_id["p2"]["status"] == "found_image_only_stem"
    assert by_id["p3"]["status"] == "missing_from_original"


def test_audit_does_not_treat_a_child_record_as_recoverable_parent(tmp_path: Path):
    orphans = tmp_path / "orphans.jsonl"
    source = tmp_path / "original.jsonl"
    _write(orphans, [{"orphan_parent_id": "p1", "quality": {"sub_question_count": 1}}])
    _write(source, [{"question_id": "p1", "parent_id": "another", "question_info": {"stem": "不是父题记录"}}])
    report = audit_orphan_parent_source(orphans, source, tmp_path / "audit", progress_every=0)
    row = json.loads((tmp_path / "audit" / "per_parent.jsonl").read_text())
    assert report["found_nonself_parent_row"] == 1
    assert row["status"] == "found_nonself_parent_row"
