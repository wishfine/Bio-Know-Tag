import json
from pathlib import Path

import pytest

from bio_know_tag.filter_processed_s85 import filter_processed_s85


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_reuses_merged_preprocess_but_restores_s85_rows_and_orphan_wrapper(tmp_path: Path):
    s85 = tmp_path / "s85.jsonl"
    updates = tmp_path / "updates.jsonl"
    processed = tmp_path / "processed.jsonl"
    _write(s85, [
        {"question_id": "p", "parent_id": "p", "question_info": {"stem": "s85 父题"}, "knw_ids": ["P"]},
        {"question_id": "c", "parent_id": "p", "question_info": {"stem": "s85 小题"}, "knw_ids": ["C"]},
        {"question_id": "s", "parent_id": "s", "question_info": {"stem": "独立题"}},
        {"question_id": "c2", "parent_id": "q", "question_info": {"stem": "待回连小题"}},
    ])
    _write(updates, [
        {"question_id": "p", "subject": "生物"},
        {"question_id": "c", "subject": "生物"},
        {"question_id": "extra", "subject": "生物"},
        {"question_id": "q", "subject": "生物"},
    ])
    _write(processed, [
        {"question_id": "p", "parent_id": "p", "stem": "更新父题", "knw_ids": ["wrong"], "sub_questions": [
            {"question_id": "c", "parent_id": "p", "stem": "更新小题", "knw_ids": ["wrong"]},
            {"question_id": "extra", "parent_id": "p", "stem": "新增小题"},
        ]},
        {"question_id": "s", "parent_id": "s", "stem": "独立题", "sub_questions": []},
        {"question_id": "q", "parent_id": "q", "stem": "新增父题，不可冒充回连材料", "sub_questions": [
            {"question_id": "c2", "parent_id": "q", "stem": "待回连小题"},
        ]},
    ])

    report = filter_processed_s85(processed, s85, updates, tmp_path / "filtered")
    rows = [json.loads(line) for line in (tmp_path / "filtered" / "questions.jsonl").read_text().splitlines()]
    assert [row["question_id"] for row in rows] == ["p", "s", "q"]
    assert rows[0]["stem"] == "s85 父题"
    assert rows[0]["knw_ids"] == ["P"]
    assert [child["question_id"] for child in rows[0]["sub_questions"]] == ["c"]
    assert rows[0]["sub_questions"][0]["stem"] == "s85 小题"
    assert rows[0]["sub_questions"][0]["knw_ids"] == ["C"]
    assert rows[2]["stem"] == ""
    assert rows[2]["sub_questions"][0]["question_id"] == "c2"
    assert report["s85_question_ids"] == 4
    assert report["restored_updated_s85_records"] == 2
    assert report["removed_non_s85_question_ids"] == 2
    assert report["orphan_context_wrappers"] == 1


def test_fails_closed_if_processed_child_relation_differs_from_s85(tmp_path: Path):
    s85 = tmp_path / "s85.jsonl"
    updates = tmp_path / "updates.jsonl"
    processed = tmp_path / "processed.jsonl"
    _write(s85, [{"question_id": "c", "parent_id": "real", "question_info": {"stem": "题"}}])
    _write(updates, [{"question_id": "c", "subject": "生物"}])
    _write(processed, [{"question_id": "wrong", "parent_id": "wrong", "stem": "", "sub_questions": [
        {"question_id": "c", "parent_id": "wrong", "stem": "题"},
    ]}])
    with pytest.raises(ValueError, match="parent relation mismatch"):
        filter_processed_s85(processed, s85, updates, tmp_path / "filtered")
    assert not (tmp_path / "filtered").exists()
