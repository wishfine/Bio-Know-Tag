import json
from pathlib import Path

from bio_know_tag.update_merge import merge_question_updates


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_merge_overlays_biology_rows_and_appends_new_rows(tmp_path: Path):
    base = tmp_path / "biology.jsonl"
    updates = tmp_path / "all.jsonl"
    output = tmp_path / "merged.jsonl"
    report_path = tmp_path / "report.json"
    _write_jsonl(
        base,
        [
            {"question_id": "q1", "subject": "生物", "knw_ids": ["old"]},
            {"question_id": "q2", "subject": "生物", "knw_ids": ["keep"]},
        ],
    )
    _write_jsonl(
        updates,
        [
            {"question_id": "geo", "subject": "地理", "knw_ids": ["ignore"]},
            {"question_id": "q1", "subject": "生物", "knw_ids": ["new"]},
            {"question_id": "q3", "subject": "生物", "knw_ids": ["added"]},
        ],
    )

    report = merge_question_updates(base, updates, output, report_path, subject="生物")

    assert _read_jsonl(output) == [
        {"question_id": "q1", "subject": "生物", "knw_ids": ["new"]},
        {"question_id": "q2", "subject": "生物", "knw_ids": ["keep"]},
        {"question_id": "q3", "subject": "生物", "knw_ids": ["added"]},
    ]
    assert report["base_rows"] == 2
    assert report["update_rows_scanned"] == 3
    assert report["subject_update_rows"] == 2
    assert report["existing_rows_updated"] == 1
    assert report["new_rows_added"] == 1
    assert report["label_changed_existing_rows"] == 1
    assert report["output_rows"] == 3
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_merge_uses_last_duplicate_update_and_reports_bad_rows(tmp_path: Path):
    base = tmp_path / "biology.jsonl"
    updates = tmp_path / "all.jsonl"
    output = tmp_path / "merged.jsonl"
    report_path = tmp_path / "report.json"
    _write_jsonl(base, [{"question_id": "q1", "subject": "生物", "knw_ids": ["old"]}])
    updates.write_text(
        json.dumps({"question_id": "q1", "subject": "生物", "knw_ids": ["first"]}, ensure_ascii=False)
        + "\n"
        + "{bad\n"
        + json.dumps({"question_id": "q1", "subject": "生物", "knw_ids": ["last"]}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    report = merge_question_updates(base, updates, output, report_path, subject="生物")

    assert _read_jsonl(output)[0]["knw_ids"] == ["last"]
    assert report["duplicate_update_question_ids"] == 1
    assert report["update_errors"] == 1

