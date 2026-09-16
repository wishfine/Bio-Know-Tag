import json
from pathlib import Path

from bio_know_tag.image_context import (
    audit_label_unit_image_context,
    iter_json_object_items,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_iter_json_object_items_streams_across_small_chunks(tmp_path: Path):
    source = tmp_path / "images.json"
    value = {
        "题目一": {"stemImageUrl": "https://x/一.png", "analysisImageUrl": ""},
        "q2": {"stemImageUrl": "", "analysisImageUrl": "https://x/a.png"},
    }
    source.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    assert list(iter_json_object_items(source, chunk_size=11)) == list(value.items())


def test_image_context_audit_separates_empty_image_only_and_review_units(tmp_path: Path):
    units = tmp_path / "units.jsonl"
    images = tmp_path / "images.json"
    run_dir = tmp_path / "run"
    _write_jsonl(
        units,
        [
            {
                "question_id": "q1",
                "parent_id": "q1",
                "unit_type": "standalone",
                "parent_stem": "",
                "stem": "",
                "options": "",
                "answer_text": "",
                "analysis": "",
            },
            {
                "question_id": "q2",
                "parent_id": "q2",
                "unit_type": "standalone",
                "parent_stem": "",
                "stem": "一段材料",
                "options": "",
                "answer_text": "",
                "analysis": "",
            },
            {
                "question_id": "c3",
                "parent_id": "p3",
                "unit_type": "sub_question",
                "parent_stem": "共同材料",
                "stem": "回答问题",
                "options": "",
                "answer_text": "答案",
                "analysis": "解析",
            },
            {
                "question_id": "c4",
                "parent_id": "p4",
                "unit_type": "sub_question",
                "parent_stem": "共同材料",
                "stem": "",
                "options": "",
                "answer_text": "",
                "analysis": "",
            },
        ],
    )
    images.write_text(
        json.dumps(
            {
                "q1": {"stemImageUrl": "https://x/q1.png", "analysisImageUrl": ""},
                "p3": {"stemImageUrl": "https://x/p3.png", "analysisImageUrl": ""},
                "other-subject": {"stemImageUrl": "https://x/other.png", "analysisImageUrl": ""},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = audit_label_unit_image_context(units, images, run_dir, progress_every=0)

    rows = [json.loads(line) for line in (run_dir / "image_context.jsonl").read_text().splitlines()]
    by_id = {row["question_id"]: row for row in rows}
    assert by_id["q1"]["content_status"] == "image_only_no_text_stem"
    assert by_id["q1"]["eligible_for_text_labeling"] is False
    assert by_id["q2"]["content_status"] == "stem_only_needs_review"
    assert by_id["q2"]["eligible_for_text_labeling"] is True
    assert by_id["q2"]["needs_content_review"] is True
    assert by_id["c3"]["parent_stem_image_url"] == "https://x/p3.png"
    assert by_id["c3"]["content_status"] == "text_with_image_context"
    assert by_id["c4"]["content_status"] == "current_question_missing"
    assert by_id["c4"]["eligible_for_text_labeling"] is False
    assert report["units"] == 4
    assert report["target_question_ids"] == 6
    assert report["image_records_scanned"] == 3
    assert report["target_ids_with_image_record"] == 2
    assert report["excluded_from_text_labeling"] == 2
    assert report["needs_content_review"] == 2
    assert len((run_dir / "excluded_units.jsonl").read_text().splitlines()) == 2
    assert len((run_dir / "review_units.jsonl").read_text().splitlines()) == 2
