import json
from pathlib import Path

from bio_know_tag.audit import audit_orphan_parents
from bio_know_tag.questions import process_jsonl


def question_info(stem: str, answer: str = "") -> dict:
    return {"stem": stem, "options": [], "analysis": "解析", "answer": answer}


def test_audit_orphan_parents_finds_missing_parent_and_checks_context(tmp_path: Path):
    raw_path = tmp_path / "raw.jsonl"
    processed_path = tmp_path / "processed.jsonl"
    preprocess_report = tmp_path / "preprocess-report.json"
    evidence_path = tmp_path / "orphan-parents.jsonl"
    audit_report = tmp_path / "audit-report.json"
    rows = [
        {
            "question_id": "p1",
            "parent_id": "p1",
            "question_info": question_info("正常父题材料"),
        },
        {
            "question_id": "c1",
            "parent_id": "p1",
            "question_info": question_info("正常小题", "A"),
        },
        {
            "question_id": "c2",
            "parent_id": "missing-parent",
            "question_info": question_info("孤儿小题", "B"),
            "parent_question_info": {
                "question_info": question_info("补全的共同材料")
            },
        },
        {
            "question_id": "solo",
            "parent_id": "solo",
            "question_info": question_info("独立题", "C"),
        },
    ]
    raw_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    process_jsonl(
        raw_path,
        processed_path,
        preprocess_report,
        progress_every=0,
    )

    report = audit_orphan_parents(
        raw_path,
        processed_path,
        evidence_path,
        audit_report,
        progress_every=0,
    )

    assert report == {
        "raw_input": 4,
        "raw_error": 0,
        "unique_question_ids": 4,
        "referenced_parent_ids": 2,
        "orphan_parent_ids": 1,
        "orphan_records_found": 1,
        "orphan_records_missing": 0,
        "orphan_with_parent_stem": 1,
        "orphan_without_parent_stem": 0,
        "orphan_child_count": 1,
        "orphan_children_without_stem": 0,
        "processed_input": 3,
        "processed_error": 0,
    }
    evidence = [
        json.loads(line)
        for line in evidence_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(evidence) == 1
    assert evidence[0]["orphan_parent_id"] == "missing-parent"
    assert evidence[0]["parent"]["stem"] == "补全的共同材料"
    assert evidence[0]["quality"]["parent_stem_present"] is True
    assert json.loads(audit_report.read_text(encoding="utf-8")) == report

