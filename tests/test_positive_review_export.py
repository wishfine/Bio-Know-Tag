import json
from pathlib import Path

import pytest

from bio_know_tag.positive_review_export import export_positive_review_by_label


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_export_writes_one_readable_json_per_label_sorted_by_score(tmp_path: Path):
    labels = tmp_path / "labels.jsonl"
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results.jsonl"
    output = tmp_path / "export"
    _write_jsonl(
        labels,
        [
            {
                "label_id": "L1",
                "label_name": "DNA/复制",
                "label_path": "知识点->DNA/复制",
                "label_type": "知识",
                "definition": "复制定义",
                "core_concepts": "复制核心",
                "common_assessments": "复制考法",
                "distinctions": "复制边界",
            },
            {
                "label_id": "L2",
                "label_name": "转录",
                "label_path": "知识点->转录",
                "label_type": "知识",
                "definition": "转录定义",
                "core_concepts": "转录核心",
                "common_assessments": "转录考法",
                "distinctions": "转录边界",
            },
        ],
    )
    _write_jsonl(
        tasks,
        [
            {
                "pair_id": "q1::L1",
                "question_id": "q1",
                "label_id": "L1",
                "label_name": "DNA/复制",
                "unit_type": "standalone",
                "parent_stem": "",
                "stem": "题目1",
                "options": "A. 1",
                "answer_text": "A",
                "analysis": "解析1",
                "flags": {"image_context_missing": False},
                "expected_relation": "legacy_positive",
            },
            {
                "pair_id": "q2::L1",
                "question_id": "q2",
                "label_id": "L1",
                "label_name": "DNA/复制",
                "unit_type": "standalone",
                "parent_stem": "",
                "stem": "题目2",
                "options": "",
                "answer_text": "B",
                "analysis": "解析2",
                "flags": {},
                "expected_relation": "legacy_positive",
            },
            {
                "pair_id": "q3::L2",
                "question_id": "q3",
                "label_id": "L2",
                "label_name": "转录",
                "unit_type": "standalone",
                "parent_stem": "",
                "stem": "题目3",
                "options": "",
                "answer_text": "C",
                "analysis": "解析3",
                "flags": {},
                "expected_relation": "legacy_positive",
            },
        ],
    )
    _write_jsonl(
        results,
        [
            {
                "task_id": "q1::L1",
                "question_id": "q1",
                "label_id": "L1",
                "match": True,
                "relevance_score": 0.9,
                "model": "fake",
            },
            {
                "task_id": "q2::L1",
                "question_id": "q2",
                "label_id": "L1",
                "match": False,
                "relevance_score": 0.0,
                "model": "fake",
            },
            {
                "task_id": "q3::L2",
                "question_id": "q3",
                "label_id": "L2",
                "match": False,
                "relevance_score": 0.4,
                "model": "fake",
            },
        ],
    )

    report = export_positive_review_by_label(tasks, results, labels, output)

    assert report["labels_exported"] == 2
    assert report["questions_exported"] == 3
    index = json.loads((output / "index.json").read_text())
    assert index["labels"][0]["label_id"] == "L1"
    assert "/" not in index["labels"][0]["file"]
    label_file = output / index["labels"][0]["file"]
    payload = json.loads(label_file.read_text())
    assert payload["label"]["definition"] == "复制定义"
    assert payload["summary"]["basically_irrelevant_count"] == 1
    assert [row["question_id"] for row in payload["questions"]] == ["q2", "q1"]
    assert payload["questions"][0]["ds_judgment"] == {
        "match": False,
        "model": "fake",
        "relevance_score": 0.0,
        "score_band": "0.00",
    }
    assert payload["questions"][0]["stem"] == "题目2"
    assert payload["questions"][0]["analysis"] == "解析2"


def test_export_rejects_missing_ds_result(tmp_path: Path):
    labels = tmp_path / "labels.jsonl"
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results.jsonl"
    _write_jsonl(
        labels,
        [{"label_id": "L1", "label_name": "L1", "definition": "d"}],
    )
    _write_jsonl(
        tasks,
        [{"pair_id": "q1::L1", "question_id": "q1", "label_id": "L1"}],
    )
    results.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="missing DS results"):
        export_positive_review_by_label(tasks, results, labels, tmp_path / "out")
