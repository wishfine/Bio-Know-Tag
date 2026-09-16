import json
from pathlib import Path

import pytest

from bio_know_tag.hard_negatives import (
    analyze_hard_negative_results,
    build_verified_sibling_hard_negatives,
    combine_positive_and_negative_assessments,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _labels() -> list[dict]:
    return [
        {"label_id": "L1", "label_name": "标签一", "label_path": "知识点->模块->父级->标签一"},
        {"label_id": "L2", "label_name": "标签二", "label_path": "知识点->模块->父级->标签二"},
        {"label_id": "L3", "label_name": "标签三", "label_path": "知识点->其他->标签三"},
    ]


def test_build_hard_negatives_uses_verified_sibling_and_excludes_existing_target(tmp_path: Path):
    tasks = tmp_path / "positive_tasks.jsonl"
    results = tmp_path / "positive_results.jsonl"
    units = tmp_path / "units.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "hard-negatives"
    _write_jsonl(labels, _labels())
    _write_jsonl(
        tasks,
        [
            {"pair_id": "q1::L1", "question_id": "q1", "label_id": "L1", "unit_type": "standalone", "stem": "题一", "options": "", "answer_text": "A", "analysis": "解析一", "parent_stem": ""},
            {"pair_id": "q2::L1", "question_id": "q2", "label_id": "L1", "unit_type": "standalone", "stem": "题二", "options": "", "answer_text": "B", "analysis": "解析二", "parent_stem": ""},
        ],
    )
    _write_jsonl(
        results,
        [
            {"task_id": "q1::L1", "label_id": "L1", "question_id": "q1", "match": True, "relevance_score": 0.95},
            {"task_id": "q2::L1", "label_id": "L1", "question_id": "q2", "match": True, "relevance_score": 0.90},
        ],
    )
    _write_jsonl(
        units,
        [
            {"question_id": "q1", "unit_type": "standalone", "legacy_candidate_ids": ["L1"]},
            {"question_id": "q2", "unit_type": "standalone", "legacy_candidate_ids": ["L1", "L2"]},
        ],
    )

    report = build_verified_sibling_hard_negatives(
        tasks,
        results,
        units,
        labels,
        output,
        negatives_per_label=10,
        min_source_score=0.8,
    )

    rows = list(map(json.loads, (output / "hard_negative_samples.jsonl").open()))
    assert len(rows) == 1
    assert rows[0]["question_id"] == "q1"
    assert rows[0]["label_id"] == "L2"
    assert rows[0]["source_label_ids"] == ["L1"]
    assert rows[0]["expected_relation"] == "verified_sibling_hard_negative"
    assert report["samples"] == 1
    assert report["target_already_in_legacy_excluded"] == 1
    assert report["labels_with_hard_negatives"] == 1


def test_build_hard_negatives_requires_complete_positive_results(tmp_path: Path):
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results.jsonl"
    units = tmp_path / "units.jsonl"
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(labels, _labels())
    _write_jsonl(tasks, [{"pair_id": "q1::L1", "question_id": "q1", "label_id": "L1", "unit_type": "standalone"}])
    _write_jsonl(results, [])
    _write_jsonl(units, [{"question_id": "q1", "legacy_candidate_ids": ["L1"]}])

    with pytest.raises(ValueError, match="positive results are incomplete"):
        build_verified_sibling_hard_negatives(tasks, results, units, labels, tmp_path / "out")


def test_analyze_hard_negative_results_reports_false_acceptance_and_confusion(tmp_path: Path):
    samples = tmp_path / "samples.jsonl"
    results = tmp_path / "results.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "analysis"
    _write_jsonl(labels, _labels())
    _write_jsonl(
        samples,
        [
            {"pair_id": "q1::L2", "question_id": "q1", "label_id": "L2", "source_label_ids": ["L1"]},
            {"pair_id": "q2::L2", "question_id": "q2", "label_id": "L2", "source_label_ids": ["L1"]},
        ],
    )
    _write_jsonl(
        results,
        [
            {"task_id": "q1::L2", "label_id": "L2", "question_id": "q1", "match": True, "relevance_score": 0.9},
            {"task_id": "q2::L2", "label_id": "L2", "question_id": "q2", "match": False, "relevance_score": 0.1},
        ],
    )

    report = analyze_hard_negative_results(samples, results, labels, output)

    assert report["false_accept_rate"] == 0.5
    row = json.loads((output / "per_label.jsonl").read_text())
    assert row["label_id"] == "L2"
    assert row["false_accept_rate"] == 0.5
    confusion = json.loads((output / "confusion_pairs.jsonl").read_text())
    assert confusion["source_label_id"] == "L1"
    assert confusion["target_label_id"] == "L2"
    assert confusion["false_accept_rate"] == 0.5


def test_combine_assessments_keeps_long_tail_separate_and_uses_two_dimensions(tmp_path: Path):
    positive = tmp_path / "positive.jsonl"
    negative = tmp_path / "negative.jsonl"
    output = tmp_path / "combined"
    _write_jsonl(
        positive,
        [
            {"label_id": "L1", "label_name": "标签一", "planned": 500, "match_rate": 0.85, "sample_tier": "CAPPED_500"},
            {"label_id": "L2", "label_name": "标签二", "planned": 20, "match_rate": 0.90, "sample_tier": "LT1_EXTREME_1_29"},
        ],
    )
    _write_jsonl(
        negative,
        [
            {"label_id": "L1", "hard_negative_total": 50, "false_accept_rate": 0.04},
            {"label_id": "L2", "hard_negative_total": 20, "false_accept_rate": 0.50},
        ],
    )

    report = combine_positive_and_negative_assessments(positive, negative, output)

    rows = {row["label_id"]: row for row in map(json.loads, (output / "label_assessments.jsonl").open())}
    assert rows["L1"]["final_screen"] == "A_STABLE_CANDIDATE"
    assert rows["L2"]["final_screen"] == "U_LONG_TAIL_REVIEW"
    assert report["screen_counts"] == {
        "A_STABLE_CANDIDATE": 1,
        "U_LONG_TAIL_REVIEW": 1,
    }
