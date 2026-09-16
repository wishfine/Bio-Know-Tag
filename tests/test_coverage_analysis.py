import json
from pathlib import Path

from bio_know_tag.coverage_analysis import analyze_coverage_snapshot


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_analyze_partial_snapshot_tracks_completion_and_issue_screens(tmp_path: Path):
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "analysis"
    _write_jsonl(
        labels,
        [
            {"label_id": "L1", "label_name": "标签一"},
            {"label_id": "L2", "label_name": "标签二"},
        ],
    )
    _write_jsonl(
        tasks,
        [
            {"pair_id": "q1::L1", "question_id": "q1", "label_id": "L1", "unit_type": "standalone", "stem": "题1"},
            {"pair_id": "q2::L1", "question_id": "q2", "label_id": "L1", "unit_type": "sub_question", "stem": "题2"},
            {"pair_id": "q3::L1", "question_id": "q3", "label_id": "L1", "unit_type": "standalone", "stem": "题3"},
            {"pair_id": "q4::L2", "question_id": "q4", "label_id": "L2", "unit_type": "standalone", "stem": "题4"},
        ],
    )
    _write_jsonl(
        results,
        [
            {"task_id": "q1::L1", "label_id": "L1", "question_id": "q1", "question_type": "standalone", "match": True, "relevance_score": 0.95},
            {"task_id": "q2::L1", "label_id": "L1", "question_id": "q2", "question_type": "sub_question", "match": False, "relevance_score": 0.0},
            {"task_id": "q4::L2", "label_id": "L2", "question_id": "q4", "question_type": "standalone", "match": False, "relevance_score": 0.5},
        ],
    )
    with results.open("a", encoding="utf-8") as handle:
        handle.write('{"partial":')

    report = analyze_coverage_snapshot(tasks, results, labels, output)

    assert report["planned_tasks"] == 4
    assert report["completed_tasks"] == 3
    assert report["completion_rate"] == 0.75
    assert report["malformed_result_lines_ignored"] == 1
    per_label = {
        row["label_id"]: row
        for row in map(json.loads, (output / "per_label.jsonl").open())
    }
    assert per_label["L1"]["planned"] == 3
    assert per_label["L1"]["completed"] == 2
    assert per_label["L1"]["match_rate"] == 0.5
    assert "incomplete" in per_label["L1"]["issue_flags"]
    assert "low_sample" in per_label["L2"]["issue_flags"]
    review_rows = list(map(json.loads, (output / "review_samples.jsonl").open()))
    assert {row["score_band"] for row in review_rows} == {"high_match", "zero", "gray"}
    assert (output / "snapshot_report.md").exists()

