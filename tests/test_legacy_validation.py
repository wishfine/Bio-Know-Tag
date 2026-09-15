import json
from pathlib import Path

from bio_know_tag.legacy_validation import (
    augment_candidates_with_legacy,
    evaluate_legacy_recall,
    valid_legacy_targets,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _labels() -> list[dict]:
    return [
        {"label_id": "L1", "label_name": "DNA复制", "label_path": "知识点->遗传->DNA复制"},
        {"label_id": "L2", "label_name": "渗透作用", "label_path": "知识点->细胞->渗透作用"},
    ]


def test_valid_legacy_targets_keeps_only_current_458_and_deduplicates():
    unit = {"legacy_knw_ids": ["OLD", "L2", "L2", "L1"]}

    valid, obsolete = valid_legacy_targets(unit, {"L1", "L2"})

    assert valid == ["L2", "L1"]
    assert obsolete == ["OLD"]


def test_evaluate_legacy_recall_excludes_obsolete_only_questions(tmp_path: Path):
    units = tmp_path / "units.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "eval"
    _write_jsonl(labels, _labels())
    _write_jsonl(
        units,
        [
            {"question_id": "q1", "legacy_knw_ids": ["L1", "L2"]},
            {"question_id": "q2", "legacy_knw_ids": ["OLD"]},
            {"question_id": "q3", "legacy_knw_ids": []},
        ],
    )
    _write_jsonl(
        candidates,
        [
            {"question_id": "q1", "candidates": [{"label_id": "L1"}, {"label_id": "X"}]},
            {"question_id": "q2", "candidates": [{"label_id": "L2"}]},
            {"question_id": "q3", "candidates": [{"label_id": "L1"}]},
        ],
    )

    report = evaluate_legacy_recall(
        units, candidates, labels, output, ks=(1, 2)
    )

    assert report["questions_scanned"] == 3
    assert report["questions_evaluated"] == 1
    assert report["obsolete_only_questions"] == 1
    assert report["no_legacy_questions"] == 1
    assert report["metrics"]["2"]["any_hit_rate"] == 1.0
    assert report["metrics"]["2"]["all_hit_rate"] == 0.0
    assert report["metrics"]["2"]["micro_recall"] == 0.5
    misses = [json.loads(line) for line in (output / "misses.jsonl").read_text().splitlines()]
    assert misses[0]["missing_valid_legacy_ids"] == ["L2"]


def test_augment_candidates_adds_only_valid_legacy_labels_without_duplicates(tmp_path: Path):
    units = tmp_path / "units.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "augmented"
    _write_jsonl(labels, _labels())
    _write_jsonl(
        units,
        [{"question_id": "q1", "legacy_knw_ids": ["L1", "OLD", "L2"]}],
    )
    _write_jsonl(
        candidates,
        [
            {
                "question_id": "q1",
                "retrieval_version": "hybrid-v1-s18-d7-k25",
                "candidates": [
                    {
                        "label_id": "L1",
                        "label_name": "DNA复制",
                        "label_path": "知识点@遗传@DNA复制",
                        "candidate_rank": 1,
                        "sources": ["sparse"],
                    }
                ],
            }
        ],
    )

    report = augment_candidates_with_legacy(units, candidates, labels, output)

    row = json.loads((output / "candidates.jsonl").read_text())
    assert [item["label_id"] for item in row["candidates"]] == ["L1", "L2"]
    assert row["candidates"][0]["sources"] == ["sparse", "legacy"]
    assert row["candidates"][1]["sources"] == ["legacy"]
    assert row["candidates"][1]["candidate_rank"] == 2
    assert "OLD" not in json.dumps(row)
    assert report["legacy_candidates_added"] == 1
    assert report["obsolete_legacy_assignments_removed"] == 1
