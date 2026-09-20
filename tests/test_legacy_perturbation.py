import json
from pathlib import Path

from bio_know_tag.legacy_perturbation import (
    GROUP_A,
    GROUP_B,
    GROUP_C,
    analyze_legacy_candidate_perturbation,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_partitions_same_expanded_and_selected_legacy(tmp_path: Path) -> None:
    top_candidates = tmp_path / "top.jsonl"
    legacy_candidates = tmp_path / "legacy.jsonl"
    top_predictions = tmp_path / "top_predictions.jsonl"
    legacy_predictions = tmp_path / "legacy_predictions.jsonl"
    labels = tmp_path / "labels.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    out = tmp_path / "out"

    _write_jsonl(
        top_candidates,
        [
            {"question_id": "qA", "candidates": [{"label_id": "L1"}]},
            {"question_id": "qB", "candidates": [{"label_id": "L1"}]},
            {"question_id": "qC", "candidates": [{"label_id": "L1"}]},
        ],
    )
    _write_jsonl(
        legacy_candidates,
        [
            {"question_id": "qA", "candidates": [{"label_id": "L1"}]},
            {
                "question_id": "qB",
                "candidates": [{"label_id": "L1"}, {"label_id": "L2"}],
            },
            {
                "question_id": "qC",
                "candidates": [{"label_id": "L1"}, {"label_id": "L2"}],
            },
        ],
    )
    _write_jsonl(
        top_predictions,
        [
            {"question_id": "qA", "selected_labels": [{"label_id": "L1"}]},
            {"question_id": "qB", "selected_labels": [{"label_id": "L1"}]},
            {"question_id": "qC", "selected_labels": []},
        ],
    )
    _write_jsonl(
        legacy_predictions,
        [
            {"question_id": "qA", "selected_labels": []},
            {"question_id": "qB", "selected_labels": [{"label_id": "L1"}]},
            {"question_id": "qC", "selected_labels": [{"label_id": "L2"}]},
        ],
    )
    _write_jsonl(
        labels,
        [
            {"label_id": "L1", "label_name": "one", "label_path": "a"},
            {"label_id": "L2", "label_name": "two", "label_path": "b"},
        ],
    )
    _write_jsonl(
        reviews,
        [
            {
                "question_id": "qC",
                "label_id": "L2",
                "decision": "DIRECT_MATCH",
            }
        ],
    )

    report = analyze_legacy_candidate_perturbation(
        top25_candidates_path=top_candidates,
        legacy_candidates_path=legacy_candidates,
        top25_predictions_path=top_predictions,
        legacy_predictions_path=legacy_predictions,
        labels_path=labels,
        luna_reviews_path=reviews,
        run_dir=out,
        progress_every=0,
    )

    assert report["groups"][GROUP_A]["questions"] == 1
    assert report["groups"][GROUP_A]["different_output"] == 1
    assert report["groups"][GROUP_B]["questions"] == 1
    assert report["groups"][GROUP_B]["same_output"] == 1
    assert report["groups"][GROUP_C]["questions"] == 1
    assert report["groups"][GROUP_C]["selected_added_legacy_count"] == 1
    assert report["groups"][GROUP_C]["luna_selected_added_decisions"] == {
        "DIRECT_MATCH": 1
    }

    groups = {
        json.loads(line)["question_id"]: json.loads(line)["perturbation_group"]
        for line in (out / "per_question.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert groups == {"qA": GROUP_A, "qB": GROUP_B, "qC": GROUP_C}


def test_reads_append_only_evidence_format(tmp_path: Path) -> None:
    top_candidates = tmp_path / "top.jsonl"
    legacy_candidates = tmp_path / "legacy.jsonl"
    top_predictions = tmp_path / "top_predictions.jsonl"
    legacy_predictions = tmp_path / "legacy_predictions.jsonl"
    out = tmp_path / "out"
    candidate_row = {"question_id": "q1", "candidates": [{"label_id": "L1"}]}
    _write_jsonl(top_candidates, [candidate_row])
    _write_jsonl(legacy_candidates, [candidate_row])
    evidence = {
        "question_id": "q1",
        "candidate_code_map": {"C01": "L1"},
        "parsed_response": {"selected": ["C01"]},
        "error": None,
    }
    _write_jsonl(top_predictions, [evidence])
    _write_jsonl(legacy_predictions, [evidence])

    report = analyze_legacy_candidate_perturbation(
        top25_candidates_path=top_candidates,
        legacy_candidates_path=legacy_candidates,
        top25_predictions_path=top_predictions,
        legacy_predictions_path=legacy_predictions,
        run_dir=out,
        progress_every=0,
    )

    assert report["groups"][GROUP_A]["same_output"] == 1

