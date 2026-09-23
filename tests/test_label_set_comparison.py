import json
from pathlib import Path

import pytest

from bio_know_tag.label_set_comparison import (
    build_legacy_id_snapshot,
    compare_label_sets,
    relationship,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_relationship_five_mutually_exclusive_categories():
    old = {"A", "B"}
    assert relationship(old, old) == "exact"
    assert relationship(old, {"A", "B", "C"}) == "contains_legacy"
    assert relationship(old, {"A"}) == "subset_of_legacy"
    assert relationship(old, {"A", "C"}) == "mixed_add_remove"
    assert relationship(old, {"C"}) == "disjoint"
    assert relationship(old, set()) == "subset_of_legacy"
    with pytest.raises(ValueError, match="nonempty"):
        relationship(set(), {"A"})


def test_compare_label_sets_reports_question_and_label_views(tmp_path):
    labels = tmp_path / "labels.jsonl"
    units = tmp_path / "units.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(labels, [{"label_id": x, "label_name": x} for x in "ABCD"])
    write_jsonl(units, [
        {"question_id": "q1", "unit_type": "standalone", "legacy_knw_ids": ["A", "B", "OLD"]},
        {"question_id": "q2", "unit_type": "standalone", "legacy_knw_ids": ["A"]},
        {"question_id": "q3", "unit_type": "sub_question", "legacy_knw_ids": ["A", "B"]},
        {"question_id": "q4", "unit_type": "standalone", "legacy_knw_ids": ["A", "B"]},
        {"question_id": "q5", "unit_type": "standalone", "legacy_knw_ids": ["A", "B"]},
        {"question_id": "q6", "unit_type": "standalone", "legacy_knw_ids": ["OLD"]},
        {"question_id": "q7", "unit_type": "standalone", "legacy_knw_ids": ["A"]},
    ])
    write_jsonl(predictions, [
        {"question_id": "q1", "selected_labels": [{"label_id": "A"}, {"label_id": "B"}]},
        {"question_id": "q2", "selected_labels": [{"label_id": "A"}, {"label_id": "C"}]},
        {"question_id": "q3", "selected_labels": [{"label_id": "A"}]},
        {"question_id": "q4", "selected_labels": [{"label_id": "A"}, {"label_id": "C"}]},
        {"question_id": "q5", "selected_labels": [{"label_id": "C"}]},
        {"question_id": "q6", "selected_labels": [{"label_id": "D"}]},
    ])
    out = tmp_path / "out"
    report = compare_label_sets(
        units_path=units, predictions_path=predictions, labels_path=labels,
        output_dir=out, expected_units=7,
    )
    counts = report["counts"]
    assert counts["units"] == 7
    assert counts["compared"] == 6
    assert counts["eligible"] == 5
    assert counts["missing_prediction"] == 1
    assert counts["obsolete_assignments_ignored"] == 2
    assert [counts[x] for x in ("exact", "contains_legacy", "subset_of_legacy", "mixed_add_remove", "disjoint")] == [1] * 5
    assert counts["no_current_legacy"] == 1
    assert counts["legacy_assignments"] == 9
    assert counts["retained_assignments"] == 5
    assert counts["missing_assignments"] == 4
    assert counts["added_assignments"] == 3
    labels_by_id = {x["label_id"]: x for x in map(json.loads, (out / "per_label.jsonl").read_text().splitlines())}
    assert len(labels_by_id) == 4
    assert labels_by_id["A"]["legacy_questions"] == 5
    assert labels_by_id["A"]["retained"] == 4
    assert labels_by_id["A"]["missed"] == 1
    assert labels_by_id["C"]["new_vs_legacy"] == 3
    assert labels_by_id["D"]["legacy_questions"] == 0
    assert labels_by_id["D"]["retention_rate"] is None


def test_compare_label_sets_rejects_extra_predictions(tmp_path):
    labels = tmp_path / "labels.jsonl"
    units = tmp_path / "units.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(labels, [{"label_id": "A"}])
    write_jsonl(units, [{"question_id": "q1", "legacy_knw_ids": ["A"]}])
    write_jsonl(predictions, [{"question_id": "q2", "selected_labels": []}])
    with pytest.raises(ValueError, match="absent from units"):
        compare_label_sets(
            units_path=units, predictions_path=predictions, labels_path=labels,
            output_dir=tmp_path / "out", expected_units=1,
        )


def test_blind_pilot_requires_recovered_legacy_ids(tmp_path):
    labels = tmp_path / "labels.jsonl"
    sample = tmp_path / "sample.jsonl"
    full = tmp_path / "full.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    snapshot = tmp_path / "legacy.jsonl"
    write_jsonl(labels, [{"label_id": "A", "label_name": "A"}])
    write_jsonl(sample, [{"question_id": "q1", "unit_type": "standalone"}])
    write_jsonl(full, [
        {"question_id": "irrelevant", "legacy_knw_ids": ["A"]},
        {"question_id": "q1", "legacy_knw_ids": ["A", "OLD"]},
    ])
    write_jsonl(predictions, [{"question_id": "q1", "selected_labels": [{"label_id": "A"}]}])
    with pytest.raises(ValueError, match="pass --legacy-snapshot"):
        compare_label_sets(
            units_path=sample, predictions_path=predictions, labels_path=labels,
            output_dir=tmp_path / "bad", expected_units=1,
        )
    snapshot_report = build_legacy_id_snapshot(
        sample_units_path=sample, full_units_path=full,
        output_path=snapshot, expected_units=1,
    )
    assert snapshot_report["matched_units"] == 1
    report = compare_label_sets(
        units_path=sample, predictions_path=predictions, labels_path=labels,
        legacy_snapshot_path=snapshot, output_dir=tmp_path / "good", expected_units=1,
    )
    assert report["counts"]["exact"] == 1
    assert report["counts"]["obsolete_assignments_ignored"] == 1
