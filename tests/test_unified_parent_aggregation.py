import json
from pathlib import Path

from bio_know_tag.unified_parent_aggregation import aggregate_unified_parent_predictions


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_parent_union_uses_children_and_extra_from_one_prediction_stream(tmp_path: Path):
    plans = tmp_path / "plans.jsonl"
    units = tmp_path / "units.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write(plans, [
        {"question_id": "p1", "child_question_ids": ["c1", "c2"]},
        {"question_id": "p2", "child_question_ids": ["c3"]},
    ])
    _write(units, [
        {"question_id": "c1", "parent_id": "p1", "unit_type": "sub_question"},
        {"question_id": "c2", "parent_id": "p1", "unit_type": "sub_question"},
        {"question_id": "c3", "parent_id": "p2", "unit_type": "sub_question"},
        {"question_id": "p1", "parent_id": "p1", "unit_type": "composite_parent_extra"},
    ])
    _write(predictions, [
        {"question_id": "c1", "parent_id": "p1", "unit_type": "sub_question", "selected_labels": [{"label_id": "L1", "label_name": "一"}], "needs_review": False, "usable_for_training": True},
        {"question_id": "c2", "parent_id": "p1", "unit_type": "sub_question", "selected_labels": [{"label_id": "L2", "label_name": "二"}], "needs_review": False, "usable_for_training": True},
        {"question_id": "c3", "parent_id": "p2", "unit_type": "sub_question", "selected_labels": [{"label_id": "L3", "label_name": "三"}], "needs_review": False, "usable_for_training": True},
        {"question_id": "p1", "parent_id": "p1", "unit_type": "composite_parent_extra", "selected_labels": [{"label_id": "L2", "label_name": "二"}, {"label_id": "L4", "label_name": "四"}], "needs_review": False, "usable_for_training": True},
    ])
    output = tmp_path / "parents.jsonl"
    report = aggregate_unified_parent_predictions(plans, units, predictions, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert report["complete_parents"] == 2
    assert [item["label_id"] for item in rows[0]["child_union_labels"]] == ["L1", "L2"]
    assert [item["label_id"] for item in rows[0]["parent_extra_labels"]] == ["L4"]
    assert [item["label_id"] for item in rows[0]["knowledge_labels"]] == ["L1", "L2", "L4"]
    assert rows[1]["knowledge_labels"][0]["label_id"] == "L3"
    assert rows[1]["parent_extra_prediction_missing"] is False


def test_parent_with_missing_child_prediction_is_blocked(tmp_path: Path):
    plans = tmp_path / "plans.jsonl"
    units = tmp_path / "units.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write(plans, [{"question_id": "p1", "child_question_ids": ["c1", "c2"]}])
    _write(units, [{"question_id": "c1", "parent_id": "p1", "unit_type": "sub_question"}, {"question_id": "c2", "parent_id": "p1", "unit_type": "sub_question"}, {"question_id": "p1", "parent_id": "p1", "unit_type": "composite_parent_extra"}])
    _write(predictions, [{"question_id": "c1", "parent_id": "p1", "unit_type": "sub_question", "selected_labels": [{"label_id": "L1", "label_name": "一"}], "needs_review": False, "usable_for_training": True}])
    output = tmp_path / "parents.jsonl"
    report = aggregate_unified_parent_predictions(plans, units, predictions, output)
    row = json.loads(output.read_text())
    assert report["incomplete_parents"] == 1
    assert row["missing_child_question_ids"] == ["c2"]
    assert row["parent_extra_prediction_missing"] is True
    assert row["needs_review"] is True
    assert row["usable_for_training"] is False


def test_parent_with_text_ineligible_child_unit_is_provisionally_incomplete(tmp_path: Path):
    plans = tmp_path / "plans.jsonl"
    units = tmp_path / "units.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write(plans, [{"question_id": "p1", "child_question_ids": ["c1", "c2"]}])
    _write(units, [
        {"question_id": "c1", "parent_id": "p1", "unit_type": "sub_question"},
        {"question_id": "p1", "parent_id": "p1", "unit_type": "composite_parent_extra"},
    ])
    _write(predictions, [
        {"question_id": "c1", "parent_id": "p1", "unit_type": "sub_question", "selected_labels": [{"label_id": "L1", "label_name": "一"}], "needs_review": False, "usable_for_training": True},
        {"question_id": "p1", "parent_id": "p1", "unit_type": "composite_parent_extra", "selected_labels": [], "needs_review": False, "usable_for_training": False},
    ])
    output = tmp_path / "parents.jsonl"
    report = aggregate_unified_parent_predictions(plans, units, predictions, output)
    row = json.loads(output.read_text())
    assert report["incomplete_parents"] == 1
    assert row["missing_child_question_ids"] == ["c2"]
    assert row["knowledge_labels"][0]["label_id"] == "L1"
    assert row["usable_for_training"] is False
