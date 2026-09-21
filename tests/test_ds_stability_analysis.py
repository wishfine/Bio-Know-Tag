import json
from pathlib import Path

from bio_know_tag.ds_stability_analysis import analyze_stability_run


def _write(path: Path, value) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in value) + "\n",
        encoding="utf-8",
    )


def _response(question_id: str, selected: list[str], *, error=None) -> dict:
    return {
        "question_id": question_id,
        "error": error,
        "perturbation_group": "A_same_candidate_set",
        "choices": [
            {
                "parse_error": None,
                "selected_label_ids": selected,
            }
        ],
        "prompt_sha256": "same",
    }


def test_reports_more_fewer_and_replacement(tmp_path: Path):
    run = tmp_path / "run"
    (run / "base").mkdir(parents=True)
    (run / "alt").mkdir(parents=True)
    _write(
        run / "run_manifest.json",
        [],
    )
    (run / "run_manifest.json").write_text(
        json.dumps({
            "input_count": 4,
            "conditions": [{"name": "base"}, {"name": "alt"}],
        }),
        encoding="utf-8",
    )
    _write(
        run / "base" / "responses.jsonl",
        [_response("q1", ["L1"]), _response("q2", ["L1", "L2"]), _response("q3", []), _response("q4", ["L1", "L2"])],
    )
    _write(
        run / "alt" / "responses.jsonl",
        [_response("q1", ["L1", "L2"]), _response("q2", ["L3", "L4"]), _response("q3", ["L4"]), _response("q4", ["L3"])],
    )
    labels = tmp_path / "labels.jsonl"
    _write(labels, [{"label_id": f"L{i}", "label_name": f"Label {i}"} for i in range(1, 5)])

    report = analyze_stability_run(run, labels_path=labels)
    stats = report["pairwise_to_baseline"][0]["overall"]
    assert stats["questions"] == 4
    assert stats["more_labels"] == 2
    assert stats["fewer_labels_with_replacements"] == 1
    assert stats["same_count_replaced"] == 1
    assert stats["added_assignments"] == 5
    assert stats["removed_assignments"] == 4
    rows = [json.loads(line) for line in (run / "detailed-analysis" / "per_question.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["comparison"] for row in rows} == {
        "more_labels", "fewer_labels_with_replacements", "same_count_replaced"
    }


def test_require_complete_rejects_missing_condition(tmp_path: Path):
    run = tmp_path / "run"
    (run / "base").mkdir(parents=True)
    (run / "alt").mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"input_count": 1, "conditions": [{"name": "base"}, {"name": "alt"}]}),
        encoding="utf-8",
    )
    _write(run / "base" / "responses.jsonl", [_response("q1", ["L1"])])
    _write(run / "alt" / "responses.jsonl", [])
    try:
        analyze_stability_run(run, require_complete=True)
    except ValueError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("expected incomplete run to fail")
