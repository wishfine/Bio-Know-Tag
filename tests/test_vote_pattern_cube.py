import json
import hashlib
from pathlib import Path

import pytest

from bio_know_tag.vote_pattern_cube import analyze_vote_pattern_cube
from bio_know_tag.adjudication import CANDIDATE_ORDER_VERSION


VOTES = ("qwen1", "qwen2", "qwen3", "ds1", "ds2", "ds3")


def _evidence(qid: str, labels: list[str]) -> dict:
    ordered = sorted(("L1", "L2", "L3"), key=lambda label_id: hashlib.sha256(
        f"{qid}\0{label_id}\0{CANDIDATE_ORDER_VERSION}".encode()
    ).digest())
    code_map = {f"C{i:02d}": label_id for i, label_id in enumerate(ordered, 1)}
    reverse = {value: key for key, value in code_map.items()}
    return {
        "question_id": qid, "error": None, "candidate_code_map": code_map,
        "parsed_response": {"selected": [reverse[x] for x in labels]},
    }


def test_vote_pattern_cube_counts_all_16_states_with_exposure(tmp_path: Path):
    selections = {
        "qwen1": {"q1": ["L1", "L3"], "q2": ["L2"]},
        "qwen2": {"q1": ["L1"], "q2": ["L2"]},
        "qwen3": {"q1": ["L1"], "q2": ["L2"]},
        "ds1": {"q1": ["L1"], "q2": ["L1"]},
        "ds2": {"q1": ["L1"], "q2": []},
        "ds3": {"q1": ["L1"], "q2": []},
    }
    for vote in VOTES:
        path = tmp_path / "votes" / vote / "evidence.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(_evidence(qid, ids)) + "\n" for qid, ids in selections[vote].items()))
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps({"label_id": x, "label_name": x}) + "\n" for x in ("L1", "L2", "L3")))
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("".join(json.dumps({"question_id": qid, "candidates": [{"label_id": x, "candidate_rank": i, "sources": ["sparse"]} for i, x in enumerate(("L1", "L2", "L3"), 1)]}) + "\n" for qid in ("q1", "q2")))
    units = tmp_path / "units.jsonl"
    units.write_text("".join(json.dumps({"question_id": qid, "unit_type": kind, "flags": {"image_context_missing": qid == "q2"}}) + "\n" for qid, kind in (("q1", "standalone"), ("q2", "sub_question"))))

    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (("units", units), ("candidates", candidates), ("labels", labels))}
    (tmp_path / "run_manifest.json").write_text(json.dumps({"input_sha256": hashes}))
    for vote in VOTES:
        (tmp_path / "votes" / vote / "run_manifest.json").write_text(json.dumps({"input_sha256": hashes}))

    report = analyze_vote_pattern_cube(tmp_path, candidates_path=candidates, units_path=units, labels_path=labels)

    assert report["common_successful_questions"] == 2
    assert report["candidate_label_exposures"] == 6
    assert report["pattern_counts"]["Q3/D3"] == 1
    assert report["pattern_counts"]["Q1/D0"] == 1
    assert report["pattern_counts"]["Q3/D0"] == 1
    assert report["pattern_counts"]["Q0/D1"] == 1
    assert report["pattern_counts"]["Q0/D0"] == 2
    by_label = {row["label_id"]: row for row in report["labels"]}
    assert by_label["L1"]["candidate_exposure"] == 2
    assert by_label["L1"]["patterns"]["Q3/D3"] == 1
    assert by_label["L1"]["patterns"]["Q0/D1"] == 1
    assert by_label["L2"]["patterns"]["Q0/D0"] == 1
    assert by_label["L3"]["patterns"]["Q0/D0"] == 1
    assert report["by_unit_type"]["standalone"]["Q3/D3"] == 1
    assert report["by_unit_type"]["sub_question"]["Q3/D0"] == 1
    assert report["by_quality_flag"]["image_context_missing"]["Q3/D0"] == 1
    assert by_label["L2"]["by_quality_flag"]["image_context_missing"]["Q3/D0"] == 1
    assert by_label["L2"]["by_candidate_rank_band"]["1-5"]["Q3/D0"] == 1
    assert by_label["L2"]["by_candidate_source"]["sparse"]["Q3/D0"] == 1
    assert by_label["L2"]["representative_samples"]["Q3/D0"][0]["question_id"] == "q2"
    assert all(row["teacher_calibration_status"] == "NO_TEACHER_GOLD" for row in report["labels"])
    from scripts.analyze_vote_pattern_cube import _markdown
    assert "Q3/D3" in _markdown(report)

    evidence_path = tmp_path / "votes" / "qwen1" / "evidence.jsonl"
    evidence_rows = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    evidence_rows[0]["candidate_code_map"]["C01"] = "L2" if evidence_rows[0]["candidate_code_map"]["C01"] != "L2" else "L1"
    evidence_path.write_text("".join(json.dumps(row) + "\n" for row in evidence_rows))
    with pytest.raises(ValueError, match="candidate_code_map differs"):
        analyze_vote_pattern_cube(tmp_path, candidates_path=candidates, units_path=units, labels_path=labels)


def test_vote_pattern_cube_requires_frozen_hashes(tmp_path: Path):
    with pytest.raises(ValueError, match="controller manifest"):
        analyze_vote_pattern_cube(tmp_path, candidates_path=tmp_path / "c", units_path=tmp_path / "u", labels_path=tmp_path / "l")
