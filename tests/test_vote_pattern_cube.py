import json
import hashlib
from pathlib import Path

import pytest

from bio_know_tag.vote_pattern_cube import analyze_vote_pattern_cube, export_label_review_packages
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
    labels.write_text("".join(json.dumps({"label_id": x, "label_name": x, "definition": f"definition of {x}"}) + "\n" for x in ("L1", "L2", "L3")))
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


def test_ten_full_review_examples_per_label_and_pattern(tmp_path: Path):
    groups = {
        "Q3/D3": [f"a{i}" for i in range(12)],
        "Q3/D0": [f"b{i}" for i in range(12)],
        "Q0/D3": [f"c{i}" for i in range(12)],
    }
    qids = [qid for group in groups.values() for qid in group]
    for vote in VOTES:
        path = tmp_path / "votes" / vote / "evidence.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(_evidence(qid, ["L1"] if (
            qid in groups["Q3/D3"] or
            (vote.startswith("qwen") and qid in groups["Q3/D0"]) or
            (vote.startswith("ds") and qid in groups["Q0/D3"])
        ) else [])) + "\n" for qid in qids))
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps({"label_id": x, "label_name": x, "definition": f"definition of {x}"}) + "\n" for x in ("L1", "L2", "L3")))
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("".join(json.dumps({"question_id": qid, "candidates": [{"label_id": x, "candidate_rank": i} for i, x in enumerate(("L1", "L2", "L3"), 1)]}) + "\n" for qid in qids))
    units = tmp_path / "units.jsonl"
    units.write_text("".join(json.dumps({"question_id": qid, "parent_id": "parent-b" if qid.startswith("b") else qid, "unit_type": "standalone", "stem": f"stem {qid}", "options": "A. yes B. no", "answer_text": "A", "analysis": "complete analysis"}) + "\n" for qid in qids))
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (("units", units), ("candidates", candidates), ("labels", labels))}
    (tmp_path / "run_manifest.json").write_text(json.dumps({"input_sha256": hashes}))
    for vote in VOTES:
        (tmp_path / "votes" / vote / "run_manifest.json").write_text(json.dumps({"input_sha256": hashes}))

    report = analyze_vote_pattern_cube(tmp_path, candidates_path=candidates, units_path=units, labels_path=labels, sample_per_label_pattern=10, full_review_context=True)
    l1 = next(row for row in report["labels"] if row["label_id"] == "L1")
    assert all(len(l1["representative_samples"][pattern]) == 10 for pattern in groups)
    assert l1["representative_samples"]["Q3/D3"][0]["options"] == "A. yes B. no"
    assert l1["representative_samples"]["Q3/D3"][0]["six_vote_selected_label_ids"]["ds3"] == ["L1"]
    assert not l1["representative_samples"]["Q2/D1"]

    zero_report = analyze_vote_pattern_cube(tmp_path, candidates_path=candidates, units_path=units, labels_path=labels, sample_per_pattern=0)
    assert not next(row for row in zero_report["labels"] if row["label_id"] == "L1")["representative_samples"]["Q3/D3"]

    images = tmp_path / "image_context.jsonl"
    sampled_a = l1["representative_samples"]["Q3/D3"][0]["question_id"]
    images.write_text(json.dumps({"question_id": sampled_a, "stem_image_url": "https://example.test/q1.png", "analysis_image_url": "https://example.test/a1.png"}) + "\n")
    output = tmp_path / "review"
    summary = export_label_review_packages(report, output, image_context_path=images)
    exported = json.loads((output / "labels" / "L1.json").read_text())
    assert all(summary["sample_counts"][pattern] == 10 for pattern in groups)
    assert exported["definition"] == "definition of L1"
    assert len(exported["examples"]["Q3/D3"]) == 10
    assert next(row for row in exported["examples"]["Q3/D3"] if row["question_id"] == sampled_a)["stem_image_url"] == "https://example.test/q1.png"
    with pytest.raises(ValueError, match="new or empty"):
        export_label_review_packages(report, output)

    raw_image_map = tmp_path / "raw_images.json"
    sampled_b = l1["representative_samples"]["Q3/D0"][0]["question_id"]
    raw_image_map.write_text(json.dumps({sampled_b: {"stemImageUrl": "https://example.test/raw-b.png", "analysisImageUrl": "https://example.test/raw-b-analysis.png"}, "parent-b": {"stemImageUrl": "https://example.test/parent.png"}}))
    raw_output = tmp_path / "review-from-raw"
    export_label_review_packages(report, raw_output, image_map_path=raw_image_map)
    raw_exported = json.loads((raw_output / "labels" / "L1.json").read_text())
    raw_sample = next(row for row in raw_exported["examples"]["Q3/D0"] if row["question_id"] == sampled_b)
    assert raw_sample["analysis_image_url"] == "https://example.test/raw-b-analysis.png"
    assert raw_sample["parent_stem_image_url"] == "https://example.test/parent.png"
