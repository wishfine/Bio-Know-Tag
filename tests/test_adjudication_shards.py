import json
import hashlib
from pathlib import Path

import pytest

from bio_know_tag.adjudication_shards import merge_shard_predictions, shard_adjudication_inputs


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_shard_inputs_keeps_mixed_units_and_candidates_aligned(tmp_path: Path):
    units = tmp_path / "units.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    _write(units, [
        {"question_id": "q1", "unit_type": "standalone"},
        {"question_id": "q2", "unit_type": "sub_question"},
        {"question_id": "p2", "unit_type": "composite_parent_extra"},
    ])
    _write(candidates, [
        {"question_id": question_id, "candidates": []}
        for question_id in ("q1", "q2", "p2")
    ])
    report = shard_adjudication_inputs(units, candidates, tmp_path / "out", shard_size=2)
    assert report["input"] == 3
    assert report["shard_count"] == 2
    assert report["unit_types"] == {
        "composite_parent_extra": 1, "standalone": 1, "sub_question": 1,
    }
    first = tmp_path / "out" / "shards" / "00001"
    second = tmp_path / "out" / "shards" / "00002"
    assert [json.loads(line)["question_id"] for line in (first / "units.jsonl").read_text().splitlines()] == ["q1", "q2"]
    assert [json.loads(line)["question_id"] for line in (first / "candidates.jsonl").read_text().splitlines()] == ["q1", "q2"]
    assert [json.loads(line)["question_id"] for line in (second / "units.jsonl").read_text().splitlines()] == ["p2"]


def test_shard_inputs_rejects_misaligned_question_ids(tmp_path: Path):
    units = tmp_path / "units.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    _write(units, [{"question_id": "q1", "unit_type": "standalone"}])
    _write(candidates, [{"question_id": "wrong", "candidates": []}])
    with pytest.raises(ValueError, match="question_id mismatch at row 1"):
        shard_adjudication_inputs(units, candidates, tmp_path / "out", shard_size=2)
    assert not (tmp_path / "out").exists()


def test_merge_shard_predictions_requires_complete_aligned_outputs(tmp_path: Path):
    units = tmp_path / "units.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    _write(units, [{"question_id": value, "unit_type": "standalone"} for value in ("q1", "q2", "q3")])
    _write(candidates, [{"question_id": value, "candidates": []} for value in ("q1", "q2", "q3")])
    shard_root = tmp_path / "sharded"
    shard_adjudication_inputs(units, candidates, shard_root, shard_size=2)
    for shard_id, question_ids in (("00001", ("q1", "q2")), ("00002", ("q3",))):
        shard_dir = shard_root / "shards" / shard_id
        vote_dir = shard_root / "shards" / shard_id / "votes" / "run1"
        vote_dir.mkdir(parents=True)
        _write(vote_dir / "predictions.jsonl", [{"question_id": value, "selected_labels": [], "prompt_version": "v1"} for value in question_ids])
        (vote_dir / "report.json").write_text(json.dumps({"input": len(question_ids), "success": len(question_ids), "error": 0}))
        (vote_dir / "run_manifest.json").write_text(json.dumps({
            "model": "fake", "prompt_version": "v1", "max_tokens": 1024,
            "input_sha256": {
                "units": hashlib.sha256((shard_dir / "units.jsonl").read_bytes()).hexdigest(),
                "candidates": hashlib.sha256((shard_dir / "candidates.jsonl").read_bytes()).hexdigest(),
                "labels": "same-labels-hash",
            },
        }))
    output = tmp_path / "merged.jsonl"
    report = merge_shard_predictions(shard_root, "run1", output)
    assert report["merged"] == 3
    assert [json.loads(line)["question_id"] for line in output.read_text().splitlines()] == ["q1", "q2", "q3"]

    _write(shard_root / "shards" / "00002" / "votes" / "run1" / "predictions.jsonl", [])
    with pytest.raises(ValueError, match="incomplete predictions in shard 00002"):
        merge_shard_predictions(shard_root, "run1", tmp_path / "bad.jsonl")


def test_merge_rejects_stale_predictions_after_candidate_change(tmp_path: Path):
    units = tmp_path / "units.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    _write(units, [{"question_id": "q1", "unit_type": "standalone"}])
    _write(candidates, [{"question_id": "q1", "candidates": []}])
    root = tmp_path / "sharded"
    shard_adjudication_inputs(units, candidates, root, shard_size=2)
    shard = root / "shards" / "00001"
    vote = shard / "votes" / "run1"
    vote.mkdir(parents=True)
    _write(vote / "predictions.jsonl", [{"question_id": "q1", "prompt_version": "v1"}])
    (vote / "report.json").write_text(json.dumps({"input": 1, "success": 1, "error": 0}))
    (vote / "run_manifest.json").write_text(json.dumps({
        "model": "fake", "prompt_version": "v1", "max_tokens": 1024,
        "input_sha256": {
            "units": hashlib.sha256((shard / "units.jsonl").read_bytes()).hexdigest(),
            "candidates": hashlib.sha256((shard / "candidates.jsonl").read_bytes()).hexdigest(),
            "labels": "same-labels-hash",
        },
    }))
    _write(shard / "candidates.jsonl", [{"question_id": "q1", "candidates": [{"label_id": "L1"}]}])
    with pytest.raises(ValueError, match="candidate input hash mismatch in shard 00001"):
        merge_shard_predictions(root, "run1", tmp_path / "merged.jsonl")
    assert not (tmp_path / "merged.jsonl").exists()
