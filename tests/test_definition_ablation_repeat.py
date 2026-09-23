import json

import pytest

from bio_know_tag.definition_ablation_repeat import analyze_repeated_ablation


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_repeated_ablation_separates_robust_changes_from_noise(tmp_path):
    sample = tmp_path / "sample.jsonl"
    _write(sample, [
        {"pair_id": "q1::L", "question_id": "q1", "label_id": "L",
         "label_name": "Label", "stratum": "low", "sample_source": "volatile_5391",
         "image_context_missing": False, "analysis": "解释"},
        {"pair_id": "q2::L", "question_id": "q2", "label_id": "L",
         "label_name": "Label", "stratum": "high", "sample_source": "fallback",
         "image_context_missing": True, "analysis": ""},
        {"pair_id": "q3::M", "question_id": "q3", "label_id": "M",
         "label_name": "Other", "stratum": "high", "sample_source": "fallback",
         "image_context_missing": False, "analysis": "解释"},
    ])
    values = {
        "name_1": [0.10, 0.10, 0.90],
        "name_2": [0.10, 0.90, 0.90],
        "definition_1": [0.90, 0.90, 0.90],
        "definition_2": [0.90, 0.90, 0.90],
    }
    paths = {}
    for arm, scores in values.items():
        path = tmp_path / f"{arm}.jsonl"
        _write(path, [
            {"task_id": f"q{i}::{label}", "question_id": f"q{i}",
             "label_id": label, "relevance_score": score}
            for i, (label, score) in enumerate(zip(("L", "L", "M"), scores), 1)
        ])
        paths[arm] = path

    report = analyze_repeated_ablation(
        sample_path=sample, result_paths=paths, output_dir=tmp_path / "analysis"
    )
    assert report["complete_pairs"] == 3
    assert report["counts"]["robust_false_to_true"] == 1
    assert report["counts"]["name_repeat_changed"] == 1
    assert report["counts"]["unstable_either_arm"] == 1
    assert report["counts"]["ablation_changed_first"] == 2
    assert report["counts"]["ablation_changed_second"] == 1
    pairs = [json.loads(x) for x in (tmp_path / "analysis/per_pair.jsonl").read_text().splitlines()]
    assert pairs[0]["robust_change"] == "false_to_true"
    assert pairs[1]["robust_change"] is None
    assert report["strata"]["image_missing:True"]["unstable_either_arm"] == 1


def test_repeated_ablation_rejects_mismatched_metadata(tmp_path):
    sample = tmp_path / "sample.jsonl"
    _write(sample, [{"pair_id": "q::L", "question_id": "q", "label_id": "L"}])
    paths = {}
    for arm in ("name_1", "name_2", "definition_1", "definition_2"):
        path = tmp_path / f"{arm}.jsonl"
        _write(path, [{"task_id": "q::L", "question_id": "wrong" if arm == "name_2" else "q",
                       "label_id": "L", "relevance_score": 0.5}])
        paths[arm] = path
    with pytest.raises(ValueError, match="metadata mismatch"):
        analyze_repeated_ablation(
            sample_path=sample, result_paths=paths, output_dir=tmp_path / "analysis"
        )
