import json
from pathlib import Path

from scripts.analyze_qwen_partial_streaming import compare


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def prediction(question_id: str, selected: list[dict], count: int) -> dict:
    return {
        "question_id": question_id,
        "selected_labels": selected,
        "candidate_count": count,
        "reason": "测试理由",
    }


def test_streaming_comparison_separates_same_set_perturbation_and_selected_addition(tmp_path: Path):
    base = tmp_path / "base.jsonl"
    augmented = tmp_path / "augmented.jsonl"
    units = tmp_path / "units.jsonl"
    labels = tmp_path / "labels.jsonl"
    write_jsonl(base, [
        prediction("A", [{"label_id": "L1"}], 2),
        prediction("B", [{"label_id": "L1"}], 2),
        prediction("C", [{"label_id": "L1"}], 2),
    ])
    write_jsonl(augmented, [
        prediction("A", [{"label_id": "L2", "candidate_rank": 2, "sources": ["sparse"]}], 2),
        prediction("B", [{"label_id": "L1", "candidate_rank": 1, "sources": ["dense"]}], 3),
        prediction("C", [{"label_id": "L1", "candidate_rank": 1, "sources": ["dense"]}, {"label_id": "L3", "candidate_rank": 3, "sources": ["legacy"]}], 3),
    ])
    write_jsonl(units, [{"question_id": qid, "stem": f"题干{qid}"} for qid in "ABC"])
    write_jsonl(labels, [{"label_id": f"L{i}", "label_name": f"标签{i}"} for i in range(1, 4)])

    report = compare(base, augmented, units, labels, sample_per_group=2)

    assert report["counts"]["common_questions"] == 3
    assert report["counts"]["different_outputs"] == 2
    assert report["groups"]["A_same_candidates"]["different_outputs"] == 1
    assert report["groups"]["B_added_not_selected"]["different_outputs"] == 0
    assert report["groups"]["C_added_legacy_selected"]["different_outputs"] == 1
    assert report["groups"]["A_same_candidates"]["share_among_changed"] == 0.5
    assert report["samples"]["C_added_legacy_selected"][0]["added_legacy_selected"] == [
        {"label_id": "L3", "label_name": "标签3"}
    ]
    assert report["samples"]["C_added_legacy_selected"][0]["stem"] == "题干C"
