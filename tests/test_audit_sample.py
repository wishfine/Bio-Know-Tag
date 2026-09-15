import json
from pathlib import Path

from bio_know_tag.audit_sample import build_adjudication_audit_sample


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_adjudication_audit_sample_is_aligned_and_deterministic(
    tmp_path: Path,
):
    units_path = tmp_path / "units.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    units = []
    candidate_rows = []
    for index in range(20):
        question_id = f"q{index:02d}"
        unit_type = ("standalone", "sub_question", "orphan_sub_question")[index % 3]
        units.append(
            {
                "question_id": question_id,
                "parent_id": question_id,
                "unit_type": unit_type,
                "stem": f"题干{index}",
                "answer_text": "" if index % 7 == 0 else "答案",
                "analysis": "" if index % 5 == 0 else "解析",
                "flags": {"image_context_missing": index % 4 == 0},
                "metadata": {
                    "difficulty": ("容易", "中等", "困难")[index % 3],
                    "structure_type": ("单选题", "填空题")[index % 2],
                },
            }
        )
        candidate_rows.append(
            {
                "question_id": question_id,
                "retrieval_version": "hybrid-test",
                "candidates": [
                    {
                        "label_id": f"L{index % 4}",
                        "label_path": f"知识点@模块{index % 4}@标签",
                        "candidate_rank": 1,
                        "sparse_rank": 1,
                        "dense_rank": 1 if index % 2 == 0 else 2,
                    },
                    {
                        "label_id": f"D{index % 4}",
                        "label_path": f"知识点@模块{index % 4}@稠密标签",
                        "candidate_rank": 2,
                        "sparse_rank": 2,
                        "dense_rank": 1 if index % 2 else 2,
                    },
                ],
            }
        )
    _write_jsonl(units_path, units)
    _write_jsonl(candidates_path, candidate_rows)

    output = tmp_path / "sample"
    report = build_adjudication_audit_sample(
        units_path,
        candidates_path,
        output,
        sample_size=9,
        representative_size=6,
        seed="test-seed",
    )

    sampled_units = [
        json.loads(line)
        for line in (output / "audit_units.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    sampled_candidates = [
        json.loads(line)
        for line in (output / "audit_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert report["sample_size"] == 9
    assert report["split_counts"] == {"representative": 6, "stress": 3}
    assert [row["question_id"] for row in sampled_units] == [
        row["question_id"] for row in sampled_candidates
    ]
    assert {row["audit_split"] for row in sampled_units} == {
        "representative",
        "stress",
    }
    assert all(row["audit_strata"] for row in sampled_units)

    repeated = tmp_path / "repeated"
    repeated_report = build_adjudication_audit_sample(
        units_path,
        candidates_path,
        repeated,
        sample_size=9,
        representative_size=6,
        seed="test-seed",
    )
    assert repeated_report == report
    assert (repeated / "audit_units.jsonl").read_bytes() == (
        output / "audit_units.jsonl"
    ).read_bytes()
