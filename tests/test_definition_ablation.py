import json
from pathlib import Path

from bio_know_tag.definition_ablation import build_definition_ablation_sample


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_definition_ablation_prioritizes_volatile_and_stratifies_labels(tmp_path: Path):
    labels = tmp_path / "labels.jsonl"
    samples = tmp_path / "samples.jsonl"
    results = tmp_path / "results.jsonl"
    volatile = tmp_path / "volatile.jsonl"
    write_jsonl(
        labels,
        [
            {
                "label_id": "L1",
                "label_name": "低匹配",
                "label_path": "a",
                "definition": "定义1",
                "core_concepts": "核心1",
                "common_assessments": "考查1",
                "distinctions": "边界1",
            },
            {
                "label_id": "L2",
                "label_name": "高匹配",
                "label_path": "b",
                "definition": "定义2",
                "core_concepts": "核心2",
                "common_assessments": "考查2",
                "distinctions": "边界2",
            },
        ],
    )
    sample_rows = []
    result_rows = []
    for label_id in ("L1", "L2"):
        for index in range(3):
            question_id = f"q-{label_id}-{index}"
            sample_rows.append(
                {
                    "question_id": question_id,
                    "label_id": label_id,
                    "unit_type": "standalone",
                    "stem": "题目",
                    "options": "",
                    "answer_text": "答案",
                    "analysis": "解析",
                    "parent_stem": "",
                    "flags": {},
                }
            )
            result_rows.append(
                {
                    "question_id": question_id,
                    "label_id": label_id,
                    "match": label_id == "L2",
                    "relevance_score": 0.9 if label_id == "L2" else 0.1,
                }
            )
    write_jsonl(samples, sample_rows)
    write_jsonl(results, result_rows)
    write_jsonl(
        volatile,
        [{"question_id": "q-L1-0", "same_output": False}],
    )

    selected, report = build_definition_ablation_sample(
        positive_samples_path=samples,
        positive_results_path=results,
        labels_path=labels,
        volatile_group_paths=[volatile],
        low_threshold=0.7,
        low_count=2,
        high_count=1,
        seed="test",
    )

    assert len(selected) == 3
    by_label = {label_id: [] for label_id in ("L1", "L2")}
    for row in selected:
        by_label[row["label_id"]].append(row)
    assert len(by_label["L1"]) == 2
    assert len(by_label["L2"]) == 1
    assert any(row["sample_source"] == "volatile_5391" for row in by_label["L1"])
    assert report["selected_pairs"] == 3

