import json
from pathlib import Path

from bio_know_tag.volatility_review_html import build_volatility_review_html


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_builds_changed_only_standalone_review(tmp_path: Path) -> None:
    group = tmp_path / "group.jsonl"
    units = tmp_path / "units.jsonl"
    labels = tmp_path / "labels.jsonl"
    images = tmp_path / "images.jsonl"
    html = tmp_path / "review.html"
    rows = tmp_path / "review.jsonl"
    _write_jsonl(
        group,
        [
            {
                "question_id": "q1",
                "perturbation_group": "A_same_candidate_set",
                "same_output": False,
                "selection_jaccard": 0.5,
                "base_candidate_count": 25,
                "augmented_candidate_count": 25,
                "added_candidate_ids": [],
                "top25_selected_label_ids": ["L1"],
                "legacy_selected_label_ids": ["L2"],
                "shared_selected_label_ids": [],
                "top25_only_selected_label_ids": ["L1"],
                "legacy_only_selected_label_ids": ["L2"],
                "selected_added_legacy_ids": [],
            },
            {
                "question_id": "q2",
                "perturbation_group": "A_same_candidate_set",
                "same_output": True,
                "top25_selected_label_ids": ["L1"],
                "legacy_selected_label_ids": ["L1"],
            },
        ],
    )
    _write_jsonl(
        units,
        [
            {
                "question_id": "q1",
                "parent_id": "p1",
                "unit_type": "sub_question",
                "parent_stem": "父题",
                "stem": "题干</script>",
                "options": "A.甲",
                "answer_text": "A",
                "analysis": "解析",
            },
            {"question_id": "q2", "stem": "ignored"},
        ],
    )
    _write_jsonl(
        labels,
        [
            {"label_id": "L1", "label_name": "标签甲", "label_path": "知识->甲"},
            {"label_id": "L2", "label_name": "标签乙", "label_path": "知识->乙"},
        ],
    )
    _write_jsonl(
        images,
        [
            {
                "question_id": "q1",
                "stem_image_url": "https://example.com/stem.png",
                "analysis_image_url": "https://example.com/analysis.png",
            }
        ],
    )
    report = build_volatility_review_html(
        group_paths=[group],
        units_path=units,
        labels_path=labels,
        image_context_path=images,
        output_html_path=html,
        output_jsonl_path=rows,
    )
    assert report["review_questions"] == 1
    assert report["group_counts"] == {"A_same_candidate_set": 1}
    content = html.read_text(encoding="utf-8")
    assert "高中生物 Label 波动复核台" in content
    assert "https://example.com/stem.png" in content
    assert "题干<\\/script>" in content
    rendered = [json.loads(line) for line in rows.read_text(encoding="utf-8").splitlines()]
    assert rendered[0]["top25_labels"][0]["label_name"] == "标签甲"
    assert rendered[0]["legacy_labels"][0]["label_name"] == "标签乙"

