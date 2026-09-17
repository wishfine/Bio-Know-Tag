import json
from pathlib import Path

from bio_know_tag.teacher_review_export import export_teacher_review_packages


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_export_teacher_review_packages_separates_positive_and_boundary(tmp_path: Path):
    labels = []
    strategies = []
    metrics = []
    corrected = []
    for label_id, name in (("L1", "正样本问题"), ("L2", "新边界风险"), ("L3", "稳定Label")):
        labels.append(
            {
                "label_id": label_id,
                "label_name": name,
                "label_path": f"知识点@{name}",
                "label_type": "知识",
                "definition": f"{name}定义",
                "core_concepts": f"{name}核心",
                "common_assessments": f"{name}考法",
                "distinctions": f"{name}边界",
            }
        )
        strategies.append(
            {
                "label_id": label_id,
                "final_strategy": {
                    "status": "confirmed",
                    "manual_followup_required": False,
                },
            }
        )
    metrics.extend(
        [
            {
                "label_id": "L1",
                "planned": 500,
                "match_rate": 0.10,
                "zero_rate": 0.70,
                "preliminary_grade": "C_NEEDS_DIAGNOSIS",
            },
            {
                "label_id": "L2",
                "planned": 500,
                "match_rate": 0.80,
                "zero_rate": 0.05,
                "preliminary_grade": "A_STABLE_CANDIDATE",
            },
            {
                "label_id": "L3",
                "planned": 500,
                "match_rate": 0.90,
                "zero_rate": 0.01,
                "preliminary_grade": "A_STABLE_CANDIDATE",
            },
        ]
    )
    corrected.extend(
        [
            {
                "label_id": "L1",
                "positive_count": 500,
                "hard_negative_total": 40,
                "first_stage_accepted": 4,
                "first_stage_rejected": 36,
                "colabel_decision_counts": {"目标Label边界过宽": 2},
                "invalid_sibling_negatives": 2,
                "unresolved": 0,
                "valid_negative_count": 38,
                "true_boundary_errors": 2,
                "corrected_boundary_error_rate": 2 / 38,
                "final_screen": "C_NARROW_OR_LEGACY_NOISE_REVIEW",
            },
            {
                "label_id": "L2",
                "positive_count": 500,
                "hard_negative_total": 30,
                "first_stage_accepted": 2,
                "first_stage_rejected": 28,
                "colabel_decision_counts": {"目标Label边界过宽": 1, "合理共标": 1},
                "invalid_sibling_negatives": 1,
                "unresolved": 0,
                "valid_negative_count": 29,
                "true_boundary_errors": 6,
                "corrected_boundary_error_rate": 0.20,
                "final_screen": "D_BROAD_BOUNDARY_REVIEW",
            },
            {
                "label_id": "L3",
                "positive_count": 500,
                "hard_negative_total": 30,
                "first_stage_accepted": 1,
                "first_stage_rejected": 29,
                "colabel_decision_counts": {"合理共标": 1},
                "invalid_sibling_negatives": 1,
                "unresolved": 0,
                "valid_negative_count": 29,
                "true_boundary_errors": 0,
                "corrected_boundary_error_rate": 0.0,
                "final_screen": "A_STABLE_CANDIDATE",
            },
        ]
    )
    positive_results = [
        {
            "task_id": f"{label_id}-{index}",
            "question_id": f"Q-{label_id}-{index}",
            "label_id": label_id,
            "relevance_score": score,
            "match": score >= 0.70,
        }
        for label_id in ("L1", "L2", "L3")
        for index, score in enumerate((0.0, 0.2, 0.65, 0.75, 0.90), 1)
    ]
    positive_tasks = [
        {
            "pair_id": row["task_id"],
            "question_id": row["question_id"],
            "label_id": row["label_id"],
            "unit_type": "standalone",
            "stem": f"{row['question_id']}题干",
        }
        for row in positive_results
    ]
    image_context = [
        {
            "question_id": row["question_id"],
            "parent_id": row["question_id"],
            "unit_type": "standalone",
            "stem_image_url": f"https://x/{row['question_id']}.png",
            "analysis_image_url": f"https://x/{row['question_id']}-analysis.png",
        }
        for row in positive_results
    ] + [
        {
            "question_id": f"NQ{index}",
            "parent_id": f"NQ{index}",
            "unit_type": "standalone",
            "stem_image_url": f"https://x/NQ{index}.png",
            "analysis_image_url": f"https://x/NQ{index}-analysis.png",
        }
        for index in range(1, 4)
    ]
    hard_samples = [
        {
            "pair_id": "HN1",
            "question_id": "NQ1",
            "label_id": "L2",
            "source_label_ids": ["L3"],
            "source_label_names": ["稳定Label"],
        },
        {
            "pair_id": "HN2",
            "question_id": "NQ2",
            "label_id": "L2",
            "source_label_ids": ["L3"],
            "source_label_names": ["稳定Label"],
        },
        {
            "pair_id": "HN3",
            "question_id": "NQ3",
            "label_id": "L2",
            "source_label_ids": ["L3"],
            "source_label_names": ["稳定Label"],
        },
    ]
    hard_results = [
        {"task_id": "HN1", "question_id": "NQ1", "relevance_score": 0.90},
        {"task_id": "HN2", "question_id": "NQ2", "relevance_score": 0.85},
        {"task_id": "HN3", "question_id": "NQ3", "relevance_score": 0.10},
    ]
    colabel_results = [
        {"task_id": "HN1", "question_id": "NQ1", "decision": "目标Label边界过宽"},
        {"task_id": "HN2", "question_id": "NQ2", "decision": "合理共标"},
    ]
    paths = {}
    for name, rows in (
        ("labels", labels),
        ("strategies", strategies),
        ("metrics", metrics),
        ("corrected", corrected),
        ("positive_tasks", positive_tasks),
        ("image_context", image_context),
        ("positive", positive_results),
        ("hard_samples", hard_samples),
        ("hard_results", hard_results),
        ("colabel", colabel_results),
    ):
        paths[name] = tmp_path / f"{name}.jsonl"
        _write_jsonl(paths[name], rows)

    output = tmp_path / "review"
    report = export_teacher_review_packages(
        labels_path=paths["labels"],
        strategies_path=paths["strategies"],
        positive_tasks_path=paths["positive_tasks"],
        positive_results_path=paths["positive"],
        positive_per_label_path=paths["metrics"],
        image_context_path=paths["image_context"],
        corrected_assessments_path=paths["corrected"],
        hard_negative_samples_path=paths["hard_samples"],
        hard_negative_results_path=paths["hard_results"],
        colabel_results_path=paths["colabel"],
        output_dir=output,
        examples_per_side=10,
    )

    assert report["positive_issue_labels"] == 1
    assert report["additional_boundary_labels"] == 1
    positive_files = list((output / "positive_issue_135").glob("*.json"))
    boundary_files = list((output / "boundary_risk_additional").glob("*.json"))
    assert len(positive_files) == 1
    assert len(boundary_files) == 1

    positive = json.loads(positive_files[0].read_text(encoding="utf-8"))
    assert positive["label"]["label_id"] == "L1"
    assert positive["positive_coverage"]["basically_irrelevant_over_total"] == "1/5"
    assert positive["positive_coverage"]["related_below_definition_over_total"] == "2/5"
    assert positive["positive_coverage"]["matched_over_total"] == "2/5"
    assert len(positive["representative_positive_ds_false"]) == 3
    assert len(positive["representative_positive_ds_true"]) == 2
    positive_item = positive["representative_positive_ds_false"][0]
    assert positive_item["label_path"] == "知识点@正样本问题"
    assert positive_item["parent_id"] == positive_item["question_id"]
    assert positive_item["question_type"] == "standalone"
    assert positive_item["stem"].endswith("题干")
    assert positive_item["stem_image_url"].startswith("https://x/")
    assert positive_item["analysis_image_url"].endswith("-analysis.png")

    boundary = json.loads(boundary_files[0].read_text(encoding="utf-8"))
    assert boundary["label"]["label_id"] == "L2"
    assert boundary["corrected_boundary_summary"]["true_boundary_errors"] == 6
    assert boundary["representative_target_excluded_after_colabel"][0]["question_id"] == "NQ1"
    assert boundary["representative_target_excluded_after_colabel"][0]["parent_id"] == "NQ1"
    assert boundary["representative_target_excluded_after_colabel"][0]["stem"] == ""
    assert boundary["representative_target_excluded_after_colabel"][0]["stem_image_url"] == "https://x/NQ1.png"
    assert boundary["representative_reasonable_colabel"][0]["question_id"] == "NQ2"
    assert boundary["representative_first_stage_target_rejected"][0]["question_id"] == "NQ3"
