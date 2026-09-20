import json
from pathlib import Path

from bio_know_tag.adjudication_review_sample import (
    build_adjudication_review_sample,
    classify_audit_tier,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prediction(question_id: str, labels: list[tuple[str, int, list[str]]]) -> dict:
    return {
        "question_id": question_id,
        "selected_labels": [
            {
                "label_id": label_id,
                "label_name": label_id,
                "candidate_rank": rank,
                "sources": sources,
                "evidence": f"evidence-{question_id}-{label_id}",
            }
            for label_id, rank, sources in labels
        ],
        "context_insufficient": False,
        "need_expand_recall": False,
        "usable_for_training": True,
    }


def test_classify_audit_tier_uses_positive_and_boundary_risk() -> None:
    assert classify_audit_tier(
        {"planned": 500, "match_rate": 0.40, "zero_rate": 0.35},
        {"final_screen": "A_STABLE_CANDIDATE"},
    )[0] == "HIGH"
    assert classify_audit_tier(
        {"planned": 500, "match_rate": 0.80, "zero_rate": 0.05},
        {"final_screen": "B_MINOR_BOUNDARY_REVIEW"},
    )[0] == "MEDIUM"
    assert classify_audit_tier(
        {"planned": 500, "match_rate": 0.85, "zero_rate": 0.05},
        {"final_screen": "A_STABLE_CANDIDATE"},
    )[0] == "STABLE"


def test_build_review_sample_prioritizes_deltas_and_duplicate_inconsistency(
    tmp_path: Path,
) -> None:
    labels = [
        {"label_id": "L1", "label_name": "高风险"},
        {"label_id": "L2", "label_name": "稳定"},
        {"label_id": "L3", "label_name": "中风险"},
    ]
    units = [
        {
            "question_id": question_id,
            "parent_id": question_id,
            "unit_type": "standalone",
            "dedupe_hash": dedupe_hash,
            "stem": f"stem-{question_id}",
            "options": "A.x",
            "answer_text": "A",
            "analysis": "analysis",
            "flags": {},
        }
        for question_id, dedupe_hash in [
            ("q1", "h1"),
            ("q2", "h2"),
            ("q3", "h3"),
            ("q4", "same"),
            ("q5", "same"),
            ("q6", "h6"),
        ]
    ]
    top25 = [
        _prediction("q1", [("L1", 1, ["sparse"])]),
        _prediction("q2", []),
        _prediction("q3", [("L1", 24, ["dense"])]),
        _prediction("q4", [("L2", 1, ["sparse"])]),
        _prediction("q5", []),
        _prediction("q6", [("L3", 2, ["sparse"])]),
    ]
    legacy = [
        _prediction("q1", [("L1", 1, ["sparse", "legacy"])]),
        _prediction("q2", [("L1", 26, ["legacy"])]),
        _prediction("q3", []),
        _prediction("q4", [("L2", 1, ["sparse", "legacy"])]),
        _prediction("q5", [("L2", 26, ["legacy"])]),
        _prediction("q6", [("L3", 2, ["sparse", "legacy"])]),
    ]
    positive = [
        {
            "label_id": "L1",
            "planned": 500,
            "match_rate": 0.40,
            "zero_rate": 0.35,
            "preliminary_grade": "C_NEEDS_DIAGNOSIS",
        },
        {
            "label_id": "L2",
            "planned": 500,
            "match_rate": 0.85,
            "zero_rate": 0.05,
            "preliminary_grade": "A_STABLE_CANDIDATE",
        },
        {
            "label_id": "L3",
            "planned": 500,
            "match_rate": 0.75,
            "zero_rate": 0.10,
            "preliminary_grade": "B_MINOR_REVIEW",
        },
    ]
    boundaries = [
        {"label_id": "L1", "final_screen": "E_BOUNDARY_CONFLICT_REVIEW"},
        {"label_id": "L2", "final_screen": "A_STABLE_CANDIDATE"},
        {"label_id": "L3", "final_screen": "B_MINOR_BOUNDARY_REVIEW"},
    ]
    image_context = [
        {
            "question_id": question_id,
            "stem_image_url": f"https://example.test/{question_id}-stem.png",
            "analysis_image_url": f"https://example.test/{question_id}-analysis.png",
            "parent_stem_image_url": "",
            "parent_analysis_image_url": "",
            "content_status": "text_with_image_context",
            "eligible_for_text_labeling": True,
            "needs_content_review": False,
        }
        for question_id, _ in [
            ("q1", "h1"),
            ("q2", "h2"),
            ("q3", "h3"),
            ("q4", "same"),
            ("q5", "same"),
            ("q6", "h6"),
        ]
    ]
    paths = {}
    for name, rows in {
        "labels": labels,
        "units": units,
        "top25": top25,
        "legacy": legacy,
        "positive": positive,
        "boundaries": boundaries,
        "image_context": image_context,
    }.items():
        paths[name] = tmp_path / f"{name}.jsonl"
        _write_jsonl(paths[name], rows)

    output = tmp_path / "output"
    report = build_adjudication_review_sample(
        paths["units"],
        paths["top25"],
        paths["legacy"],
        paths["labels"],
        paths["positive"],
        paths["boundaries"],
        output,
        image_context_path=paths["image_context"],
        high_count=3,
        medium_count=2,
        stable_count=2,
        seed="test-seed",
    )

    assert report["questions"] == 6
    assert report["tier_counts"] == {"HIGH": 1, "MEDIUM": 1, "STABLE": 1}
    per_label = {
        row["label_id"]: row
        for row in map(json.loads, (output / "per_label.jsonl").read_text().splitlines())
    }
    assert per_label["L1"]["production"]["legacy_only_selected"] == 1
    assert per_label["L1"]["production"]["top25_only_selected"] == 1
    assert per_label["L2"]["production"]["duplicate_inconsistency"] == 2
    tasks = [json.loads(line) for line in (output / "review_tasks.jsonl").read_text().splitlines()]
    l1_tasks = [row for row in tasks if row["label_id"] == "L1"]
    assert {row["selection_status"] for row in l1_tasks} == {
        "both_selected",
        "legacy_only_selected",
        "top25_only_selected",
    }
    l2_tasks = [row for row in tasks if row["label_id"] == "L2"]
    assert any(row["duplicate_inconsistency"] for row in l2_tasks)
    assert l1_tasks[0]["stem_image_url"].startswith("https://example.test/")
    assert "image_context" in report["input_sha256"]
    assert (output / "labels" / "L1.json").is_file()
