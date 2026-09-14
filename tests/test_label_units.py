import json
from pathlib import Path

from bio_know_tag.label_units import answer_to_text, build_labeling_derivatives


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_answer_to_text_recursively_cleans_fill_blank_answers():
    answer = [
        {"answers": ["<p>甲&nbsp;</p>", "乙<sup>2</sup>"], "blankId": "101", "index": 0},
        {"answers": ["CO<sub>2</sub>"], "blankId": "102", "index": 1},
    ]

    assert answer_to_text(answer) == "空1：甲 / 乙^{2}；空2：CO_{2}"


def test_build_derivatives_flattens_units_skips_synthetic_parent_and_tracks_duplicates(
    tmp_path: Path,
):
    processed = tmp_path / "questions.jsonl"
    labels = tmp_path / "labels.jsonl"
    orphan_audit = tmp_path / "orphan-parents.jsonl"
    run_dir = tmp_path / "run"
    duplicate_answer = [
        {"answers": ["<p>A&nbsp;</p>"], "blankId": "101", "index": 0}
    ]
    _write_jsonl(
        processed,
        [
            {
                "question_id": "solo",
                "parent_id": "solo",
                "stem": "如图，独立题",
                "options": "",
                "answer": duplicate_answer,
                "analysis": "解析",
                "knw_ids": ["L1"],
                "sub_questions": [],
            },
            {
                "question_id": "parent",
                "parent_id": "parent",
                "stem": "共同材料",
                "options": "",
                "answer": "",
                "analysis": "",
                "knw_ids": ["L1", "L2"],
                "sub_questions": [
                    {
                        "question_id": "child-1",
                        "parent_id": "parent",
                        "stem": "小题一",
                        "options": "A. 甲",
                        "answer": "A",
                        "analysis": "解析一",
                        "knw_ids": ["L1"],
                    },
                    {
                        "question_id": "child-2",
                        "parent_id": "parent",
                        "stem": "小题二",
                        "options": "",
                        "answer": "B",
                        "analysis": "解析二",
                        "knw_ids": ["old-unmatched"],
                    },
                ],
            },
            {
                "question_id": "missing-parent",
                "parent_id": "missing-parent",
                "stem": "",
                "options": "",
                "answer": "",
                "analysis": "",
                "sub_questions": [
                    {
                        "question_id": "orphan-child",
                        "parent_id": "missing-parent",
                        "stem": "如图，独立题",
                        "options": "",
                        "answer": duplicate_answer,
                        "analysis": "解析",
                        "knw_ids": ["L2"],
                    }
                ],
            },
        ],
    )
    _write_jsonl(
        labels,
        [
            {
                "label_id": "L1",
                "label_name": "标签一",
                "reference_strategy": {"关键词策略代码": "K1"},
            },
            {
                "label_id": "L2",
                "label_name": "标签二",
                "reference_strategy": {"关键词策略代码": "K2"},
            },
        ],
    )
    _write_jsonl(
        orphan_audit,
        [{"orphan_parent_id": "missing-parent", "parent": {}}],
    )

    report = build_labeling_derivatives(
        processed,
        labels,
        orphan_audit,
        run_dir,
        progress_every=0,
    )

    units = [
        json.loads(line)
        for line in (run_dir / "label_units.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_id = {unit["question_id"]: unit for unit in units}
    parents = [
        json.loads(line)
        for line in (run_dir / "parent_aggregation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    duplicate_groups = [
        json.loads(line)
        for line in (run_dir / "duplicate_groups.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert report["label_units"] == 4
    assert report["standalone_units"] == 1
    assert report["sub_question_units"] == 2
    assert report["orphan_sub_question_units"] == 1
    assert report["real_compound_parents"] == 1
    assert report["synthetic_parent_containers_skipped"] == 1
    assert report["parent_context_missing_units"] == 1
    assert report["image_context_missing_units"] == 2
    assert [parent["question_id"] for parent in parents] == ["parent"]
    assert parents[0]["child_question_ids"] == ["child-1", "child-2"]
    assert by_id["child-1"]["parent_stem"] == "共同材料"
    assert by_id["orphan-child"]["parent_stem"] == ""
    assert by_id["orphan-child"]["unit_type"] == "orphan_sub_question"
    assert by_id["orphan-child"]["flags"]["parent_context_missing"] is True
    assert by_id["solo"]["answer_text"] == "空1：A"
    assert by_id["solo"]["flags"]["image_context_missing"] is True
    assert by_id["child-1"]["proposed_route"] == "R1"
    assert by_id["child-2"]["proposed_route"] == "R2"
    assert len(duplicate_groups) == 1
    assert duplicate_groups[0]["question_ids"] == ["solo", "orphan-child"]
    assert duplicate_groups[0]["duplicate_label_conflict"] is True
    assert by_id["solo"]["flags"]["duplicate_label_conflict"] is True
    assert by_id["orphan-child"]["flags"]["duplicate_label_conflict"] is True
    assert {by_id["solo"]["flags"]["duplicate_of"], by_id["orphan-child"]["flags"]["duplicate_of"]} == {
        None,
        "solo",
    }
    assert json.loads((run_dir / "build_report.json").read_text(encoding="utf-8")) == report
    route_report = json.loads((run_dir / "route_report.json").read_text(encoding="utf-8"))
    assert sum(route_report["route_counts"].values()) == 4
    assert not list(run_dir.glob("*.tmp"))
