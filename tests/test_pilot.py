import json
from pathlib import Path

from bio_know_tag.pilot import build_pilot_package


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _unit(
    question_id: str,
    *,
    parent_id: str | None = None,
    unit_type: str = "standalone",
    candidates: list[str] | None = None,
    unmatched: list[str] | None = None,
    route: str = "R0",
    stem: str = "题干",
    flags: dict | None = None,
) -> dict:
    return {
        "question_id": question_id,
        "parent_id": parent_id or question_id,
        "unit_type": unit_type,
        "parent_stem": "父题材料" if unit_type == "sub_question" else "",
        "sibling_question_ids": [],
        "stem": stem,
        "options": "",
        "answer": "A",
        "answer_text": "A",
        "analysis": "解析",
        "legacy_knw_ids": (candidates or []) + (unmatched or []),
        "legacy_candidate_ids": candidates or [],
        "legacy_unmatched_ids": unmatched or [],
        "legacy_strategy_codes": ["K1"] if candidates else [],
        "proposed_route": route,
        "route_reason": "测试",
        "dedupe_hash": f"hash-{question_id}",
        "flags": {
            "parent_context_missing": False,
            "image_context_missing": False,
            "duplicate_of": None,
            "duplicate_label_conflict": False,
            **(flags or {}),
        },
        "metadata": {},
    }


def test_build_pilot_package_keeps_complete_parent_group_and_audit_samples(
    tmp_path: Path,
):
    units_path = tmp_path / "label_units.jsonl"
    parents_path = tmp_path / "parent_aggregation.jsonl"
    duplicates_path = tmp_path / "duplicate_groups.jsonl"
    output = tmp_path / "pilot"
    units = [
        _unit("solo-l1", candidates=["L1"]),
        _unit("solo-l2", candidates=["L2"]),
        _unit("missing-stem", candidates=["L1"], stem=""),
        _unit("unmatched", unmatched=["OLD"], route="R2"),
        _unit("dup-a", candidates=["L1"], route="R1"),
        _unit("dup-b", candidates=["L2"], route="R1"),
        _unit("child-a", parent_id="parent", unit_type="sub_question", candidates=["L1"], route="R1"),
        _unit("child-b", parent_id="parent", unit_type="sub_question", candidates=["L2"], route="R1"),
        _unit(
            "orphan",
            parent_id="missing-parent",
            unit_type="orphan_sub_question",
            candidates=["L1"],
            flags={"parent_context_missing": True},
        ),
    ]
    _write_jsonl(units_path, units)
    _write_jsonl(
        parents_path,
        [
            {
                "question_id": "parent",
                "parent_id": "parent",
                "unit_type": "composite_parent",
                "parent_stem": "父题材料",
                "child_question_ids": ["child-a", "child-b"],
                "legacy_candidate_ids": ["L1", "L2"],
            }
        ],
    )
    _write_jsonl(
        duplicates_path,
        [
            {
                "dedupe_hash": "same",
                "primary_question_id": "dup-a",
                "question_ids": ["dup-a", "dup-b"],
                "member_count": 2,
                "duplicate_label_conflict": True,
                "legacy_knw_id_sets": [["L1"], ["L2"]],
            }
        ],
    )
    report = build_pilot_package(
        units_path,
        parents_path,
        duplicates_path,
        output,
        target_size=8,
        parent_groups=1,
        duplicate_groups=1,
        audit_sample_size=1,
        stratum_sample_size=1,
    )

    pilot = [
        json.loads(line)
        for line in (output / "pilot_units.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_id = {record["question_id"]: record for record in pilot}
    parent_pilot = [
        json.loads(line)
        for line in (output / "pilot_parents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    duplicate_samples = [
        json.loads(line)
        for line in (output / "duplicate_samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert {"child-a", "child-b"}.issubset(by_id)
    assert by_id["child-a"]["sample_reasons"] == sorted(by_id["child-a"]["sample_reasons"])
    assert "complete_parent_group" in by_id["child-a"]["sample_reasons"]
    assert "empty_stem" in by_id["missing-stem"]["sample_reasons"]
    assert {"dup-a", "dup-b"}.issubset(by_id)
    assert "exact_duplicate_sample" in by_id["dup-a"]["sample_reasons"]
    assert "orphan_sub_question" in by_id["orphan"]["sample_reasons"]
    assert parent_pilot[0]["question_id"] == "parent"
    assert duplicate_samples[0]["dedupe_hash"] == "same"
    assert "legacy_candidate_ids" not in by_id["solo-l1"]
    assert "legacy_knw_id_sets" not in duplicate_samples[0]
    assert "duplicate_label_conflict" not in by_id["dup-a"]["flags"]
    assert report["legacy_fields_used"] is False
    assert report["pilot_parent_groups"] == 1
    assert report["duplicate_groups_sampled"] == 1
    assert report["pilot_units"] == len(pilot)
    assert json.loads((output / "pilot_report.json").read_text(encoding="utf-8")) == report
    assert not list(output.glob("*.tmp"))

    for unit in units:
        unit["legacy_knw_ids"] = ["completely-different"]
        unit["legacy_candidate_ids"] = ["completely-different"]
        unit["legacy_unmatched_ids"] = []
        unit["legacy_strategy_codes"] = ["KE"]
        unit["proposed_route"] = "R2"
        unit["route_reason"] = "这些字段不得影响抽样"
    _write_jsonl(units_path, units)
    repeated_output = tmp_path / "pilot-repeated"
    repeated_report = build_pilot_package(
        units_path,
        parents_path,
        duplicates_path,
        repeated_output,
        target_size=8,
        parent_groups=1,
        duplicate_groups=1,
        audit_sample_size=1,
        stratum_sample_size=1,
    )
    assert repeated_report == report
    assert (repeated_output / "pilot_units.jsonl").read_bytes() == (
        output / "pilot_units.jsonl"
    ).read_bytes()
