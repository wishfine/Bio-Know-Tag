import json
from pathlib import Path

import pytest

from bio_know_tag.ds_stability import (
    choose_unstable_rows,
    default_conditions,
    jaccard,
    parse_choice,
    parse_condition,
    strict_majority_ids,
)


def test_parse_condition_and_defaults():
    condition = parse_condition("temp01-n4:0.1:4:30:42")
    assert condition.name == "temp01-n4"
    assert condition.temperature == 0.1
    assert condition.n == 4
    assert condition.workers == 30
    assert condition.seed == 42
    assert [item.name for item in default_conditions()] == [
        "temp0-workers1",
        "temp0-workers10",
        "temp0-workers30",
        "temp0-seed42",
    ]


def test_parse_condition_rejects_invalid_values():
    with pytest.raises(ValueError):
        parse_condition("bad:0.1:4")
    with pytest.raises(ValueError):
        parse_condition("bad:-0.1:4:1")
    with pytest.raises(ValueError):
        parse_condition("bad:0.1:0:1")


def test_majority_and_jaccard():
    assert strict_majority_ids([{"A", "B"}, {"A"}, {"A", "C"}]) == {"A"}
    assert strict_majority_ids([{"A"}, set(), {"B"}, {"B"}]) == set()
    assert jaccard({"A"}, {"A", "B"}) == 0.5
    assert jaccard(set(), set()) == 1.0


def test_parse_choice_maps_short_codes_and_preserves_errors():
    parsed, selected, error = parse_choice(
        json.dumps(
            {
                "selected": ["C01"],
                "evidence": {"C01": "题干证据"},
                "context_insufficient": False,
                "need_expand_recall": False,
                "reason": "直接考查。",
            },
            ensure_ascii=False,
        ),
        {"C01": "label-a"},
    )
    assert error is None
    assert parsed["selected"] == ["C01"]
    assert selected == {"label-a"}

    parsed, selected, error = parse_choice("not-json", {"C01": "label-a"})
    assert parsed is None
    assert selected == set()
    assert error.startswith("ValueError:")


def test_choose_unstable_rows_is_deterministic_and_excludes_same_output(tmp_path: Path):
    path = tmp_path / "A.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "question_id": question_id,
                    "same_output": same,
                    "perturbation_group": "A_same_candidate_set",
                }
            )
            for question_id, same in (("q1", False), ("q2", True), ("q3", False))
        )
        + "\n",
        encoding="utf-8",
    )
    first = choose_unstable_rows([path], limit=1, seed=7)
    second = choose_unstable_rows([path], limit=1, seed=7)
    assert first == second
    assert first[0]["same_output"] is False
