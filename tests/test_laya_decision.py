import json

from bio_know_tag.laya_decision import (
    LAYA_NUOL_INSTRUCTION,
    build_laya_state_and_questions,
    laya_answers_to_scores,
    laya_noul_to_match_scores,
)


def _unit():
    return {
        "question_id": "q1",
        "unit_type": "standalone",
        "parent_stem": "",
        "stem": "当前小题",
        "options": "A.甲\nB.乙",
        "answer_text": "A",
        "analysis": "解析",
        "flags": {"image_context_missing": False},
    }


def _labels():
    return {
        "L1": {
            "label_id": "L1",
            "label_name": "标签一",
            "label_path": "知识点->模块->标签一",
            "definition": "标签一定义",
            "core_concepts": "标签一核心",
            "distinctions": "标签一边界",
        },
        "L2": {
            "label_id": "L2",
            "label_name": "标签二",
            "label_path": "知识点->模块->标签二",
            "definition": "标签二定义",
            "core_concepts": "标签二核心",
            "distinctions": "标签二边界",
        },
    }


def test_laya_questions_use_ds_candidate_order_and_all_card_fields():
    candidates = [
        {"label_id": "L2", "candidate_rank": 1},
        {"label_id": "L1", "candidate_rank": 2},
    ]
    state, questions, code_map = build_laya_state_and_questions(
        _unit(), candidates, _labels()
    )

    assert state["stem"] == "当前小题"
    assert list(questions) == list(code_map)
    assert set(questions) == {"C01", "C02"}
    assert all(q["type"] == "noul" for q in questions.values())
    assert LAYA_NUOL_INSTRUCTION in questions["C01"]["instructions"]
    assert '"label_name":"标签二"' in questions["C01"]["instructions"]
    assert '"core_concepts":"标签二核心"' in questions["C01"]["instructions"]
    assert "common_assessments" not in questions["C01"]["instructions"]


def test_laya_answers_to_scores_validates_and_preserves_codes():
    scores = laya_answers_to_scores(
        {
            "C01": {"type": "noul", "noul": 0.2},
            "C02": {"type": "noul", "noul": 0.95},
        },
        {"C01": "L1", "C02": "L2"},
    )
    assert scores == {"C01": 0.2, "C02": 0.95}


def test_laya_noul_scores_are_inverted_for_match_selection():
    noul_scores = {"C01": 0.03, "C02": 0.95}

    assert laya_noul_to_match_scores(noul_scores) == {
        "C01": 0.97,
        "C02": 0.05,
    }


def test_laya_candidate_card_is_json_and_no_reasoning_is_required():
    _, questions, _ = build_laya_state_and_questions(
        _unit(), [{"label_id": "L1"}], _labels()
    )
    text = questions["C01"]["instructions"]
    card_line = next(
        line for line in text.splitlines() if line.startswith("待判断候选Label卡片：")
    )
    assert json.loads(card_line.split("：", 1)[1])
