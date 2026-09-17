import json
from pathlib import Path

from bio_know_tag.definition_verifier import (
    PROMPT_VERSION,
    build_definition_verifier_prompt,
    run_definition_verifier,
    validate_definition_verifier_result,
)


def _unit() -> dict:
    return {
        "question_id": "q1",
        "parent_id": "p1",
        "unit_type": "sub_question",
        "parent_stem": "父题背景",
        "stem": "当前小题",
        "options": "",
        "answer_text": "答案",
        "analysis": "解析",
    }


def _labels() -> dict[str, dict]:
    return {
        "L1": {
            "label_id": "L1",
            "label_name": "标签一",
            "label_path": "知识点->模块->标签一",
            "definition": "标签一的定义",
            "distinctions": "标签一的边界",
            "core_concepts": "不应发给复核器的核心概念",
        },
        "L2": {
            "label_id": "L2",
            "label_name": "标签二",
            "label_path": "知识点->模块->标签二",
            "definition": "标签二的定义",
            "distinctions": "标签二的边界",
            "core_concepts": "也不应发给复核器",
        },
    }


def _prediction() -> dict:
    return {
        "question_id": "q1",
        "reason": "第一阶段理由不应发给复核器",
        "selected_labels": [
            {"label_id": "L1", "label_name": "标签一", "candidate_rank": 1},
            {"label_id": "L2", "label_name": "标签二", "candidate_rank": 2},
        ],
        "usable_for_training": True,
        "needs_review": False,
        "need_expand_recall": False,
        "context_insufficient": False,
    }


def test_definition_verifier_prompt_is_independent_and_definition_only():
    prompt, code_map = build_definition_verifier_prompt(
        _unit(), _prediction(), _labels()
    )

    assert PROMPT_VERSION == "definition-verifier-v1.1-explicit-subcase"
    assert set(code_map.values()) == {"L1", "L2"}
    assert "当前小题" in prompt
    assert "父题背景" in prompt
    assert "标签一的定义" in prompt
    assert "标签一的边界" in prompt
    assert "第一阶段理由不应发给复核器" not in prompt
    assert "不应发给复核器的核心概念" not in prompt
    assert "candidate_rank" not in prompt
    assert "不得因第一阶段已选中而倾向于保留" in prompt
    assert "只看definition和distinctions" in prompt
    assert "EXPLICIT_SUBCASE" in prompt
    assert "UNLISTED_SIBLING" in prompt
    assert "定义明确列出的一个子项" in prompt
    assert "未在定义中出现的同类方法" in prompt


def test_validate_definition_verifier_requires_every_code_once():
    result = validate_definition_verifier_result(
        {
            "results": [
                {
                    "code": "V02",
                    "match": False,
                    "coverage_relation": "DIFFERENT_TASK",
                    "question_target": "当前任务二",
                    "definition_target": "定义任务二",
                    "reason": "任务不同",
                },
                {
                    "code": "V01",
                    "match": True,
                    "coverage_relation": "EXPLICIT_SUBCASE",
                    "question_target": "当前任务一",
                    "definition_target": "定义任务一",
                    "reason": "完全一致",
                },
            ]
        },
        {"V01", "V02"},
    )

    assert [item["code"] for item in result] == ["V01", "V02"]
    assert result[0]["match"] is True
    assert result[0]["coverage_relation"] == "EXPLICIT_SUBCASE"


def test_run_definition_verifier_filters_rejected_label(tmp_path: Path):
    units_path = tmp_path / "units.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    output = tmp_path / "verified"
    units_path.write_text(json.dumps(_unit(), ensure_ascii=False) + "\n")
    predictions_path.write_text(json.dumps(_prediction(), ensure_ascii=False) + "\n")
    labels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _labels().values())
    )
    _, code_map = build_definition_verifier_prompt(
        _unit(), _prediction(), _labels()
    )
    code_by_label = {label_id: code for code, label_id in code_map.items()}

    class Response:
        content = json.dumps(
            {
                "results": [
                    {
                        "code": code_by_label["L1"],
                        "match": True,
                        "coverage_relation": "EXPLICIT_SUBCASE",
                        "question_target": "当前任务一",
                        "definition_target": "定义任务一",
                        "reason": "完全一致",
                    },
                    {
                        "code": code_by_label["L2"],
                        "match": False,
                        "coverage_relation": "UNLISTED_SIBLING",
                        "question_target": "当前任务二",
                        "definition_target": "定义任务二",
                        "reason": "仅共享背景",
                    },
                ]
            },
            ensure_ascii=False,
        )
        endpoint = "fake"
        attempts = 1
        latency_seconds = 0.01

    class Client:
        def chat(self, messages, *, max_tokens):
            return Response()

    report = run_definition_verifier(
        units_path,
        predictions_path,
        labels_path,
        output,
        Client(),
        model="fake-model",
    )

    verified = json.loads((output / "predictions.jsonl").read_text())
    assert [item["label_id"] for item in verified["selected_labels"]] == ["L1"]
    assert verified["definition_rejected_labels"][0]["label_id"] == "L2"
    assert verified["usable_for_training"] is True
    assert verified["stage1_reason"] == _prediction()["reason"]
    assert report["labels_kept"] == 1
    assert report["labels_rejected"] == 1
    assert report["questions_all_rejected"] == 0
