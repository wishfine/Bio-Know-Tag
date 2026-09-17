import json
import hashlib
import threading
import time
from pathlib import Path

import pytest

from bio_know_tag.adjudication import (
    PROMPT_VERSION,
    build_adjudication_prompt,
    run_adjudication,
    validate_adjudication_result,
)
from bio_know_tag.ds import DSRequestError


def _label(label_id: str, name: str) -> dict:
    return {
        "label_id": label_id,
        "label_name": name,
        "label_path": f"知识点->模块->{name}",
        "definition": f"{name}定义",
        "core_concepts": f"{name}核心",
        "common_assessments": f"{name}考查",
        "distinctions": f"{name}边界",
    }


def _unit() -> dict:
    return {
        "question_id": "q1",
        "parent_id": "q1",
        "unit_type": "standalone",
        "parent_stem": "",
        "stem": "题干",
        "options": "",
        "answer_text": "答案",
        "analysis": "解析",
        "flags": {},
    }


def _candidate(index: int) -> dict:
    return {
        "label_id": f"L{index}",
        "label_name": f"标签{index}",
        "label_path": f"知识点@模块@标签{index}",
        "candidate_rank": index,
        "sources": ["sparse"] if index < 20 else ["dense"],
        "sparse_rank": index if index < 20 else None,
        "dense_rank": index if index >= 20 else None,
    }


def test_adjudication_prompt_uses_short_codes_and_teacher_definitions():
    labels = {"L1": _label("L1", "标签一"), "L2": _label("L2", "标签二")}
    prompt, code_map = build_adjudication_prompt(
        _unit(), [_candidate(1), _candidate(2)], labels
    )

    assert set(code_map.values()) == {"L1", "L2"}
    assert "C01" in prompt
    assert "标签一定义" in prompt
    assert "标签一核心" in prompt
    assert "标签一边界" in prompt
    assert "标签一考查" not in prompt
    assert "知识点@模块@标签一" in prompt
    assert "旧knw_ids" not in prompt
    assert "candidate_rank" not in prompt
    assert "多Label时" in prompt
    assert "可以少选、selected=[]或要求扩召" in prompt
    assert "删除测试" not in prompt
    assert '"parent_context_missing": false' in prompt
    assert '"image_context_missing": false' in prompt
    assert '"context_insufficient": false' in prompt
    assert "rejected_close_codes" not in prompt
    assert '"necessity"' not in prompt
    assert '"reason": "当前设问直接考查……"' in prompt
    assert '"none_of_candidates"' not in prompt
    assert '"missing_knowledge"' not in prompt


def test_adjudication_prompt_deterministically_shuffles_candidate_positions():
    labels = {
        f"L{index}": _label(f"L{index}", f"标签{index}")
        for index in range(1, 4)
    }
    candidates = [_candidate(index) for index in range(1, 4)]

    _, q1_first = build_adjudication_prompt(_unit(), candidates, labels)
    _, q1_second = build_adjudication_prompt(_unit(), candidates, labels)
    other_orders = []
    for index in range(2, 10):
        _, code_map = build_adjudication_prompt(
            {**_unit(), "question_id": f"q{index}"}, candidates, labels
        )
        other_orders.append(list(code_map.values()))

    assert q1_first == q1_second
    assert any(list(q1_first.values()) != order for order in other_orders)


def test_candidate_order_uses_v83_seed_for_clean_reason_ablation():
    labels = {
        f"L{index}": _label(f"L{index}", f"标签{index}")
        for index in range(1, 5)
    }
    candidates = [_candidate(index) for index in range(1, 5)]

    _, code_map = build_adjudication_prompt(_unit(), candidates, labels)
    expected = sorted(
        (candidate["label_id"] for candidate in candidates),
        key=lambda label_id: hashlib.sha256(
            f"q1\0{label_id}\0candidate-adjudication-v8.3-internal-reflection".encode(
                "utf-8"
            )
        ).digest(),
    )

    assert list(code_map.values()) == expected


def test_v91d_prompt_aligns_method_purpose_and_prioritizes_current_question():
    labels = {"L1": _label("L1", "标签一"), "L2": _label("L2", "标签二")}
    prompt, _ = build_adjudication_prompt(
        _unit(), [_candidate(1), _candidate(2)], labels
    )

    assert PROMPT_VERSION == "candidate-adjudication-v9.1d-method-purpose-alignment"
    assert "错标的代价远高于漏标" in prompt
    assert "硬否决：任意一项成立就拒绝，后续不得翻回" in prompt
    assert "反证复核" in prompt
    assert "具体对象A横向迁移到具体对象B" in prompt
    assert "题目只是使用已知结论完成推断" in prompt
    assert "生命层级、结构或作用通道不一致" in prompt
    assert "parent_stem只能在当前小题存在" in prompt
    assert "当前小题与父题背景的考查方向不同或冲突时" in prompt
    assert "parent_stem不得覆盖、扩张或替代当前设问的考点" in prompt
    assert "方法目的或结果指标不一致" in prompt
    assert "对象是什么、要得到什么指标或结论、使用什么方法" in prompt
    assert "调查对象、目标指标或结果含义不同，必须拒绝" in prompt
    assert "具体对象A只能上溯到通用机制Label" in prompt
    assert "实验、方法、观察、调查、测定、制作、构建、判定类Label" in prompt
    assert "Label的对象、方法目的和结果指标均与当前任务一致" in prompt
    assert "综合Label" in prompt
    assert "多个彼此独立的子知识，不等于考查综合Label" in prompt
    assert "施肥过多" not in prompt
    assert "果蝇白眼" not in prompt
    assert "人类红绿色盲" not in prompt
    assert "小分子跨膜" not in prompt
    assert "固定化脂酶" not in prompt
    assert "rejected_risky" not in prompt
    assert "core_concepts只用于解释该范围内的概念和机制" in prompt
    assert "evidence必须支持该Label的直接考查" in prompt
    assert "1至2句话、不超过120字" in prompt
    assert '"reason": "当前设问直接考查……"' in prompt
    assert '"selected": ["C01", "C05"]' in prompt
    schema = prompt.split("只输出一个JSON对象：", 1)[1]
    assert schema.index('"selected"') < schema.index('"evidence"')
    assert schema.index('"evidence"') < schema.index('"context_insufficient"')
    assert schema.index('"context_insufficient"') < schema.index('"need_expand_recall"')
    assert schema.index('"need_expand_recall"') < schema.index('"reason"')
    assert '"evidence": {"C01": "题目原文", "C05": "题目原文"}' in prompt


def test_run_adjudication_filters_units_without_question_text(tmp_path: Path):
    units_path = tmp_path / "units.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    output = tmp_path / "judge"
    unit = {**_unit(), "stem": "", "parent_stem": ""}
    units_path.write_text(json.dumps(unit, ensure_ascii=False) + "\n", encoding="utf-8")
    candidates_path.write_text(
        json.dumps({"question_id": "q1", "candidates": [_candidate(1)]}) + "\n",
        encoding="utf-8",
    )
    labels_path.write_text(
        json.dumps(_label("L1", "标签1"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    class Response:
        content = json.dumps(
            {
                "reason": "解析中存在相关知识。",
                "selected": ["C01"],
                "evidence": {"C01": "解析"},
                "need_expand_recall": False,
                "context_insufficient": False,
            },
            ensure_ascii=False,
        )
        endpoint = "fake"
        attempts = 1
        latency_seconds = 0.01

    class Client:
        def chat(self, messages, *, max_tokens):
            return Response()

    report = run_adjudication(
        units_path,
        candidates_path,
        labels_path,
        output,
        Client(),
        model="fake-model",
    )
    prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
    assert prediction["text_content_missing"] is True
    assert prediction["usable_for_training"] is False
    assert prediction["needs_review"] is True
    assert report["training_filter_reasons"]["missing_question_text"] == 1
    assert report["usable_for_training"] == 0


def test_prompt_restores_single_v83_question_object():
    labels = {"L1": _label("L1", "标签一")}
    unit = {
        **_unit(),
        "parent_stem": "父题背景唯一标识",
        "stem": "当前小题唯一标识",
        "answer_text": "当前答案唯一标识",
        "analysis": "当前解析唯一标识",
    }

    prompt, _ = build_adjudication_prompt(unit, [_candidate(1)], labels)

    assert "【唯一判标对象：当前小题】" not in prompt
    assert "【仅用于补全指代，不得作为独立判标依据】" not in prompt
    question_section = prompt.split("题目：", 1)[1].split("候选Label", 1)[0]
    assert question_section.index("父题背景唯一标识") < question_section.index(
        "当前小题唯一标识"
    )
    assert question_section.index("当前小题唯一标识") < question_section.index(
        "当前答案唯一标识"
    )
    assert question_section.index("当前答案唯一标识") < question_section.index(
        "当前解析唯一标识"
    )


def test_validate_v91_adjudication_accepts_reason_evidence_and_final_selected_codes():
    result = validate_adjudication_result(
        {
            "reason": "当前设问直接考查标签一和标签二。",
            "selected": ["C02", "C01", "C02"],
            "evidence": {"C01": "题干", "C02": "答案"},
            "need_expand_recall": False,
            "context_insufficient": False,
        },
        {"C01", "C02"},
    )

    assert result["reason"] == "当前设问直接考查标签一和标签二。"
    assert result["selected"] == ["C02", "C01"]
    assert result["evidence"] == {"C02": "答案", "C01": "题干"}
    assert "rejected_risky" not in result
    assert result["none_of_candidates"] is False
    with pytest.raises(ValueError, match="short codes"):
        validate_adjudication_result(
            {
                "reason": "理由",
                "selected": [{"code": "C01", "evidence": "题干"}],
                "evidence": {"C01": "题干"},
                "need_expand_recall": False,
                "context_insufficient": False,
            },
            {"C01"},
        )


def test_validate_adjudication_derives_empty_state():
    valid = {
        "reason": "理由",
        "selected": ["C01"],
        "evidence": {"C01": "题干"},
        "need_expand_recall": False,
        "context_insufficient": False,
    }
    assert validate_adjudication_result(valid, {"C01", "C02"})["selected"] == [
        "C01"
    ]
    empty = validate_adjudication_result(
        {**valid, "selected": [], "evidence": {}},
        {"C01", "C02"},
    )
    assert empty["none_of_candidates"] is True


@pytest.mark.parametrize("reason", [None, "", "   ", 123])
def test_validate_adjudication_requires_nonempty_string_reason(reason):
    with pytest.raises(ValueError, match="reason"):
        validate_adjudication_result(
            {
                "reason": reason,
                "selected": [],
                "evidence": {},
                "need_expand_recall": True,
                "context_insufficient": False,
            },
            {"C01"},
        )


def test_validate_adjudication_drops_unknown_answer_code_when_expanding_recall():
    result = validate_adjudication_result(
        {
            "selected": ["D09"],
            "evidence": {"D09": "题干"},
            "context_insufficient": False,
            "need_expand_recall": True,
            "reason": "正确知识点不在候选中，需要扩召。",
        },
        {"C01", "C02"},
    )

    assert result["selected"] == []
    assert result["none_of_candidates"] is True
    assert result["unknown_selected_codes_dropped"] == ["D09"]


def test_validate_adjudication_safely_drops_unknown_code_and_forces_expansion():
    result = validate_adjudication_result(
        {
            "selected": ["D09"],
            "evidence": {"D09": "题干"},
            "context_insufficient": False,
            "need_expand_recall": False,
            "reason": "选择D09。",
        },
        {"C01", "C02"},
    )

    assert result["selected"] == []
    assert result["unknown_selected_codes_dropped"] == ["D09"]
    assert result["need_expand_recall"] is True
    assert result["none_of_candidates"] is True


@pytest.mark.parametrize(
    ("selected", "expand", "context", "valid"),
    [
        (
            ["C01"],
            False,
            False,
            True,
        ),
        (
            ["C01"],
            True,
            False,
            True,
        ),
        ([], True, False, True),
        ([], False, True, True),
        ([], False, False, True),
        (["C01"], False, False, True),
        ("not-a-list", False, False, False),
    ],
)
def test_validate_adjudication_candidate_and_context_states(
    selected, expand, context, valid
):
    value = {
        "reason": "简短理由",
        "selected": selected,
        "evidence": {
            code: "题干" for code in selected if isinstance(code, str)
        } if isinstance(selected, list) else {},
        "need_expand_recall": expand,
        "context_insufficient": context,
    }
    if valid:
        result = validate_adjudication_result(value, {"C01"})
        assert result["context_insufficient"] is context
    else:
        with pytest.raises(ValueError):
            validate_adjudication_result(value, {"C01"})


def test_run_adjudication_records_tail_candidate_usage(tmp_path: Path):
    units_path = tmp_path / "units.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    output = tmp_path / "judge"
    units_path.write_text(json.dumps(_unit(), ensure_ascii=False) + "\n", encoding="utf-8")
    candidates = [_candidate(index) for index in range(1, 31)]
    labels = {
        f"L{index}": _label(f"L{index}", f"标签{index}")
        for index in range(1, 31)
    }
    _, shuffled_code_map = build_adjudication_prompt(
        _unit(), candidates, labels
    )
    code_for_l28 = next(
        code for code, label_id in shuffled_code_map.items() if label_id == "L28"
    )
    candidates_path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "retrieval_version": "hybrid-v1-s20-d10-k30",
                "candidates": candidates,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    labels_path.write_text(
        "".join(
            json.dumps(_label(f"L{index}", f"标签{index}"), ensure_ascii=False) + "\n"
            for index in range(1, 31)
        ),
        encoding="utf-8",
    )

    class Response:
        content = json.dumps(
            {
                "reason": "当前题目直接考查标签28。",
                "selected": [code_for_l28],
                "evidence": {code_for_l28: "题干"},
                "need_expand_recall": False,
                "context_insufficient": False,
            },
            ensure_ascii=False,
        )
        endpoint = "fake"
        attempts = 1
        latency_seconds = 0.1
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }
        reasoning = None
        response_message_keys = ("role", "content", "reasoning")
        retry_errors = (
            {
                "attempt": 1,
                "endpoint": "fake",
                "error_type": "ConnectionResetError",
                "error": "reset",
            },
        )

    class Client:
        def chat(self, messages, *, max_tokens):
            assert max_tokens == 1024
            return Response()

    report = run_adjudication(
        units_path,
        candidates_path,
        labels_path,
        output,
        Client(),
        model="fake-model",
    )

    prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
    assert prediction["selected_labels"][0]["label_id"] == "L28"
    assert prediction["reason"] == "当前题目直接考查标签28。"
    assert prediction["selected_labels"][0]["candidate_rank"] == 28
    assert "rejected_risky_labels" not in prediction
    assert "output_conflict" not in prediction
    assert "output_normalization" not in prediction
    assert prediction["selected_labels"][0]["label_path"] == "知识点@模块@标签28"
    assert prediction["candidate_count"] == 30
    assert prediction["retrieval_version"] == "hybrid-v1-s20-d10-k30"
    assert report["success"] == 1
    assert report["selected_candidate_rank_distribution"] == {"28": 1}
    assert report["max_selected_candidate_rank"] == 28
    assert report["selected_from_rank_21_25"] == 0
    assert report["questions_using_rank_21_25"] == 0
    assert report["selected_from_rank_21_plus"] == 1
    assert report["questions_using_rank_21_plus"] == 1
    assert report["selected_from_rank_26_30"] == 1
    assert report["questions_using_rank_26_30"] == 1
    assert "rejected_risky_count" not in report
    assert "questions_with_rejected_risky" not in report
    assert "output_conflict" not in report
    assert "output_normalization_counts" not in report
    assert report["need_expand_recall"] == 0
    assert report["context_insufficient"] == 0
    assert report["unknown_selected_codes_dropped"] == 0
    assert report["questions_with_unknown_selected_codes"] == 0
    assert report["usable_for_training"] == 1
    assert report["filtered_from_training"] == 0
    assert report["token_usage"]["requests_with_usage"] == 1
    assert report["token_usage"]["mean_prompt_tokens"] == 100.0
    assert report["token_usage"]["mean_completion_tokens"] == 20.0
    assert report["requests_retried"] == 1
    assert report["retry_error_types"] == {"ConnectionResetError": 1}
    assert "unverified_evidence_items" not in report
    assert "questions_with_unverified_evidence" not in report
    assert prediction["selected_labels"][0]["evidence"] == "题干"
    assert "evidence_verified" not in prediction["selected_labels"][0]
    assert "necessity" not in prediction["selected_labels"][0]
    assert prediction["usable_for_training"] is True
    assert "missing_knowledge" not in prediction
    assert prediction["needs_review"] is False
    evidence = json.loads((output / "evidence.jsonl").read_text(encoding="utf-8"))
    assert evidence["usage"]["total_tokens"] == 120
    assert evidence["reasoning"] is None
    assert evidence["retry_errors"][0]["error_type"] == "ConnectionResetError"
    tail = json.loads((output / "tail_selected.jsonl").read_text(encoding="utf-8"))
    assert tail["question"]["question_id"] == "q1"
    assert tail["prediction"]["selected_labels"][0]["candidate_rank"] == 28


def test_run_adjudication_can_issue_requests_concurrently(tmp_path: Path):
    units_path = tmp_path / "units.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    output = tmp_path / "judge"
    units = [{**_unit(), "question_id": question_id} for question_id in ("q1", "q2")]
    units_path.write_text(
        "".join(json.dumps(unit, ensure_ascii=False) + "\n" for unit in units),
        encoding="utf-8",
    )
    candidates_path.write_text(
        "".join(
            json.dumps(
                {"question_id": unit["question_id"], "candidates": [_candidate(1)]},
                ensure_ascii=False,
            )
            + "\n"
            for unit in units
        ),
        encoding="utf-8",
    )
    labels_path.write_text(
        json.dumps(_label("L1", "标签1"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    class Response:
        content = json.dumps(
            {
                "reason": "当前题目直接考查标签1。",
                "selected": ["C01"],
                "evidence": {"C01": "题干"},
                "need_expand_recall": False,
                "context_insufficient": False,
            },
            ensure_ascii=False,
        )
        endpoint = "fake"
        attempts = 1
        latency_seconds = 0.01

    class Client:
        def __init__(self):
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def chat(self, messages, *, max_tokens):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            return Response()

    client = Client()
    report = run_adjudication(
        units_path,
        candidates_path,
        labels_path,
        output,
        client,
        model="fake-model",
        workers=2,
    )

    assert report["success"] == 2
    assert report["workers"] == 2
    assert report["run_wall_seconds"] > 0
    assert report["requests_per_second_this_run"] > 0
    assert report["request_latency_seconds"]["count"] == 2
    assert client.max_active == 2


def test_run_adjudication_refuses_resume_with_changed_candidates(tmp_path: Path):
    units_path = tmp_path / "units.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    output = tmp_path / "judge"
    units_path.write_text(json.dumps(_unit(), ensure_ascii=False) + "\n", encoding="utf-8")
    labels_path.write_text(
        "".join(
            json.dumps(_label(f"L{index}", f"标签{index}"), ensure_ascii=False) + "\n"
            for index in (1, 2)
        ),
        encoding="utf-8",
    )
    candidates_path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "retrieval_version": "hybrid-v1-s18-d7-k25",
                "candidates": [_candidate(1)],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class Response:
        content = json.dumps(
            {
                "reason": "当前题目直接考查标签1。",
                "selected": ["C01"],
                "evidence": {"C01": "题干"},
                "need_expand_recall": False,
                "context_insufficient": False,
            },
            ensure_ascii=False,
        )
        endpoint = "fake"
        attempts = 1
        latency_seconds = 0.01

    class Client:
        def chat(self, messages, *, max_tokens):
            return Response()

    run_adjudication(
        units_path,
        candidates_path,
        labels_path,
        output,
        Client(),
        model="fake-model",
    )
    candidates_path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "retrieval_version": "changed",
                "candidates": [_candidate(2)],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run manifest mismatch"):
        run_adjudication(
            units_path,
            candidates_path,
            labels_path,
            output,
            Client(),
            model="fake-model",
        )


@pytest.mark.parametrize(
    ("selected", "expand", "context", "expected_reason"),
    [
        ([], False, False, "empty_selected"),
        (["C01"], True, False, "need_expand_recall"),
        (["C01"], False, True, "context_insufficient"),
    ],
)
def test_run_adjudication_filters_risky_training_rows(
    tmp_path: Path,
    selected,
    expand,
    context,
    expected_reason,
):
    units_path = tmp_path / "units.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    output = tmp_path / "judge"
    units_path.write_text(json.dumps(_unit(), ensure_ascii=False) + "\n", encoding="utf-8")
    candidates_path.write_text(
        json.dumps({"question_id": "q1", "candidates": [_candidate(1)]}) + "\n",
        encoding="utf-8",
    )
    labels_path.write_text(
        json.dumps(_label("L1", "标签1"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    class Response:
        content = json.dumps(
            {
                "reason": "根据题目证据作出判断。",
                "selected": selected,
                "evidence": {code: "题干" for code in selected},
                "need_expand_recall": expand,
                "context_insufficient": context,
            },
            ensure_ascii=False,
        )
        endpoint = "fake"
        attempts = 1
        latency_seconds = 0.01

    class Client:
        def chat(self, messages, *, max_tokens):
            return Response()

    report = run_adjudication(
        units_path,
        candidates_path,
        labels_path,
        output,
        Client(),
        model="fake-model",
    )
    prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))

    assert prediction["usable_for_training"] is False
    assert "output_conflict" not in prediction
    assert report["usable_for_training"] == 0
    assert report["filtered_from_training"] == 1
    assert report["training_filter_reasons"][expected_reason] == 1


def test_run_adjudication_records_terminal_request_diagnostics(tmp_path: Path):
    units_path = tmp_path / "units.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    output = tmp_path / "judge"
    units_path.write_text(json.dumps(_unit(), ensure_ascii=False) + "\n", encoding="utf-8")
    candidates_path.write_text(
        json.dumps({"question_id": "q1", "candidates": [_candidate(1)]}) + "\n",
        encoding="utf-8",
    )
    labels_path.write_text(
        json.dumps(_label("L1", "标签1"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    class Client:
        def chat(self, messages, *, max_tokens):
            raise DSRequestError(
                "chat completion failed after 5 attempts: reset",
                attempts=5,
                endpoint="http://ds.test/v1/chat/completions",
                latency_seconds=31.2,
                retry_errors=[
                    {
                        "attempt": attempt,
                        "endpoint": "http://ds.test/v1/chat/completions",
                        "error_type": "ConnectionResetError",
                        "error": "reset",
                    }
                    for attempt in range(1, 6)
                ],
            )

    report = run_adjudication(
        units_path,
        candidates_path,
        labels_path,
        output,
        Client(),
        model="fake-model",
    )
    evidence = json.loads((output / "evidence.jsonl").read_text(encoding="utf-8"))

    assert report["success"] == 0
    assert evidence["attempts"] == 5
    assert evidence["endpoint"] == "http://ds.test/v1/chat/completions"
    assert evidence["latency_seconds"] == 31.2
    assert len(evidence["retry_errors"]) == 5
