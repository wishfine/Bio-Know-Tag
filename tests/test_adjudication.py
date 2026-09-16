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
    assert "标签一考查" in prompt
    assert "知识点@模块@标签一" in prompt
    assert "旧knw_ids" not in prompt
    assert "candidate_rank" not in prompt
    assert "合理多标可以保留" in prompt
    assert "可以少选、置空或扩召" in prompt
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


def test_v85_prompt_restores_v83_layout_and_puts_reason_first():
    labels = {"L1": _label("L1", "标签一"), "L2": _label("L2", "标签二")}
    prompt, _ = build_adjudication_prompt(
        _unit(), [_candidate(1), _candidate(2)], labels
    )

    assert PROMPT_VERSION == "candidate-adjudication-v8.5-v83-reason-first"
    assert "错标的代价远高于漏标" in prompt
    assert "五道硬门槛" in prompt
    assert "反证复核" in prompt
    assert "只要任意一道不能确定通过" in prompt
    assert "不得因为研究对象、题干关键词或所属章节相同" in prompt
    assert "施肥过多" not in prompt
    assert "小分子跨膜" not in prompt
    assert "固定化脂酶" not in prompt
    assert "rejected_risky" not in prompt
    assert "reason用1至2句话、不超过120字" in prompt
    assert '"reason": "当前设问直接考查……"' in prompt
    assert '"selected": ["C01", "C05"]' in prompt
    schema = prompt.split("只输出一个JSON对象：", 1)[1]
    assert schema.index('"reason"') < schema.index('"selected"')
    assert schema.index('"selected"') < schema.index('"context_insufficient"')
    assert schema.index('"context_insufficient"') < schema.index('"need_expand_recall"')
    assert "evidence" not in prompt


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


def test_validate_v85_adjudication_accepts_reason_and_final_selected_codes():
    result = validate_adjudication_result(
        {
            "reason": "当前设问直接考查标签一和标签二。",
            "selected": ["C02", "C01", "C02"],
            "need_expand_recall": False,
            "context_insufficient": False,
        },
        {"C01", "C02"},
    )

    assert result["reason"] == "当前设问直接考查标签一和标签二。"
    assert result["selected"] == ["C02", "C01"]
    assert "rejected_risky" not in result
    assert result["none_of_candidates"] is False
    with pytest.raises(ValueError, match="short codes"):
        validate_adjudication_result(
            {
                "reason": "理由",
                "selected": [{"code": "C01", "evidence": "题干"}],
                "need_expand_recall": False,
                "context_insufficient": False,
            },
            {"C01"},
        )


def test_validate_adjudication_derives_empty_state():
    valid = {
        "reason": "理由",
        "selected": ["C01"],
        "need_expand_recall": False,
        "context_insufficient": False,
    }
    assert validate_adjudication_result(valid, {"C01", "C02"})["selected"] == [
        "C01"
    ]
    empty = validate_adjudication_result(
        {**valid, "selected": []},
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
                "need_expand_recall": True,
                "context_insufficient": False,
            },
            {"C01"},
        )


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
    candidates = [_candidate(index) for index in range(1, 26)]
    labels = {
        f"L{index}": _label(f"L{index}", f"标签{index}")
        for index in range(1, 26)
    }
    _, shuffled_code_map = build_adjudication_prompt(
        _unit(), candidates, labels
    )
    code_for_l23 = next(
        code for code, label_id in shuffled_code_map.items() if label_id == "L23"
    )
    candidates_path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "retrieval_version": "hybrid-v1-s18-d7-k25",
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
            for index in range(1, 26)
        ),
        encoding="utf-8",
    )

    class Response:
        content = json.dumps(
            {
                "reason": "当前题目直接考查标签23。",
                "selected": [code_for_l23],
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
    assert prediction["selected_labels"][0]["label_id"] == "L23"
    assert prediction["reason"] == "当前题目直接考查标签23。"
    assert prediction["selected_labels"][0]["candidate_rank"] == 23
    assert "rejected_risky_labels" not in prediction
    assert "output_conflict" not in prediction
    assert "output_normalization" not in prediction
    assert prediction["selected_labels"][0]["label_path"] == "知识点@模块@标签23"
    assert prediction["candidate_count"] == 25
    assert prediction["retrieval_version"] == "hybrid-v1-s18-d7-k25"
    assert report["success"] == 1
    assert report["selected_candidate_rank_distribution"] == {"23": 1}
    assert report["max_selected_candidate_rank"] == 23
    assert report["selected_from_rank_21_25"] == 1
    assert report["questions_using_rank_21_25"] == 1
    assert "rejected_risky_count" not in report
    assert "questions_with_rejected_risky" not in report
    assert "output_conflict" not in report
    assert "output_normalization_counts" not in report
    assert report["need_expand_recall"] == 0
    assert report["context_insufficient"] == 0
    assert report["usable_for_training"] == 1
    assert report["filtered_from_training"] == 0
    assert report["token_usage"]["requests_with_usage"] == 1
    assert report["token_usage"]["mean_prompt_tokens"] == 100.0
    assert report["token_usage"]["mean_completion_tokens"] == 20.0
    assert report["requests_retried"] == 1
    assert report["retry_error_types"] == {"ConnectionResetError": 1}
    assert "unverified_evidence_items" not in report
    assert "questions_with_unverified_evidence" not in report
    assert "evidence" not in prediction["selected_labels"][0]
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
    assert tail["prediction"]["selected_labels"][0]["candidate_rank"] == 23


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
