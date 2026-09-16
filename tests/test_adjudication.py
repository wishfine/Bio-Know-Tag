import json
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
    assert '"reason"' not in prompt
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
    _, q2 = build_adjudication_prompt(
        {**_unit(), "question_id": "q2"}, candidates, labels
    )

    assert q1_first == q1_second
    assert list(q1_first.values()) != list(q2.values())


def test_v8_prompt_uses_asymmetric_loss_and_adversarial_self_check():
    labels = {"L1": _label("L1", "标签一"), "L2": _label("L2", "标签二")}
    prompt, _ = build_adjudication_prompt(
        _unit(), [_candidate(1), _candidate(2)], labels
    )

    assert PROMPT_VERSION == "candidate-adjudication-v8.2-selected-first-ablation"
    assert "错标的代价远高于漏标" in prompt
    assert "五道硬门槛" in prompt
    assert "反证复核" in prompt
    assert "只要任意一道不能确定通过" in prompt
    assert "不得因为研究对象、题干关键词或所属章节相同" in prompt
    assert "施肥过多" not in prompt
    assert "小分子跨膜" not in prompt
    assert "固定化脂酶" not in prompt
    assert "rejected_risky不是所有未选候选的列表" in prompt
    assert "输出前做最终协议检查" in prompt
    assert '"rejected_risky": ["C03"]' in prompt
    assert '"selected": ["C01", "C05"]' in prompt
    schema = prompt.split("只输出一个JSON对象：", 1)[1]
    assert schema.index('"selected"') < schema.index('"rejected_risky"')
    assert schema.index('"rejected_risky"') < schema.index('"context_insufficient"')
    assert schema.index('"context_insufficient"') < schema.index('"need_expand_recall"')
    assert "evidence" not in prompt


def test_validate_v8_adjudication_separates_selected_and_rejected_risky():
    result = validate_adjudication_result(
        {
            "selected": ["C02", "C01", "C02"],
            "rejected_risky": [],
            "need_expand_recall": False,
            "context_insufficient": False,
        },
        {"C01", "C02"},
    )

    assert result["selected"] == ["C02", "C01"]
    assert result["rejected_risky"] == []
    assert result["none_of_candidates"] is False
    with pytest.raises(ValueError, match="short codes"):
        validate_adjudication_result(
            {
                "selected": [{"code": "C01", "evidence": "题干"}],
                "rejected_risky": [],
                "need_expand_recall": False,
                "context_insufficient": False,
            },
            {"C01"},
        )


def test_validate_adjudication_derives_empty_state():
    valid = {
        "selected": ["C01"],
        "rejected_risky": [],
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


def test_validate_v8_safely_normalizes_overlap_and_excess_risk_codes():
    base = {
        "selected": ["C01"],
        "rejected_risky": ["C02"],
        "need_expand_recall": False,
        "context_insufficient": False,
    }
    result = validate_adjudication_result(base, {"C01", "C02", "C03", "C04"})
    assert result["rejected_risky"] == ["C02"]

    normalized = validate_adjudication_result(
        {**base, "rejected_risky": ["C01", "C02", "C03", "C04"]},
        {"C01", "C02", "C03", "C04"},
    )

    assert normalized["selected"] == []
    assert normalized["rejected_risky"] == ["C01", "C02", "C03"]
    assert normalized["need_expand_recall"] is True
    assert normalized["output_conflict"] is True
    assert normalized["normalization"] == {
        "rejected_risky_truncated": 1,
        "selected_removed_as_rejected": ["C01"],
        "need_expand_recall_forced": True,
    }


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
        "selected": selected,
        "rejected_risky": [],
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
    code_for_l22 = next(
        code for code, label_id in shuffled_code_map.items() if label_id == "L22"
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
                "selected": [code_for_l23],
                "rejected_risky": [code_for_l22],
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
    assert prediction["selected_labels"][0]["candidate_rank"] == 23
    assert prediction["rejected_risky_labels"][0]["label_id"] == "L22"
    assert prediction["rejected_risky_labels"][0]["candidate_rank"] == 22
    assert prediction["selected_labels"][0]["label_path"] == "知识点@模块@标签23"
    assert prediction["candidate_count"] == 25
    assert prediction["retrieval_version"] == "hybrid-v1-s18-d7-k25"
    assert report["success"] == 1
    assert report["selected_candidate_rank_distribution"] == {"23": 1}
    assert report["max_selected_candidate_rank"] == 23
    assert report["selected_from_rank_21_25"] == 1
    assert report["questions_using_rank_21_25"] == 1
    assert report["rejected_risky_count"] == 1
    assert report["questions_with_rejected_risky"] == 1
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
                "selected": ["C01"],
                "rejected_risky": [],
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
                "selected": ["C01"],
                "rejected_risky": [],
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
    ("selected", "rejected", "expand", "context", "expected_reason", "output_conflict"),
    [
        ([], [], False, False, "empty_selected", False),
        (["C01"], [], True, False, "need_expand_recall", False),
        (["C01"], [], False, True, "context_insufficient", False),
        (["C01"], ["C01"], False, False, "output_conflict", True),
    ],
)
def test_run_adjudication_filters_risky_training_rows(
    tmp_path: Path,
    selected,
    rejected,
    expand,
    context,
    expected_reason,
    output_conflict,
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
                "selected": selected,
                "rejected_risky": rejected,
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
    assert prediction["output_conflict"] is output_conflict
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
