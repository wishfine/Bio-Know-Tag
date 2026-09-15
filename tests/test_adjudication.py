import json
import threading
import time
from pathlib import Path

import pytest

from bio_know_tag.adjudication import (
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
    assert "错选一个Label比漏选更严重" in prompt
    assert "允许少选，也允许selected为空" in prompt
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
    assert list(q1_first.values()) != ["L1", "L2", "L3"]
    assert list(q1_first.values()) != list(q2.values())


def test_validate_adjudication_requires_evidence_and_consistent_empty_state():
    valid = {
        "selected": [
            {
                "code": "C01",
                "evidence": "解析",
            }
        ],
        "need_expand_recall": False,
        "context_insufficient": False,
    }
    validated = validate_adjudication_result(
        valid, {"C01", "C02"}, question_evidence_text="题干 答案 解析"
    )
    assert validated["selected"][0]["question_evidence_verified"] is True

    with pytest.raises(ValueError, match="evidence"):
        validate_adjudication_result(
            {
                **valid,
                "selected": [
                    {
                        "code": "C01",
                        "evidence": "",
                    }
                ],
            },
            {"C01", "C02"},
        )
    unverified = validate_adjudication_result(
        {
            **valid,
            "selected": [
                {
                    "code": "C01",
                    "evidence": "补写的上下文；题干",
                }
            ],
        },
        {"C01", "C02"},
        question_evidence_text="题干 答案",
    )
    assert unverified["selected"][0]["question_evidence_verified"] is False
    empty = validate_adjudication_result(
        {**valid, "selected": []},
        {"C01", "C02"},
    )
    assert empty["none_of_candidates"] is True


@pytest.mark.parametrize(
    ("selected", "expand", "context", "valid"),
    [
        (
            [{"code": "C01", "evidence": "题干"}],
            False,
            False,
            True,
        ),
        (
            [{"code": "C01", "evidence": "题干"}],
            True,
            False,
            True,
        ),
        ([], True, False, True),
        ([], False, True, True),
        ([], False, False, True),
        ([{"code": "C01", "evidence": "题干"}], False, False, True),
        ("not-a-list", False, False, False),
    ],
)
def test_validate_adjudication_candidate_and_context_states(
    selected, expand, context, valid
):
    value = {
        "selected": selected,
        "need_expand_recall": expand,
        "context_insufficient": context,
    }
    if valid:
        result = validate_adjudication_result(
            value, {"C01"}, question_evidence_text="题干"
        )
        assert result["context_insufficient"] is context
    else:
        with pytest.raises(ValueError):
            validate_adjudication_result(
                value, {"C01"}, question_evidence_text="题干"
            )


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
                "selected": [
                    {
                        "code": code_for_l23,
                        "evidence": "根据解析可知",
                    }
                ],
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
    assert prediction["selected_labels"][0]["label_path"] == "知识点@模块@标签23"
    assert prediction["candidate_count"] == 25
    assert prediction["retrieval_version"] == "hybrid-v1-s18-d7-k25"
    assert report["success"] == 1
    assert report["selected_candidate_rank_distribution"] == {"23": 1}
    assert report["max_selected_candidate_rank"] == 23
    assert report["selected_from_rank_21_25"] == 1
    assert report["questions_using_rank_21_25"] == 1
    assert report["need_expand_recall"] == 0
    assert report["context_insufficient"] == 0
    assert report["usable_for_training"] == 1
    assert report["filtered_from_training"] == 0
    assert report["token_usage"]["requests_with_usage"] == 1
    assert report["token_usage"]["mean_prompt_tokens"] == 100.0
    assert report["token_usage"]["mean_completion_tokens"] == 20.0
    assert report["requests_retried"] == 1
    assert report["retry_error_types"] == {"ConnectionResetError": 1}
    assert report["unverified_evidence_items"] == 1
    assert report["questions_with_unverified_evidence"] == 1
    assert prediction["selected_labels"][0]["evidence_verified"] is False
    assert "necessity" not in prediction["selected_labels"][0]
    assert prediction["usable_for_training"] is True
    assert "missing_knowledge" not in prediction
    # Evidence substring matching is diagnostic only. A semantically useful
    # paraphrase must not create a manual-review task by itself.
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
                "selected": [
                    {
                        "code": "C01",
                        "evidence": "题干",
                    }
                ],
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
                "selected": [{"code": "C01", "evidence": "题干"}],
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
        ([{"code": "C01", "evidence": "题干"}], True, False, "need_expand_recall"),
        ([{"code": "C01", "evidence": "题干"}], False, True, "context_insufficient"),
    ],
)
def test_run_adjudication_filters_risky_training_rows(
    tmp_path: Path, selected, expand, context, expected_reason
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
