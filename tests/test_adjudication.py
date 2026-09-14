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

    assert code_map == {"C01": "L1", "C02": "L2"}
    assert "C01" in prompt
    assert "标签一定义" in prompt
    assert "知识点@模块@标签一" in prompt
    assert "旧knw_ids" in prompt
    assert "candidate_rank" not in prompt


def test_validate_adjudication_requires_evidence_and_consistent_empty_state():
    valid = {
        "selected": [
            {
                "code": "C01",
                "question_evidence": "解析",
                "necessity": "覆盖当前设问",
            }
        ],
        "rejected_close_codes": ["C02"],
        "none_of_candidates": False,
        "need_expand_recall": False,
        "reason": "C01是完成设问所需的最小知识点。",
    }
    assert validate_adjudication_result(
        valid, {"C01", "C02"}, question_evidence_text="题干 答案 解析"
    ) == valid

    with pytest.raises(ValueError, match="question_evidence"):
        validate_adjudication_result(
            {
                **valid,
                "selected": [
                    {
                        "code": "C01",
                        "question_evidence": "",
                        "necessity": "覆盖当前设问",
                    }
                ],
            },
            {"C01", "C02"},
        )
    with pytest.raises(ValueError, match="not quoted"):
        validate_adjudication_result(
            valid,
            {"C01", "C02"},
            question_evidence_text="题干 答案",
        )
    with pytest.raises(ValueError, match="none_of_candidates"):
        validate_adjudication_result(
            {**valid, "selected": [], "none_of_candidates": False},
            {"C01", "C02"},
        )


def test_run_adjudication_records_tail_candidate_usage(tmp_path: Path):
    units_path = tmp_path / "units.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    output = tmp_path / "judge"
    units_path.write_text(json.dumps(_unit(), ensure_ascii=False) + "\n", encoding="utf-8")
    candidates = [_candidate(index) for index in range(1, 26)]
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
                        "code": "C23",
                        "question_evidence": "解析",
                        "necessity": "该知识直接覆盖设问",
                    }
                ],
                "rejected_close_codes": ["C01"],
                "none_of_candidates": False,
                "need_expand_recall": False,
                "reason": "标签23最符合设问。",
            },
            ensure_ascii=False,
        )
        endpoint = "fake"
        attempts = 1
        latency_seconds = 0.1

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
                        "question_evidence": "题干",
                        "necessity": "覆盖当前设问",
                    }
                ],
                "rejected_close_codes": [],
                "none_of_candidates": False,
                "need_expand_recall": False,
                "reason": "最小充分集合。",
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
