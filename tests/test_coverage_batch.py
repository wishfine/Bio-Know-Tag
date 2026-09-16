import json
from pathlib import Path

import pytest

from bio_know_tag.coverage_batch import (
    build_batch_prompt,
    build_batches,
    normalize_batch_response,
    run_coverage_batches,
)


def _task(index: int, label_id: str = "L1", context: str = "题目") -> dict:
    return {
        "pair_id": f"q{index}::{label_id}",
        "question_id": f"q{index}",
        "label_id": label_id,
        "unit_type": "standalone",
        "parent_stem": "",
        "stem": context,
        "options": "A. 选项",
        "answer_text": "A",
        "analysis": "解析",
    }


def _label() -> dict:
    return {
        "label_id": "L1",
        "label_name": "DNA复制",
        "definition": "DNA以半保留方式复制。",
        "core_concepts": "模板链和新链。",
        "common_assessments": "考查复制次数。",
        "distinctions": "与转录区分。",
    }


def test_build_batch_prompt_reuses_one_label_card_and_requests_compact_results():
    prompt = build_batch_prompt([_task(1), _task(2)], _label())

    assert prompt.count("Label名称：DNA复制") == 1
    assert '"task_id": "q1::L1"' in prompt
    assert '"task_id": "q2::L1"' in prompt
    assert "results 数量必须等于输入题目数量 2" in prompt
    assert '"relevance_score":0.92' in prompt
    assert "reason" not in prompt


def test_build_batches_never_mixes_labels_and_respects_size():
    tasks = [_task(1), _task(2), _task(3, "L2"), _task(4, "L2")]

    batches = build_batches(tasks, max_batch_size=1, char_budget=10_000)

    assert len(batches) == 4
    assert all(len(batch) == 1 for batch in batches)
    assert all(len({task["label_id"] for task in batch}) == 1 for batch in batches)


def test_normalize_batch_response_corrects_match_threshold():
    tasks = [_task(1), _task(2)]
    value = {
        "results": [
            {"task_id": "q1::L1", "question_id": "q1", "match": False, "relevance_score": 0.92},
            {"task_id": "q2::L1", "question_id": "q2", "match": True, "relevance_score": 0.2},
        ]
    }

    results = normalize_batch_response(value, tasks)

    assert [result["match"] for result in results] == [True, False]
    assert [result["relevance_score"] for result in results] == [0.92, 0.2]


def test_normalize_batch_response_rejects_reordered_ids():
    with pytest.raises(ValueError, match="mismatch"):
        normalize_batch_response(
            {
                "results": [
                    {"task_id": "wrong", "question_id": "q1", "match": True, "relevance_score": 0.9}
                ]
            },
            [_task(1)],
        )


def test_run_coverage_batches_writes_resumable_compact_results(tmp_path: Path):
    tasks_path = tmp_path / "tasks.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    output = tmp_path / "run"
    tasks_path.write_text(json.dumps(_task(1), ensure_ascii=False) + "\n", encoding="utf-8")
    labels_path.write_text(json.dumps(_label(), ensure_ascii=False) + "\n", encoding="utf-8")

    class Response:
        content = json.dumps(
            {
                "results": [
                    {
                        "task_id": "q1::L1",
                        "question_id": "q1",
                        "match": True,
                        "relevance_score": 0.93,
                    }
                ]
            }
        )
        endpoint = "fake"
        attempts = 1
        latency_seconds = 0.1
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        retry_errors = ()

    class Client:
        calls = 0

        def chat(self, messages, *, max_tokens):
            self.calls += 1
            assert max_tokens == 7000
            return Response()

    client = Client()
    first = run_coverage_batches(tasks_path, labels_path, output, client, model="fake")
    second = run_coverage_batches(tasks_path, labels_path, output, client, model="fake")

    assert first["success"] == 1
    assert first["match_rate"] == 1.0
    assert second["success"] == 1
    assert client.calls == 1
    result = json.loads((output / "results.jsonl").read_text())
    assert result["relevance_score"] == 0.93
    assert "reason" not in result
