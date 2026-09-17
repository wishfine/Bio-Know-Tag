import json
from pathlib import Path

import pytest

from bio_know_tag.co_label_judge import (
    DECISIONS,
    build_batch_prompt,
    build_batches,
    build_judge_tasks,
    combine_corrected_boundary_assessments,
    normalize_batch_response,
    run_co_label_judge,
)
from bio_know_tag.ds import DSRequestError


def _label(label_id: str, name: str) -> dict:
    return {
        "label_id": label_id,
        "label_name": name,
        "definition": f"{name}定义",
        "core_concepts": f"{name}核心",
        "common_assessments": f"{name}考法",
        "distinctions": f"{name}边界",
    }


def _sample(index: int = 1) -> dict:
    return {
        "pair_id": f"q{index}::T",
        "question_id": f"q{index}",
        "label_id": "T",
        "source_label_ids": ["S"],
        "source_label_names": ["源Label"],
        "source_positive_score": 0.95,
        "unit_type": "standalone",
        "parent_stem": "",
        "stem": f"题目{index}",
        "options": "A. 选项",
        "answer_text": "A",
        "analysis": "解析",
    }


def test_build_judge_tasks_only_keeps_first_stage_acceptances():
    samples = {_sample(1)["pair_id"]: _sample(1), _sample(2)["pair_id"]: _sample(2)}
    results = {
        "q1::T": {"task_id": "q1::T", "relevance_score": 0.95},
        "q2::T": {"task_id": "q2::T", "relevance_score": 0.2},
    }

    tasks = build_judge_tasks(samples, results)

    assert [task["pair_id"] for task in tasks] == ["q1::T"]
    assert tasks[0]["first_stage_score"] == 0.95


def test_prompt_defines_minimal_sufficient_colabel_decisions():
    labels = {"S": _label("S", "源Label"), "T": _label("T", "目标Label")}
    prompt = build_batch_prompt([_sample()], labels)

    assert "最小充分知识点集合" in prompt
    assert "上位Label不得仅因包含当前知识就共标" in prompt
    assert "综合Label必须真正要求多个子模块联动" in prompt
    assert all(decision in prompt for decision in DECISIONS)
    assert "first_stage_score" not in prompt


def test_normalize_rejects_unknown_decision():
    with pytest.raises(ValueError, match="unknown decision"):
        normalize_batch_response(
            {
                "results": [
                    {
                        "task_id": "q1::T",
                        "question_id": "q1",
                        "decision": "可能合理",
                    }
                ]
            },
            [_sample()],
        )


def test_build_batches_respects_complete_prompt_budget():
    labels = {"S": _label("S", "源Label"), "T": _label("T", "目标Label")}
    tasks = [_sample(1), _sample(2)]
    tasks[0]["analysis"] = "长" * 2000
    tasks[1]["analysis"] = "长" * 2000
    one_prompt_size = len(build_batch_prompt([tasks[0]], labels))

    batches = build_batches(
        tasks,
        labels=labels,
        max_batch_size=20,
        char_budget=one_prompt_size + 500,
    )

    assert [len(batch) for batch in batches] == [1, 1]
    assert all(
        len(build_batch_prompt(batch, labels)) <= one_prompt_size + 500
        for batch in batches
    )


def test_run_is_resumable_and_adaptively_splits_http_400(tmp_path: Path):
    samples_path = tmp_path / "samples.jsonl"
    first_stage_path = tmp_path / "first.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    run_dir = tmp_path / "run"
    samples_path.write_text(
        "".join(json.dumps(_sample(i), ensure_ascii=False) + "\n" for i in (1, 2)),
        encoding="utf-8",
    )
    first_stage_path.write_text(
        "".join(
            json.dumps(
                {"task_id": f"q{i}::T", "question_id": f"q{i}", "relevance_score": 0.95}
            )
            + "\n"
            for i in (1, 2)
        ),
        encoding="utf-8",
    )
    labels_path.write_text(
        json.dumps(_label("S", "源Label"), ensure_ascii=False)
        + "\n"
        + json.dumps(_label("T", "目标Label"), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    class Response:
        endpoint = "fake"
        attempts = 1
        latency_seconds = 0.1
        usage = None
        retry_errors = ()

        def __init__(self, question_id: str):
            self.content = json.dumps(
                {
                    "results": [
                        {
                            "task_id": f"{question_id}::T",
                            "question_id": question_id,
                            "decision": "合理共标",
                        }
                    ]
                },
                ensure_ascii=False,
            )

    class Client:
        calls = 0

        def chat(self, messages, *, max_tokens):
            self.calls += 1
            prompt = messages[0]["content"]
            if '"task_id": "q1::T"' in prompt and '"task_id": "q2::T"' in prompt:
                raise DSRequestError(
                    "HTTP Error 400: Bad Request",
                    attempts=3,
                    endpoint="fake",
                    latency_seconds=0.1,
                    retry_errors=[],
                )
            return Response("q1" if '"task_id": "q1::T"' in prompt else "q2")

    client = Client()
    first = run_co_label_judge(
        samples_path,
        first_stage_path,
        labels_path,
        run_dir,
        client,
        model="fake",
        workers=1,
        max_batch_size=2,
    )
    second = run_co_label_judge(
        samples_path,
        first_stage_path,
        labels_path,
        run_dir,
        client,
        model="fake",
        workers=1,
        max_batch_size=2,
    )

    assert first["input"] == first["success"] == 2
    assert first["decision_counts"] == {"合理共标": 2}
    assert first["true_boundary_error_rate"] == 0.0
    assert second["pending"] == 0
    assert client.calls == 3


def test_corrected_boundary_rate_removes_invalid_sibling_negatives(tmp_path: Path):
    positive = tmp_path / "positive.jsonl"
    negative = tmp_path / "negative.jsonl"
    colabel = tmp_path / "colabel.jsonl"
    output = tmp_path / "combined"
    positive.write_text(
        json.dumps(
            {
                "label_id": "T",
                "label_name": "目标",
                "planned": 500,
                "match_rate": 0.8,
                "sample_tier": "CAPPED_500",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    negative.write_text(
        json.dumps(
            {
                "label_id": "T",
                "hard_negative_total": 50,
                "false_accept": 20,
                "false_accept_rate": 0.4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    colabel.write_text(
        json.dumps(
            {
                "target_label_id": "T",
                "total": 20,
                "decision_counts": {
                    "合理共标": 14,
                    "目标Label边界过宽": 3,
                    "来源Label不足以描述该题": 1,
                    "无法判断": 2,
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = combine_corrected_boundary_assessments(
        positive, negative, colabel, output
    )

    row = json.loads((output / "label_assessments.jsonl").read_text())
    assert row["first_stage_rejected"] == 30
    assert row["invalid_sibling_negatives"] == 15
    assert row["valid_negative_count"] == 33
    assert row["corrected_boundary_error_rate"] == pytest.approx(3 / 33, abs=1e-6)
    assert row["final_screen"] == "A_STABLE_CANDIDATE"
    assert report["labels"] == 1
