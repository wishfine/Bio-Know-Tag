import json
from pathlib import Path

import pytest

from bio_know_tag.ds import DSResponse
from bio_know_tag.full_adjudication import run_full_adjudication


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    units, candidates, labels = (tmp_path / name for name in ("units.jsonl", "candidates.jsonl", "labels.jsonl"))
    _write(units, [
        {"question_id": "q1", "parent_id": "q1", "unit_type": "standalone", "stem": "酶", "options": "", "answer_text": "", "analysis": ""},
        {"question_id": "p2", "parent_id": "p2", "unit_type": "composite_parent_extra", "stem": "光合作用", "options": "", "answer_text": "", "analysis": ""},
    ])
    _write(candidates, [
        {"question_id": qid, "retrieval_version": "test+legacy", "candidates": [{"label_id": "L1", "candidate_rank": 1, "sources": ["legacy"]}]}
        for qid in ("q1", "p2")
    ])
    _write(labels, [{"label_id": "L1", "label_name": "酶", "label_path": "知识点@酶", "definition": "", "core_concepts": [], "distinctions": []}])
    return units, candidates, labels


class Client:
    def __init__(self):
        self.calls = 0
    def chat(self, messages, *, max_tokens, stream):
        self.calls += 1
        assert stream is True
        assert max_tokens == 64
        return DSResponse(
            content=json.dumps({
                "selected": ["C01"], "evidence": {"C01": "酶"},
                "context_insufficient": False, "need_expand_recall": False,
                "reason": "直接考查",
            }, ensure_ascii=False),
            endpoint="http://fake/v1/chat/completions", attempts=1,
            latency_seconds=0.01, finish_reason="stop",
        )


def test_full_input_streams_to_one_vote_dir_and_resumes(tmp_path: Path):
    units, candidates, labels = _inputs(tmp_path)
    client = Client()
    output = tmp_path / "vote"
    report = run_full_adjudication(units, candidates, labels, output, client, model="model-test", workers=2, max_tokens=64)
    assert report["input"] == report["success"] == 2
    assert report["error"] == 0
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert [row["question_id"] for row in predictions] == ["q1", "p2"]
    assert [row["unit_type"] for row in predictions] == ["standalone", "composite_parent_extra"]
    assert predictions[0]["selected_labels"][0]["label_id"] == "L1"
    assert predictions[1]["prompt_version"] != predictions[0]["prompt_version"]
    assert client.calls == 2
    second = run_full_adjudication(units, candidates, labels, output, client, model="model-test", workers=2, max_tokens=64)
    assert second["success"] == 2
    assert client.calls == 2


def test_full_input_rejects_question_id_mismatch_without_requests(tmp_path: Path):
    units, candidates, labels = _inputs(tmp_path)
    _write(candidates, [{"question_id": "wrong", "candidates": []}, {"question_id": "p2", "candidates": []}])
    client = Client()
    with pytest.raises(ValueError, match="question_id mismatch"):
        run_full_adjudication(units, candidates, labels, tmp_path / "vote", client, model="model-test", workers=2, max_tokens=64)
    assert client.calls == 0


def test_full_input_only_retries_failed_question_on_resume(tmp_path: Path):
    units, candidates, labels = _inputs(tmp_path)
    class FailingOnce(Client):
        def __init__(self):
            super().__init__()
            self.failed = False
        def chat(self, messages, *, max_tokens, stream):
            if "光合作用" in messages[-1]["content"] and not self.failed:
                self.failed = True
                self.calls += 1
                return DSResponse(
                    content='{"selected":', endpoint="http://fake/v1/chat/completions",
                    attempts=1, latency_seconds=0.01, finish_reason="length",
                )
            return super().chat(messages, max_tokens=max_tokens, stream=stream)
    client = FailingOnce()
    output = tmp_path / "vote"
    first = run_full_adjudication(units, candidates, labels, output, client, model="model-test", workers=2, max_tokens=64)
    assert first["success"] == 1
    assert first["error"] == 1
    assert client.calls == 2
    second = run_full_adjudication(units, candidates, labels, output, client, model="model-test", workers=2, max_tokens=64)
    assert second["success"] == 2
    assert second["error"] == 0
    assert client.calls == 3
    evidence = [json.loads(line) for line in (output / "evidence.jsonl").read_text().splitlines()]
    assert len(evidence) == 3
    assert any("finish_reason=length" in (row.get("error") or "") for row in evidence)


def test_full_input_refuses_request_configuration_change_on_resume(tmp_path: Path):
    units, candidates, labels = _inputs(tmp_path)
    client = Client()
    output = tmp_path / "vote"
    run_full_adjudication(
        units, candidates, labels, output, client,
        model="model-test", workers=2, max_tokens=64,
        request_config={"endpoint": "http://first", "retries": 5},
    )
    with pytest.raises(ValueError, match="manifest"):
        run_full_adjudication(
            units, candidates, labels, output, client,
            model="model-test", workers=2, max_tokens=64,
            request_config={"endpoint": "http://second", "retries": 5},
        )
    assert client.calls == 2
