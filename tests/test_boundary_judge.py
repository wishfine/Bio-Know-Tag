import json
from pathlib import Path

import pytest

from bio_know_tag.boundary_judge import (
    build_boundary_prompt,
    build_boundary_sample,
    run_boundary_judge,
    validate_boundary_result,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _labels() -> list[dict]:
    return [
        {
            "label_id": "L1",
            "label_name": "DNA复制",
            "label_path": "知识点->遗传->DNA复制",
            "definition": "DNA以半保留方式复制。",
            "core_concepts": "模板链和新链。",
            "common_assessments": "考查复制次数和同位素分布。",
            "distinctions": "与转录区分。",
        },
        {
            "label_id": "L2",
            "label_name": "转录",
            "label_path": "知识点->遗传->转录",
            "definition": "以DNA为模板合成RNA。",
            "core_concepts": "RNA聚合酶。",
            "common_assessments": "考查碱基互补。",
            "distinctions": "与DNA复制区分。",
        },
    ]


def test_build_boundary_sample_uses_only_current_label_ids(tmp_path: Path):
    units = tmp_path / "units.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "sample"
    _write_jsonl(labels, _labels())
    _write_jsonl(
        units,
        [
            {"question_id": "q1", "stem": "DNA复制几次", "legacy_knw_ids": ["L1", "OLD"]},
            {"question_id": "q2", "stem": "转录生成RNA", "legacy_knw_ids": ["L2"]},
        ],
    )

    report = build_boundary_sample(
        units, labels, output, positive_per_label=2, negative_per_label=0
    )

    rows = [json.loads(line) for line in (output / "boundary_samples.jsonl").read_text().splitlines()]
    assert {(row["question_id"], row["label_id"]) for row in rows} == {("q1", "L1"), ("q2", "L2")}
    assert all(row["expected_relation"] == "legacy_positive" for row in rows)
    assert report["obsolete_legacy_assignments_ignored"] == 1


def test_boundary_prompt_contains_teacher_fields_and_no_legacy_ids():
    sample = {
        "question_id": "q1",
        "stem": "DNA复制几次？",
        "options": "",
        "answer_text": "3次",
        "analysis": "根据半保留复制计算。",
        "parent_stem": "",
    }
    prompt = build_boundary_prompt(sample, _labels()[0])

    assert "Label名称：DNA复制" in prompt
    assert "definition：DNA以半保留方式复制。" in prompt
    assert "core_concepts：模板链和新链。" in prompt
    assert "common_assessments：考查复制次数和同位素分布。" in prompt
    assert "distinctions：与转录区分。" in prompt
    assert "legacy_knw_ids" not in prompt
    assert "常见考查方式是示例而不是穷举清单" in prompt
    assert '"score":0.92' in prompt
    assert '"difference_type":"matched"' in prompt


def test_validate_boundary_result_derives_match_from_score():
    result = validate_boundary_result(
        {
            "score": 0.82,
            "difference_type": "matched",
            "reason": "题目直接考查DNA半保留复制。",
        }
    )

    assert result == {
        "score": 0.82,
        "match": True,
        "difference_type": "matched",
        "reason": "题目直接考查DNA半保留复制。",
    }


@pytest.mark.parametrize("score,expected", [(0.7, True), (0.69, False), (0.0, False), (1.0, True)])
def test_validate_boundary_result_uses_fixed_match_threshold(score, expected):
    result = validate_boundary_result(
        {"score": score, "difference_type": "boundary_ambiguous", "reason": "边界测试"}
    )
    assert result["match"] is expected


def test_validate_boundary_result_rejects_unknown_difference_type():
    with pytest.raises(ValueError, match="difference_type"):
        validate_boundary_result(
            {"score": 0.5, "difference_type": "unknown", "reason": "理由"}
        )


def test_run_boundary_judge_resumes_successful_pairs(tmp_path: Path):
    samples = tmp_path / "samples.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "judge"
    _write_jsonl(labels, _labels())
    _write_jsonl(
        samples,
        [
            {
                "pair_id": "q1::L1",
                "question_id": "q1",
                "label_id": "L1",
                "expected_relation": "legacy_positive",
                "stem": "DNA复制几次？",
                "options": "",
                "answer_text": "3次",
                "analysis": "根据半保留复制计算。",
                "parent_stem": "",
            }
        ],
    )

    class Response:
        content = json.dumps(
            {
                "score": 0.91,
                "difference_type": "matched",
                "reason": "题目直接考查DNA复制。",
            },
            ensure_ascii=False,
        )
        endpoint = "fake"
        attempts = 1
        latency_seconds = 0.1
        usage = {"prompt_tokens": 10, "completion_tokens": 4}
        reasoning = None
        response_message_keys = ("role", "content")
        retry_errors = ()

    class Client:
        calls = 0

        def chat(self, messages, *, max_tokens):
            self.calls += 1
            assert max_tokens == 256
            return Response()

    client = Client()
    first = run_boundary_judge(samples, labels, output, client, model="fake", max_tokens=256)
    second = run_boundary_judge(samples, labels, output, client, model="fake", max_tokens=256)

    assert first["success"] == first["input"] == 1
    assert first["legacy_positive_match_rate"] == 1.0
    assert first["score_distribution"][">=0.90"] == 1
    assert second["success"] == 1
    assert client.calls == 1
    prediction = json.loads((output / "predictions.jsonl").read_text())
    assert prediction["match"] is True
    assert prediction["score"] == 0.91
    assert prediction["difference_type"] == "matched"
