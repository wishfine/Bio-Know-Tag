import json
from pathlib import Path
from types import SimpleNamespace

from bio_know_tag.adjudication import PARENT_PROMPT_VERSION, PROMPT_VERSION, run_adjudication


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_mixed_child_and_parent_use_their_own_prompts_and_resume(tmp_path: Path):
    units = tmp_path / "units.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    labels = tmp_path / "labels.jsonl"
    run_dir = tmp_path / "run"
    _write(units, [
        {
            "question_id": "child", "parent_id": "parent", "unit_type": "sub_question",
            "parent_stem": "实验背景", "stem": "子题考查渗透作用", "options": "",
            "answer_text": "渗透作用", "analysis": "子题考查渗透作用", "flags": {},
        },
        {
            "question_id": "parent", "parent_id": "parent", "unit_type": "composite_parent_extra",
            "parent_stem": "", "stem": "实验背景", "options": "",
            "answer_text": "", "analysis": "", "flags": {},
        },
    ])
    _write(candidates, [
        {"question_id": question_id, "retrieval_version": "hybrid-test", "candidates": [
            {"label_id": "L1", "label_name": "渗透作用", "label_path": "知识点@渗透作用", "rank": 1}
        ]}
        for question_id in ("child", "parent")
    ])
    _write(labels, [{
        "label_id": "L1", "label_name": "渗透作用", "label_path": "知识点@渗透作用",
        "definition": "水分子跨膜移动", "core_concepts": "渗透", "distinctions": "区别于扩散",
    }])

    class Client:
        def __init__(self):
            self.calls = []

        def chat(self, messages, *, max_tokens):
            prompt = messages[1]["content"]
            self.calls.append(prompt)
            if "父题材料自身额外考查" in prompt:
                result = {"reason": "公共材料只有背景。", "selected": [], "evidence": {},
                          "context_insufficient": False, "need_expand_recall": False}
            else:
                result = {"reason": "当前小题直接考查渗透。", "selected": ["C01"],
                          "evidence": {"C01": "子题考查渗透作用"},
                          "context_insufficient": False, "need_expand_recall": False}
            return SimpleNamespace(
                content=json.dumps(result, ensure_ascii=False), endpoint="fake", attempts=1,
                latency_seconds=0.01, usage=None, reasoning=None,
                response_message_keys=(), retry_errors=(),
            )

    client = Client()
    report = run_adjudication(units, candidates, labels, run_dir, client, model="test")
    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    by_id = {row["question_id"]: row for row in rows}
    assert len(client.calls) == 2
    assert report["prompt_version"] == "candidate-adjudication-mixed-v1"
    assert by_id["child"]["prompt_version"] == PROMPT_VERSION
    assert by_id["parent"]["prompt_version"] == PARENT_PROMPT_VERSION
    assert [item["label_id"] for item in by_id["child"]["selected_labels"]] == ["L1"]
    assert by_id["parent"]["selected_labels"] == []
    assert by_id["parent"]["needs_review"] is False

    resumed = run_adjudication(units, candidates, labels, run_dir, client, model="test")
    assert resumed["success"] == 2
    assert len(client.calls) == 2
