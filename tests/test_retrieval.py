import json
from pathlib import Path

import pytest

from bio_know_tag.retrieval import (
    BM25Retriever,
    build_coarse_recall_prompt,
    build_label_cards,
    build_query_tokens,
    format_label_path,
    reciprocal_rank_fusion,
    run_ds_coarse_recall,
    run_sparse_retrieval,
    validate_coarse_recall_result,
)


def _labels():
    return [
        {
            "label_id": "L1",
            "label_name": "基因表达载体的构建",
            "label_path": "知识点->生物技术与工程->基因工程->基因表达载体的构建",
            "definition": "将目的基因与启动子、终止子和标记基因组成表达载体。",
            "core_concepts": "启动子控制转录起始。",
            "common_assessments": "判断表达载体的结构。",
            "distinctions": "区别于将目的基因导入受体细胞。",
        },
        {
            "label_id": "L2",
            "label_name": "胚胎移植",
            "label_path": "知识点->生物技术与工程->胚胎工程->胚胎移植",
            "definition": "将早期胚胎移入生理状态相同的受体。",
            "core_concepts": "同期发情、供体、受体。",
            "common_assessments": "判断操作步骤。",
            "distinctions": "区别于胚胎分割。",
        },
    ]


def _unit(question_id: str = "q1"):
    return {
        "question_id": question_id,
        "parent_id": question_id,
        "unit_type": "standalone",
        "parent_stem": "现代生物科技专题",
        "stem": "构建载体时应将目的基因和什么调控元件连接？",
        "options": "",
        "answer_text": "启动子",
        "analysis": "启动子使目的基因在乳腺中特异表达。",
        "flags": {},
        "metadata": {},
    }


def test_format_label_path_only_replaces_ascii_hierarchy_separator():
    assert format_label_path("知识点->遗传与进化->DNA复制") == "知识点@遗传与进化@DNA复制"
    assert format_label_path("DNA→RNA→蛋白质") == "DNA→RNA→蛋白质"


def test_sparse_bm25_retrieves_relevant_label_without_legacy_ids():
    cards = build_label_cards(_labels())
    retriever = BM25Retriever(cards)

    results = retriever.search(build_query_tokens(_unit()), top_k=2)

    assert results[0]["label_id"] == "L1"
    assert results[0]["label_path"] == "知识点@生物技术与工程@基因工程@基因表达载体的构建"
    assert results[0]["score"] > results[1]["score"]


def test_coarse_prompt_uses_at_paths_and_not_teacher_definitions():
    cards = build_label_cards(_labels())
    prompt = build_coarse_recall_prompt([_unit()], cards, top_k=20)

    assert "知识点@生物技术与工程@基因工程@基因表达载体的构建" in prompt
    assert "启动子控制转录起始" not in prompt
    assert '"question_id": "q1"' in prompt
    assert '"code": "B001"' in prompt
    assert '"label_id"' not in prompt


def test_validate_coarse_result_requires_exact_questions_and_known_labels():
    value = {"results": [{"question_id": "q1", "candidate_codes": ["B001", "B002"]}]}
    assert validate_coarse_recall_result(value, ["q1"], {"B001", "B002"}, top_k=20) == value

    with pytest.raises(ValueError, match="unknown candidate code"):
        validate_coarse_recall_result(
            {"results": [{"question_id": "q1", "candidate_codes": ["missing"]}]},
            ["q1"],
            {"B001", "B002"},
            top_k=20,
        )


def test_validate_coarse_result_stably_deduplicates_candidates():
    value = {
        "results": [
            {
                "question_id": "q1",
                "candidate_codes": ["B001", "B002", "B001", "B002"],
            }
        ]
    }

    validated = validate_coarse_recall_result(
        value, ["q1"], {"B001", "B002"}, top_k=20
    )

    assert validated["results"][0]["candidate_codes"] == ["B001", "B002"]


def test_validate_coarse_result_truncates_ranked_codes_to_top_k():
    value = {
        "results": [
            {
                "question_id": "q1",
                "candidate_codes": ["B001", "B002", "B003"],
            }
        ]
    }

    validated = validate_coarse_recall_result(
        value, ["q1"], {"B001", "B002", "B003"}, top_k=2
    )

    assert validated["results"][0]["candidate_codes"] == ["B001", "B002"]
    assert validated["normalization"]["candidate_codes_truncated"] == 1


def test_rrf_fuses_rankings_without_requiring_all_methods():
    fused = reciprocal_rank_fusion(
        [["L1", "L2"], ["L2", "L1"], ["L2"]],
        top_k=2,
        rank_constant=60,
    )
    assert [item["label_id"] for item in fused] == ["L2", "L1"]
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_run_sparse_retrieval_writes_candidate_sidecar_and_report(tmp_path: Path):
    units_path = tmp_path / "pilot.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    output = tmp_path / "sparse"
    units_path.write_text(json.dumps(_unit(), ensure_ascii=False) + "\n", encoding="utf-8")
    labels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _labels()),
        encoding="utf-8",
    )

    report = run_sparse_retrieval(units_path, labels_path, output, top_k=2)

    candidate = json.loads((output / "candidates.jsonl").read_text(encoding="utf-8"))
    assert candidate["question_id"] == "q1"
    assert candidate["method"] == "char_ngram_bm25"
    assert candidate["candidates"][0]["label_id"] == "L1"
    assert candidate["candidates"][0]["label_path"].count("->") == 0
    assert report["processed"] == report["input"] == 1
    assert report["error"] == 0
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report


def test_run_ds_coarse_recall_batches_and_writes_at_paths(tmp_path: Path):
    units_path = tmp_path / "pilot.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    output = tmp_path / "coarse"
    units = [_unit("q1"), _unit("q2")]
    units_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in units),
        encoding="utf-8",
    )
    labels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _labels()),
        encoding="utf-8",
    )

    class Response:
        content = json.dumps(
            {
                "results": [
                    {"question_id": "q1", "candidate_codes": ["B001", "B002"]},
                    {"question_id": "q2", "candidate_codes": ["B002"]},
                ]
            },
            ensure_ascii=False,
        )
        endpoint = "fake"
        attempts = 1
        latency_seconds = 0.01

    class Client:
        def chat(self, messages, *, max_tokens):
            assert "老师原释义" not in messages[1]["content"]
            assert max_tokens == 512
            return Response()

    report = run_ds_coarse_recall(
        units_path,
        labels_path,
        output,
        Client(),
        model="fake-model",
        top_k=20,
        batch_size=2,
        max_tokens=512,
    )

    candidates = [
        json.loads(line)
        for line in (output / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert report["success"] == 2
    assert report["error"] == 0
    assert report["requests_succeeded"] == 1
    assert [row["question_id"] for row in candidates] == ["q1", "q2"]
    assert candidates[0]["candidates"][0]["label_path"] == (
        "知识点@生物技术与工程@基因工程@基因表达载体的构建"
    )
    assert candidates[0]["method"] == "ds_all_label_paths"
    evidence = [
        json.loads(line)
        for line in (output / "evidence.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(evidence) == 1
    assert evidence[0]["prompt_version"] == "coarse-all-paths-v2"
