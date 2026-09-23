import json
from pathlib import Path

import pytest

from bio_know_tag.retrieval import (
    BM25Retriever,
    build_coarse_recall_prompt,
    build_label_cards,
    build_query_tokens,
    build_dense_label_text,
    build_dense_query_text,
    compare_candidate_runs,
    quota_fuse_candidates,
    format_label_path,
    reciprocal_rank_fusion,
    run_candidate_reranking,
    run_ds_coarse_recall,
    run_dense_retrieval,
    run_hybrid_retrieval,
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


def test_dense_texts_use_teacher_fields_and_question_context():
    card = build_label_cards(_labels())[0]
    label_text = build_dense_label_text(card)
    query_text = build_dense_query_text(_unit())

    assert "基因表达载体的构建" in label_text
    assert "启动子控制转录起始" in label_text
    assert "知识点@生物技术与工程" in label_text
    assert "构建载体时" in query_text
    assert "启动子使目的基因" in query_text
    assert "现代生物科技专题" in query_text


def test_dense_query_keeps_the_end_of_a_long_question():
    unit = {
        **_unit(),
        "stem": "背景" * 500 + "请绘制实验组的柱形图",
        "analysis": "分析" * 500 + "最终需要比较数据变化趋势",
    }

    query_text = build_dense_query_text(unit)

    assert "请绘制实验组的柱形图" in query_text
    assert "最终需要比较数据变化趋势" in query_text


def test_run_dense_retrieval_uses_configurable_encoder_and_writes_sidecar(tmp_path: Path):
    units_path = tmp_path / "pilot.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    output = tmp_path / "dense"
    units_path.write_text(json.dumps(_unit(), ensure_ascii=False) + "\n", encoding="utf-8")
    labels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _labels()),
        encoding="utf-8",
    )

    class Encoder:
        model_name = "fake-embedding"

        def index(self, label_texts):
            assert len(label_texts) == 2

        def search(self, query_texts, *, top_k):
            assert len(query_texts) == 1
            assert top_k == 2
            return [[("L1", 0.91), ("L2", 0.42)]]

    report = run_dense_retrieval(
        units_path,
        labels_path,
        output,
        Encoder(),
        top_k=2,
        batch_size=8,
    )

    row = json.loads((output / "candidates.jsonl").read_text(encoding="utf-8"))
    assert row["method"] == "dense_embedding"
    assert row["candidates"][0]["label_id"] == "L1"
    assert row["candidates"][0]["label_path"].count("->") == 0
    assert report["input"] == report["processed"] == 1
    assert report["error"] == 0
    assert report["model"] == "fake-embedding"
    assert report["retrieval_version"] == "dense-v2-head-tail"
    assert len(report["input_sha256"]["units"]) == 64


def test_run_dense_retrieval_keeps_failed_batch_out_of_candidates(tmp_path: Path):
    units_path = tmp_path / "pilot.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    output = tmp_path / "dense"
    units_path.write_text(
        "".join(json.dumps(_unit(qid), ensure_ascii=False) + "\n" for qid in ("q1", "q2")),
        encoding="utf-8",
    )
    labels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _labels()),
        encoding="utf-8",
    )

    class Encoder:
        model_name = "broken"

        def index(self, label_texts):
            return None

        def search(self, query_texts, *, top_k):
            return [[("L1", 0.9)], [("unknown", 0.8)]]

    report = run_dense_retrieval(
        units_path, labels_path, output, Encoder(), top_k=2, batch_size=2
    )

    assert report["processed"] == 0
    assert report["error"] == 2
    assert (output / "candidates.jsonl").read_text(encoding="utf-8") == ""
    error = json.loads((output / "errors.jsonl").read_text(encoding="utf-8"))
    assert error["question_ids"] == ["q1", "q2"]
    assert "unknown label_id" in error["error"]


def test_compare_candidate_runs_outputs_overlap_and_disagreement_samples(tmp_path: Path):
    units_path = tmp_path / "units.jsonl"
    sparse_path = tmp_path / "sparse.jsonl"
    dense_path = tmp_path / "dense.jsonl"
    output = tmp_path / "compare"
    units = [_unit("q1"), _unit("q2")]
    units_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in units),
        encoding="utf-8",
    )
    sparse = [
        {"question_id": "q1", "candidates": [{"label_id": "L1"}, {"label_id": "L2"}]},
        {"question_id": "q2", "candidates": [{"label_id": "L1"}, {"label_id": "L2"}]},
    ]
    dense = [
        {"question_id": "q1", "candidates": [{"label_id": "L1"}, {"label_id": "L2"}]},
        {"question_id": "q2", "candidates": [{"label_id": "L2"}, {"label_id": "L3"}]},
    ]
    sparse_path.write_text(
        "".join(json.dumps(row) + "\n" for row in sparse), encoding="utf-8"
    )
    dense_path.write_text(
        "".join(json.dumps(row) + "\n" for row in dense), encoding="utf-8"
    )

    report = compare_candidate_runs(
        units_path, sparse_path, dense_path, output, top_k=2, sample_size=1
    )

    assert report["questions"] == 2
    assert report["top1_agreement"] == 1
    assert report["top1_agreement_rate"] == 0.5
    assert report["mean_overlap_at_k"] == 1.5
    samples = [
        json.loads(line)
        for line in (output / "disagreement_samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(samples) == 1
    assert samples[0]["question_id"] == "q2"


def test_quota_fusion_prioritizes_sparse_and_adds_dense_only_candidates():
    sparse = [
        {"label_id": "L1", "rank": 1, "score": 10.0},
        {"label_id": "L2", "rank": 2, "score": 9.0},
        {"label_id": "L3", "rank": 3, "score": 8.0},
    ]
    dense = [
        {"label_id": "L2", "rank": 1, "score": 0.9},
        {"label_id": "L4", "rank": 2, "score": 0.8},
        {"label_id": "L5", "rank": 3, "score": 0.7},
    ]

    fused = quota_fuse_candidates(
        sparse, dense, sparse_quota=2, dense_quota=2, top_k=4
    )

    assert [item["label_id"] for item in fused] == ["L1", "L2", "L4", "L5"]
    assert fused[1]["sources"] == ["sparse", "dense"]
    assert fused[2]["sources"] == ["dense"]
    assert fused[3]["sources"] == ["dense"]


def test_run_hybrid_retrieval_writes_25_candidate_sidecar(tmp_path: Path):
    sparse_path = tmp_path / "sparse.jsonl"
    dense_path = tmp_path / "dense.jsonl"
    output = tmp_path / "hybrid"
    sparse = {
        "question_id": "q1",
        "candidates": [
            {"label_id": f"S{i}", "label_name": f"稀疏{i}", "label_path": f"知识点@稀疏{i}", "rank": i, "score": 100-i}
            for i in range(1, 21)
        ],
    }
    dense = {
        "question_id": "q1",
        "candidates": [
            {"label_id": f"D{i}", "label_name": f"稠密{i}", "label_path": f"知识点@稠密{i}", "rank": i, "score": 1-i/100}
            for i in range(1, 21)
        ],
    }
    sparse_path.write_text(json.dumps(sparse, ensure_ascii=False) + "\n", encoding="utf-8")
    dense_path.write_text(json.dumps(dense, ensure_ascii=False) + "\n", encoding="utf-8")

    report = run_hybrid_retrieval(
        sparse_path,
        dense_path,
        output,
        top_k=25,
        sparse_quota=18,
        dense_quota=7,
    )

    row = json.loads((output / "candidates.jsonl").read_text(encoding="utf-8"))
    assert report["processed"] == 1
    assert report["candidate_count_distribution"] == {"25": 1}
    assert len(row["candidates"]) == 25
    assert row["candidates"][0]["label_id"] == "S1"
    assert row["candidates"][18]["label_id"] == "D1"
    assert row["retrieval_version"] == "hybrid-v1-s18-d7-k25"


def test_run_hybrid_retrieval_rejects_row_order_mismatch(tmp_path: Path):
    sparse_path = tmp_path / "sparse.jsonl"
    dense_path = tmp_path / "dense.jsonl"
    sparse_path.write_text(
        ''.join(json.dumps({"question_id": question_id, "candidates": []}) + "\n" for question_id in ("q1", "q2")),
        encoding="utf-8",
    )
    dense_path.write_text(
        ''.join(json.dumps({"question_id": question_id, "candidates": []}) + "\n" for question_id in ("q2", "q1")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="question IDs differ at row 1"):
        run_hybrid_retrieval(sparse_path, dense_path, tmp_path / "hybrid")
    assert not (tmp_path / "hybrid" / "candidates.jsonl").exists()


def test_candidate_reranking_unions_sparse_and_dense_before_top_k(tmp_path: Path):
    units_path = tmp_path / "units.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    sparse_path = tmp_path / "sparse.jsonl"
    dense_path = tmp_path / "dense.jsonl"
    output = tmp_path / "reranked"
    units_path.write_text(json.dumps(_unit(), ensure_ascii=False) + "\n", encoding="utf-8")
    labels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _labels()),
        encoding="utf-8",
    )
    sparse_path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "candidates": [
                    {"label_id": "L1", "rank": 1, "score": 10.0},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dense_path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "candidates": [
                    {"label_id": "L2", "rank": 1, "score": 0.9},
                    {"label_id": "L1", "rank": 2, "score": 0.8},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class Reranker:
        model_name = "fake-reranker"

        def score(self, pairs):
            assert len(pairs) == 2
            return [0.2 if "基因表达载体" in label else 0.9 for _, label in pairs]

    report = run_candidate_reranking(
        units_path,
        labels_path,
        sparse_path,
        dense_path,
        output,
        Reranker(),
        sparse_pool=30,
        dense_pool=30,
        top_k=2,
    )

    row = json.loads((output / "candidates.jsonl").read_text(encoding="utf-8"))
    assert [item["label_id"] for item in row["candidates"]] == ["L2", "L1"]
    assert row["candidates"][0]["sources"] == ["dense"]
    assert row["candidates"][1]["sources"] == ["sparse", "dense"]
    assert row["retrieval_version"] == "rerank-v1-s30-d30-k2"
    assert report["candidate_count_distribution"] == {"2": 1}
    assert report["model"] == "fake-reranker"
    assert report["upstream_retrieval_versions"] == {
        "dense": [""],
        "sparse": [""],
    }
    assert len(report["input_sha256"]["units"]) == 64
    assert len(report["input_sha256"]["labels"]) == 64
