import json

from bio_know_tag.adjudication import build_adjudication_inputs
from bio_know_tag.reranker import (
    RERANKER_INSTRUCTION,
    TransformerCrossEncoderReranker,
    build_qwen3_reranker_token_ids,
    build_reranker_pairs,
    extract_binary_logprobs,
    normalize_chat_template_token_ids,
)


class _FakeTokenizer:
    def __init__(self):
        self.prepared_pairs = []

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text)

    def num_special_tokens_to_add(self, *, pair):
        assert pair is True
        return 3

    def prepare_for_model(
        self, tokens, *, pair_ids, add_special_tokens, truncation
    ):
        assert add_special_tokens is True
        assert truncation is False
        self.prepared_pairs.append((tokens, pair_ids))
        return {"input_ids": tokens + ["SEP"] + pair_ids}

    def pad(self, encoded, *, padding, return_tensors):
        assert padding is True
        assert return_tensors == "pt"
        return {"input_ids": [item["input_ids"] for item in encoded]}


def test_reranker_token_budget_preserves_query_head_and_tail():
    reranker = TransformerCrossEncoderReranker.__new__(
        TransformerCrossEncoderReranker
    )
    reranker.tokenizer = _FakeTokenizer()
    reranker.max_length = 64
    query = "QUERY_HEAD_" + "x" * 200 + "_QUERY_TAIL"
    label = "HEAD_" + "y" * 200 + "_TAIL"

    encoded = reranker._tokenize_pairs([(query, label)])

    prepared_query, prepared_label = reranker.tokenizer.prepared_pairs[0]
    assert "QUERY_HEAD" in "".join(prepared_query)
    assert "QUERY_TAIL" in "".join(prepared_query)
    assert "HEAD" in "".join(prepared_label)
    assert "TAIL" in "".join(prepared_label)
    assert "SEP" in encoded["input_ids"][0]


def _unit():
    return {
        "question_id": "q1",
        "unit_type": "sub_question",
        "parent_stem": "父题" * 2000,
        "stem": "当前小题",
        "options": "A.甲\nB.乙",
        "answer_text": "A",
        "analysis": "解析",
        "flags": {
            "parent_context_missing": False,
            "image_context_missing": True,
        },
    }


def _labels():
    return {
        "L1": {
            "label_id": "L1",
            "label_name": "标签一",
            "label_path": "知识点->模块->标签一",
            "definition": "标签一定义",
            "core_concepts": "标签一核心",
            "common_assessments": "不应发送给DS或Reranker",
            "distinctions": "标签一边界",
        },
        "L2": {
            "label_id": "L2",
            "label_name": "标签二",
            "label_path": "知识点->模块->标签二",
            "definition": "标签二定义",
            "core_concepts": "标签二核心",
            "common_assessments": "同样不发送",
            "distinctions": "标签二边界",
        },
    }


def _candidates():
    return [
        {"label_id": "L2", "candidate_rank": 1, "sources": ["dense"]},
        {"label_id": "L1", "candidate_rank": 2, "sources": ["legacy"]},
    ]


def test_reranker_uses_exact_ds_question_and_candidate_cards():
    unit = _unit()
    labels = _labels()
    candidates = _candidates()

    question, cards, code_map = build_adjudication_inputs(
        unit, candidates, labels
    )
    pairs = build_reranker_pairs(unit, candidates, labels)

    assert [pair["code"] for pair in pairs] == list(code_map)
    assert [pair["label_id"] for pair in pairs] == list(code_map.values())
    assert all(json.loads(pair["query"]) == question for pair in pairs)
    assert [json.loads(pair["document"]) for pair in pairs] == cards


def test_shared_inputs_preserve_ds_fields_limits_and_actual_label_schema():
    question, cards, _ = build_adjudication_inputs(
        _unit(), _candidates(), _labels()
    )

    assert len(question["parent_stem"]) == 3000
    assert question["image_context_missing"] is True
    assert list(cards[0]) == [
        "code",
        "label_name",
        "label_path",
        "definition",
        "core_concepts",
        "distinctions",
    ]
    assert "common_assessments" not in cards[0]


def test_reranker_instruction_keeps_precision_first_empty_output_policy():
    assert "false positives" in RERANKER_INSTRUCTION.lower()
    assert "no candidate" in RERANKER_INSTRUCTION.lower()
    assert "parent" in RERANKER_INSTRUCTION.lower()
    assert "distinctions" in RERANKER_INSTRUCTION


def test_normalize_chat_template_token_ids_accepts_transformers_5_mapping():
    value = {"input_ids": [[1, 2, 3], [4, 5]]}

    assert normalize_chat_template_token_ids(value) == [[1, 2, 3], [4, 5]]


def test_normalize_chat_template_token_ids_keeps_legacy_list_output():
    assert normalize_chat_template_token_ids([[1, 2], [3]]) == [[1, 2], [3]]


def test_explicit_qwen3_token_format_contains_each_query_and_document():
    class Tokenizer:
        def __init__(self):
            self.texts = []

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            self.texts.append(text)
            return [ord(character) for character in text]

    tokenizer = Tokenizer()
    pairs = [
        {"query": '{"question_id":"q1"}', "document": '{"label_name":"甲"}'},
        {"query": '{"question_id":"q2"}', "document": '{"label_name":"乙"}'},
    ]

    prompts = build_qwen3_reranker_token_ids(
        tokenizer,
        pairs,
        max_model_len=4096,
    )

    assert len(prompts) == 2
    assert prompts[0] != prompts[1]
    assert any('"question_id":"q1"' in text for text in tokenizer.texts)
    assert any('"label_name":"甲"' in text for text in tokenizer.texts)
    assert any('"question_id":"q2"' in text for text in tokenizer.texts)
    assert any('"label_name":"乙"' in text for text in tokenizer.texts)


def test_extract_binary_logprobs_uses_official_floor_for_missing_token():
    class Value:
        def __init__(self, logprob):
            self.logprob = logprob

    assert extract_binary_logprobs(
        {7: Value(-0.2)},
        true_token=7,
        false_token=8,
    ) == (-0.2, -10.0)
    assert extract_binary_logprobs(
        {8: Value(-0.3)},
        true_token=7,
        false_token=8,
    ) == (-10.0, -0.3)
