from bio_know_tag.reranker import TransformerCrossEncoderReranker


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
