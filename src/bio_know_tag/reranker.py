"""Rerank retrieval candidates and build DS-aligned Qwen3 inputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Iterable

from bio_know_tag.adjudication import build_adjudication_inputs


RERANKER_INPUT_VERSION = "qwen3-reranker-ds-aligned-v2-explicit-token-format"
RERANKER_INSTRUCTION = """Determine whether the candidate Label is directly assessed by the current high-school biology question. False positives are substantially more costly than false negatives. The Label scope is jointly defined by label_name, label_path, definition, and distinctions; distinctions are hard exclusion boundaries, while core_concepts may explain but must not expand that scope. Reject a Label that is only related by terminology, chapter, hierarchy, background, shared mechanism, or a commonly co-occurring concept. Reject object, task, dimension, biological level, experimental purpose, method, or application mismatches. Judge only the current question; parent_stem may resolve an explicit reference but must not create an independent assessed concept. A wrong option supports a Label only when evaluating that option genuinely requires the Label. Each Label must independently support a real key judgment. It is valid for no candidate Label to match."""
QWEN3_RERANKER_PREFIX = """<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>
<|im_start|>user
"""
QWEN3_RERANKER_SUFFIX = """<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


def normalize_chat_template_token_ids(value: Any) -> list[list[int]]:
    """Normalize Transformers 4/5 chat-template batch return values."""
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("chat template mapping is missing input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError("chat template output must contain token ID sequences")
    if not value:
        return []
    if isinstance(value[0], int):
        return [list(value)]
    return [list(token_ids) for token_ids in value]


def build_qwen3_reranker_token_ids(
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    *,
    max_model_len: int,
) -> list[list[int]]:
    """Encode pairs with Qwen's official explicit reranker prompt format."""
    prefix_tokens = tokenizer.encode(
        QWEN3_RERANKER_PREFIX,
        add_special_tokens=False,
    )
    suffix_tokens = tokenizer.encode(
        QWEN3_RERANKER_SUFFIX,
        add_special_tokens=False,
    )
    prompts = []
    for pair in pairs:
        body = (
            f"<Instruct>: {RERANKER_INSTRUCTION}\n\n"
            f"<Query>: {pair['query']}\n\n"
            f"<Document>: {pair['document']}"
        )
        body_tokens = tokenizer.encode(body, add_special_tokens=False)
        token_ids = prefix_tokens + body_tokens + suffix_tokens
        if len(token_ids) > max_model_len:
            raise ValueError(
                "reranker input would be truncated, violating DS alignment: "
                f"{pair['question_id']}::{pair['label_id']} has "
                f"{len(token_ids)} tokens > {max_model_len}"
            )
        prompts.append(token_ids)
    return prompts


def extract_binary_logprobs(
    final_logprobs: Mapping[int, Any],
    *,
    true_token: int,
    false_token: int,
    missing_floor: float = -10.0,
) -> tuple[float, float]:
    """Extract yes/no logprobs using Qwen's official missing-token floor."""
    true_value = final_logprobs.get(true_token)
    false_value = final_logprobs.get(false_token)
    true_logprob = (
        float(true_value.logprob) if true_value is not None else missing_floor
    )
    false_logprob = (
        float(false_value.logprob) if false_value is not None else missing_floor
    )
    return true_logprob, false_logprob


class TransformerCrossEncoderReranker:
    """Score question/Label pairs with a sequence-classification model."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda:0",
        batch_size: int = 64,
        max_length: int = 512,
        revision: str | None = None,
        local_files_only: bool = False,
        use_fp16: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("reranking requires torch and transformers") from exc
        self.torch = torch
        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_revision = revision
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, local_files_only=local_files_only
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=revision, local_files_only=local_files_only
        ).to(self.device)
        self.model_commit_hash = getattr(self.model.config, "_commit_hash", None)
        if self.use_fp16:
            self.model = self.model.half()
        self.model.eval()

    @staticmethod
    def _crop_tokens(tokens: list[int], budget: int) -> list[int]:
        if len(tokens) <= budget:
            return tokens
        head = max(1, budget // 3)
        return tokens[:head] + tokens[-(budget - head) :]

    def _tokenize_pairs(self, pairs: list[tuple[str, str]]) -> object:
        # Crop token IDs on each side independently so generic pair truncation
        # cannot erase the question tail. Building from IDs also avoids a lossy
        # decode/re-tokenize round trip.
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)
        content_budget = max(2, self.max_length - special_tokens)
        query_budget = max(1, int(content_budget * 0.62))
        label_budget = max(1, content_budget - query_budget)
        prepared = []
        for query, label in pairs:
            query_ids = self.tokenizer.encode(query, add_special_tokens=False)
            label_ids = self.tokenizer.encode(label, add_special_tokens=False)
            prepared.append(
                self.tokenizer.prepare_for_model(
                    self._crop_tokens(query_ids, query_budget),
                    pair_ids=self._crop_tokens(label_ids, label_budget),
                    add_special_tokens=True,
                    truncation=False,
                )
            )
        return self.tokenizer.pad(
            prepared,
            padding=True,
            return_tensors="pt",
        )

    def score(self, pairs: Iterable[tuple[str, str]]) -> list[float]:
        torch = self.torch
        pair_list = list(pairs)
        scores: list[float] = []
        for offset in range(0, len(pair_list), self.batch_size):
            batch = pair_list[offset : offset + self.batch_size]
            inputs = self._tokenize_pairs(batch)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs).logits.reshape(-1)
            scores.extend(float(value) for value in logits.float().cpu().tolist())
        return scores


def build_reranker_pairs(
    unit: dict[str, Any],
    candidates: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create one query-document pair per candidate from shared DS payloads."""
    question, candidate_cards, code_map = build_adjudication_inputs(
        unit,
        candidates,
        labels_by_id,
    )
    query = json.dumps(question, ensure_ascii=False, separators=(",", ":"))
    pairs = []
    for card in candidate_cards:
        code = str(card["code"])
        pairs.append(
            {
                "question_id": str(unit.get("question_id") or ""),
                "code": code,
                "label_id": code_map[code],
                "query": query,
                "document": json.dumps(
                    card,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    return pairs
