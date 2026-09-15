"""Lazy Transformers cross-encoder used to rerank retrieval candidates."""

from __future__ import annotations

from typing import Iterable


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
