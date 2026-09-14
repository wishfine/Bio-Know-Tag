"""Lazy Transformers implementation for dense Label retrieval."""

from __future__ import annotations

from typing import Iterable


DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class TransformerDenseRetriever:
    """Encode BGE-style CLS vectors and perform exact cosine retrieval."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda:0",
        encode_batch_size: int = 128,
        max_length: int = 512,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        local_files_only: bool = False,
        use_fp16: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "dense retrieval requires torch and transformers"
            ) from exc
        self.torch = torch
        self.model_name = model_name
        self.device = torch.device(device)
        self.encode_batch_size = encode_batch_size
        self.max_length = max_length
        self.query_instruction = query_instruction
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            model_name, local_files_only=local_files_only
        ).to(self.device)
        if use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()
        self.model.eval()
        self.label_ids: list[str] = []
        self.label_embeddings = None

    def _encode(self, texts: list[str], *, is_query: bool) -> object:
        torch = self.torch
        vectors = []
        for offset in range(0, len(texts), self.encode_batch_size):
            batch = texts[offset : offset + self.encode_batch_size]
            if is_query and self.query_instruction:
                batch = [f"{self.query_instruction}{text}" for text in batch]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = self.model(**inputs)
                embedding = output.last_hidden_state[:, 0]
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            vectors.append(embedding)
        return torch.cat(vectors, dim=0)

    def index(self, label_entries: Iterable[tuple[str, str]]) -> None:
        entries = list(label_entries)
        if not entries:
            raise ValueError("label entries must not be empty")
        self.label_ids = [str(label_id) for label_id, _ in entries]
        self.label_embeddings = self._encode(
            [text for _, text in entries], is_query=False
        )

    def search(
        self, query_texts: list[str], *, top_k: int
    ) -> list[list[tuple[str, float]]]:
        if self.label_embeddings is None:
            raise RuntimeError("index must be called before search")
        query_embeddings = self._encode(query_texts, is_query=True)
        scores = query_embeddings @ self.label_embeddings.T
        limit = min(top_k, len(self.label_ids))
        values, indices = self.torch.topk(scores, k=limit, dim=1)
        results = []
        for row_values, row_indices in zip(values.cpu().tolist(), indices.cpu().tolist()):
            results.append(
                [
                    (self.label_ids[index], float(score))
                    for score, index in zip(row_values, row_indices)
                ]
            )
        return results
