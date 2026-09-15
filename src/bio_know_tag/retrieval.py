"""No-legacy candidate retrieval primitives and experiment runners."""

from __future__ import annotations

import json
import math
import re
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from bio_know_tag.ds import append_evidence, parse_json_content


CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", re.IGNORECASE)


def format_label_path(path: Any) -> str:
    """Render taxonomy hierarchy with @ without touching biological arrows."""
    return str(path or "").replace("->", "@")


def text_tokens(value: Any) -> list[str]:
    """Tokenize Chinese text with character n-grams plus ASCII terms."""
    text = str(value or "").lower()
    tokens = ASCII_TOKEN_RE.findall(text)
    for run in CJK_RUN_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
        if len(run) >= 3:
            tokens.extend(run[index : index + 3] for index in range(len(run) - 2))
        if len(run) <= 12:
            tokens.append(run)
    return tokens


def _weighted_tokens(*fields: tuple[Any, int]) -> list[str]:
    result: list[str] = []
    for value, weight in fields:
        tokens = text_tokens(value)
        for _ in range(max(0, weight)):
            result.extend(tokens)
    return result


def build_label_cards(labels: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    cards = []
    seen_ids: set[str] = set()
    for index, label in enumerate(labels, 1):
        label_id = str(label.get("label_id") or "").strip()
        if not label_id:
            raise ValueError("label_id is required")
        if label_id in seen_ids:
            raise ValueError(f"duplicate label_id: {label_id}")
        seen_ids.add(label_id)
        card = {
            "code": f"B{index:03d}",
            "label_id": label_id,
            "label_name": str(label.get("label_name") or "").strip(),
            "label_path": format_label_path(label.get("label_path")),
            "definition": str(label.get("definition") or "").strip(),
            "core_concepts": str(label.get("core_concepts") or "").strip(),
            "common_assessments": str(label.get("common_assessments") or "").strip(),
            "distinctions": str(label.get("distinctions") or "").strip(),
        }
        card["_tokens"] = _weighted_tokens(
            (card["label_name"], 4),
            (card["label_path"].replace("@", " "), 2),
            (card["definition"], 2),
            (card["core_concepts"], 2),
            (card["common_assessments"], 1),
            (card["distinctions"], 2),
        )
        cards.append(card)
    return cards


def build_query_tokens(unit: dict[str, Any]) -> list[str]:
    return _weighted_tokens(
        (unit.get("stem"), 2),
        (unit.get("answer_text"), 2),
        (unit.get("analysis"), 2),
        (unit.get("options"), 1),
        (unit.get("parent_stem"), 1),
    )


def build_dense_label_text(card: dict[str, Any]) -> str:
    """Compose the teacher-authoritative text embedded for one Label."""
    clip = lambda value, size: str(value or "")[:size]
    return "\n".join(
        (
            f"知识点名称：{clip(card.get('label_name'), 50)}",
            f"知识点路径：{clip(format_label_path(card.get('label_path')), 100)}",
            f"定义：{clip(card.get('definition'), 120)}",
            f"核心概念：{clip(card.get('core_concepts'), 130)}",
            f"易混淆边界：{clip(card.get('distinctions'), 80)}",
        )
    )


def _head_tail(value: Any, size: int) -> str:
    """Keep both the setup and the actual request at the end of long fields."""
    text = str(value or "")
    if len(text) <= size:
        return text
    head_size = max(1, size // 3)
    tail_size = max(1, size - head_size)
    return f"{text[:head_size]}…[中间省略]…{text[-tail_size:]}"


def build_dense_query_text(unit: dict[str, Any]) -> str:
    """Compose a compact query without dropping late subquestions or conclusions."""
    return "\n".join(
        (
            f"当前题干：{_head_tail(unit.get('stem'), 260)}",
            f"答案：{_head_tail(unit.get('answer_text'), 100)}",
            f"解析：{_head_tail(unit.get('analysis'), 240)}",
            f"选项：{_head_tail(unit.get('options'), 120)}",
            f"父题公共材料（仅作语境）：{_head_tail(unit.get('parent_stem'), 100)}",
        )
    )


class BM25Retriever:
    """Small in-memory BM25 index for the 458 teacher Label Cards."""

    def __init__(self, cards: Iterable[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75):
        self.cards = [dict(card) for card in cards]
        if not self.cards:
            raise ValueError("at least one label card is required")
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(card.get("_tokens") or []) for card in self.cards]
        self.lengths = [sum(counter.values()) for counter in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths) or 1.0
        document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())
        document_count = len(self.cards)
        self.idf = {
            token: math.log(
                1 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for token, frequency in document_frequency.items()
        }

    def search(self, query_tokens: Iterable[str], *, top_k: int = 20) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_frequencies = Counter(query_tokens)
        scored = []
        for card, frequencies, length in zip(
            self.cards, self.term_frequencies, self.lengths
        ):
            score = 0.0
            norm = self.k1 * (
                1 - self.b + self.b * length / self.average_length
            )
            for token, query_weight in query_frequencies.items():
                term_frequency = frequencies.get(token, 0)
                if not term_frequency:
                    continue
                score += (
                    self.idf.get(token, 0.0)
                    * (term_frequency * (self.k1 + 1))
                    / (term_frequency + norm)
                    * query_weight
                )
            scored.append((score, card))
        scored.sort(key=lambda item: (-item[0], item[1]["label_id"]))
        results = []
        for score, card in scored:
            if score <= 0 or len(results) >= top_k:
                break
            results.append(
                {
                    "label_id": card["label_id"],
                    "label_name": card["label_name"],
                    "label_path": card["label_path"],
                    "rank": len(results) + 1,
                    "score": round(score, 8),
                }
            )
        return results


def _coarse_unit(unit: dict[str, Any]) -> dict[str, str]:
    return {
        "question_id": str(unit.get("question_id") or ""),
        "parent_stem": str(unit.get("parent_stem") or "")[:3000],
        "stem": str(unit.get("stem") or "")[:4000],
        "options": str(unit.get("options") or "")[:3000],
        "answer_text": str(unit.get("answer_text") or "")[:2000],
        "analysis": str(unit.get("analysis") or "")[:5000],
    }


def build_coarse_recall_prompt(
    units: Iterable[dict[str, Any]],
    cards: Iterable[dict[str, Any]],
    *,
    top_k: int = 20,
) -> str:
    compact_catalog = [
        {
            "code": card["code"],
            "label_path": card["label_path"],
        }
        for card in cards
    ]
    compact_units = [_coarse_unit(unit) for unit in units]
    return f"""你是高中生物知识点候选召回器。本轮只负责高召回地圈定候选，不做最终打标。

规则：
1. 对每道题分别判断；父题材料只提供语境，重点看当前题干、答案和解析。
2. 候选应覆盖完成设问可能需要的知识点，包括容易混淆的相邻Label。
3. 不因背景词、实验工具或错误选项机械加入Label。
4. 每题最多返回{top_k}个label_id，按可能性从高到低排列；确实没有合适项时允许空数组。
5. 只能使用目录中存在的短代码（B001至B458），不要抄写19位label_id，不要使用旧knw_ids。

Label目录（仅名称路径，不含释义）：
{json.dumps(compact_catalog, ensure_ascii=False)}

题目：
{json.dumps(compact_units, ensure_ascii=False)}

只输出一个JSON对象，不要输出Markdown或解释：
{{"results":[{{"question_id":"原question_id","candidate_codes":["按可能性排序的短代码"]}}]}}"""


def validate_coarse_recall_result(
    value: dict[str, Any],
    expected_question_ids: Iterable[str],
    known_candidate_codes: set[str],
    *,
    top_k: int,
) -> dict[str, Any]:
    results = value.get("results")
    if not isinstance(results, list):
        raise ValueError("results must be a list")
    expected = [str(question_id) for question_id in expected_question_ids]
    actual = []
    duplicates_removed = 0
    codes_truncated = 0
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("each result must be an object")
        question_id = str(result.get("question_id") or "")
        actual.append(question_id)
        candidates = result.get("candidate_codes")
        if not isinstance(candidates, list) or any(
            not isinstance(label_id, str) or not label_id
            for label_id in candidates
        ):
            raise ValueError("candidate_codes must be a list of strings")
        unique_candidates = list(dict.fromkeys(candidates))
        duplicates_removed += len(candidates) - len(unique_candidates)
        unknown = [
            code for code in unique_candidates if code not in known_candidate_codes
        ]
        if unknown:
            raise ValueError(f"unknown candidate code: {unknown[0]}")
        result["candidate_codes"] = unique_candidates[:top_k]
        codes_truncated += max(0, len(unique_candidates) - top_k)
    if actual != expected:
        raise ValueError("results must contain every requested question exactly once and in order")
    if duplicates_removed or codes_truncated:
        value["normalization"] = {}
        if duplicates_removed:
            value["normalization"]["duplicate_candidate_codes_removed"] = duplicates_removed
        if codes_truncated:
            value["normalization"]["candidate_codes_truncated"] = codes_truncated
    return value


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[str]],
    *,
    top_k: int = 20,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    scores: Counter[str] = Counter()
    method_hits: Counter[str] = Counter()
    for ranking in rankings:
        seen: set[str] = set()
        for rank, label_id in enumerate(ranking, 1):
            label_id = str(label_id)
            if label_id in seen:
                continue
            seen.add(label_id)
            scores[label_id] += 1.0 / (rank_constant + rank)
            method_hits[label_id] += 1
    ordered = sorted(scores, key=lambda label_id: (-scores[label_id], label_id))
    return [
        {
            "label_id": label_id,
            "rank": rank,
            "rrf_score": round(scores[label_id], 10),
            "method_hits": method_hits[label_id],
        }
        for rank, label_id in enumerate(ordered[:top_k], 1)
    ]


def _read_objects(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            yield value


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_sparse_retrieval(
    units_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    *,
    top_k: int = 20,
    limit: int | None = None,
    progress_every: int = 500,
) -> dict[str, Any]:
    labels = list(_read_objects(labels_path))
    cards = build_label_cards(labels)
    retriever = BM25Retriever(cards)
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidates.jsonl"
    temporary = candidate_path.with_name(f".{candidate_path.name}.tmp")
    processed = 0
    errors = 0
    candidate_counts: Counter[str] = Counter()
    score_zero = 0
    started_at = datetime.now(timezone.utc).isoformat()
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for index, unit in enumerate(_read_objects(units_path), 1):
            if limit is not None and index > limit:
                break
            try:
                question_id = str(unit.get("question_id") or "").strip()
                if not question_id:
                    raise ValueError("question_id is required")
                candidates = retriever.search(
                    build_query_tokens(unit), top_k=top_k
                )
                candidate_counts[str(len(candidates))] += 1
                score_zero += int(not candidates or candidates[0]["score"] <= 0)
                record = {
                    "question_id": question_id,
                    "method": "char_ngram_bm25",
                    "retrieval_version": "sparse-v1",
                    "candidates": candidates,
                }
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                output.write("\n")
                processed += 1
            except (TypeError, ValueError, json.JSONDecodeError):
                errors += 1
            if progress_every and index % progress_every == 0:
                print(
                    f"sparse recall: input={index}, processed={processed}, error={errors}",
                    flush=True,
                )
    temporary.replace(candidate_path)
    report = {
        "input": processed + errors,
        "processed": processed,
        "error": errors,
        "top_k": top_k,
        "labels": len(cards),
        "zero_top_score": score_zero,
        "candidate_count_distribution": dict(
            sorted(candidate_counts.items(), key=lambda item: int(item[0]))
        ),
        "method": "char_ngram_bm25",
        "retrieval_version": "sparse-v1",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report


def _latest_coarse_results(
    evidence_path: Path, code_to_id: dict[str, str]
) -> tuple[dict[str, list[str]], int]:
    latest: dict[str, list[str]] = {}
    evidence_rows = 0
    if not evidence_path.exists():
        return latest, evidence_rows
    for record in _read_objects(evidence_path):
        evidence_rows += 1
        parsed = record.get("parsed_response")
        if record.get("error") or not isinstance(parsed, dict):
            continue
        for result in parsed.get("results") or []:
            if isinstance(result, dict) and result.get("question_id"):
                codes = [str(value) for value in result.get("candidate_codes") or []]
                latest[str(result["question_id"])] = [code_to_id[code] for code in codes]
    return latest, evidence_rows


def run_ds_coarse_recall(
    units_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    client: Any,
    *,
    model: str,
    top_k: int = 20,
    batch_size: int = 5,
    limit: int | None = None,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Run the all-Label-name/path DS recall baseline with resume evidence."""
    if top_k < 1 or batch_size < 1:
        raise ValueError("top_k and batch_size must be positive")
    labels = list(_read_objects(labels_path))
    cards = build_label_cards(labels)
    cards_by_id = {card["label_id"]: card for card in cards}
    cards_by_code = {card["code"]: card for card in cards}
    code_to_id = {code: card["label_id"] for code, card in cards_by_code.items()}
    known_codes = set(cards_by_code)
    units = []
    for index, unit in enumerate(_read_objects(units_path), 1):
        if limit is not None and index > limit:
            break
        units.append(unit)
    expected_ids = [str(unit.get("question_id") or "") for unit in units]
    if any(not question_id for question_id in expected_ids):
        raise ValueError("every unit must have question_id")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("duplicate question_id in units")

    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "evidence.jsonl"
    completed, _ = _latest_coarse_results(evidence_path, code_to_id)
    pending = [unit for unit in units if str(unit["question_id"]) not in completed]
    requests_succeeded = 0
    requests_failed = 0

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        question_ids = [str(unit["question_id"]) for unit in batch]
        prompt = build_coarse_recall_prompt(batch, cards, top_k=top_k)
        record = {
            "stage": "ds_coarse_recall",
            "prompt_version": "coarse-all-paths-v2",
            "question_ids": question_ids,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_response": None,
            "parsed_response": None,
            "endpoint": None,
            "attempts": 0,
            "latency_seconds": None,
            "error": None,
        }
        try:
            response = client.chat(
                [
                    {
                        "role": "system",
                        "content": "你是严谨的高中生物知识点候选召回器，只输出JSON。",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
            record.update(
                {
                    "raw_response": response.content,
                    "endpoint": response.endpoint,
                    "attempts": response.attempts,
                    "latency_seconds": response.latency_seconds,
                }
            )
            parsed = validate_coarse_recall_result(
                parse_json_content(response.content),
                question_ids,
                known_codes,
                top_k=top_k,
            )
            record.update(
                {
                    "parsed_response": parsed,
                }
            )
            requests_succeeded += 1
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            requests_failed += 1
        append_evidence(evidence_path, record)
        completed_now, evidence_rows = _latest_coarse_results(evidence_path, code_to_id)
        interim = {
            "input": len(units),
            "processed": len(completed_now),
            "success": len(completed_now),
            "error": len(units) - len(completed_now),
            "pending": len(units) - len(completed_now),
            "evidence_rows": evidence_rows,
            "requests_succeeded_this_run": requests_succeeded,
            "requests_failed_this_run": requests_failed,
            "top_k": top_k,
            "batch_size": batch_size,
            "labels": len(cards),
            "model": model,
            "method": "ds_all_label_paths",
            "prompt_version": "coarse-all-paths-v2",
        }
        _write_json_atomic(output_dir / "report.json", interim)
        print(
            f"DS coarse recall: success={len(completed_now)}/{len(units)}, batch={question_ids[0]}..{question_ids[-1]}, {'ERROR' if record['error'] else 'OK'}",
            flush=True,
        )

    completed, evidence_rows = _latest_coarse_results(evidence_path, code_to_id)
    candidate_path = output_dir / "candidates.jsonl"
    temporary = candidate_path.with_name(f".{candidate_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for question_id in expected_ids:
            if question_id not in completed:
                continue
            candidates = []
            for rank, label_id in enumerate(completed[question_id], 1):
                card = cards_by_id[label_id]
                candidates.append(
                    {
                        "label_id": label_id,
                        "label_name": card["label_name"],
                        "label_path": card["label_path"],
                        "rank": rank,
                    }
                )
            output.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "method": "ds_all_label_paths",
                        "retrieval_version": "ds-coarse-v2",
                        "candidates": candidates,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            output.write("\n")
    temporary.replace(candidate_path)
    success = len(completed)
    report = {
        "input": len(units),
        "processed": success,
        "success": success,
        "error": len(units) - success,
        "pending": len(units) - success,
        "evidence_rows": evidence_rows,
        "requests_succeeded": requests_succeeded,
        "requests_failed": requests_failed,
        "top_k": top_k,
        "batch_size": batch_size,
        "labels": len(cards),
        "model": model,
        "method": "ds_all_label_paths",
        "retrieval_version": "ds-coarse-v2",
        "prompt_version": "coarse-all-paths-v2",
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report


def run_dense_retrieval(
    units_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    encoder: Any,
    *,
    top_k: int = 20,
    batch_size: int = 64,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run a configurable dense encoder and write the standard candidate sidecar."""
    if top_k < 1 or batch_size < 1:
        raise ValueError("top_k and batch_size must be positive")
    cards = build_label_cards(_read_objects(labels_path))
    cards_by_id = {card["label_id"]: card for card in cards}
    encoder.index(
        [(card["label_id"], build_dense_label_text(card)) for card in cards]
    )
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidates.jsonl"
    temporary = candidate_path.with_name(f".{candidate_path.name}.tmp")
    processed = 0
    errors = 0
    input_count = 0
    error_records: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    batch: list[dict[str, Any]] = []

    def write_batch(output, units: list[dict[str, Any]]) -> tuple[int, int]:
        if not units:
            return 0, 0
        try:
            rankings = encoder.search(
                [build_dense_query_text(unit) for unit in units], top_k=top_k
            )
            if len(rankings) != len(units):
                raise ValueError("encoder returned wrong number of rankings")
            serialized_records = []
            for unit, ranking in zip(units, rankings):
                candidates = []
                seen: set[str] = set()
                for label_id, score in ranking:
                    label_id = str(label_id)
                    if label_id in seen:
                        continue
                    if label_id not in cards_by_id:
                        raise ValueError(f"encoder returned unknown label_id: {label_id}")
                    seen.add(label_id)
                    card = cards_by_id[label_id]
                    candidates.append(
                        {
                            "label_id": label_id,
                            "label_name": card["label_name"],
                            "label_path": card["label_path"],
                            "rank": len(candidates) + 1,
                            "score": round(float(score), 8),
                        }
                    )
                    if len(candidates) >= top_k:
                        break
                serialized_records.append(
                    json.dumps(
                        {
                            "question_id": str(unit["question_id"]),
                            "method": "dense_embedding",
                            "retrieval_version": "dense-v2-head-tail",
                            "model": str(encoder.model_name),
                            "candidates": candidates,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            for serialized in serialized_records:
                output.write(serialized)
                output.write("\n")
            return len(units), 0
        except Exception as exc:
            error_records.append(
                {
                    "question_ids": [str(unit.get("question_id") or "") for unit in units],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return 0, len(units)

    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for unit in _read_objects(units_path):
            if limit is not None and input_count >= limit:
                break
            input_count += 1
            if not unit.get("question_id"):
                errors += 1
                continue
            batch.append(unit)
            if len(batch) >= batch_size:
                succeeded, failed = write_batch(output, batch)
                processed += succeeded
                errors += failed
                batch.clear()
                print(
                    f"dense recall: input={input_count}, processed={processed}, error={errors}",
                    flush=True,
                )
        succeeded, failed = write_batch(output, batch)
        processed += succeeded
        errors += failed
    temporary.replace(candidate_path)
    error_path = output_dir / "errors.jsonl"
    error_temporary = error_path.with_name(f".{error_path.name}.tmp")
    with error_temporary.open("w", encoding="utf-8", newline="\n") as output:
        for record in error_records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            output.write("\n")
    error_temporary.replace(error_path)
    report = {
        "input": input_count,
        "processed": processed,
        "error": errors,
        "top_k": top_k,
        "batch_size": batch_size,
        "labels": len(cards),
        "method": "dense_embedding",
        "retrieval_version": "dense-v2-head-tail",
        "model": str(encoder.model_name),
        "model_revision": getattr(encoder, "model_revision", None),
        "model_commit_hash": getattr(encoder, "model_commit_hash", None),
        "device": str(getattr(encoder, "device", "")),
        "max_length": getattr(encoder, "max_length", None),
        "query_instruction": getattr(encoder, "query_instruction", None),
        "precision": "fp16" if getattr(encoder, "use_fp16", False) else "fp32",
        "input_paths": {
            "units": str(Path(units_path)),
            "labels": str(Path(labels_path)),
        },
        "input_sha256": {
            "units": _file_sha256(units_path),
            "labels": _file_sha256(labels_path),
        },
        "error_batches": len(error_records),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report


def compare_candidate_runs(
    units_path: str | Path,
    sparse_candidates_path: str | Path,
    dense_candidates_path: str | Path,
    run_dir: str | Path,
    *,
    top_k: int = 20,
    sample_size: int = 200,
) -> dict[str, Any]:
    """Compare two retrieval rankings without pretending overlap is accuracy."""
    units = {
        str(unit["question_id"]): unit for unit in _read_objects(units_path)
    }
    sparse = {
        str(row["question_id"]): row for row in _read_objects(sparse_candidates_path)
    }
    dense = {
        str(row["question_id"]): row for row in _read_objects(dense_candidates_path)
    }
    common_ids = sorted(set(units) & set(sparse) & set(dense))
    if not common_ids:
        raise ValueError("candidate runs have no common questions")
    comparisons = []
    top1_agreement = 0
    overlap_sum = 0
    jaccard_sum = 0.0
    for question_id in common_ids:
        sparse_ids = [
            str(item["label_id"])
            for item in (sparse[question_id].get("candidates") or [])[:top_k]
        ]
        dense_ids = [
            str(item["label_id"])
            for item in (dense[question_id].get("candidates") or [])[:top_k]
        ]
        sparse_set = set(sparse_ids)
        dense_set = set(dense_ids)
        overlap = len(sparse_set & dense_set)
        union = len(sparse_set | dense_set)
        jaccard = overlap / union if union else 1.0
        agrees = bool(sparse_ids and dense_ids and sparse_ids[0] == dense_ids[0])
        top1_agreement += int(agrees)
        overlap_sum += overlap
        jaccard_sum += jaccard
        comparisons.append(
            {
                "question_id": question_id,
                "jaccard_at_k": jaccard,
                "overlap_at_k": overlap,
                "top1_agree": agrees,
                "unit": units[question_id],
                "sparse_candidates": sparse[question_id].get("candidates") or [],
                "dense_candidates": dense[question_id].get("candidates") or [],
            }
        )
    comparisons.sort(key=lambda row: (row["jaccard_at_k"], row["question_id"]))
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "disagreement_samples.jsonl"
    temporary = sample_path.with_name(f".{sample_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in comparisons[:sample_size]:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            output.write("\n")
    temporary.replace(sample_path)
    question_count = len(common_ids)
    report = {
        "questions": question_count,
        "top_k": top_k,
        "top1_agreement": top1_agreement,
        "top1_agreement_rate": round(top1_agreement / question_count, 6),
        "mean_overlap_at_k": round(overlap_sum / question_count, 6),
        "mean_jaccard_at_k": round(jaccard_sum / question_count, 6),
        "disagreement_samples": min(sample_size, question_count),
        "note": "排名重合度不是准确率；必须人工确认分歧样本后才能判断Dense是否有增益。",
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report


def run_candidate_reranking(
    units_path: str | Path,
    labels_path: str | Path,
    sparse_candidates_path: str | Path,
    dense_candidates_path: str | Path,
    run_dir: str | Path,
    reranker: Any,
    *,
    sparse_pool: int = 30,
    dense_pool: int = 30,
    top_k: int = 30,
) -> dict[str, Any]:
    """Rerank the union of broad sparse and dense pools with a cross-encoder."""
    if min(sparse_pool, dense_pool, top_k) < 1:
        raise ValueError("pool sizes and top_k must be positive")
    units = {str(row["question_id"]): row for row in _read_objects(units_path)}
    labels = {
        card["label_id"]: card
        for card in build_label_cards(_read_objects(labels_path))
    }
    sparse_rows = list(_read_objects(sparse_candidates_path))
    dense_rows = list(_read_objects(dense_candidates_path))
    sparse = {str(row["question_id"]): row for row in sparse_rows}
    dense = {str(row["question_id"]): row for row in dense_rows}
    if set(units) != set(sparse) or set(units) != set(dense):
        raise ValueError("units, sparse, and dense runs must contain identical question IDs")

    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "candidates.jsonl"
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    candidate_counts: Counter[str] = Counter()
    pool_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    version = f"rerank-v1-s{sparse_pool}-d{dense_pool}-k{top_k}"
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for index, (question_id, unit) in enumerate(units.items(), 1):
            sparse_items = (sparse[question_id].get("candidates") or [])[:sparse_pool]
            dense_items = (dense[question_id].get("candidates") or [])[:dense_pool]
            sparse_by_id = {str(item["label_id"]): item for item in sparse_items}
            dense_by_id = {str(item["label_id"]): item for item in dense_items}
            pool_ids = list(sparse_by_id)
            pool_ids.extend(label_id for label_id in dense_by_id if label_id not in sparse_by_id)
            unknown = [label_id for label_id in pool_ids if label_id not in labels]
            if unknown:
                raise ValueError(f"candidate uses unknown label_id: {unknown[0]}")
            query_text = build_dense_query_text(unit)
            scores = reranker.score(
                [(query_text, build_dense_label_text(labels[label_id])) for label_id in pool_ids]
            )
            if len(scores) != len(pool_ids):
                raise ValueError("reranker returned wrong number of scores")
            ordered = sorted(
                zip(pool_ids, scores),
                key=lambda item: (-float(item[1]), item[0]),
            )[:top_k]
            candidates = []
            for rank, (label_id, score) in enumerate(ordered, 1):
                sparse_item = sparse_by_id.get(label_id)
                dense_item = dense_by_id.get(label_id)
                card = labels[label_id]
                sources = [
                    method
                    for method, present in (("sparse", sparse_item), ("dense", dense_item))
                    if present is not None
                ]
                source_counts["+".join(sources)] += 1
                candidates.append(
                    {
                        "label_id": label_id,
                        "label_name": card["label_name"],
                        "label_path": card["label_path"],
                        "rank": rank,
                        "candidate_rank": rank,
                        "rerank_score": round(float(score), 8),
                        "sources": sources,
                        "sparse_rank": sparse_item.get("rank") if sparse_item else None,
                        "sparse_score": sparse_item.get("score") if sparse_item else None,
                        "dense_rank": dense_item.get("rank") if dense_item else None,
                        "dense_score": dense_item.get("score") if dense_item else None,
                    }
                )
            pool_counts[str(len(pool_ids))] += 1
            candidate_counts[str(len(candidates))] += 1
            output.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "method": "sparse_dense_cross_encoder_rerank",
                        "retrieval_version": version,
                        "candidate_pool_count": len(pool_ids),
                        "candidates": candidates,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            output.write("\n")
            if index % 50 == 0 or index == len(units):
                print(
                    f"rerank: processed={index}/{len(units)}, pool={len(pool_ids)}, output={len(candidates)}",
                    flush=True,
                )
    temporary.replace(output_path)
    report = {
        "input": len(units),
        "processed": len(units),
        "error": 0,
        "sparse_pool": sparse_pool,
        "dense_pool": dense_pool,
        "top_k": top_k,
        "model": str(reranker.model_name),
        "model_revision": getattr(reranker, "model_revision", None),
        "model_commit_hash": getattr(reranker, "model_commit_hash", None),
        "device": str(getattr(reranker, "device", "")),
        "batch_size": getattr(reranker, "batch_size", None),
        "max_length": getattr(reranker, "max_length", None),
        "precision": "fp16" if getattr(reranker, "use_fp16", False) else "fp32",
        "method": "sparse_dense_cross_encoder_rerank",
        "retrieval_version": version,
        "upstream_retrieval_versions": {
            "sparse": sorted({str(row.get("retrieval_version") or "") for row in sparse_rows}),
            "dense": sorted({str(row.get("retrieval_version") or "") for row in dense_rows}),
        },
        "input_paths": {
            "units": str(Path(units_path)),
            "labels": str(Path(labels_path)),
            "sparse_candidates": str(Path(sparse_candidates_path)),
            "dense_candidates": str(Path(dense_candidates_path)),
        },
        "input_sha256": {
            "units": _file_sha256(units_path),
            "labels": _file_sha256(labels_path),
            "sparse_candidates": _file_sha256(sparse_candidates_path),
            "dense_candidates": _file_sha256(dense_candidates_path),
        },
        "pool_count_distribution": dict(
            sorted(pool_counts.items(), key=lambda item: int(item[0]))
        ),
        "candidate_count_distribution": dict(
            sorted(candidate_counts.items(), key=lambda item: int(item[0]))
        ),
        "candidate_source_counts": dict(sorted(source_counts.items())),
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report


def quota_fuse_candidates(
    sparse_candidates: list[dict[str, Any]],
    dense_candidates: list[dict[str, Any]],
    *,
    sparse_quota: int = 18,
    dense_quota: int = 7,
    top_k: int = 25,
) -> list[dict[str, Any]]:
    """Reserve BM25/Dense rank slots, then use RRF only to fill vacancies."""
    if min(sparse_quota, dense_quota, top_k) < 0 or top_k < 1:
        raise ValueError("quotas must be non-negative and top_k must be positive")
    sparse_by_id = {str(item["label_id"]): item for item in sparse_candidates}
    dense_by_id = {str(item["label_id"]): item for item in dense_candidates}
    ordered_ids: list[str] = []
    seen: set[str] = set()

    def add(items: Iterable[dict[str, Any]]) -> None:
        for item in items:
            label_id = str(item["label_id"])
            if label_id not in seen and len(ordered_ids) < top_k:
                seen.add(label_id)
                ordered_ids.append(label_id)

    add(sparse_candidates[:sparse_quota])
    dense_added = 0
    for candidate in dense_candidates:
        label_id = str(candidate["label_id"])
        if label_id in seen:
            continue
        add([candidate])
        dense_added += 1
        if dense_added >= dense_quota or len(ordered_ids) >= top_k:
            break
    rrf = reciprocal_rank_fusion(
        [list(sparse_by_id), list(dense_by_id)], top_k=len(seen) + top_k
    )
    add(
        {"label_id": item["label_id"]}
        for item in rrf
        if item["label_id"] not in seen
    )
    add(sparse_candidates)
    add(dense_candidates)

    fused = []
    for candidate_rank, label_id in enumerate(ordered_ids, 1):
        sparse = sparse_by_id.get(label_id)
        dense = dense_by_id.get(label_id)
        source = sparse or dense or {"label_id": label_id}
        fused.append(
            {
                "label_id": label_id,
                "label_name": source.get("label_name", ""),
                "label_path": format_label_path(source.get("label_path")),
                "candidate_rank": candidate_rank,
                "sources": [
                    method
                    for method, present in (("sparse", sparse), ("dense", dense))
                    if present is not None
                ],
                "sparse_rank": sparse.get("rank") if sparse else None,
                "sparse_score": sparse.get("score") if sparse else None,
                "dense_rank": dense.get("rank") if dense else None,
                "dense_score": dense.get("score") if dense else None,
            }
        )
    return fused


def run_hybrid_retrieval(
    sparse_candidates_path: str | Path,
    dense_candidates_path: str | Path,
    run_dir: str | Path,
    *,
    top_k: int = 25,
    sparse_quota: int = 18,
    dense_quota: int = 7,
) -> dict[str, Any]:
    sparse = {
        str(row["question_id"]): row for row in _read_objects(sparse_candidates_path)
    }
    dense = {
        str(row["question_id"]): row for row in _read_objects(dense_candidates_path)
    }
    if set(sparse) != set(dense):
        raise ValueError("sparse and dense runs must contain identical question IDs")
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "candidates.jsonl"
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    candidate_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    version = f"hybrid-v1-s{sparse_quota}-d{dense_quota}-k{top_k}"
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for question_id in sparse:
            candidates = quota_fuse_candidates(
                sparse[question_id].get("candidates") or [],
                dense[question_id].get("candidates") or [],
                sparse_quota=sparse_quota,
                dense_quota=dense_quota,
                top_k=top_k,
            )
            candidate_counts[str(len(candidates))] += 1
            for candidate in candidates:
                source_counts["+".join(candidate["sources"])] += 1
            output.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "method": "sparse_dense_quota_fusion",
                        "retrieval_version": version,
                        "candidates": candidates,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            output.write("\n")
    temporary.replace(output_path)
    report = {
        "input": len(sparse),
        "processed": len(sparse),
        "error": 0,
        "top_k": top_k,
        "sparse_quota": sparse_quota,
        "dense_quota": dense_quota,
        "retrieval_version": version,
        "candidate_count_distribution": dict(
            sorted(candidate_counts.items(), key=lambda item: int(item[0]))
        ),
        "candidate_source_counts": dict(sorted(source_counts.items())),
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report
