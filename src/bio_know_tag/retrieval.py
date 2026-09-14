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
    for label in labels:
        label_id = str(label.get("label_id") or "").strip()
        if not label_id:
            raise ValueError("label_id is required")
        if label_id in seen_ids:
            raise ValueError(f"duplicate label_id: {label_id}")
        seen_ids.add(label_id)
        card = {
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
            "label_id": card["label_id"],
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
5. 只能使用目录中存在的label_id。不要使用旧knw_ids。

Label目录（仅名称路径，不含释义）：
{json.dumps(compact_catalog, ensure_ascii=False)}

题目：
{json.dumps(compact_units, ensure_ascii=False)}

只输出一个JSON对象，不要输出Markdown或解释：
{{"results":[{{"question_id":"原question_id","candidate_label_ids":["按可能性排序的label_id"]}}]}}"""


def validate_coarse_recall_result(
    value: dict[str, Any],
    expected_question_ids: Iterable[str],
    known_label_ids: set[str],
    *,
    top_k: int,
) -> dict[str, Any]:
    results = value.get("results")
    if not isinstance(results, list):
        raise ValueError("results must be a list")
    expected = [str(question_id) for question_id in expected_question_ids]
    actual = []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("each result must be an object")
        question_id = str(result.get("question_id") or "")
        actual.append(question_id)
        candidates = result.get("candidate_label_ids")
        if not isinstance(candidates, list) or any(
            not isinstance(label_id, str) or not label_id
            for label_id in candidates
        ):
            raise ValueError("candidate_label_ids must be a list of strings")
        if len(candidates) > top_k:
            raise ValueError("candidate_label_ids exceeds top_k")
        if len(candidates) != len(set(candidates)):
            raise ValueError("candidate_label_ids contains duplicates")
        unknown = [label_id for label_id in candidates if label_id not in known_label_ids]
        if unknown:
            raise ValueError(f"unknown label_id: {unknown[0]}")
    if actual != expected:
        raise ValueError("results must contain every requested question exactly once and in order")
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


def _latest_coarse_results(evidence_path: Path) -> tuple[dict[str, list[str]], int]:
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
                latest[str(result["question_id"])] = [
                    str(value) for value in result.get("candidate_label_ids") or []
                ]
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
    known_ids = set(cards_by_id)
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
    completed, _ = _latest_coarse_results(evidence_path)
    pending = [unit for unit in units if str(unit["question_id"]) not in completed]
    requests_succeeded = 0
    requests_failed = 0

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        question_ids = [str(unit["question_id"]) for unit in batch]
        prompt = build_coarse_recall_prompt(batch, cards, top_k=top_k)
        record = {
            "stage": "ds_coarse_recall",
            "prompt_version": "coarse-all-paths-v1",
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
            parsed = validate_coarse_recall_result(
                parse_json_content(response.content),
                question_ids,
                known_ids,
                top_k=top_k,
            )
            record.update(
                {
                    "raw_response": response.content,
                    "parsed_response": parsed,
                    "endpoint": response.endpoint,
                    "attempts": response.attempts,
                    "latency_seconds": response.latency_seconds,
                }
            )
            requests_succeeded += 1
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            requests_failed += 1
        append_evidence(evidence_path, record)
        completed_now, evidence_rows = _latest_coarse_results(evidence_path)
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
            "prompt_version": "coarse-all-paths-v1",
        }
        _write_json_atomic(output_dir / "report.json", interim)
        print(
            f"DS coarse recall: success={len(completed_now)}/{len(units)}, batch={question_ids[0]}..{question_ids[-1]}, {'ERROR' if record['error'] else 'OK'}",
            flush=True,
        )

    completed, evidence_rows = _latest_coarse_results(evidence_path)
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
                        "retrieval_version": "ds-coarse-v1",
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
        "retrieval_version": "ds-coarse-v1",
        "prompt_version": "coarse-all-paths-v1",
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report
