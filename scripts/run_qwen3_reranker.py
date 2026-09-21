#!/usr/bin/env python3
"""Score DS-aligned question/Label pairs with Qwen3-Reranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from bio_know_tag.reranker import (
    RERANKER_INPUT_VERSION,
    RERANKER_INSTRUCTION,
    build_reranker_pairs,
)
from bio_know_tag.retrieval import format_label_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--question-id", action="append", dest="question_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--pair-batch-size", type=int, default=128)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            rows.append(value)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def existing_scores(path: Path) -> dict[tuple[str, str], dict]:
    completed = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            key = (str(value["question_id"]), str(value["label_id"]))
            completed[key] = value
    return completed


def stable_binary_score(true_logprob: float, false_logprob: float) -> float:
    difference = false_logprob - true_logprob
    if difference >= 0:
        exp_value = math.exp(-difference)
        return exp_value / (1.0 + exp_value)
    exp_value = math.exp(difference)
    return 1.0 / (1.0 + exp_value)


def main() -> int:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold must be between 0 and 1")
    for name in ("pair_batch_size", "tensor_parallel_size", "max_model_len", "max_num_seqs"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")
    if not args.model_path.is_dir():
        raise SystemExit(f"model path does not exist: {args.model_path}")

    selected_question_ids = set(args.question_ids or [])
    units = read_jsonl(args.units)
    if selected_question_ids:
        units = [
            unit
            for unit in units
            if str(unit.get("question_id")) in selected_question_ids
        ]
        found = {str(unit.get("question_id")) for unit in units}
        missing = selected_question_ids - found
        if missing:
            raise SystemExit(f"question IDs not found in units: {sorted(missing)}")
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        units = units[: args.limit]
    if not units:
        raise SystemExit("no units selected")

    unit_ids = {str(unit["question_id"]) for unit in units}
    candidate_rows = {
        str(row["question_id"]): row
        for row in read_jsonl(args.candidates)
        if str(row.get("question_id")) in unit_ids
    }
    missing_candidates = unit_ids - set(candidate_rows)
    if missing_candidates:
        raise SystemExit(
            f"question IDs not found in candidates: {sorted(missing_candidates)}"
        )
    labels_by_id = {
        str(row["label_id"]): row for row in read_jsonl(args.labels)
    }

    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-qwen3-reranker"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_version": RERANKER_INPUT_VERSION,
        "instruction": RERANKER_INSTRUCTION,
        "model_path": str(args.model_path),
        "units": str(args.units),
        "candidates": str(args.candidates),
        "labels": str(args.labels),
        "input_sha256": {
            "units": file_sha256(args.units),
            "candidates": file_sha256(args.candidates),
            "labels": file_sha256(args.labels),
        },
        "question_ids": sorted(selected_question_ids),
        "limit": args.limit,
        "threshold": args.threshold,
        "pair_batch_size": args.pair_batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
    }
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("run manifest mismatch; use a new run directory")
    else:
        write_json(manifest_path, manifest)

    all_pairs = []
    candidate_metadata = {}
    for unit in units:
        question_id = str(unit["question_id"])
        candidates = candidate_rows[question_id].get("candidates") or []
        pairs = build_reranker_pairs(unit, candidates, labels_by_id)
        all_pairs.extend(pairs)
        candidate_metadata[question_id] = {
            str(candidate["label_id"]): candidate for candidate in candidates
        }

    scores_path = run_dir / "scores.jsonl"
    completed = existing_scores(scores_path)
    pending = [
        pair
        for pair in all_pairs
        if (pair["question_id"], pair["label_id"]) not in completed
    ]

    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    if pending:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.inputs.data import TokensPrompt

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_path),
            padding_side="left",
            trust_remote_code=True,
        )
        llm = LLM(
            model=str(args.model_path),
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            enable_prefix_caching=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            trust_remote_code=True,
        )
        true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
        false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        sampling = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[true_token, false_token],
        )

        with scores_path.open("a", encoding="utf-8", newline="\n") as output:
            for offset in range(0, len(pending), args.pair_batch_size):
                batch = pending[offset : offset + args.pair_batch_size]
                messages = [
                    [
                        {
                            "role": "system",
                            "content": (
                                "Judge whether the Document meets the requirements "
                                "based on the Query and the Instruct provided. Note "
                                "that the answer can only be \"yes\" or \"no\"."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"<Instruct>: {RERANKER_INSTRUCTION}\n\n"
                                f"<Query>: {pair['query']}\n\n"
                                f"<Document>: {pair['document']}"
                            ),
                        },
                    ]
                    for pair in batch
                ]
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                prompts = []
                prompt_lengths = []
                for pair, ids in zip(batch, token_ids, strict=True):
                    total_length = len(ids) + len(suffix_tokens)
                    if total_length > args.max_model_len:
                        raise ValueError(
                            "reranker input would be truncated, violating DS alignment: "
                            f"{pair['question_id']}::{pair['label_id']} has "
                            f"{total_length} tokens > {args.max_model_len}"
                        )
                    prompts.append(TokensPrompt(prompt_token_ids=ids + suffix_tokens))
                    prompt_lengths.append(total_length)

                batch_started = time.monotonic()
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                batch_seconds = time.monotonic() - batch_started
                for pair, prompt_length, response in zip(
                    batch, prompt_lengths, outputs, strict=True
                ):
                    final_logprobs = response.outputs[0].logprobs[-1]
                    if true_token not in final_logprobs or false_token not in final_logprobs:
                        raise ValueError("yes/no tokens missing from vLLM logprobs")
                    true_logprob = float(final_logprobs[true_token].logprob)
                    false_logprob = float(final_logprobs[false_token].logprob)
                    record = {
                        "question_id": pair["question_id"],
                        "code": pair["code"],
                        "label_id": pair["label_id"],
                        "score": stable_binary_score(true_logprob, false_logprob),
                        "true_logprob": true_logprob,
                        "false_logprob": false_logprob,
                        "logit_difference": true_logprob - false_logprob,
                        "prompt_tokens": prompt_length,
                        "batch_seconds": batch_seconds,
                        "input_version": RERANKER_INPUT_VERSION,
                    }
                    output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                    output.write("\n")
                output.flush()
                print(
                    f"pairs={min(offset + len(batch), len(pending))}/{len(pending)} "
                    f"batch_seconds={batch_seconds:.3f}",
                    flush=True,
                )

    completed = existing_scores(scores_path)
    predictions_path = run_dir / "predictions.jsonl"
    selected_distribution = Counter()
    with predictions_path.open("w", encoding="utf-8", newline="\n") as output:
        for unit in units:
            question_id = str(unit["question_id"])
            rows = [
                completed[(pair["question_id"], pair["label_id"])]
                for pair in all_pairs
                if pair["question_id"] == question_id
            ]
            rows.sort(key=lambda row: (-float(row["score"]), row["code"]))
            selected_labels = []
            for row in rows:
                if float(row["score"]) < args.threshold:
                    continue
                label_id = str(row["label_id"])
                label = labels_by_id[label_id]
                candidate = candidate_metadata[question_id][label_id]
                selected_labels.append(
                    {
                        "label_id": label_id,
                        "label_name": label.get("label_name", ""),
                        "label_path": format_label_path(label.get("label_path")),
                        "score": row["score"],
                        "candidate_rank": candidate.get("candidate_rank")
                        or candidate.get("rank"),
                        "sources": candidate.get("sources", []),
                        "sparse_rank": candidate.get("sparse_rank"),
                        "dense_rank": candidate.get("dense_rank"),
                    }
                )
            selected_labels.sort(
                key=lambda row: (
                    int(row.get("candidate_rank") or 10**9),
                    row["label_id"],
                )
            )
            selected_distribution[str(len(selected_labels))] += 1
            prediction = {
                "question_id": question_id,
                "parent_id": unit.get("parent_id", question_id),
                "unit_type": unit.get("unit_type", ""),
                "selected_labels": selected_labels,
                "none_of_candidates": not selected_labels,
                "max_score": max((float(row["score"]) for row in rows), default=None),
                "threshold": args.threshold,
                "candidate_count": len(rows),
                "retrieval_version": candidate_rows[question_id].get(
                    "retrieval_version", ""
                ),
                "model": args.model_path.name,
                "input_version": RERANKER_INPUT_VERSION,
            }
            output.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True))
            output.write("\n")

    elapsed = time.monotonic() - started
    score_rows = list(completed.values())
    report = {
        "input_questions": len(units),
        "input_pairs": len(all_pairs),
        "processed_pairs": len(score_rows),
        "pending_pairs": len(all_pairs) - len(score_rows),
        "threshold": args.threshold,
        "selected_count_distribution": dict(sorted(selected_distribution.items())),
        "questions_with_no_selected_label": selected_distribution.get("0", 0),
        "mean_prompt_tokens_per_pair": (
            round(sum(int(row["prompt_tokens"]) for row in score_rows) / len(score_rows), 3)
            if score_rows
            else None
        ),
        "run_started_at": started_at,
        "run_wall_seconds": round(elapsed, 3),
        "pairs_per_second_this_run": (
            round(len(pending) / elapsed, 4) if elapsed else None
        ),
        "model_path": str(args.model_path),
        "input_version": RERANKER_INPUT_VERSION,
    }
    write_json(run_dir / "report.json", report)
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0 if report["pending_pairs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
