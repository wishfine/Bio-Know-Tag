#!/usr/bin/env python3
"""Run Laya multilingual as independent binary decisions for candidate Labels.

Each question is evaluated with one Laya ``noul`` question per candidate. The
question payload and candidate cards come from ``build_adjudication_inputs``,
which is also used by the DS adjudicator and the Qwen reranker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bio_know_tag.adjudication import build_adjudication_inputs
from bio_know_tag.laya_decision import (
    LAYA_INPUT_VERSION,
    build_laya_state_and_questions,
    laya_answers_to_scores,
    laya_noul_to_match_scores,
)
from bio_know_tag.retrieval import format_label_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument(
        "--head-max-len",
        type=int,
        default=1024,
        help="Laya instruction/option budget; 1024 preserves full biology Label cards.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-id", action="append", dest="question_ids")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    if args.max_len < 1 or args.head_max_len < 1:
        raise SystemExit("--max-len and --head-max-len must be positive")

    units = read_jsonl(args.units)
    selected_ids = {str(value) for value in (args.question_ids or [])}
    if selected_ids:
        units = [row for row in units if str(row.get("question_id")) in selected_ids]
        found = {str(row.get("question_id")) for row in units}
        missing = selected_ids - found
        if missing:
            raise SystemExit(f"question IDs not found in units: {sorted(missing)}")
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        units = units[: args.limit]
    if not units:
        raise SystemExit("no units selected")

    unit_ids = {str(row["question_id"]) for row in units}
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

    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_version": LAYA_INPUT_VERSION,
        "model_path": args.model_path,
        "subfolder": args.subfolder,
        "device": args.device,
        "units": str(args.units),
        "candidates": str(args.candidates),
        "labels": str(args.labels),
        "input_sha256": {
            "units": sha256(args.units),
            "candidates": sha256(args.candidates),
            "labels": sha256(args.labels),
        },
        "question_ids": sorted(selected_ids),
        "limit": args.limit,
        "threshold": args.threshold,
        "max_len": args.max_len,
        "head_max_len": args.head_max_len,
    }
    manifest_path = args.run_dir / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("run manifest mismatch; use a new run directory")
    else:
        write_json(manifest_path, manifest)

    predictions_path = args.run_dir / "predictions.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if predictions_path.exists():
        for row in read_jsonl(predictions_path):
            if row.get("question_id"):
                existing[str(row["question_id"])] = row

    try:
        import laya
    except ImportError as exc:
        raise SystemExit(
            "Laya is not installed. Install it in the active environment with: pip install laya"
        ) from exc

    agent = laya.load(args.model_path, device=args.device, subfolder=args.subfolder)
    agent.cfg["max_len"] = args.max_len
    agent.cfg["head_max_len"] = args.head_max_len

    selected_distribution = Counter()
    score_rows = 0
    processed = 0
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    with predictions_path.open("a", encoding="utf-8", newline="\n") as output:
        for unit in units:
            question_id = str(unit["question_id"])
            if question_id in existing:
                prediction = existing[question_id]
                selected_distribution[str(len(prediction.get("selected_labels") or []))] += 1
                continue

            candidates = candidate_rows[question_id].get("candidates") or []
            state, questions, code_map = build_laya_state_and_questions(
                unit, candidates, labels_by_id
            )
            result = agent.predict(state, questions)
            noul_scores = laya_answers_to_scores(
                result.get("answers") or {}, code_map
            )
            match_scores = laya_noul_to_match_scores(noul_scores)
            score_rows += len(noul_scores)

            candidate_by_id = {
                str(candidate["label_id"]): candidate for candidate in candidates
            }
            selected_labels = []
            all_scores = []
            for code, label_id in code_map.items():
                noul_score = noul_scores[code]
                match_score = match_scores[code]
                candidate = candidate_by_id[label_id]
                label = labels_by_id[label_id]
                all_scores.append(match_score)
                if match_score < args.threshold:
                    continue
                selected_labels.append(
                    {
                        "label_id": label_id,
                        "label_name": label.get("label_name", ""),
                        "label_path": format_label_path(label.get("label_path")),
                        "score": match_score,
                        "match_score": match_score,
                        "noul_score": noul_score,
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
                "scores": [
                    {
                        "code": code,
                        "label_id": label_id,
                        "score": match_scores[code],
                        "match_score": match_scores[code],
                        "noul_score": noul_scores[code],
                        "candidate_rank": (
                            candidate_by_id[label_id].get("candidate_rank")
                            or candidate_by_id[label_id].get("rank")
                        ),
                    }
                    for code, label_id in code_map.items()
                ],
                "selected_labels": selected_labels,
                "none_of_candidates": not selected_labels,
                "max_score": max(all_scores, default=None),
                "score_semantics": "match_score=noul=P(true); higher means the Label matches",
                "noul_semantics": "noul is Laya's true/match probability",
                "threshold": args.threshold,
                "candidate_count": len(code_map),
                "retrieval_version": candidate_rows[question_id].get(
                    "retrieval_version", ""
                ),
                "model": args.model_path,
                "input_version": LAYA_INPUT_VERSION,
                "laya_usage": result.get("usage", {}),
            }
            output.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True))
            output.write("\n")
            output.flush()
            existing[question_id] = prediction
            processed += 1
            print(
                f"questions={processed}/{len(units)} pairs={score_rows} "
                f"selected={len(selected_labels)}",
                flush=True,
            )

    elapsed = time.monotonic() - started
    report = {
        "input_questions": len(units),
        "processed_questions_this_run": processed,
        "completed_questions": len(existing),
        "input_pairs": sum(
            len(candidate_rows[str(unit["question_id"])].get("candidates") or [])
            for unit in units
        ),
        "processed_pairs_this_run": score_rows,
        "threshold": args.threshold,
        "score_semantics": "match_score=noul=P(true); higher means the Label matches",
        "noul_semantics": "noul is Laya's true/match probability",
        "noul_match_threshold": args.threshold,
        "max_len": args.max_len,
        "head_max_len": args.head_max_len,
        "selected_count_distribution": dict(sorted(selected_distribution.items())),
        "questions_with_no_selected_label": selected_distribution.get("0", 0),
        "run_started_at": started_at,
        "run_wall_seconds": round(elapsed, 3),
        "questions_per_second_this_run": (
            round(processed / elapsed, 4) if elapsed else None
        ),
        "model_path": args.model_path,
        "input_version": LAYA_INPUT_VERSION,
    }
    write_json(args.run_dir / "report.json", report)
    print(json.dumps({"run_dir": str(args.run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
