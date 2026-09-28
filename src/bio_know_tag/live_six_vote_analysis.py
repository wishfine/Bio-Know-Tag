"""Snapshot analysis of Qwen and DS's live append-only six-vote evidence."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VOTES = ("qwen1", "qwen2", "qwen3", "ds1", "ds2", "ds3")
MODELS = {"qwen": VOTES[:3], "ds": VOTES[3:]}


def _load_vote(
    path: Path, *, max_success: int | None = None,
) -> tuple[dict[str, frozenset[str]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    selected_by_question: dict[str, frozenset[str]] = {}
    counts: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    with path.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                counts["incomplete_tail_ignored"] += 1
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts["malformed_complete_lines"] += 1
                continue
            if not isinstance(row, dict):
                counts["malformed_complete_lines"] += 1
                continue
            counts["evidence_rows"] += 1
            finish_reasons[str(row.get("finish_reason"))] += 1
            question_id = str(row.get("question_id") or "")
            if not question_id:
                counts["missing_question_id"] += 1
                continue
            if row.get("error"):
                errors[str(row["error"]).split(":", 1)[0]] += 1
                continue
            parsed = row.get("parsed_response")
            code_map = row.get("candidate_code_map")
            if not isinstance(parsed, dict) or not isinstance(code_map, dict):
                counts["invalid_success_rows"] += 1
                continue
            codes = parsed.get("selected")
            if not isinstance(codes, list) or any(
                not isinstance(code, str) or code not in code_map for code in codes
            ):
                counts["invalid_success_rows"] += 1
                continue
            label_ids = frozenset(str(code_map[code]) for code in codes)
            if question_id in selected_by_question:
                counts["repeat_successes"] += 1
            selected_by_question[question_id] = label_ids
            if max_success is not None and len(selected_by_question) > max_success:
                raise ValueError(
                    f"{path.parent.name} exceeds in-memory snapshot memory cap of "
                    f"{max_success:,} successful questions"
                )
    return selected_by_question, {
        **dict(counts),
        "successful_questions": len(selected_by_question),
        "errors": sum(errors.values()),
        "error_types": dict(errors.most_common()),
        "finish_reasons": dict(finish_reasons.most_common()),
    }


def _majority(sets: tuple[frozenset[str], ...]) -> frozenset[str]:
    counts = Counter(label_id for chosen in sets for label_id in chosen)
    return frozenset(label_id for label_id, votes in counts.items() if votes >= 2)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _sample(
    bucket: list[dict[str, Any]], item: dict[str, Any],
    seen: int, limit: int, rng: random.Random,
) -> None:
    if len(bucket) < limit:
        bucket.append(item)
    elif limit:
        replace = rng.randrange(seen)
        if replace < limit:
            bucket[replace] = item


def _label_names(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    labels = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                labels[str(row["label_id"])] = str(row.get("label_name") or "")
    return labels


def _check_manifests(root: Path) -> dict[str, Any]:
    parent_path = root / "run_manifest.json"
    if not parent_path.exists():
        return {"controller_manifest_found": False, "vote_manifests_checked": 0}
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    expected_hashes = parent.get("input_sha256") or {}
    checked = 0
    for vote in VOTES:
        path = root / "votes" / vote / "run_manifest.json"
        if not path.exists():
            raise ValueError(f"missing vote manifest for {vote}: {path}")
        child = json.loads(path.read_text(encoding="utf-8"))
        actual_hashes = child.get("input_sha256") or {}
        for field, expected in expected_hashes.items():
            if actual_hashes.get(field) != expected:
                raise ValueError(f"{vote} {field} input SHA256 differs from controller manifest")
        model_group = "qwen" if vote.startswith("qwen") else "ds"
        service = (parent.get("services") or {}).get(model_group) or {}
        if service.get("model") and child.get("model") != service["model"]:
            raise ValueError(f"{vote} model differs from controller manifest")
        thinking = (parent.get("thinking_override") or {}).get(model_group)
        request_config = child.get("request_config") or {}
        if thinking is not None and request_config.get("enable_thinking") is not thinking:
            raise ValueError(f"{vote} thinking override differs from controller manifest")
        checked += 1
    return {"controller_manifest_found": True, "vote_manifests_checked": checked}


def analyze_live_six_vote(
    run_dir: str | Path,
    *,
    labels_path: str | Path | None = None,
    sample_limit: int = 20,
    max_success_per_vote: int = 500000,
) -> dict[str, Any]:
    if sample_limit < 0:
        raise ValueError("sample_limit must be nonnegative")
    if max_success_per_vote < 1:
        raise ValueError("max_success_per_vote must be positive")
    root = Path(run_dir)
    manifest_alignment = _check_manifests(root)
    labels = _label_names(labels_path)
    data: dict[str, dict[str, frozenset[str]]] = {}
    vote_reports: dict[str, dict[str, Any]] = {}
    for vote in VOTES:
        data[vote], vote_reports[vote] = _load_vote(
            root / "votes" / vote / "evidence.jsonl",
            max_success=max_success_per_vote,
        )
    common = set.intersection(*(set(rows) for rows in data.values()))
    count = len(common)
    within: dict[str, Counter[str]] = {model: Counter() for model in MODELS}
    cross: Counter[str] = Counter()
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    pairwise: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = {
        "stable_cross_model_difference": [],
        "qwen_volatility": [],
        "ds_volatility": [],
    }
    sample_seen: Counter[str] = Counter()
    rng = random.Random(20260928)
    for question_id in sorted(common):
        qs = tuple(data[vote][question_id] for vote in MODELS["qwen"])
        ds = tuple(data[vote][question_id] for vote in MODELS["ds"])
        qm, dm = _majority(qs), _majority(ds)
        selections = {"qwen": qs, "ds": ds}
        for model, sets in selections.items():
            stats = within[model]
            distinct = len(set(sets))
            stats["questions"] += 1
            stats[{1: "all_same", 2: "two_same_one_different", 3: "all_three_different"}[distinct]] += 1
            stats["all_empty"] += int(not any(sets))
            stats["first_two_same"] += int(sets[0] == sets[1])
            stats["first_two_same_but_third_differs"] += int(
                sets[0] == sets[1] and sets[2] != sets[0]
            )
            if sets[0] != sets[1]:
                stats["first_two_different"] += 1
                stats["third_agrees_first_when_split"] += int(sets[2] == sets[0])
                stats["third_agrees_second_when_split"] += int(sets[2] == sets[1])
                stats["third_differs_both_when_split"] += int(sets[2] not in (sets[0], sets[1]))
            label_union = set().union(*sets)
            label_intersection = set.intersection(*(set(chosen) for chosen in sets))
            stats["any_label_instability"] += int(label_union != label_intersection)
            stats["majority_selected_assignments"] += len(_majority(sets))
            for first in range(3):
                for second in range(first + 1, 3):
                    key = f"{model}{first + 1}-{model}{second + 1}"
                    pairwise[f"{key}:equal"] += int(sets[first] == sets[second])
                    pairwise[f"{key}:jaccard_sum"] += _jaccard(sets[first], sets[second])
            if distinct > 1:
                key = f"{model}_volatility"
                sample_seen[key] += 1
                _sample(samples[key], {
                    "question_id": question_id,
                    "vote_label_ids": [sorted(chosen) for chosen in sets],
                }, sample_seen[key], sample_limit, rng)

        cross["questions"] += 1
        cross["majority_same"] += int(qm == dm)
        cross["majority_different"] += int(qm != dm)
        cross["both_majority_empty"] += int(not qm and not dm)
        cross["qwen_only_additions"] += int(bool(qm - dm) and not (dm - qm))
        cross["ds_only_additions"] += int(bool(dm - qm) and not (qm - dm))
        cross["bidirectional_difference"] += int(bool(qm - dm) and bool(dm - qm))
        cross["qwen_majority_assignments"] += len(qm)
        cross["ds_majority_assignments"] += len(dm)
        cross["qwen_only_assignments"] += len(qm - dm)
        cross["ds_only_assignments"] += len(dm - qm)
        cross["shared_assignments"] += len(qm & dm)
        cross["majority_jaccard_sum"] += _jaccard(qm, dm)
        both_unanimous = len(set(qs)) == len(set(ds)) == 1
        cross["both_unanimous"] += int(both_unanimous)
        cross["both_unanimous_same"] += int(both_unanimous and qm == dm)
        cross["both_unanimous_different"] += int(both_unanimous and qm != dm)
        if both_unanimous and qm != dm:
            key = "stable_cross_model_difference"
            sample_seen[key] += 1
            _sample(samples[key], {
                "question_id": question_id,
                "qwen_label_ids": sorted(qm),
                "ds_label_ids": sorted(dm),
            }, sample_seen[key], sample_limit, rng)
        for label_id in set().union(*qs, *ds):
            stats = per_label[label_id]
            q_votes = sum(label_id in chosen for chosen in qs)
            d_votes = sum(label_id in chosen for chosen in ds)
            stats["seen_in_six_votes"] += 1
            stats["qwen_any"] += int(q_votes > 0)
            stats["ds_any"] += int(d_votes > 0)
            stats["qwen_unstable"] += int(0 < q_votes < 3)
            stats["ds_unstable"] += int(0 < d_votes < 3)
            stats["qwen_3_of_3"] += int(q_votes == 3)
            stats["ds_3_of_3"] += int(d_votes == 3)
            stats["qwen_majority_only"] += int(q_votes >= 2 and d_votes < 2)
            stats["ds_majority_only"] += int(d_votes >= 2 and q_votes < 2)
            stats["both_majority"] += int(q_votes >= 2 and d_votes >= 2)
        for qi, qset in enumerate(qs, 1):
            for di, dset in enumerate(ds, 1):
                key = f"qwen{qi}-ds{di}"
                pairwise[f"{key}:equal"] += int(qset == dset)
                pairwise[f"{key}:jaccard_sum"] += _jaccard(qset, dset)

    pairwise_rows = {
        key: {
            "equal": pairwise[f"{key}:equal"],
            "equal_rate": round(pairwise[f"{key}:equal"] / count, 6) if count else None,
            "mean_jaccard": round(pairwise[f"{key}:jaccard_sum"] / count, 6) if count else None,
        }
        for key in sorted({key.split(":", 1)[0] for key in pairwise})
    }
    for model in MODELS:
        stats = within[model]
        stats["all_same_rate"] = round(stats["all_same"] / count, 6) if count else None
        stats["any_label_instability_rate"] = round(stats["any_label_instability"] / count, 6) if count else None
        stats["first_two_same_rate"] = round(stats["first_two_same"] / count, 6) if count else None
    cross["majority_same_rate"] = round(cross["majority_same"] / count, 6) if count else None
    cross["both_unanimous_different_rate"] = round(cross["both_unanimous_different"] / count, 6) if count else None
    cross["majority_mean_jaccard"] = round(cross["majority_jaccard_sum"] / count, 6) if count else None
    cross.pop("majority_jaccard_sum", None)
    for bucket in samples.values():
        for item in bucket:
            for field in ("qwen_label_ids", "ds_label_ids"):
                if field in item:
                    item[field] = [{"label_id": value, "label_name": labels.get(value, "")} for value in item[field]]
            if "vote_label_ids" in item:
                item["vote_label_ids"] = [
                    [{"label_id": value, "label_name": labels.get(value, "")} for value in values]
                    for values in item["vote_label_ids"]
                ]
    label_rows = [
        {"label_id": label_id, "label_name": labels.get(label_id, ""), **dict(stats)}
        for label_id, stats in per_label.items()
    ]
    for row in label_rows:
        row["qwen_unstable_given_seen_rate"] = round(
            row.get("qwen_unstable", 0) / row["qwen_any"], 6
        ) if row.get("qwen_any") else None
        row["ds_unstable_given_seen_rate"] = round(
            row.get("ds_unstable", 0) / row["ds_any"], 6
        ) if row.get("ds_any") else None
        row["cross_majority_difference_rate_given_seen"] = round(
            (row.get("qwen_majority_only", 0) + row.get("ds_majority_only", 0))
            / row["seen_in_six_votes"], 6
        )
    label_rows.sort(key=lambda item: (-(item.get("qwen_unstable", 0) + item.get("ds_unstable", 0)), item["label_id"]))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(root),
        "snapshot_warning": "Live files were read one after another; only questions successful in all six votes form comparison denominators. This is not a quality/gold-label evaluation.",
        "max_success_per_vote": max_success_per_vote,
        "votes": vote_reports,
        "manifest_alignment": manifest_alignment,
        "all_six_common_questions": count,
        "within_model": {model: dict(stats) for model, stats in within.items()},
        "cross_model": dict(cross),
        "pairwise": pairwise_rows,
        "labels": label_rows,
        "samples": samples,
    }
