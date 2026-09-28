"""Generic Q0–Q3 by D0–D3 analysis for candidate-exposed labels."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bio_know_tag.adjudication import CANDIDATE_ORDER_VERSION
from bio_know_tag.live_six_vote_analysis import VOTES, _check_manifests, _load_vote


PATTERNS = tuple(f"Q{q}/D{d}" for q in range(4) for d in range(4))
REVIEW_PATTERNS = ("Q3/D3", "Q3/D0", "Q0/D3", "Q2/D1", "Q1/D2", "Q0/D0")
QUALITY_FLAGS = (
    "image_context_missing", "parent_context_missing", "duplicate_label_conflict",
)


def _load_labels(path: str | Path) -> dict[str, dict[str, Any]]:
    labels = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                labels[str(row["label_id"])] = row
    return labels


def _expected_hashes(root: Path) -> dict[str, str]:
    path = root / "run_manifest.json"
    if not path.is_file():
        raise ValueError(f"missing controller manifest: {path}")
    hashes = json.loads(path.read_text(encoding="utf-8")).get("input_sha256") or {}
    if any(not isinstance(hashes.get(name), str) or len(hashes[name]) != 64 for name in ("units", "candidates", "labels")):
        raise ValueError("controller manifest requires units, candidates and labels SHA256 hashes")
    return hashes


def _candidate_signature(question_id: str, candidate_ids: list[str]) -> bytes:
    ordered = sorted(candidate_ids, key=lambda label_id: hashlib.sha256(
        f"{question_id}\0{label_id}\0{CANDIDATE_ORDER_VERSION}".encode("utf-8")
    ).digest())
    return hashlib.blake2b("\0".join(ordered).encode("utf-8"), digest_size=16).digest()


def _rank_band(rank: Any) -> str:
    try:
        value = int(rank)
    except (TypeError, ValueError):
        return "unknown"
    if value <= 0:
        return "unknown"
    if value <= 5:
        return "1-5"
    if value <= 10:
        return "6-10"
    if value <= 25:
        return "11-25"
    return "26+"


def _reservoir(
    bucket: list[dict[str, Any]], item: dict[str, Any],
    seen: int, limit: int, rng: random.Random,
) -> None:
    if len(bucket) < limit:
        bucket.append(item)
    elif limit:
        replacement = rng.randrange(seen)
        if replacement < limit:
            bucket[replacement] = item


def analyze_vote_pattern_cube(
    votes_root: str | Path,
    *,
    candidates_path: str | Path,
    units_path: str | Path,
    labels_path: str | Path,
    sample_per_pattern: int = 10,
    max_success_per_vote: int = 500000,
    positive_per_label_path: str | Path | None = None,
    boundary_assessments_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one reusable 16-state cube; never infer teacher truth from votes."""
    if sample_per_pattern < 0 or max_success_per_vote < 1:
        raise ValueError("sample_per_pattern must be nonnegative; max_success_per_vote positive")
    root = Path(votes_root)
    expected = _expected_hashes(root)
    manifest_alignment = _check_manifests(root)
    labels = _load_labels(labels_path)
    if hashlib.sha256(Path(labels_path).read_bytes()).hexdigest() != expected["labels"]:
        raise ValueError("labels SHA256 differs from vote controller manifest")
    vote_maps: dict[str, dict[str, frozenset[str]]] = {}
    vote_reports = {}
    vote_candidate_signatures: dict[str, dict[str, bytes]] = {}
    for vote in VOTES:
        signatures: dict[str, bytes] = {}
        vote_maps[vote], vote_reports[vote] = _load_vote(
            root / "votes" / vote / "evidence.jsonl",
            max_success=max_success_per_vote,
            candidate_signatures_out=signatures,
        )
        vote_candidate_signatures[vote] = signatures
    common = set.intersection(*(set(rows) for rows in vote_maps.values()))
    pattern_counts: Counter[str] = Counter()
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    exposure: Counter[str] = Counter()
    by_unit_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_quality_flag: dict[str, Counter[str]] = defaultdict(Counter)
    label_by_unit_type: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    label_by_quality_flag: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    label_by_rank_band: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    label_by_legacy_assignment: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    label_by_candidate_source: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    by_rank_band: dict[str, Counter[str]] = defaultdict(Counter)
    by_legacy_assignment: dict[str, Counter[str]] = defaultdict(Counter)
    by_candidate_source: dict[str, Counter[str]] = defaultdict(Counter)
    samples: dict[str, list[dict[str, Any]]] = {pattern: [] for pattern in PATTERNS}
    sample_seen: Counter[str] = Counter()
    label_samples: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    label_sample_seen: Counter[tuple[str, str]] = Counter()
    rng = random.Random(20260928)
    seen_common: set[str] = set()
    candidate_digest = hashlib.sha256()
    unit_digest = hashlib.sha256()
    candidate_rows_scanned = 0
    unit_rows_scanned = 0

    with Path(candidates_path).open("rb") as candidate_file, Path(units_path).open("rb") as unit_file:
        for candidate_line, unit_line in itertools.zip_longest(candidate_file, unit_file):
            if candidate_line is None or unit_line is None:
                raise ValueError("candidate and unit JSONL row counts differ")
            candidate_digest.update(candidate_line)
            unit_digest.update(unit_line)
            if not candidate_line.strip() and not unit_line.strip():
                continue
            candidate = json.loads(candidate_line)
            unit = json.loads(unit_line)
            candidate_rows_scanned += 1
            unit_rows_scanned += 1
            qid = str(unit.get("question_id") or "")
            if not qid or qid != str(candidate.get("question_id") or ""):
                raise ValueError(f"unit/candidate question_id mismatch at row {unit_rows_scanned}")
            if qid not in common:
                continue
            if qid in seen_common:
                raise ValueError(f"duplicate question_id in frozen candidate file: {qid}")
            seen_common.add(qid)
            qs = [vote_maps[vote][qid] for vote in VOTES[:3]]
            ds = [vote_maps[vote][qid] for vote in VOTES[3:]]
            selected_union = frozenset.union(*qs, *ds)
            candidate_items = candidate.get("candidates")
            if not isinstance(candidate_items, list):
                raise ValueError(f"missing candidate list for {qid}")
            candidate_ids = [str(item.get("label_id") or "") for item in candidate_items]
            if len(candidate_ids) != len(set(candidate_ids)) or any(not value or value not in labels for value in candidate_ids):
                raise ValueError(f"duplicate or unknown candidate label for {qid}")
            if not selected_union.issubset(candidate_ids):
                raise ValueError(f"selected label not exposed as candidate for {qid}")
            signature = _candidate_signature(qid, candidate_ids)
            for vote in VOTES:
                if vote_candidate_signatures[vote][qid] != signature:
                    raise ValueError(f"{vote} candidate_code_map differs from sidecar candidates for {qid}")
            unit_type = str(unit.get("unit_type") or "unknown")
            flags = unit.get("flags") or {}
            if not isinstance(flags, dict):
                flags = {}
            legacy = {str(value) for value in unit.get("legacy_knw_ids") or []}
            for item, label_id in zip(candidate_items, candidate_ids):
                qvotes = sum(label_id in chosen for chosen in qs)
                dvotes = sum(label_id in chosen for chosen in ds)
                pattern = f"Q{qvotes}/D{dvotes}"
                pattern_counts[pattern] += 1
                by_label[label_id][pattern] += 1
                exposure[label_id] += 1
                by_unit_type[unit_type][pattern] += 1
                label_by_unit_type[label_id][unit_type][pattern] += 1
                for name in QUALITY_FLAGS:
                    if flags.get(name):
                        by_quality_flag[name][pattern] += 1
                        label_by_quality_flag[label_id][name][pattern] += 1
                rank_band = _rank_band(item.get("candidate_rank"))
                by_rank_band[rank_band][pattern] += 1
                label_by_rank_band[label_id][rank_band][pattern] += 1
                legacy_status = "legacy_id_present" if label_id in legacy else "legacy_id_absent"
                by_legacy_assignment[legacy_status][pattern] += 1
                label_by_legacy_assignment[label_id][legacy_status][pattern] += 1
                source = "+".join(sorted(str(s) for s in item.get("sources") or [])) or "unknown"
                by_candidate_source[source][pattern] += 1
                label_by_candidate_source[label_id][source][pattern] += 1
                sample_seen[pattern] += 1
                sample = {
                    "question_id": qid,
                    "label_id": label_id,
                    "label_name": str(labels[label_id].get("label_name") or ""),
                    "unit_type": unit_type,
                    "candidate_rank": item.get("candidate_rank"),
                    "candidate_sources": item.get("sources"),
                    "legacy_id_present": label_id in legacy,
                    "flags": {name: bool(flags.get(name)) for name in QUALITY_FLAGS},
                    "qwen_votes": qvotes,
                    "ds_votes": dvotes,
                    "pattern": pattern,
                    "stem": str(unit.get("stem") or "")[:500],
                    "parent_stem": str(unit.get("parent_stem") or "")[:500],
                    "analysis": str(unit.get("analysis") or "")[:800],
                }
                _reservoir(samples[pattern], sample, sample_seen[pattern], sample_per_pattern, rng)
                if pattern in REVIEW_PATTERNS:
                    key = (label_id, pattern)
                    label_sample_seen[key] += 1
                    _reservoir(label_samples[label_id][pattern], sample, label_sample_seen[key], min(sample_per_pattern, 3), rng)

    if seen_common != common:
        raise ValueError(f"{len(common - seen_common)} six-vote successful questions missing from candidates/units")
    if candidate_digest.hexdigest() != expected["candidates"]:
        raise ValueError("candidates SHA256 differs from vote controller manifest")
    if unit_digest.hexdigest() != expected["units"]:
        raise ValueError("units SHA256 differs from vote controller manifest")
    if sum(pattern_counts.values()) != sum(exposure.values()):
        raise AssertionError("every candidate exposure must have exactly one pattern")

    priors: dict[str, dict[str, Any]] = defaultdict(dict)
    for path, name, fields in (
        (positive_per_label_path, "positive_coverage_weak_supervision", (
            "completed", "match_rate", "zero_rate", "weak_related_rate", "sample_tier", "issue_flags", "preliminary_grade",
        )),
        (boundary_assessments_path, "hard_negative_weak_supervision", (
            "hard_negative_count", "hard_negative_false_accept_rate", "final_screen",
        )),
    ):
        if path is None:
            continue
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    label_id = str(row.get("label_id") or "")
                    if label_id in labels:
                        priors[label_id][name] = {field: row.get(field) for field in fields}

    label_rows = []
    for label_id, label in labels.items():
        counts = by_label[label_id]
        exposed = exposure[label_id]
        any_selected = exposed - counts["Q0/D0"]
        strong_conflict = counts["Q3/D0"] + counts["Q0/D3"]
        row = {
            "label_id": label_id,
            "label_name": str(label.get("label_name") or ""),
            "label_path": label.get("label_path"),
            "candidate_exposure": exposed,
            "any_vote_selected": any_selected,
            "patterns": {pattern: counts[pattern] for pattern in PATTERNS},
            "q3d3_given_exposure_rate": round(counts["Q3/D3"] / exposed, 6) if exposed else None,
            "q3d3_given_any_selected_rate": round(counts["Q3/D3"] / any_selected, 6) if any_selected else None,
            "stable_cross_model_conflict_count": strong_conflict,
            "stable_cross_model_conflict_given_exposure_rate": round(strong_conflict / exposed, 6) if exposed else None,
            "boundary_split_count": counts["Q2/D1"] + counts["Q1/D2"],
            "by_unit_type": {
                unit_type: {pattern: values[pattern] for pattern in PATTERNS}
                for unit_type, values in sorted(label_by_unit_type[label_id].items())
            },
            "by_quality_flag": {name: {pattern: values[pattern] for pattern in PATTERNS} for name, values in sorted(label_by_quality_flag[label_id].items())},
            "by_candidate_rank_band": {name: {pattern: values[pattern] for pattern in PATTERNS} for name, values in sorted(label_by_rank_band[label_id].items())},
            "by_legacy_assignment_weak_supervision": {name: {pattern: values[pattern] for pattern in PATTERNS} for name, values in sorted(label_by_legacy_assignment[label_id].items())},
            "by_candidate_source": {name: {pattern: values[pattern] for pattern in PATTERNS} for name, values in sorted(label_by_candidate_source[label_id].items())},
            "prior_model_evidence_not_gold": priors.get(label_id, {}),
            "teacher_calibration_status": "NO_TEACHER_GOLD",
            "representative_samples": {pattern: label_samples[label_id].get(pattern, []) for pattern in REVIEW_PATTERNS},
        }
        label_rows.append(row)
    label_rows.sort(key=lambda row: row["label_id"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "votes_root": str(root),
        "manifest_alignment": manifest_alignment,
        "common_successful_questions": len(common),
        "candidate_rows_scanned": candidate_rows_scanned,
        "unit_rows_scanned": unit_rows_scanned,
        "candidate_label_exposures": sum(exposure.values()),
        "pattern_counts": {pattern: pattern_counts[pattern] for pattern in PATTERNS},
        "by_unit_type": {
            name: {pattern: counter[pattern] for pattern in PATTERNS}
            for name, counter in sorted(by_unit_type.items())
        },
        "by_quality_flag": {
            name: {pattern: by_quality_flag[name][pattern] for pattern in PATTERNS}
            for name in QUALITY_FLAGS
        },
        "by_candidate_rank_band": {
            name: {pattern: counter[pattern] for pattern in PATTERNS}
            for name, counter in sorted(by_rank_band.items())
        },
        "by_legacy_assignment_weak_supervision": {
            name: {pattern: counter[pattern] for pattern in PATTERNS}
            for name, counter in sorted(by_legacy_assignment.items())
        },
        "by_candidate_source": {
            name: {pattern: counter[pattern] for pattern in PATTERNS}
            for name, counter in sorted(by_candidate_source.items())
        },
        "votes": vote_reports,
        "labels": label_rows,
        "samples": samples,
        "warning": "Pattern counts measure vote behavior, not correctness. Q0/D0 is defined only for candidate-exposed labels on all-six successful questions. Previous DS coverage and old knw_ids are weak supervision, not teacher gold.",
    }
