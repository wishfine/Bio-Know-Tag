"""Analyze repeated name-only versus full-definition judgments."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ARMS = ("name_1", "name_2", "definition_1", "definition_2")


def _read_jsonl(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number} is not an object")
                yield value


def _load_unique(path: str | Path, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        value = str(row.get(key) or "")
        if not value or value in rows:
            raise ValueError(f"missing or duplicate {key}: {value!r} in {path}")
        rows[value] = row
    return rows


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _direction(left: bool, right: bool) -> str:
    if left == right:
        return "same_match" if left else "same_nonmatch"
    return "false_to_true" if right else "true_to_false"


def analyze_repeated_ablation(
    *,
    sample_path: str | Path,
    result_paths: dict[str, str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    if set(result_paths) != set(ARMS):
        raise ValueError(f"Expected result arms: {ARMS}")
    sample = _load_unique(sample_path, "pair_id")
    results = {arm: _load_unique(result_paths[arm], "task_id") for arm in ARMS}
    manifests = {}
    for arm in ARMS:
        manifest_path = Path(result_paths[arm]).with_name("run_manifest.json")
        if manifest_path.exists():
            manifests[arm] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifests and len(manifests) != len(ARMS):
        raise ValueError("all four run manifests must be present together")
    if manifests:
        common_fields = ("model", "prompt_version", "max_batch_size", "char_budget", "max_tokens", "limit")
        baseline = {field: manifests["name_1"].get(field) for field in common_fields}
        for arm, manifest in manifests.items():
            current = {field: manifest.get(field) for field in common_fields}
            if current != baseline:
                raise ValueError(f"{arm} run manifest differs from name_1")
        for first, second in (("name_1", "name_2"), ("definition_1", "definition_2")):
            if manifests[first].get("input_sha256") != manifests[second].get("input_sha256"):
                raise ValueError(f"{first} and {second} input hashes differ")
    expected = set(sample)
    for arm, rows in results.items():
        unknown = set(rows) - expected
        if unknown:
            raise ValueError(f"{arm} contains {len(unknown)} unknown task IDs")

    counts = Counter()
    strata: dict[str, Counter[str]] = defaultdict(Counter)
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    complete_ids = expected.intersection(*(set(rows) for rows in results.values()))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pair_path = output / "per_pair.jsonl"
    with pair_path.open("w", encoding="utf-8") as handle:
        for pair_id in sorted(complete_ids):
            task = sample[pair_id]
            arm_rows = {arm: results[arm][pair_id] for arm in ARMS}
            for arm, row in arm_rows.items():
                if str(row.get("question_id")) != str(task.get("question_id")) or str(
                    row.get("label_id")
                ) != str(task.get("label_id")):
                    raise ValueError(f"{arm} metadata mismatch for {pair_id}")
            score = {arm: float(row["relevance_score"]) for arm, row in arm_rows.items()}
            if any(not 0.0 <= value <= 1.0 for value in score.values()):
                raise ValueError(f"invalid score for {pair_id}")
            match = {arm: value >= 0.70 for arm, value in score.items()}
            name_stable = match["name_1"] == match["name_2"]
            definition_stable = match["definition_1"] == match["definition_2"]
            first_change = _direction(match["name_1"], match["definition_1"])
            second_change = _direction(match["name_2"], match["definition_2"])
            robust_change = (
                first_change
                if name_stable and definition_stable and first_change != "same_match"
                and first_change != "same_nonmatch"
                else None
            )
            stratum = str(task.get("stratum") or "unknown")
            source = str(task.get("sample_source") or "unknown")
            image_missing = bool(task.get("image_context_missing"))
            analysis_missing = not bool(str(task.get("analysis") or "").strip())
            row = {
                "pair_id": pair_id,
                "question_id": task["question_id"],
                "label_id": task["label_id"],
                "label_name": task.get("label_name", ""),
                "stratum": stratum,
                "sample_source": source,
                "image_context_missing": image_missing,
                "analysis_missing": analysis_missing,
                "scores": score,
                "matches": match,
                "name_repeat_stable": name_stable,
                "definition_repeat_stable": definition_stable,
                "first_change": first_change,
                "second_change": second_change,
                "robust_change": robust_change,
                "mean_score_delta": round(
                    (score["definition_1"] + score["definition_2"]
                     - score["name_1"] - score["name_2"]) / 2, 6
                ),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["complete"] += 1
            counts["name_repeat_changed"] += not name_stable
            counts["definition_repeat_changed"] += not definition_stable
            counts["ablation_changed_first"] += first_change in ("false_to_true", "true_to_false")
            counts["ablation_changed_second"] += second_change in ("false_to_true", "true_to_false")
            counts["robust_false_to_true"] += robust_change == "false_to_true"
            counts["robust_true_to_false"] += robust_change == "true_to_false"
            counts["unstable_either_arm"] += not (name_stable and definition_stable)
            keys = (f"stratum:{stratum}", f"source:{source}",
                    f"image_missing:{image_missing}", f"analysis_missing:{analysis_missing}")
            for key in keys:
                strata[key]["complete"] += 1
                strata[key]["robust_changed"] += robust_change is not None
                strata[key]["unstable_either_arm"] += not (name_stable and definition_stable)
            label = by_label[str(task["label_id"])]
            label["complete"] += 1
            label["robust_false_to_true"] += robust_change == "false_to_true"
            label["robust_true_to_false"] += robust_change == "true_to_false"
            label["unstable_either_arm"] += not (name_stable and definition_stable)

    with (output / "per_label.jsonl").open("w", encoding="utf-8") as handle:
        for label_id, item in sorted(by_label.items()):
            handle.write(json.dumps({"label_id": label_id, **item}, ensure_ascii=False) + "\n")
    report = {
        "sample_pairs": len(sample),
        "complete_pairs": counts["complete"],
        "missing_by_arm": {arm: len(expected - set(rows)) for arm, rows in results.items()},
        "counts": dict(counts),
        "strata": {key: dict(value) for key, value in sorted(strata.items())},
        "input_sha256": {"sample": _sha256(sample_path), **{
            arm: _sha256(path) for arm, path in result_paths.items()
        }},
        "run_manifests": manifests,
        "interpretation": "A/B flips alone are not definition effects. Robust flips require both same-condition repeats to agree.",
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
