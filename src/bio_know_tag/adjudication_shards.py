"""Split aligned units and candidate rows for bounded-memory adjudication."""

from __future__ import annotations

import json
import hashlib
import shutil
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Iterator


def _rows(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield row


def shard_adjudication_inputs(
    units_path: str | Path,
    candidates_path: str | Path,
    output_dir: str | Path,
    *,
    shard_size: int = 10_000,
) -> dict[str, Any]:
    """Publish aligned shards only after every source row has been validated."""
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"shard output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(mkdtemp(prefix=f".{output.name}.sharding-", dir=output.parent))
    try:
        report = _build_shards(units_path, candidates_path, work, output, shard_size)
        if output.exists():
            raise ValueError(f"shard output appeared during build: {output}")
        work.replace(output)
        return report
    finally:
        if work.exists():
            shutil.rmtree(work)


def _build_shards(
    units_path: str | Path,
    candidates_path: str | Path,
    output: Path,
    published_output: Path,
    shard_size: int,
) -> dict[str, Any]:
    """Write into a private sibling directory; caller publishes on success."""
    output.mkdir(parents=True, exist_ok=True)
    shards_dir = output / "shards"
    shards_dir.mkdir(exist_ok=True)

    counts: Counter[str] = Counter()
    shards: list[dict[str, Any]] = []
    units_out = candidates_out = None
    try:
        for index, (unit, candidate) in enumerate(
            zip_longest(_rows(units_path), _rows(candidates_path)), 1
        ):
            if unit is None or candidate is None:
                raise ValueError(f"unit/candidate row counts differ at row {index}")
            question_id = str(unit.get("question_id") or "")
            if not question_id or question_id != str(candidate.get("question_id") or ""):
                raise ValueError(f"question_id mismatch at row {index}")
            if (index - 1) % shard_size == 0:
                if units_out is not None:
                    units_out.close()
                    candidates_out.close()
                shard_number = len(shards) + 1
                shard_dir = shards_dir / f"{shard_number:05d}"
                shard_dir.mkdir()
                units_out = (shard_dir / "units.jsonl").open("w", encoding="utf-8", newline="\n")
                candidates_out = (shard_dir / "candidates.jsonl").open("w", encoding="utf-8", newline="\n")
                shards.append({
                    "shard_id": f"{shard_number:05d}",
                    "units": str(published_output / "shards" / f"{shard_number:05d}" / "units.jsonl"),
                    "candidates": str(published_output / "shards" / f"{shard_number:05d}" / "candidates.jsonl"),
                    "count": 0,
                })
            serialized_unit = json.dumps(unit, ensure_ascii=False, sort_keys=True) + "\n"
            serialized_candidate = json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n"
            units_out.write(serialized_unit)
            candidates_out.write(serialized_candidate)
            shards[-1]["count"] += 1
            counts["input"] += 1
            counts[str(unit.get("unit_type") or "unknown")] += 1
    finally:
        if units_out is not None:
            units_out.close()
            candidates_out.close()
    if not counts["input"]:
        raise ValueError("no input units")
    report = {
        "input": counts["input"],
        "shard_size": shard_size,
        "shard_count": len(shards),
        "unit_types": {key: counts[key] for key in sorted(counts) if key != "input"},
        "source_units": str(units_path),
        "source_candidates": str(candidates_path),
        "shards": shards,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_shard_predictions(
    shard_root: str | Path,
    vote_name: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Merge one vote only when every shard has every prediction in input order."""
    if not vote_name or "/" in vote_name or vote_name in {".", ".."}:
        raise ValueError("vote_name must be one directory name")
    root = Path(shard_root)
    manifest = json.loads((root / "report.json").read_text(encoding="utf-8"))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    merged = 0
    expected_config: tuple[Any, ...] | None = None
    prompt_versions_by_type: dict[str, str] = {}
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            for shard in manifest["shards"]:
                shard_id = str(shard["shard_id"])
                shard_dir = root / "shards" / shard_id
                vote_dir = shard_dir / "votes" / vote_name
                report = json.loads((vote_dir / "report.json").read_text(encoding="utf-8"))
                run_manifest = json.loads((vote_dir / "run_manifest.json").read_text(encoding="utf-8"))
                input_sha = run_manifest.get("input_sha256") or {}
                if input_sha.get("units") != _sha256(shard_dir / "units.jsonl"):
                    raise ValueError(f"unit input hash mismatch in shard {shard_id}")
                if input_sha.get("candidates") != _sha256(shard_dir / "candidates.jsonl"):
                    raise ValueError(f"candidate input hash mismatch in shard {shard_id}")
                config = (
                    run_manifest.get("model"), input_sha.get("labels"),
                    run_manifest.get("max_tokens"),
                    json.dumps(run_manifest.get("audited_exclusions"), sort_keys=True),
                    json.dumps(run_manifest.get("chat_template_kwargs"), sort_keys=True),
                )
                if not config[0] or not config[1]:
                    raise ValueError(f"incomplete run manifest in shard {shard_id}")
                if expected_config is None:
                    expected_config = config
                elif config != expected_config:
                    raise ValueError(f"run configuration mismatch in shard {shard_id}")
                expected = int(shard["count"])
                if report.get("input") != expected or report.get("success") != expected or report.get("error") != 0:
                    raise ValueError(f"incomplete predictions in shard {shard_id}")
                observed = 0
                for index, (unit, prediction) in enumerate(
                    zip_longest(
                        _rows(shard_dir / "units.jsonl"),
                        _rows(vote_dir / "predictions.jsonl"),
                    ),
                    1,
                ):
                    if unit is None or prediction is None:
                        raise ValueError(f"incomplete predictions in shard {shard_id}")
                    if str(unit.get("question_id") or "") != str(prediction.get("question_id") or ""):
                        raise ValueError(f"prediction question_id mismatch in shard {shard_id} row {index}")
                    unit_type = str(unit.get("unit_type") or "")
                    per_type = run_manifest.get("prompt_versions_by_unit_type") or {}
                    prompt_version = str(per_type.get(unit_type) or run_manifest.get("prompt_version") or "")
                    if not prompt_version:
                        raise ValueError(f"missing prompt version in shard {shard_id}")
                    prior_version = prompt_versions_by_type.setdefault(unit_type, prompt_version)
                    if prior_version != prompt_version:
                        raise ValueError(f"prompt version mismatch for {unit_type} in shard {shard_id}")
                    if prediction.get("prompt_version") != prompt_version:
                        raise ValueError(f"prediction prompt version mismatch in shard {shard_id} row {index}")
                    destination.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
                    observed += 1
                if observed != expected:
                    raise ValueError(f"incomplete predictions in shard {shard_id}")
                merged += observed
        if merged != manifest["input"]:
            raise ValueError(f"merged {merged} predictions, expected {manifest['input']}")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "vote_name": vote_name,
        "merged": merged,
        "shard_count": len(manifest["shards"]),
        "output": str(output),
    }
