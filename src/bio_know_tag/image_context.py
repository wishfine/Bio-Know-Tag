"""Stream a large question-image map and audit text-only labeling eligibility."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator


def iter_json_object_items(
    path: str | Path, *, chunk_size: int = 1024 * 1024
) -> Iterator[tuple[str, Any]]:
    """Yield a top-level JSON object's items without loading the full object."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def refill() -> None:
            nonlocal buffer, position, eof
            buffer = buffer[position:]
            position = 0
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        def ensure_character() -> None:
            while position >= len(buffer) and not eof:
                refill()

        def skip_whitespace() -> None:
            nonlocal position
            while True:
                ensure_character()
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return

        def decode_value() -> Any:
            nonlocal position
            while True:
                skip_whitespace()
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    return value
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError("invalid or truncated JSON object") from exc
                    refill()

        refill()
        skip_whitespace()
        if position >= len(buffer) or buffer[position] != "{":
            raise ValueError("image map must be a top-level JSON object")
        position += 1
        first = True
        while True:
            skip_whitespace()
            ensure_character()
            if position < len(buffer) and buffer[position] == "}":
                position += 1
                break
            if not first:
                if position >= len(buffer) or buffer[position] != ",":
                    raise ValueError("expected comma between JSON object items")
                position += 1
            key = decode_value()
            if not isinstance(key, str):
                raise ValueError("image map keys must be strings")
            skip_whitespace()
            ensure_character()
            if position >= len(buffer) or buffer[position] != ":":
                raise ValueError("expected colon after JSON object key")
            position += 1
            value = decode_value()
            yield key, value
            first = False
            if position > chunk_size:
                refill()
        skip_whitespace()
        if position < len(buffer) or not eof:
            while not eof:
                refill()
                skip_whitespace()
            if position < len(buffer):
                raise ValueError("unexpected content after top-level JSON object")


def _read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL row must be an object")
                yield value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _url(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def audit_label_unit_image_context(
    units_path: str | Path,
    image_map_path: str | Path,
    run_dir: str | Path,
    *,
    progress_every: int = 100_000,
) -> dict[str, Any]:
    """Create a question-aligned image/context sidecar and a quality report."""
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_dir / "image_context.jsonl"
    sidecar_tmp = sidecar_path.with_name(f".{sidecar_path.name}.tmp")
    excluded_path = output_dir / "excluded_units.jsonl"
    excluded_tmp = excluded_path.with_name(f".{excluded_path.name}.tmp")
    review_path = output_dir / "review_units.jsonl"
    review_tmp = review_path.with_name(f".{review_path.name}.tmp")
    sqlite_path = output_dir / ".image_context.sqlite3.tmp"
    for path in (sidecar_tmp, excluded_tmp, review_tmp, sqlite_path):
        if path.exists():
            path.unlink()

    target_ids: set[str] = set()
    units_scanned = 0
    for unit in _read_jsonl(units_path):
        units_scanned += 1
        for field in ("question_id", "parent_id"):
            value = str(unit.get(field) or "").strip()
            if value:
                target_ids.add(value)

    connection = sqlite3.connect(sqlite_path)
    connection.execute(
        "CREATE TABLE images (question_id TEXT PRIMARY KEY, stem_url TEXT NOT NULL, analysis_url TEXT NOT NULL)"
    )
    image_records_scanned = malformed_image_records = matched_images = 0
    batch: list[tuple[str, str, str]] = []
    try:
        for question_id, value in iter_json_object_items(image_map_path):
            image_records_scanned += 1
            if question_id not in target_ids:
                continue
            if not isinstance(value, dict):
                malformed_image_records += 1
                continue
            batch.append(
                (
                    question_id,
                    _url(value.get("stemImageUrl")),
                    _url(value.get("analysisImageUrl")),
                )
            )
            if len(batch) >= 10_000:
                connection.executemany("INSERT OR REPLACE INTO images VALUES (?, ?, ?)", batch)
                connection.commit()
                matched_images += len(batch)
                batch.clear()
            if progress_every and image_records_scanned % progress_every == 0:
                print(
                    f"image scan: {image_records_scanned} records, "
                    f"{matched_images + len(batch)} target matches",
                    flush=True,
                )
        if batch:
            connection.executemany("INSERT OR REPLACE INTO images VALUES (?, ?, ?)", batch)
            connection.commit()
            matched_images += len(batch)
            batch.clear()
        matched_images = int(connection.execute("SELECT COUNT(*) FROM images").fetchone()[0])

        @lru_cache(maxsize=100_000)
        def image_for(question_id: str) -> tuple[str, str]:
            if not question_id:
                return "", ""
            row = connection.execute(
                "SELECT stem_url, analysis_url FROM images WHERE question_id = ?",
                (question_id,),
            ).fetchone()
            return (str(row[0]), str(row[1])) if row else ("", "")

        counters: Counter[str] = Counter()
        processed_units = 0
        with (
            sidecar_tmp.open("w", encoding="utf-8", newline="\n") as output,
            excluded_tmp.open("w", encoding="utf-8", newline="\n") as excluded_output,
            review_tmp.open("w", encoding="utf-8", newline="\n") as review_output,
        ):
            for unit in _read_jsonl(units_path):
                processed_units += 1
                question_id = str(unit.get("question_id") or "").strip()
                parent_id = str(unit.get("parent_id") or question_id).strip()
                stem = str(unit.get("stem") or "").strip()
                parent_stem = str(unit.get("parent_stem") or "").strip()
                options = str(unit.get("options") or "").strip()
                answer_text = str(unit.get("answer_text") or "").strip()
                analysis = str(unit.get("analysis") or "").strip()
                stem_url, analysis_url = image_for(question_id)
                parent_stem_url, parent_analysis_url = (
                    image_for(parent_id) if parent_id != question_id else ("", "")
                )
                has_image = bool(
                    stem_url or analysis_url or parent_stem_url or parent_analysis_url
                )
                supporting_text = bool(options or answer_text or analysis)
                if not stem and not parent_stem:
                    if has_image:
                        status = "image_only_no_text_stem"
                        review = True
                    else:
                        status = "both_stems_empty_no_image"
                        review = False
                    eligible = False
                elif not stem:
                    if stem_url or analysis_url:
                        status = "current_stem_image_only"
                        review = True
                    else:
                        status = "current_question_missing"
                        review = False
                    eligible = False
                elif not supporting_text:
                    status = "stem_only_needs_review"
                    eligible = True
                    review = True
                elif has_image:
                    status = "text_with_image_context"
                    eligible = True
                    review = False
                else:
                    status = "text_only"
                    eligible = True
                    review = False
                counters[status] += 1
                counters["excluded_from_text_labeling"] += int(not eligible)
                counters["needs_content_review"] += int(review)
                counters["units_with_any_image_url"] += int(has_image)
                row = {
                    "question_id": question_id,
                    "parent_id": parent_id,
                    "unit_type": unit.get("unit_type"),
                    "stem_image_url": stem_url,
                    "analysis_image_url": analysis_url,
                    "parent_stem_image_url": parent_stem_url,
                    "parent_analysis_image_url": parent_analysis_url,
                    "image_url_available": has_image,
                    "image_not_provided_to_text_model": has_image,
                    "content_status": status,
                    "eligible_for_text_labeling": eligible,
                    "needs_content_review": review,
                }
                rendered = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                output.write(rendered)
                if not eligible:
                    excluded_output.write(rendered)
                if review:
                    review_output.write(rendered)
                if progress_every and processed_units % progress_every == 0:
                    print(
                        f"unit audit: {processed_units} units, "
                        f"{counters['excluded_from_text_labeling']} excluded, "
                        f"{counters['needs_content_review']} review",
                        flush=True,
                    )
        sidecar_tmp.replace(sidecar_path)
        excluded_tmp.replace(excluded_path)
        review_tmp.replace(review_path)
    finally:
        connection.close()
        if sqlite_path.exists():
            sqlite_path.unlink()

    report = {
        "units": units_scanned,
        "target_question_ids": len(target_ids),
        "image_records_scanned": image_records_scanned,
        "target_ids_with_image_record": matched_images,
        "malformed_target_image_records": malformed_image_records,
        "units_with_any_image_url": counters["units_with_any_image_url"],
        "excluded_from_text_labeling": counters["excluded_from_text_labeling"],
        "needs_content_review": counters["needs_content_review"],
        "content_status_counts": {
            key: counters[key]
            for key in sorted(counters)
            if key not in {
                "units_with_any_image_url",
                "excluded_from_text_labeling",
                "needs_content_review",
            }
        },
        "note": "Image URLs are retained as a sidecar; no image was sent to the text-only DS service.",
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report
