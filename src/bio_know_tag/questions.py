"""Clean question text and aggregate child questions under their parents."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from bs4 import BeautifulSoup


METADATA_FIELDS = ("structure_type", "answered_count", "percent_correct", "difficulty")


def _convert_sup_sub(text: str) -> str:
    text = re.sub(r"</sup>\s*<sup>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</sub>\s*<sub>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<sup>\s*(.*?)\s*</sup>", r"^{\1}", text, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(
        r"<sub>\s*(.*?)\s*</sub>", r"_{\1}", text, flags=re.IGNORECASE | re.DOTALL
    )


def normalize_text(value: Any) -> str:
    """Normalize HTML-rich question text while retaining scientific notation."""
    if not isinstance(value, str) or not value:
        return ""
    text = re.sub(
        r"(?:\$\s*)*<img[^>]*>(?:\s*<br\s*/?>)*", " ", value, flags=re.IGNORECASE
    )
    text = _convert_sup_sub(text)
    text = text.replace("$(\\qquad)$", "(   )")
    text = re.sub(
        r'<span\s+class=["\'][^"\']*underline[^"\']*fillblank[^"\']*["\'][^>]*>.*?</span>',
        "______",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = BeautifulSoup(text, "html.parser").get_text(" ")
    cleaned = re.sub(r"(?:\$\s*)+", " ", cleaned)
    cleaned = re.sub(r"\s*(?:\\!\s*)*%\s*(?:\\!\s*)*(?:\\text\{\s*\})?", r"\\%", cleaned)

    keep_full_width = {"，", "。", "；", "：", "？", "！"}
    characters: list[str] = []
    for character in cleaned:
        code = ord(character)
        if 0xFF01 <= code <= 0xFF5E and character not in keep_full_width:
            characters.append(chr(code - 0xFEE0))
        else:
            characters.append(character)
    normalized = "".join(characters)
    for old, new in {
        "﹣": "-",
        "−": "-",
        "～": "~",
        "≥": ">=",
        "≤": "<=",
        "≠": "!=",
        "\u3000": " ",
        "\u00a0": " ",
        "\n": " ",
        "\r": " ",
    }.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = normalized.replace("()", "(   )")
    return re.sub(r"\s+(?=[。，！？；：\"'）】])", "", normalized).strip()


def _parse_question_info(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("question_info must be a JSON object or object-encoded string")


def _format_options(options: Any) -> str:
    if not isinstance(options, list):
        return ""
    formatted: list[str] = []
    for option in options:
        if not isinstance(option, dict):
            continue
        title = normalize_text(option.get("title"))
        content = normalize_text(option.get("htmlCode"))
        if title and content:
            formatted.append(f"{title}. {content}")
    return "\n".join(formatted)


def clean_question_info(value: Any) -> dict[str, str]:
    """Parse and normalize the prompt fields used by downstream labeling."""
    info = _parse_question_info(value)
    return {
        "stem": normalize_text(info.get("stem")),
        "options": _format_options(info.get("options")),
        "analysis": normalize_text(info.get("analysis")),
    }


def _base_question(row: dict[str, Any], question_id: str, parent_id: str) -> dict[str, Any]:
    cleaned = clean_question_info(row.get("question_info"))
    result: dict[str, Any] = {
        "parent_id": parent_id,
        "question_id": question_id,
        **cleaned,
    }
    for field in METADATA_FIELDS:
        result[field] = row.get(field, "" if field != "answered_count" else 0)
    return result


def _placeholder_parent(row: dict[str, Any], parent_id: str) -> dict[str, Any]:
    parent_info = row.get("parent_question_info")
    if not isinstance(parent_info, dict):
        parent_info = {}
    cleaned = clean_question_info(parent_info.get("question_info"))
    result: dict[str, Any] = {
        "parent_id": parent_id,
        "question_id": parent_id,
        **cleaned,
    }
    for field in METADATA_FIELDS:
        result[field] = parent_info.get(field, row.get(field, ""))
    result["sub_questions"] = []
    return result


def aggregate_questions(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Globally aggregate rows, so input order need not place parents first."""
    parents: dict[str, dict[str, Any]] = {}
    seen_question_ids: set[str] = set()
    stats = {"input": 0, "processed": 0, "skipped": 0, "error": 0}
    child_count = 0

    for row in rows:
        stats["input"] += 1
        try:
            if not isinstance(row, dict):
                raise ValueError("question row must be an object")
            question_id = str(row.get("question_id") or "").strip()
            if not question_id:
                raise ValueError("question_id is required")
            if question_id in seen_question_ids:
                raise ValueError(f"duplicate question_id: {question_id}")
            parent_id = str(row.get("parent_id") or question_id).strip()

            if parent_id == question_id:
                current = _base_question(row, question_id, parent_id)
                current["sub_questions"] = parents.get(parent_id, {}).get(
                    "sub_questions", []
                )
                parents[parent_id] = current
            else:
                if parent_id not in parents:
                    parents[parent_id] = _placeholder_parent(row, parent_id)
                child = _base_question(row, question_id, parent_id)
                parents[parent_id]["sub_questions"].append(child)
                child_count += 1
            seen_question_ids.add(question_id)
            stats["processed"] += 1
        except (TypeError, ValueError, json.JSONDecodeError):
            stats["skipped"] += 1
            stats["error"] += 1

    report = {
        **stats,
        "parent_count": len(parents),
        "child_count": child_count,
    }
    report = {
        key: report[key]
        for key in (
            "input",
            "processed",
            "parent_count",
            "child_count",
            "skipped",
            "error",
        )
    }
    return list(parents.values()), report


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def process_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, int]:
    """Process a JSONL file and atomically publish data plus quality report."""
    source = Path(input_path)
    parse_stats = {"input": 0, "skipped": 0, "error": 0}

    def valid_rows() -> Iterator[dict[str, Any]]:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if limit is not None and parse_stats["input"] >= limit:
                    break
                parse_stats["input"] += 1
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"line {line_number} is not an object")
                    yield value
                except (json.JSONDecodeError, ValueError):
                    parse_stats["skipped"] += 1
                    parse_stats["error"] += 1

    parents, aggregate_report = aggregate_questions(valid_rows())
    report = {
        **aggregate_report,
        "input": parse_stats["input"],
        "skipped": aggregate_report["skipped"] + parse_stats["skipped"],
        "error": aggregate_report["error"] + parse_stats["error"],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for parent in parents:
            handle.write(json.dumps(parent, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(output)
    _atomic_write_json(Path(report_path), report)
    return report

