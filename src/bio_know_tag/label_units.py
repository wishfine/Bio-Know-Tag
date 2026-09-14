"""Build immutable-baseline derivatives used by the labeling pipeline."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from bio_know_tag.questions import normalize_text


HIGH_RISK_CODES = {"KC", "KM", "KE"}
IMAGE_REFERENCE_RE = re.compile(
    r"(?:如图|图中|下图|上图|图示|示意图|曲线|坐标图|柱状图|图甲|图乙)"
)
METADATA_FIELDS = (
    "question_index",
    "subject",
    "business_type",
    "difficulty",
    "structure_type",
    "answered_count",
    "percent_correct",
)


def answer_to_text(value: Any) -> str:
    """Recursively normalize heterogeneous answer values into readable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        if value and all(isinstance(item, dict) and "answers" in item for item in value):
            ordered = sorted(
                enumerate(value),
                key=lambda pair: (
                    pair[1].get("index", pair[0])
                    if isinstance(pair[1].get("index", pair[0]), int)
                    else pair[0]
                ),
            )
            blanks = []
            for display_index, (_, item) in enumerate(ordered, 1):
                rendered = answer_to_text(item.get("answers"))
                if rendered:
                    blanks.append(f"空{display_index}：{rendered.replace('；', ' / ')}")
            return "；".join(blanks)
        rendered_items = [answer_to_text(item) for item in value]
        return "；".join(item for item in rendered_items if item)
    if isinstance(value, dict):
        if "answers" in value:
            return answer_to_text(value.get("answers"))
        rendered_items = []
        for key in sorted(value):
            rendered = answer_to_text(value[key])
            if rendered:
                rendered_items.append(f"{normalize_text(str(key))}：{rendered}")
        return "；".join(rendered_items)
    return normalize_text(str(value))


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_label_catalog(path: str | Path) -> dict[str, dict[str, str]]:
    catalog = {}
    for record in _read_jsonl(path):
        label_id = str(record.get("label_id") or "").strip()
        if not label_id:
            continue
        reference = record.get("reference_strategy") or {}
        catalog[label_id] = {
            "label_name": str(record.get("label_name") or ""),
            "strategy_code": str(reference.get("关键词策略代码") or ""),
        }
    return catalog


def _load_orphan_parent_ids(path: str | Path) -> set[str]:
    result = set()
    for record in _read_jsonl(path):
        parent_id = str(record.get("orphan_parent_id") or "").strip()
        if parent_id:
            result.add(parent_id)
    return result


def _legacy_fields(
    question: dict[str, Any], catalog: dict[str, dict[str, str]]
) -> tuple[list[str], list[str], list[str], list[str]]:
    legacy_ids = [str(value) for value in (question.get("knw_ids") or []) if value]
    matched = [label_id for label_id in legacy_ids if label_id in catalog]
    unmatched = [label_id for label_id in legacy_ids if label_id not in catalog]
    strategy_codes = sorted(
        {
            catalog[label_id]["strategy_code"]
            for label_id in matched
            if catalog[label_id]["strategy_code"]
        }
    )
    return legacy_ids, matched, unmatched, strategy_codes


def _proposed_route(
    unit_type: str,
    matched: list[str],
    unmatched: list[str],
    strategy_codes: list[str],
) -> tuple[str, str]:
    if not matched:
        return "R2", "没有命中新458图谱的旧候选"
    if unmatched:
        return "R2", "旧标签同时包含未映射ID，候选集合可能不完整"
    if "KE" in strategy_codes:
        return "R2", "候选包含KE图谱冲突Label"
    if unit_type == "sub_question":
        return "R1", "组合题小题需从父题级旧候选中选择自身最小Label集合"
    if len(matched) > 1:
        return "R1", "旧标签为多标签，需要候选裁决"
    if HIGH_RISK_CODES.intersection(strategy_codes):
        return "R1", "候选包含KC/KM高风险或独立维度Label"
    return "R0", "独立打标单元且旧标签完整命中；仍需通过按Label抽样验证"


def _content_hash(parent_stem: str, question: dict[str, Any], answer_text: str) -> str:
    fields = [
        normalize_text(parent_stem),
        normalize_text(question.get("stem")),
        normalize_text(question.get("options")),
        answer_text,
        normalize_text(question.get("analysis")),
    ]
    canonical = json.dumps(fields, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_unit(
    question: dict[str, Any],
    *,
    unit_type: str,
    parent_stem: str,
    sibling_question_ids: list[str],
    catalog: dict[str, dict[str, str]],
) -> dict[str, Any]:
    question_id = str(question.get("question_id") or "").strip()
    if not question_id:
        raise ValueError("question_id is required")
    parent_id = str(question.get("parent_id") or question_id).strip()
    answer_text = answer_to_text(question.get("answer"))
    legacy_ids, matched, unmatched, strategy_codes = _legacy_fields(question, catalog)
    route, route_reason = _proposed_route(
        unit_type, matched, unmatched, strategy_codes
    )
    stem = str(question.get("stem") or "").strip()
    options = str(question.get("options") or "").strip()
    analysis = str(question.get("analysis") or "").strip()
    normalized_parent_stem = str(parent_stem or "").strip()
    image_reference_text = " ".join(
        (normalized_parent_stem, stem, options, analysis)
    )
    unit = {
        "question_id": question_id,
        "parent_id": parent_id,
        "unit_type": unit_type,
        "parent_stem": normalized_parent_stem,
        "sibling_question_ids": sibling_question_ids,
        "stem": stem,
        "options": options,
        "answer": question.get("answer", ""),
        "answer_text": answer_text,
        "analysis": analysis,
        "legacy_knw_ids": legacy_ids,
        "legacy_candidate_ids": matched,
        "legacy_unmatched_ids": unmatched,
        "legacy_strategy_codes": strategy_codes,
        "proposed_route": route,
        "route_reason": route_reason,
        "dedupe_hash": _content_hash(normalized_parent_stem, question, answer_text),
        "flags": {
            "parent_context_missing": unit_type == "orphan_sub_question",
            "image_context_missing": bool(
                IMAGE_REFERENCE_RE.search(image_reference_text)
            ),
            "duplicate_of": None,
            "duplicate_label_conflict": False,
        },
        "metadata": {
            field: question.get(field)
            for field in METADATA_FIELDS
            if field in question
        },
    }
    return unit


def _write_line(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
    handle.write("\n")


def build_labeling_derivatives(
    processed_path: str | Path,
    labels_path: str | Path,
    orphan_audit_path: str | Path,
    run_dir: str | Path,
    *,
    limit: int | None = None,
    progress_every: int = 100_000,
) -> dict[str, Any]:
    """Flatten labeling units, build parent plans, deduplicate, and route locally."""
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = _load_label_catalog(labels_path)
    orphan_parent_ids = _load_orphan_parent_ids(orphan_audit_path)
    if not catalog:
        raise ValueError("label catalog is empty")

    stage_path = output_dir / ".label_units.stage.tmp"
    final_tmp_path = output_dir / ".label_units.jsonl.tmp"
    parents_tmp_path = output_dir / ".parent_aggregation.jsonl.tmp"
    duplicates_tmp_path = output_dir / ".duplicate_groups.jsonl.tmp"
    sqlite_path = output_dir / ".dedupe.sqlite3.tmp"
    for path in (
        stage_path,
        final_tmp_path,
        parents_tmp_path,
        duplicates_tmp_path,
        sqlite_path,
    ):
        if path.exists():
            path.unlink()

    connection = sqlite3.connect(sqlite_path)
    connection.execute(
        "CREATE TABLE units (seq INTEGER PRIMARY KEY AUTOINCREMENT, content_hash TEXT NOT NULL, question_id TEXT NOT NULL, label_signature TEXT NOT NULL)"
    )
    counters: Counter[str] = Counter()
    unmatched_ids: Counter[str] = Counter()
    matched_assignments = 0
    dedupe_batch: list[tuple[str, str, str]] = []

    try:
        with stage_path.open("w", encoding="utf-8", newline="\n") as stage, parents_tmp_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as parent_output:
            for record_index, parent in enumerate(_read_jsonl(processed_path), 1):
                if limit is not None and record_index > limit:
                    break
                counters["processed_parent_records"] += 1
                try:
                    if not isinstance(parent, dict):
                        raise ValueError("processed row must be an object")
                    parent_id = str(parent.get("question_id") or "").strip()
                    if not parent_id:
                        raise ValueError("parent question_id is required")
                    children = parent.get("sub_questions") or []
                    if not isinstance(children, list):
                        raise ValueError("sub_questions must be a list")
                    parent_stem = str(parent.get("stem") or "").strip()

                    if not children:
                        unit_specs = [(parent, "standalone", "", [])]
                    else:
                        child_ids = [
                            str(child.get("question_id") or "").strip()
                            for child in children
                            if isinstance(child, dict)
                            and str(child.get("question_id") or "").strip()
                        ]
                        is_synthetic = parent_id in orphan_parent_ids
                        unit_type = (
                            "orphan_sub_question" if is_synthetic else "sub_question"
                        )
                        effective_parent_stem = "" if is_synthetic else parent_stem
                        unit_specs = [
                            (
                                child,
                                unit_type,
                                effective_parent_stem,
                                [other for other in child_ids if other != child_id],
                            )
                            for child, child_id in (
                                (child, str(child.get("question_id") or "").strip())
                                for child in children
                                if isinstance(child, dict)
                            )
                        ]
                        if is_synthetic:
                            counters["synthetic_parent_containers_skipped"] += 1
                        else:
                            counters["real_compound_parents"] += 1
                            legacy_ids, matched, unmatched, strategy_codes = _legacy_fields(
                                parent, catalog
                            )
                            _write_line(
                                parent_output,
                                {
                                    "question_id": parent_id,
                                    "parent_id": parent_id,
                                    "unit_type": "composite_parent",
                                    "parent_stem": parent_stem,
                                    "options": str(parent.get("options") or "").strip(),
                                    "answer": parent.get("answer", ""),
                                    "answer_text": answer_to_text(parent.get("answer")),
                                    "analysis": str(parent.get("analysis") or "").strip(),
                                    "child_question_ids": child_ids,
                                    "legacy_knw_ids": legacy_ids,
                                    "legacy_candidate_ids": matched,
                                    "legacy_unmatched_ids": unmatched,
                                    "legacy_strategy_codes": strategy_codes,
                                    "flags": {"synthetic_parent": False},
                                },
                            )

                    for question, unit_type, effective_parent_stem, sibling_ids in unit_specs:
                        unit = _build_unit(
                            question,
                            unit_type=unit_type,
                            parent_stem=effective_parent_stem,
                            sibling_question_ids=sibling_ids,
                            catalog=catalog,
                        )
                        _write_line(stage, unit)
                        counters["label_units"] += 1
                        counters[f"{unit_type}_units"] += 1
                        matched_assignments += len(unit["legacy_candidate_ids"])
                        unmatched_ids.update(unit["legacy_unmatched_ids"])
                        dedupe_batch.append(
                            (
                                unit["dedupe_hash"],
                                unit["question_id"],
                                json.dumps(
                                    sorted(unit["legacy_knw_ids"]),
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            )
                        )
                        if len(dedupe_batch) >= 10_000:
                            connection.executemany(
                                "INSERT INTO units(content_hash, question_id, label_signature) VALUES (?, ?, ?)",
                                dedupe_batch,
                            )
                            connection.commit()
                            dedupe_batch.clear()
                except (TypeError, ValueError, json.JSONDecodeError):
                    counters["error"] += 1

                if progress_every and record_index % progress_every == 0:
                    print(
                        f"build: parent_records={record_index}, units={counters['label_units']}, error={counters['error']}",
                        flush=True,
                    )

        if dedupe_batch:
            connection.executemany(
                "INSERT INTO units(content_hash, question_id, label_signature) VALUES (?, ?, ?)",
                dedupe_batch,
            )
            connection.commit()
        connection.execute("CREATE INDEX units_hash_index ON units(content_hash)")
        connection.commit()

        duplicate_info: dict[str, tuple[str, bool, int]] = {}
        with duplicates_tmp_path.open("w", encoding="utf-8", newline="\n") as duplicate_output:
            duplicate_members_cursor = connection.execute(
                """
                SELECT units.content_hash, units.question_id, units.label_signature
                FROM units
                JOIN (
                    SELECT content_hash
                    FROM units
                    GROUP BY content_hash
                    HAVING COUNT(*) > 1
                ) AS duplicated USING (content_hash)
                ORDER BY units.content_hash, units.seq
                """
            )
            for content_hash, grouped_rows in itertools.groupby(
                duplicate_members_cursor, key=lambda row: row[0]
            ):
                members = [
                    (question_id, signature)
                    for _, question_id, signature in grouped_rows
                ]
                count = len(members)
                signature_count = len(
                    {signature for _, signature in members}
                )
                question_ids = [question_id for question_id, _ in members]
                conflict = signature_count > 1
                duplicate_info[content_hash] = (question_ids[0], conflict, count)
                _write_line(
                    duplicate_output,
                    {
                        "dedupe_hash": content_hash,
                        "primary_question_id": question_ids[0],
                        "question_ids": question_ids,
                        "member_count": count,
                        "duplicate_label_conflict": conflict,
                        "legacy_knw_id_sets": [json.loads(signature) for _, signature in members],
                    },
                )

        route_counts: Counter[str] = Counter()
        route_reason_counts: Counter[str] = Counter()
        candidate_count_distribution: Counter[str] = Counter()
        unit_type_counts: Counter[str] = Counter()
        missing_state_counts: Counter[str] = Counter()
        duplicate_members = 0
        duplicate_conflict_groups = sum(info[1] for info in duplicate_info.values())
        with stage_path.open("r", encoding="utf-8") as stage, final_tmp_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as final_output:
            for line in stage:
                unit = json.loads(line)
                info = duplicate_info.get(unit["dedupe_hash"])
                if info:
                    primary_id, conflict, _ = info
                    duplicate_members += 1
                    unit["flags"]["duplicate_of"] = (
                        None if unit["question_id"] == primary_id else primary_id
                    )
                    unit["flags"]["duplicate_label_conflict"] = conflict
                    if conflict and unit["proposed_route"] == "R0":
                        unit["proposed_route"] = "R1"
                        unit["route_reason"] = (
                            "精确重复题的旧候选Label不一致，禁止直接继承"
                        )
                _write_line(final_output, unit)
                route_counts[unit["proposed_route"]] += 1
                route_reason_counts[unit["route_reason"]] += 1
                candidate_count_distribution[
                    str(len(unit["legacy_candidate_ids"]))
                ] += 1
                unit_type_counts[unit["unit_type"]] += 1
                missing_state_counts["parent_context_missing_units"] += int(
                    unit["flags"]["parent_context_missing"]
                )
                missing_state_counts["image_context_missing_units"] += int(
                    unit["flags"]["image_context_missing"]
                )
                missing_state_counts["empty_stem_units"] += int(not unit["stem"])
                missing_state_counts["empty_answer_text_units"] += int(
                    not unit["answer_text"]
                )
                missing_state_counts["empty_analysis_units"] += int(
                    not unit["analysis"]
                )

        report = {
            "processed_parent_records": counters["processed_parent_records"],
            "label_units": counters["label_units"],
            "standalone_units": counters["standalone_units"],
            "sub_question_units": counters["sub_question_units"],
            "orphan_sub_question_units": counters["orphan_sub_question_units"],
            "real_compound_parents": counters["real_compound_parents"],
            "synthetic_parent_containers_skipped": counters[
                "synthetic_parent_containers_skipped"
            ],
            "exact_duplicate_groups": len(duplicate_info),
            "exact_duplicate_members": duplicate_members,
            "ds_calls_saved_by_exact_dedupe": duplicate_members
            - len(duplicate_info),
            "duplicate_label_conflict_groups": duplicate_conflict_groups,
            "matched_legacy_assignments": matched_assignments,
            "unmatched_legacy_assignments": sum(unmatched_ids.values()),
            "different_unmatched_legacy_ids": len(unmatched_ids),
            "parent_context_missing_units": missing_state_counts[
                "parent_context_missing_units"
            ],
            "image_context_missing_units": missing_state_counts[
                "image_context_missing_units"
            ],
            "empty_stem_units": missing_state_counts["empty_stem_units"],
            "empty_answer_text_units": missing_state_counts[
                "empty_answer_text_units"
            ],
            "empty_analysis_units": missing_state_counts["empty_analysis_units"],
            "error": counters["error"],
        }
        route_report = {
            "route_counts": dict(sorted(route_counts.items())),
            "route_reason_counts": dict(route_reason_counts.most_common()),
            "unit_type_counts": dict(sorted(unit_type_counts.items())),
            "missing_state_counts": dict(sorted(missing_state_counts.items())),
            "candidate_count_distribution": dict(
                sorted(candidate_count_distribution.items(), key=lambda item: int(item[0]))
            ),
            "matched_legacy_assignments": matched_assignments,
            "unmatched_legacy_assignments": sum(unmatched_ids.values()),
            "unmatched_legacy_ids": dict(unmatched_ids.most_common()),
            "notes": {
                "R0": "仅表示可直接继承候选；必须先完成按Label分层抽样验证。",
                "R1": "使用旧Label局部候选，由DS选择当前题最小Label集合。",
                "R2": "旧候选为空、不完整或包含KE冲突，需要局部重新召回。",
            },
        }

        final_tmp_path.replace(output_dir / "label_units.jsonl")
        parents_tmp_path.replace(output_dir / "parent_aggregation.jsonl")
        duplicates_tmp_path.replace(output_dir / "duplicate_groups.jsonl")
        _write_json_atomic(output_dir / "route_report.json", route_report)
        _write_json_atomic(output_dir / "build_report.json", report)
        return report
    finally:
        connection.close()
        for path in (stage_path, sqlite_path):
            if path.exists():
                path.unlink()
