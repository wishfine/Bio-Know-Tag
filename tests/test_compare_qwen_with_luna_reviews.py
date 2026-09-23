import json
from pathlib import Path

from scripts.compare_qwen_with_luna_reviews import compare


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def evidence(qid: str, candidates: dict[str, str], selected: list[str]) -> dict:
    return {
        "question_id": qid,
        "candidate_code_map": candidates,
        "parsed_response": {"selected": selected, "reason": f"理由{qid}"},
        "error": None,
    }


def test_luna_pair_comparison_separates_adjudication_errors_and_recall_gains(tmp_path: Path):
    luna, top, legacy = (tmp_path / name for name in ("luna.jsonl", "top.jsonl", "legacy.jsonl"))
    labels, units = (tmp_path / name for name in ("labels.jsonl", "units.jsonl"))
    write_jsonl(luna, [
        {"question_id": "q1", "label_id": "L1", "decision": "DIRECT_MATCH"},
        {"question_id": "q2", "label_id": "L2", "decision": "WRONG_LABEL"},
        {"question_id": "q3", "label_id": "L3", "decision": "REASONABLE_CO_LABEL"},
        {"question_id": "q4", "label_id": "L4", "decision": "DIRECT_MATCH"},
        {"question_id": "q5", "label_id": "L5", "decision": "WRONG_LABEL"},
        {"question_id": "q6", "label_id": "L6", "decision": "INSUFFICIENT_CONTEXT"},
    ])
    write_jsonl(top, [
        evidence("q1", {"C01": "L1"}, ["C01"]),
        evidence("q2", {"C01": "L2"}, ["C01"]),
        evidence("q3", {"C01": "L3"}, ["C01"]),
        evidence("q4", {"C01": "other"}, []),
        evidence("q5", {"C01": "other"}, []),
        evidence("q6", {"C01": "L6"}, []),
    ])
    write_jsonl(legacy, [
        evidence("q1", {"C01": "L1"}, ["C01"]),
        evidence("q2", {"C01": "L2"}, ["C01"]),
        evidence("q3", {"C01": "L3"}, []),
        evidence("q4", {"C01": "other", "C02": "L4"}, ["C02"]),
        evidence("q5", {"C01": "other", "C02": "L5"}, ["C02"]),
        evidence("q6", {"C01": "L6"}, []),
    ])
    write_jsonl(labels, [{"label_id": f"L{i}", "label_name": f"标签{i}"} for i in range(1, 7)])
    write_jsonl(units, [{"question_id": f"q{i}", "stem": f"题干{i}"} for i in range(1, 7)])

    report = compare(luna, top, legacy, labels, units, tmp_path / "out", sample_per_category=2)

    assert report["luna_audit_pairs"] == 6
    assert report["paired_question_count"] == 6
    assert report["pair_counts"]["paired_insufficient_context"] == 1
    assert report["pair_counts"]["added_legacy_target_selected_positive"] == 1
    assert report["pair_counts"]["added_legacy_target_selected_wrong"] == 1
    assert report["models"]["top25"]["positive_selected"] == 2
    assert report["models"]["top25"]["wrong_selected"] == 1
    assert report["models"]["top25"]["positive_missing_from_candidates"] == 1
    assert report["models"]["top25_plus_legacy"]["positive_selected"] == 2
    assert report["models"]["top25_plus_legacy"]["wrong_selected"] == 2
    samples = [json.loads(line) for line in (tmp_path / "out" / "review_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["category"] == "added_legacy_wrong" and row["stem"] == "题干5" for row in samples)
