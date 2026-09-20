import json
from pathlib import Path

from bio_know_tag.adjudication_snapshot import build_paired_adjudication_snapshot


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_snapshot_keeps_common_latest_success_and_materializes_labels(tmp_path: Path):
    units = [
        {"question_id": "q1", "stem": "题一", "dedupe_hash": "h1"},
        {"question_id": "q2", "stem": "题二", "dedupe_hash": "h2"},
    ]
    labels = [{"label_id": "L1", "label_name": "标签一", "label_path": "知识点->一"}]
    candidates = [
        {
            "question_id": question_id,
            "candidates": [
                {
                    "label_id": "L1",
                    "label_name": "标签一",
                    "label_path": "知识点@一",
                    "candidate_rank": 1,
                    "sources": ["sparse"],
                }
            ],
        }
        for question_id in ("q1", "q2")
    ]
    top_evidence = [
        {
            "question_id": "q1",
            "candidate_code_map": {"C01": "L1"},
            "parsed_response": {
                "selected": ["C01"],
                "evidence": {"C01": "题一"},
                "reason": "考查标签一",
                "context_insufficient": False,
                "need_expand_recall": False,
            },
            "error": None,
            "prompt_version": "v1",
        },
        {"question_id": "q2", "parsed_response": None, "error": "failed"},
    ]
    legacy_evidence = [
        {
            "question_id": "q1",
            "candidate_code_map": {"C01": "L1"},
            "parsed_response": {
                "selected": [],
                "evidence": {},
                "reason": "不选择",
                "context_insufficient": False,
                "need_expand_recall": False,
            },
            "error": None,
            "prompt_version": "v1",
        },
        {
            "question_id": "q2",
            "candidate_code_map": {"C01": "L1"},
            "parsed_response": {"selected": ["C01"]},
            "error": None,
            "prompt_version": "v1",
        },
    ]
    paths = {}
    for name, rows in {
        "units": units,
        "labels": labels,
        "top_candidates": candidates,
        "legacy_candidates": candidates,
        "top_evidence": top_evidence,
        "legacy_evidence": legacy_evidence,
    }.items():
        paths[name] = tmp_path / f"{name}.jsonl"
        _write(paths[name], rows)

    output = tmp_path / "snapshot"
    report = build_paired_adjudication_snapshot(
        paths["units"],
        paths["top_evidence"],
        paths["legacy_evidence"],
        paths["top_candidates"],
        paths["legacy_candidates"],
        paths["labels"],
        output,
    )

    assert report["common_success"] == 1
    assert report["top25_only_success"] == 0
    assert report["legacy_only_success"] == 1
    top = json.loads((output / "top25_predictions.jsonl").read_text())
    legacy = json.loads((output / "legacy_predictions.jsonl").read_text())
    assert top["selected_labels"][0]["label_id"] == "L1"
    assert top["selected_labels"][0]["evidence"] == "题一"
    assert legacy["selected_labels"] == []
    assert json.loads((output / "units.jsonl").read_text())["question_id"] == "q1"
