import json
from pathlib import Path

from bio_know_tag.questions import (
    aggregate_questions,
    clean_question_info,
    normalize_text,
    process_jsonl,
)


def qinfo(
    stem: str,
    *,
    analysis: str = "解析",
    answer: str = "答案",
    options=None,
) -> str:
    return json.dumps(
        {
            "stem": stem,
            "analysis": analysis,
            "answer": answer,
            "options": options or [],
        },
        ensure_ascii=False,
    )


def parent_row() -> dict:
    return {
        "question_id": "p1",
        "parent_id": "p1",
        "question_info": qinfo("<p>大题 CO<sub>2</sub></p>"),
        "structure_type": "组合题",
        "answered_count": 10,
        "percent_correct": 0.5,
        "difficulty": "中等",
        "question_index": 7,
        "subject": "高中生物",
        "business_type": "题库",
        "knw_ids": ["old-1", "old-2"],
    }


def child_row() -> dict:
    return {
        "question_id": "c1",
        "parent_id": "p1",
        "question_info": qinfo(
            "<p>小题 10<sup>3</sup></p>",
            options=[
                {"title": "A", "htmlCode": "<span>正确</span>"},
                {"title": "B", "htmlCode": "<img src='x'>干扰"},
            ],
        ),
        "parent_question_info": {
            "question_id": "p1",
            "question_info": qinfo("<p>备用大题</p>"),
            "structure_type": "组合题",
        },
    }


def test_normalize_text_preserves_scientific_super_and_subscripts():
    assert normalize_text("CO<sub>2</sub> 与 10<sup>3</sup>") == "CO_{2} 与 10^{3}"


def test_clean_question_info_removes_images_and_formats_options():
    cleaned = clean_question_info(child_row()["question_info"])

    assert cleaned["stem"] == "小题 10^{3}"
    assert cleaned["options"] == "A. 正确\nB. 干扰"
    assert cleaned["analysis"] == "解析"
    assert cleaned["answer"] == "答案"


def test_aggregate_supports_child_before_parent():
    parents, report = aggregate_questions([child_row(), parent_row()])

    assert parents[0]["question_id"] == "p1"
    assert parents[0]["stem"] == "大题 CO_{2}"
    assert parents[0]["answer"] == "答案"
    assert parents[0]["question_index"] == 7
    assert parents[0]["subject"] == "高中生物"
    assert parents[0]["business_type"] == "题库"
    assert parents[0]["knw_ids"] == ["old-1", "old-2"]
    assert [question["question_id"] for question in parents[0]["sub_questions"]] == [
        "c1"
    ]
    assert report == {
        "input": 2,
        "processed": 2,
        "parent_count": 1,
        "child_count": 1,
        "skipped": 0,
        "error": 0,
    }


def test_aggregate_treats_missing_parent_id_as_standalone():
    row = {
        "question_id": "solo",
        "parent_id": "",
        "question_info": {"stem": "独立题", "options": [], "analysis": ""},
    }

    parents, report = aggregate_questions([row])

    assert parents[0]["question_id"] == "solo"
    assert parents[0]["parent_id"] == "solo"
    assert parents[0]["sub_questions"] == []
    assert report["processed"] == 1


def test_aggregate_reports_duplicate_question_id():
    parents, report = aggregate_questions([parent_row(), parent_row()])

    assert len(parents) == 1
    assert report["processed"] == 1
    assert report["skipped"] == 1
    assert report["error"] == 1


def test_process_jsonl_reports_malformed_json_and_writes_atomically(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "runtime" / "questions.jsonl"
    report_path = tmp_path / "runtime" / "report.json"
    input_path.write_text(
        json.dumps(parent_row(), ensure_ascii=False) + "\n{bad json\n",
        encoding="utf-8",
    )

    report = process_jsonl(input_path, output_path, report_path)

    assert report["input"] == 2
    assert report["processed"] == 1
    assert report["parent_count"] == 1
    assert report["skipped"] == 1
    assert report["error"] == 1
    assert output_path.read_text(encoding="utf-8").count("\n") == 1
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert not list(output_path.parent.glob("*.tmp"))


def test_process_jsonl_uses_streaming_path_instead_of_in_memory_aggregate(
    tmp_path: Path, monkeypatch
):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "questions.jsonl"
    report_path = tmp_path / "report.json"
    rows = [
        {
            "question_id": f"q{index}",
            "parent_id": f"q{index}",
            "question_info": {"stem": f"题目{index}", "options": [], "analysis": "", "answer": "A"},
        }
        for index in range(5)
    ]
    input_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    import bio_know_tag.questions as questions

    def fail_if_called(_rows):
        raise AssertionError("process_jsonl must not collect all rows via aggregate_questions")

    monkeypatch.setattr(questions, "aggregate_questions", fail_if_called)

    report = process_jsonl(input_path, output_path, report_path)

    assert report["processed"] == 5
    assert output_path.read_text(encoding="utf-8").count("\n") == 5


def test_process_jsonl_streaming_path_aggregates_child_before_parent(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "questions.jsonl"
    report_path = tmp_path / "report.json"
    standalone = {
        "question_id": "solo",
        "parent_id": "solo",
        "question_info": {
            "stem": "独立题",
            "options": [],
            "analysis": "解析",
            "answer": "A",
        },
    }
    input_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in (child_row(), standalone, parent_row())
        ),
        encoding="utf-8",
    )

    report = process_jsonl(input_path, output_path, report_path)
    output = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    by_id = {row["question_id"]: row for row in output}

    assert report["parent_count"] == 2
    assert report["child_count"] == 1
    assert by_id["p1"]["stem"] == "大题 CO_{2}"
    assert by_id["p1"]["sub_questions"][0]["question_id"] == "c1"
    assert by_id["p1"]["knw_ids"] == ["old-1", "old-2"]
