import json

from openpyxl import load_workbook

from bio_know_tag.label_review_workbook import build_workbook_rows, write_workbook


def _record():
    return {
        "label_id": "1",
        "label_name": "DNA半保留复制",
        "label_path": "知识点->遗传信息的传递",
        "label_type": "知识",
        "definition": "亲代DNA双链分开后分别作为模板合成子链。",
        "core_concepts": "半保留、模板链",
        "common_assessments": "过程判断",
        "distinctions": "与全保留复制区分",
        "stage1": {
            "core_meaning": "DNA复制方式",
            "included_content": ["亲代链保留"],
            "excluded_content": ["全保留复制"],
        },
        "stage2_category": "L1",
        "stage2_judge": {
            "alignment_score": 5,
            "name_sufficiency": "名称本身足够",
            "audit_decision": "两者基本等价",
            "audit_reason": "核心一致",
            "boundary_differences": [],
            "omissions": [],
            "expansions": [],
        },
        "second_review": {
            "status": "confirmed",
            "confidence": "high",
            "previous_mode": "name_only",
            "final_mode": "name_plus_boundary",
            "adjusted_from_previous": False,
            "rationale": "名称清楚，但保留边界。",
            "manual_followup_required": False,
            "operational_boundary": None,
        },
        "strategy": {"reason": "保留边界"},
        "final_strategy": {
            "mode": "name_plus_boundary",
            "automation": "名称召回并附一条边界后由LLM裁决",
            "prompt_fields": ["label_name", "distinctions"],
            "manual_followup_required": False,
            "confidence": "high",
            "operational_boundary": None,
        },
        "taxonomy_issue": None,
    }


def test_build_workbook_rows_contains_summary_and_detail_columns():
    rows = build_workbook_rows([_record()])
    assert rows["summary"][0]["Label名称"] == "DNA半保留复制"
    assert "核心含义：DNA复制方式" in rows["summary"][0]["DS释义"]
    assert "分数：5" in rows["summary"][0]["DS Judge结果"]
    assert rows["detail"][0]["最终处理策略"] == "name_plus_boundary"
    assert rows["manual"] == []


def test_write_workbook_creates_four_readable_sheets(tmp_path):
    output = tmp_path / "label-review.xlsx"
    report = write_workbook([_record()], output)

    assert report["rows"] == 1
    workbook = load_workbook(output, data_only=False)
    assert workbook.sheetnames == ["逐Label总表", "逐Label详情", "人工跟进", "统计汇总", "字段说明"]
    assert workbook["逐Label总表"].max_row == 2
    assert workbook["逐Label总表"].freeze_panes == "A2"
    assert workbook["逐Label总表"].auto_filter.ref.startswith("A1:")
    assert workbook["逐Label详情"]["C2"].value == "DNA半保留复制"
    assert workbook["统计汇总"]["C2"].value == 1
    assert workbook["字段说明"]["A1"].value == "字段"
