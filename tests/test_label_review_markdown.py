from bio_know_tag.label_review_markdown import build_markdown, write_markdown


def _record():
    return {
        "label_id": "1",
        "label_name": "DNA半保留复制",
        "label_path": "知识点->遗传信息的传递",
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
            "adjusted_from_previous": True,
            "rationale": "名称清楚，但保留边界。",
            "manual_followup_required": False,
        },
        "strategy": {"reason": "保留边界"},
        "final_strategy": {
            "mode": "name_plus_boundary",
            "automation": "名称召回并附一条边界后由LLM裁决",
            "prompt_fields": ["label_name", "distinctions"],
            "manual_followup_required": False,
            "confidence": "high",
        },
        "taxonomy_issue": None,
    }


def test_build_markdown_has_one_row_and_escapes_table_breaks():
    markdown = build_markdown([_record()])
    assert "| 序号 | Label ID | Label名称 |" in markdown
    assert "DNA半保留复制" in markdown
    assert "DS Judge结果" in markdown
    assert "GPT Judge结果（二次复核）" in markdown
    assert "name_plus_boundary" in markdown
    assert "<br>" in markdown
    assert "## L1 / L2 / L3 是什么" in markdown
    assert "## DS 释义 Prompt" in markdown
    assert "## DS Judge Prompt 与分数" in markdown
    assert "## GPT Judge 状态" in markdown
    assert "## 最终处理策略" in markdown
    assert "## 需要人工跟进的 Label" in markdown


def test_write_markdown_returns_row_count(tmp_path):
    output = tmp_path / "label-review.md"
    report = write_markdown([_record()], output)
    assert report == {"output": str(output), "rows": 1}
    assert output.read_text(encoding="utf-8").startswith("# 高中生物 458 个 Label")
