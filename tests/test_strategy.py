from bio_know_tag.strategy import build_strategy_records, choose_strategy, strategy_reason


def _label(**overrides):
    value = {
        "label_id": "1",
        "label_name": "DNA半保留复制",
        "label_path": "知识点->分子与细胞->DNA半保留复制",
        "label_type": "知识",
        "definition": "亲代DNA双链分开后分别作为模板合成子链。",
        "core_concepts": "半保留、模板链、互补配对",
        "common_assessments": "实验与过程判断",
        "distinctions": "与全保留复制区分",
    }
    value.update(overrides)
    return value


def _stage1():
    return {
        "core_meaning": "DNA复制方式",
        "included_content": ["亲代链保留"],
        "excluded_content": ["全保留复制"],
    }


def _stage2(**overrides):
    value = {
        "alignment_score": 5,
        "omissions": [],
        "expansions": [],
        "boundary_differences": [],
        "audit_decision": "两者基本等价",
        "audit_reason": "核心一致",
        "name_sufficiency": "名称本身足够",
    }
    value.update(overrides)
    return value


def test_p0_or_taxonomy_conflict_is_held_for_manual_review():
    result = choose_strategy(
        _label(label_name="蛋白质病毒的增殖"),
        _stage1(),
        _stage2(alignment_score=1, audit_decision="原释义更准确", name_sufficiency="需要原释义"),
        reference={"关键词策略代码": "KE", "Prompt是否需要释义": "先修图谱；当前释义不足以稳定区分"},
        issue={"风险级别": "P0"},
    )

    assert result["mode"] == "taxonomy_hold"
    assert result["automation"] == "暂停自动最终打标"
    assert result["manual_review_required"] is True


def test_structural_axis_is_emitted_separately():
    result = choose_strategy(
        _label(label_name="文字信息类"),
        _stage1(),
        _stage2(alignment_score=2, name_sufficiency="需要原释义"),
        reference={"关键词策略代码": "KM", "Prompt是否需要释义": "必须给“维度定义+与知识标签关系”"},
        issue={"风险级别": "结构性"},
    )

    assert result["mode"] == "separate_dimension"
    assert result["automation"] == "作为独立维度预测，不写入知识标签"
    assert result["manual_review_required"] is False


def test_l3_uses_teacher_definition_with_review_flag():
    result = choose_strategy(
        _label(label_name="水在细胞中的功能"),
        _stage1(),
        _stage2(alignment_score=3, audit_decision="原释义更准确", name_sufficiency="需要原释义"),
        reference={"关键词策略代码": "K2", "Prompt是否需要释义": "可不给长释义，但建议保留1句边界"},
    )

    assert result["mode"] == "strict_definition"
    assert result["automation"] == "提供老师边界后由LLM最终裁决"
    assert result["manual_review_required"] is True


def test_safe_l1_can_use_name_plus_one_boundary_for_efficiency():
    result = choose_strategy(
        _label(),
        _stage1(),
        _stage2(),
        reference={"关键词策略代码": "K1", "Prompt是否需要释义": "可不给长释义，但建议保留1句边界"},
    )

    assert result["mode"] == "name_plus_boundary"
    assert result["automation"] == "名称召回并附一条边界后由LLM裁决"
    assert result["manual_review_required"] is False
    assert "L1" in strategy_reason(result, "L1")


def test_l2_adds_compact_definition():
    result = choose_strategy(
        _label(),
        _stage1(),
        _stage2(name_sufficiency="需要原释义"),
        reference={"关键词策略代码": "K2", "Prompt是否需要释义": "建议给精简释义：突出任务形态/步骤边界"},
    )

    assert result["mode"] == "compact_definition"
    assert result["automation"] == "提供精简老师释义后由LLM裁决"
    assert result["manual_review_required"] is False


def test_l1_with_reference_request_for_compact_definition_uses_it():
    result = choose_strategy(
        _label(),
        _stage1(),
        _stage2(),
        reference={"关键词策略代码": "K1", "Prompt是否需要释义": "建议给精简释义：突出任务形态/步骤边界"},
    )

    assert result["mode"] == "compact_definition"
    assert result["teacher_definition_required"] is True


def test_build_strategy_records_keeps_both_ds_stages_and_reference():
    label = _label()
    stage1_record = {"label_id": "1", "parsed_response": _stage1()}
    stage2_record = {"label_id": "1", "category": "L1", "parsed_response": _stage2()}

    records = build_strategy_records(
        [label],
        {"1": stage1_record},
        {"1": stage2_record},
        reference_by_id={"1": {"关键词策略代码": "K1"}},
    )

    assert records[0]["stage1"]["core_meaning"] == "DNA复制方式"
    assert records[0]["stage2_judge"]["alignment_score"] == 5
    assert records[0]["reference_strategy"]["关键词策略代码"] == "K1"
    assert records[0]["strategy"]["prompt_fields"] == ["label_name"]
