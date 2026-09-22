#!/usr/bin/env python3
"""Render a generic teacher gold-label review HTML from a selection JSONL.

The input schema is deliberately subject-agnostic.  Required fields are
``question_id`` and ``involved_label_ids``; question text, image URLs, source
labels, and review reasons are optional.  The output stores teacher choices in
browser localStorage and exports them as JSONL.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from generate_label_gold_review_html import patch_template


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["label_id"]): row for row in read_jsonl(path) if row.get("label_id")}


def normalize_card(value: Any, labels: dict[str, dict[str, Any]], source: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        label_id = str(value.get("label_id") or "")
        card = dict(value)
        card["label_id"] = label_id
        card["label_name"] = str(value.get("label_name") or labels.get(label_id, {}).get("label_name") or "未知Label")
        card["label_path"] = str(value.get("label_path") or labels.get(label_id, {}).get("label_path") or "").replace("->", "@")
        if source:
            card.setdefault("source", source)
        return card
    label_id = str(value)
    return {
        "label_id": label_id,
        "label_name": str(labels.get(label_id, {}).get("label_name") or "未知Label"),
        "label_path": str(labels.get(label_id, {}).get("label_path") or "").replace("->", "@"),
        "source": source,
    }


def normalize_row(row: dict[str, Any], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = dict(row)
    for field in (
        "top25_labels", "legacy_labels", "shared_labels", "top25_only_labels",
        "legacy_only_labels", "added_candidate_labels", "selected_added_legacy_labels",
    ):
        result[field] = [normalize_card(value, labels, field) for value in row.get(field) or []]
    original_ids = {str(value) for value in row.get("original_knw_label_ids") or []}
    involved: dict[str, dict[str, Any]] = {}
    for field, source in (
        ("top25_labels", "Top25召回"),
        ("legacy_labels", "Top25+旧knw_ids结果"),
        ("added_candidate_labels", "新增旧knw_ids候选"),
        ("selected_added_legacy_labels", "新增旧Label被选中"),
    ):
        for value in result[field]:
            label_id = str(value["label_id"])
            item = involved.setdefault(label_id, dict(value))
            item.setdefault("sources", [])
            if source not in item["sources"]:
                item["sources"].append(source)
    for label_id in original_ids:
        item = involved.setdefault(label_id, normalize_card(label_id, labels, "原始knw_ids（可识别）"))
        item["is_original"] = True
        item.setdefault("sources", []).append("原始knw_ids（可识别）")
    for value in row.get("definition_changed_labels") or []:
        label_id = str(value.get("label_id") or "")
        if not label_id:
            continue
        item = involved.setdefault(label_id, normalize_card(value, labels, "释义消融变化"))
        item.setdefault("sources", []).append("释义消融变化")
    for item in involved.values():
        item["sources"] = list(dict.fromkeys(item.get("sources") or []))
        item["is_original"] = bool(item.get("is_original") or item["label_id"] in original_ids)
    result["involved_labels"] = sorted(
        involved.values(),
        key=lambda value: (not value.get("is_original"), value.get("label_name", ""), value["label_id"]),
    )
    result.setdefault("source_reasons", [])
    result.setdefault("original_knw_label_note", "请根据数据来源确认原始knw_ids；橙色项是当前可识别的原始/旧Label。")
    result.setdefault("unit_type", "")
    result.setdefault("stem", "")
    result.setdefault("options", "")
    result.setdefault("answer_text", "")
    result.setdefault("analysis", "")
    for field in ("stem_image_url", "analysis_image_url", "parent_stem_image_url", "parent_analysis_image_url"):
        result.setdefault(field, "")
    result.setdefault("added_candidate_labels", [])
    return result


def patch_gold_only_template(template: str) -> str:
    """Keep the image-only review surface and show only the gold-label form."""
    template = template.replace(
        ".gold-other{margin-top:10px}",
        ".gold-other{margin-top:10px}.gold-other input{width:100%;display:block}.gold-empty{display:block;margin-top:10px;color:#dfeae2;font-size:12px}.gold-empty input{accent-color:#e4b58e;margin-right:6px}",
    )
    template = template.replace(
        "function currentReview(qid){return reviews[qid]||{decision:'',correct_labels:[],note:'',reviewed_at:''}}",
        "function currentReview(qid){return reviews[qid]||{decision:'',gold_label_ids:[],other_label_text:'',no_label:false,note:'',reviewed_at:''}}",
    )
    render_function = r'''function renderGoldOptions(q,r){return (q.involved_labels||[]).map(x=>{const checked=(r.gold_label_ids||[]).includes(x.label_id);const original=x.is_original?'<span class=original-badge>原始 knw_id</span>':'';const sources=(x.sources||[]).map(s=>`<span class=source-badge>${esc(s)}</span>`).join('');return `<label class="gold-option ${x.is_original?'original':''}"><input type=checkbox ${checked?'checked':''} onchange="toggleGold('${q.question_id}','${x.label_id}',this.checked)"><span><b>${esc(x.label_name)} ${original}</b><small>${sources}<br>${esc(x.label_path)}</small></span></label>`}).join('')||'<div class=empty>没有可用候选Label</div>'}function renderCard(q,index){const r=currentReview(q.question_id),g=GROUP_META[q.perturbation_group]||GROUP_META.D_definition_ablation,imgs=[imageBox(q.parent_stem_image_url,'父题题干图'),imageBox(q.stem_image_url,'当前题干图'),imageBox(q.parent_analysis_image_url,'父题解析图'),imageBox(q.analysis_image_url,'当前题解析图')].join('')||'<div class=no-image>该题没有可加载的图片</div>',sourceNote=(q.source_reasons||[]).map(x=>`<span class=source-badge>${esc(x)}</span>`).join(''),definitionNote=(q.original_knw_label_note||q.original_knw_note||'');return `<article class=card id="q-${esc(q.question_id)}"><header class=card-head><span class=qid>#${index+1} · ID ${esc(q.question_id)}</span><span class="pill ${g.short}">${g.short} · ${esc(g.name)}</span><span class=meta>${esc(q.unit_type)} · 来源 ${sourceNote}</span></header><div class=body><div class=images>${imgs}</div><section class=gold-panel><div class=gold-title>最终金标 Label（请勾选）</div><div class=gold-note>所有涉及到的 Label 都列在下面。橙色边框表示可识别的原始/旧 knw_id。${definitionNote?`<br>${esc(definitionNote)}`:''}</div><div class=gold-grid>${renderGoldOptions(q,r)}</div><div class=gold-other><input value="${esc(r.other_label_text||'')}" placeholder="其他 Label / 老师建议（名称、路径或ID）" oninput="setOtherLabel('${q.question_id}',this.value)"></div><label class=gold-empty><input type=checkbox ${r.no_label?'checked':''} onchange="setNoLabel('${q.question_id}',this.checked)">确认：本题不应标任何 Label</label><textarea rows=3 placeholder="备注：为什么修改、Label边界问题、建议补充的释义…" oninput="setNote('${q.question_id}',this.value)">${esc(r.note||'')}</textarea></section></div></article>`}function render()'''
    template = re.sub(
        r"function renderCard\(q,index\)\{.*?function render\(\)",
        render_function,
        template,
        flags=re.S,
    )
    template = template.replace(
        "function setDecision(qid,v){const r=currentReview(qid);r.decision=v;r.reviewed_at=new Date().toISOString();reviews[qid]=r;save();render()}function toggleGold",
        "function setDecision(qid,v){const r=currentReview(qid);r.decision=v;r.reviewed_at=new Date().toISOString();reviews[qid]=r;save();render()}function setNoLabel(qid,v){const r=currentReview(qid);r.no_label=v;if(v)r.gold_label_ids=[];r.reviewed_at=new Date().toISOString();reviews[qid]=r;save();render()}function toggleGold",
    )
    template = template.replace(
        "const reviewed=filtered.filter(q=>reviews[q.question_id]?.decision).length",
        "const reviewed=filtered.filter(q=>{const r=reviews[q.question_id]||{};return r.no_label||(r.gold_label_ids||[]).length||String(r.other_label_text||'').trim()}).length",
    )
    return template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-jsonl", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--title", default="Label 人工金标审阅台")
    parser.add_argument("--storage-key", default="label-gold-review-v1")
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--recommended-only", action="store_true", help="render only rows marked recommended=true")
    args = parser.parse_args()
    if args.page_size < 1:
        raise SystemExit("--page-size must be positive")

    labels = load_labels(args.labels)
    input_rows = read_jsonl(args.selection_jsonl)
    if args.recommended_only:
        input_rows = [row for row in input_rows if row.get("recommended")]
    rows = [normalize_row(row, labels) for row in input_rows]
    catalog = [normalize_card(label_id, labels) for label_id in sorted(labels)]
    template = Path("src/bio_know_tag/volatility_review_batch_template.html").read_text(encoding="utf-8")
    template = patch_gold_only_template(patch_template(template))
    template = template.replace("高中生物 Label 波动复核台", args.title)
    template = template.replace("每页 50 题 · 图片审核", f"每页 {args.page_size} 题 · 人工金标")
    template = template.replace("50 QUESTIONS / PAGE · IMAGE ONLY", f"{args.page_size} QUESTIONS / PAGE · GOLD LABEL REVIEW")
    template = template.replace("PAGE_SIZE=50", f"PAGE_SIZE={args.page_size}")
    template = template.replace("bio-label-volatility-review-v2-image-batch", args.storage_key)
    template = template.replace("__REVIEW_DATA__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/"))
    template = template.replace("__LABEL_CATALOG__", json.dumps(catalog, ensure_ascii=False).replace("</", "<\\/"))
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(template, encoding="utf-8")
    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "output_html": str(args.output_html),
        "output_jsonl": str(args.output_jsonl) if args.output_jsonl else None,
        "questions": len(rows),
        "labels": len(catalog),
        "page_size": args.page_size,
        "storage_key": args.storage_key,
        "selection_input": str(args.selection_jsonl),
        "recommended_only": args.recommended_only,
    }
    args.output_html.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
