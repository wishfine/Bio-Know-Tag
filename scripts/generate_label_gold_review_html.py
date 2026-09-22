#!/usr/bin/env python3
"""Build the combined volatility/definition-ablation teacher gold-label HTML."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_labels(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row["label_id"]): row for row in read_jsonl(path) if row.get("label_id")}


def card(label_id: str, labels: dict[str, dict[str, Any]], *, source: str = "") -> dict[str, Any]:
    label = labels.get(str(label_id), {})
    return {
        "label_id": str(label_id),
        "label_name": str(label.get("label_name") or "未知Label"),
        "label_path": str(label.get("label_path") or "").replace("->", "@"),
        "source": source,
    }


def dedupe_involved(
    row: dict[str, Any], labels: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}

    def add(label_id: str, source: str, original: bool = False) -> None:
        label_id = str(label_id)
        if not label_id:
            return
        value = by_id.setdefault(label_id, card(label_id, labels, source=source))
        sources = value.setdefault("sources", [])
        if source and source not in sources:
            sources.append(source)
        value["is_original"] = bool(value.get("is_original") or original)

    for key, source in (
        ("top25_labels", "Top25召回"),
        ("legacy_labels", "Top25+旧knw_ids结果"),
        ("added_candidate_labels", "新增旧knw_ids候选"),
        ("selected_added_legacy_labels", "新增旧Label被选中"),
        ("definition_changed_labels", "释义消融变化"),
    ):
        for item in row.get(key) or []:
            add(str(item.get("label_id") or ""), source)

    for label_id in row.get("original_knw_label_ids") or []:
        add(str(label_id), "原始knw_ids（可识别）", original=True)

    result = list(by_id.values())
    result.sort(key=lambda value: (not value.get("is_original"), value["label_name"], value["label_id"]))
    return result


def load_definition_changes(
    root: Path,
    labels: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for task in read_jsonl(root / "paired_tasks.jsonl"):
        key = str(task.get("pair_id") or "")
        if key and (key not in tasks or task.get("condition") == "name_plus_definition"):
            tasks[key] = task
    name = {str(row.get("task_id") or ""): row for row in read_jsonl(root / "ds-name-only/results.jsonl")}
    definition = {str(row.get("task_id") or ""): row for row in read_jsonl(root / "ds-name-plus-definition/results.jsonl")}
    by_question: dict[str, dict[str, Any]] = {}
    for pair_id in sorted(set(name) & set(definition)):
        left, right = name[pair_id], definition[pair_id]
        task = tasks.get(pair_id, {})
        left_match, right_match = bool(left.get("match")), bool(right.get("match"))
        left_score = float(left.get("relevance_score") or 0.0)
        right_score = float(right.get("relevance_score") or 0.0)
        delta = right_score - left_score
        if left_match == right_match and abs(delta) < 0.20:
            continue
        qid = str(task.get("question_id") or left.get("question_id") or "")
        if not qid:
            continue
        label_id = str(task.get("label_id") or left.get("label_id") or "")
        change = "false_to_true" if not left_match and right_match else "true_to_false" if left_match and not right_match else "score_shift"
        item = by_question.setdefault(qid, {
            "question_id": qid,
            "parent_id": qid,
            "unit_type": str(task.get("unit_type") or "standalone"),
            "perturbation_group": "D_definition_ablation",
            "selection_jaccard": None,
            "base_candidate_count": 0,
            "augmented_candidate_count": 0,
            "flags": {"definition_ablation": True},
            "stem_image_url": "",
            "analysis_image_url": "",
            "parent_stem_image_url": "",
            "parent_analysis_image_url": "",
            "stem": str(task.get("stem") or (task.get("question") or {}).get("stem") or ""),
            "options": str(task.get("options") or (task.get("question") or {}).get("options") or ""),
            "answer_text": str(task.get("answer_text") or (task.get("question") or {}).get("answer_text") or ""),
            "analysis": str(task.get("analysis") or (task.get("question") or {}).get("analysis") or ""),
            "top25_labels": [],
            "legacy_labels": [],
            "shared_labels": [],
            "top25_only_labels": [],
            "legacy_only_labels": [],
            "added_candidate_labels": [],
            "selected_added_legacy_labels": [],
            "original_knw_label_ids": [],
            "original_knw_note": "该题来自释义消融正样本，测试中的历史Label视为原始knw_id相关Label。",
            "definition_changed_labels": [],
            "definition_changes": [],
            "source_reasons": ["definition_ablation"],
        })
        if label_id not in item["original_knw_label_ids"]:
            item["original_knw_label_ids"].append(label_id)
        if label_id not in [x.get("label_id") for x in item["definition_changed_labels"]]:
            item["definition_changed_labels"].append(card(label_id, labels, source="释义消融变化"))
        item["definition_changes"].append({
            "pair_id": pair_id,
            "label_id": label_id,
            "label_name": str(task.get("label_name") or labels.get(label_id, {}).get("label_name") or ""),
            "change": change,
            "name_only_match": left_match,
            "definition_match": right_match,
            "name_only_score": left_score,
            "definition_score": right_score,
            "score_delta": round(delta, 6),
            "sample_source": task.get("sample_source"),
            "stratum": task.get("stratum"),
        })
    return by_question


def merge_rows(
    volatility_html: Path,
    definition_root: Path,
    labels_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text = volatility_html.read_text(encoding="utf-8")
    match = re.search(r'<script id="review-data" type="application/json">(.*?)</script>', text, re.S)
    if not match:
        raise ValueError("volatility HTML has no embedded review-data")
    volatility_rows = json.loads(match.group(1))
    labels = load_labels(labels_path)
    definition_rows = load_definition_changes(definition_root, labels)
    by_question = {str(row["question_id"]): row for row in volatility_rows}
    for row in volatility_rows:
        row.setdefault("source_reasons", ["volatility"])
        row["original_knw_label_ids"] = [
            str(item.get("label_id"))
            for item in row.get("legacy_only_labels", [])
            if item.get("label_id")
        ]
        row["original_knw_note"] = "当前导出无法区分共享候选中的原始knw_id；高亮的是可识别的旧Label/旧ID相关集合。"
    for qid, definition_row in definition_rows.items():
        if qid in by_question:
            row = by_question[qid]
            row.setdefault("source_reasons", []).append("definition_ablation")
            row.setdefault("definition_changes", []).extend(definition_row["definition_changes"])
            row.setdefault("definition_changed_labels", []).extend(definition_row["definition_changed_labels"])
            row["original_knw_label_ids"] = sorted(set(row.get("original_knw_label_ids", [])) | set(definition_row["original_knw_label_ids"]))
            row["original_knw_note"] = "该题同时出现在波动实验和释义消融实验；原始Label高亮包含可识别旧Label及释义消融中的历史Label。"
        else:
            by_question[qid] = definition_row
    rows = list(by_question.values())
    for row in rows:
        row["involved_labels"] = dedupe_involved(row, labels)
    group_order = {"A_same_candidate_set": 0, "B_candidate_set_expanded_added_not_selected": 1, "C_added_legacy_selected": 2, "D_definition_ablation": 3}
    rows.sort(key=lambda row: (group_order.get(row.get("perturbation_group"), 99), str(row.get("question_id"))))
    catalog = [card(label_id, labels) for label_id in sorted(labels)]
    return rows, catalog


def patch_template(template: str) -> str:
    extra_css = ".gold-panel{padding:15px;border:1px solid #56695f;border-radius:14px;background:#14201a}.gold-title{font-weight:900;color:#f4ead2}.gold-note{margin:6px 0 12px;color:#b8c6be;font-size:11px;line-height:1.6}.gold-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.gold-option{display:flex;gap:9px;align-items:flex-start;padding:10px;border:1px solid #506058;background:#24332c;border-radius:10px;color:#eef4ef}.gold-option.original{border-color:#e4b58e;background:#3b3024}.gold-option input{margin-top:3px;accent-color:#9fc6b1}.gold-option b{display:block}.gold-option small{display:block;color:#b8c6be;font-size:10px;line-height:1.4;margin-top:3px}.original-badge{display:inline-block;margin-left:5px;padding:2px 5px;border-radius:5px;background:#e4b58e;color:#382718;font-size:10px;font-weight:900}.source-badge{display:inline-block;margin:2px 4px 0 0;padding:2px 5px;border-radius:5px;background:#506058;color:#dcebe2;font-size:9px}.gold-other{margin-top:10px}.pill.D{background:#eadff1;color:#704b7f}@media(max-width:900px){.gold-grid{grid-template-columns:1fr}}"
    template = template.replace('</style>', extra_css + '</style>')
    template = template.replace(
        '<button class="filter" data-group="C_added_legacy_selected">C · 新旧Label被选中 <b id="countC">0</b></button>',
        '<button class="filter" data-group="C_added_legacy_selected">C · 新旧Label被选中 <b id="countC">0</b></button><button class="filter" data-group="D_definition_ablation">D · 释义导致变化 <b id="countD">0</b></button>'
    )
    template = template.replace(
        "const GROUP_META={A_same_candidate_set:{short:'A',name:'候选集合未变化'},B_candidate_set_expanded_added_not_selected:{short:'B',name:'增加候选但未选中'},C_added_legacy_selected:{short:'C',name:'新增旧 Label 被选中'}}",
        "const GROUP_META={A_same_candidate_set:{short:'A',name:'候选集合未变化'},B_candidate_set_expanded_added_not_selected:{short:'B',name:'增加候选但未选中'},C_added_legacy_selected:{short:'C',name:'新增旧 Label 被选中'},D_definition_ablation:{short:'D',name:'释义导致变化'}}"
    )
    template = template.replace(
        "function currentReview(qid){return reviews[qid]||{decision:'',correct_labels:[],note:'',reviewed_at:''}}",
        "function currentReview(qid){return reviews[qid]||{decision:'',gold_label_ids:[],other_label_text:'',note:'',reviewed_at:''}}"
    )
    template = re.sub(
        r"function renderCard\(q,index\)\{.*?\nfunction render\(\)",
        """function renderGoldOptions(q,r){return (q.involved_labels||[]).map(x=>{const checked=(r.gold_label_ids||[]).includes(x.label_id);const original=x.is_original?'<span class=original-badge>原始 knw_id</span>':'';const sources=(x.sources||[]).map(s=>`<span class=source-badge>${esc(s)}</span>`).join('');return `<label class=\"gold-option ${x.is_original?'original':''}\"><input type=checkbox ${checked?'checked':''} onchange=\"toggleGold('${q.question_id}','${x.label_id}',this.checked)\"><span><b>${esc(x.label_name)} ${original}</b><small>${sources}<br>${esc(x.label_path)}</small></span></label>`}).join('')||'<div class=empty>没有可用候选Label</div>'}function renderCard(q,index){const r=currentReview(q.question_id),g=GROUP_META[q.perturbation_group]||GROUP_META.D_definition_ablation,imgs=[imageBox(q.parent_stem_image_url,'父题题干图'),imageBox(q.stem_image_url,'当前题干图'),imageBox(q.parent_analysis_image_url,'父题解析图'),imageBox(q.analysis_image_url,'当前题解析图')].join('')||'<div class=no-image>该题没有可加载的题干/解析图片</div>',added=q.added_candidate_labels.length?`<div class=added><b>旧 knw_ids 新增候选：</b>${q.added_candidate_labels.map(x=>esc(x.label_name)).join('、')}</div>`:'',sourceNote=(q.source_reasons||[]).map(x=>`<span class=source-badge>${esc(x)}</span>`).join(''),definitionNote=(q.original_knw_note||'');return `<article class=card id=\"q-${esc(q.question_id)}\"><header class=card-head><span class=qid>#${index+1} · ID ${esc(q.question_id)}</span><span class=\"pill ${g.short}\">${g.short} · ${esc(g.name)}</span><span class=meta>${esc(q.unit_type)} · 来源 ${sourceNote}</span></header><div class=body><div class=images>${imgs}</div>${q.stem?`<div><div class=review-note>题干文本</div><div class=text-box>${esc(q.stem)}</div></div>`:''}${q.options?`<div><div class=review-note>选项</div><div class=text-box>${esc(q.options)}</div></div>`:''}${q.answer_text?`<div><div class=review-note>答案</div><div class=\"text-box answer\">${esc(q.answer_text)}</div></div>`:''}${q.analysis?`<div><div class=review-note>解析</div><div class=\"text-box analysis\">${esc(q.analysis)}</div></div>`:''}${added}<div class=compare><div class=\"run before\"><h3>纯 Hybrid Top25</h3><div class=run-sub>橙色为仅本次选中；灰色为两次共有</div>${labelList(q.top25_labels,q.top25_only_labels.map(x=>x.label_id))}</div><div class=\"run after\"><h3>Top25 + 旧 knw_ids</h3><div class=run-sub>橙色为仅本次选中；灰色为两次共有</div>${labelList(q.legacy_labels,q.legacy_only_labels.map(x=>x.label_id))}</div></div><section class=gold-panel><div class=gold-title>最终金标 Label（请勾选）</div><div class=gold-note>所有涉及到的 Label 都列在下面。橙色边框表示可识别的原始/旧 knw_id；共享候选中的原始ID可能无法从当前导出中完全区分。${definitionNote?`<br>${esc(definitionNote)}`:''}</div><div class=gold-grid>${renderGoldOptions(q,r)}</div><div class=gold-other><input value=\"${esc(r.other_label_text||'')}\" placeholder=\"其他 Label（可输入名称、路径或ID）\" oninput=\"setOtherLabel('${q.question_id}',this.value)\"></div></section></div><section class=review><div class=review-head><h2>教师审核</h2><span class=saved>${r.reviewed_at?'已保存':'未审核'}</span></div><div class=review-note>先判断两组结果，再勾选最终金标。可多选；如果候选中没有合理Label，请填写“其他 Label”。</div><div class=choices>${DECISIONS.map(([v,l])=>`<button class=\"choice ${r.decision===v?'selected':''}\" onclick=\"setDecision('${q.question_id}','${v}')\">${l}</button>`).join('')}</div><div class=details><textarea rows=3 placeholder=\"备注：为什么修改、Label边界问题、建议补充的释义…\" oninput=\"setNote('${q.question_id}',this.value)\">${esc(r.note||'')}</textarea></div></section></article>`}function render()""",
        template,
        flags=re.S,
    )
    template = template.replace(
        "function setDecision(qid,v){const r=currentReview(qid);r.decision=v;r.reviewed_at=new Date().toISOString();if(v!=='BOTH_WRONG')r.correct_labels=[];reviews[qid]=r;save();render()}function setNote(qid,v){const r=currentReview(qid);r.note=v;r.reviewed_at=new Date().toISOString();reviews[qid]=r;save()}",
        "function setDecision(qid,v){const r=currentReview(qid);r.decision=v;r.reviewed_at=new Date().toISOString();reviews[qid]=r;save();render()}function toggleGold(qid,id,checked){const r=currentReview(qid);r.gold_label_ids=r.gold_label_ids||[];if(checked&&!r.gold_label_ids.includes(id))r.gold_label_ids.push(id);if(!checked)r.gold_label_ids=r.gold_label_ids.filter(x=>x!==id);r.reviewed_at=new Date().toISOString();reviews[qid]=r;save();render()}function setOtherLabel(qid,v){const r=currentReview(qid);r.other_label_text=v;r.reviewed_at=new Date().toISOString();reviews[qid]=r;save()}function setNote(qid,v){const r=currentReview(qid);r.note=v;r.reviewed_at=new Date().toISOString();reviews[qid]=r;save()}"
    )
    template = template.replace(
        "const c={A:0,B:0,C:0};DATA.forEach(q=>c[GROUP_META[q.perturbation_group].short]++);document.getElementById('countALL').textContent=DATA.length;for(const k of ['A','B','C'])document.getElementById('count'+k).textContent=c[k]",
        "const c={A:0,B:0,C:0,D:0};DATA.forEach(q=>c[(GROUP_META[q.perturbation_group]||GROUP_META.D_definition_ablation).short]++);document.getElementById('countALL').textContent=DATA.length;for(const k of ['A','B','C','D'])document.getElementById('count'+k).textContent=c[k]"
    )
    template = template.replace(
        "top25_label_ids:q.top25_labels.map(x=>x.label_id),legacy_label_ids:q.legacy_labels.map(x=>x.label_id),...reviews[q.question_id]",
        "top25_label_ids:q.top25_labels.map(x=>x.label_id),legacy_label_ids:q.legacy_labels.map(x=>x.label_id),original_knw_label_ids:q.original_knw_label_ids||[],involved_label_ids:(q.involved_labels||[]).map(x=>x.label_id),...reviews[q.question_id]"
    )
    return template


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volatility-html", type=Path, required=True)
    parser.add_argument("--definition-run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path)
    args = parser.parse_args()
    rows, catalog = merge_rows(args.volatility_html, args.definition_run, args.labels)
    template = Path("src/bio_know_tag/volatility_review_batch_template.html").read_text(encoding="utf-8")
    template = patch_template(template)
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    catalog_payload = json.dumps(catalog, ensure_ascii=False).replace("</", "<\\/")
    html = template.replace("__REVIEW_DATA__", payload).replace("__LABEL_CATALOG__", catalog_payload)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html, encoding="utf-8")
    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "output_html": str(args.output_html),
        "output_jsonl": str(args.output_jsonl) if args.output_jsonl else None,
        "review_questions": len(rows),
        "volatility_questions": sum("volatility" in row.get("source_reasons", []) for row in rows),
        "definition_ablation_questions": sum("definition_ablation" in row.get("source_reasons", []) for row in rows),
        "group_counts": dict(__import__("collections").Counter(row.get("perturbation_group") for row in rows)),
        "labels": len(catalog),
    }
    args.output_html.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
