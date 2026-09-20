"""Build a standalone teacher-review HTML for volatile label assignments."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable


GROUP_ORDER = {
    "A_same_candidate_set": 0,
    "B_candidate_set_expanded_added_not_selected": 1,
    "C_added_legacy_selected": 2,
}


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield value


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_labels(path: str | Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["label_id"]): row
        for row in _read_jsonl(path)
        if row.get("label_id")
    }


def _load_changed_groups(
    group_paths: Iterable[str | Path],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    seen: set[str] = set()
    for raw_path in group_paths:
        path = Path(raw_path)
        hashes[str(path)] = _file_sha256(path)
        for row in _read_jsonl(path):
            question_id = str(row.get("question_id") or "")
            if not question_id:
                raise ValueError(f"{path} contains a row without question_id")
            if row.get("same_output") is True:
                continue
            if question_id in seen:
                raise ValueError(f"duplicate changed question: {question_id}")
            seen.add(question_id)
            rows.append(row)
    rows.sort(
        key=lambda row: (
            GROUP_ORDER.get(str(row.get("perturbation_group")), 99),
            str(row.get("question_id")),
        )
    )
    return rows, hashes


def _index_selected_rows(
    path: str | Path,
    wanted_ids: set[str],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        question_id = str(row.get("question_id") or "")
        if question_id in wanted_ids:
            index[question_id] = row
    return index


def _label_cards(
    label_ids: Iterable[str], labels: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    cards = []
    for label_id in label_ids:
        label = labels.get(str(label_id), {})
        cards.append(
            {
                "label_id": str(label_id),
                "label_name": str(label.get("label_name") or "未知Label"),
                "label_path": str(label.get("label_path") or "").replace("->", "@"),
            }
        )
    return cards


def _build_review_row(
    comparison: dict[str, Any],
    unit: dict[str, Any],
    images: dict[str, Any],
    labels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    top25_ids = [str(value) for value in comparison.get("top25_selected_label_ids", [])]
    legacy_ids = [
        str(value) for value in comparison.get("legacy_selected_label_ids", [])
    ]
    shared_ids = [
        str(value) for value in comparison.get("shared_selected_label_ids", [])
    ]
    top25_only_ids = [
        str(value) for value in comparison.get("top25_only_selected_label_ids", [])
    ]
    legacy_only_ids = [
        str(value) for value in comparison.get("legacy_only_selected_label_ids", [])
    ]
    added_candidate_ids = [
        str(value) for value in comparison.get("added_candidate_ids", [])
    ]
    return {
        "question_id": str(comparison["question_id"]),
        "parent_id": str(unit.get("parent_id") or comparison["question_id"]),
        "unit_type": str(unit.get("unit_type") or ""),
        "perturbation_group": str(comparison.get("perturbation_group") or ""),
        "selection_jaccard": comparison.get("selection_jaccard"),
        "base_candidate_count": comparison.get("base_candidate_count"),
        "augmented_candidate_count": comparison.get("augmented_candidate_count"),
        "flags": unit.get("flags") or {},
        "stem_image_url": str(images.get("stem_image_url") or ""),
        "analysis_image_url": str(images.get("analysis_image_url") or ""),
        "parent_stem_image_url": str(images.get("parent_stem_image_url") or ""),
        "parent_analysis_image_url": str(
            images.get("parent_analysis_image_url") or ""
        ),
        "top25_labels": _label_cards(top25_ids, labels),
        "legacy_labels": _label_cards(legacy_ids, labels),
        "shared_labels": _label_cards(shared_ids, labels),
        "top25_only_labels": _label_cards(top25_only_ids, labels),
        "legacy_only_labels": _label_cards(legacy_only_ids, labels),
        "added_candidate_labels": _label_cards(added_candidate_ids, labels),
        "selected_added_legacy_labels": _label_cards(
            comparison.get("selected_added_legacy_ids", []), labels
        ),
    }


def build_volatility_review_html(
    *,
    group_paths: Iterable[str | Path],
    units_path: str | Path,
    labels_path: str | Path,
    output_html_path: str | Path,
    image_context_path: str | Path | None = None,
    output_jsonl_path: str | Path | None = None,
    limit: int | None = None,
    seed: int = 20260920,
) -> dict[str, Any]:
    comparisons, group_hashes = _load_changed_groups(group_paths)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rng = random.Random(seed)
        comparisons = rng.sample(comparisons, min(limit, len(comparisons)))
        comparisons.sort(
            key=lambda row: (
                GROUP_ORDER.get(str(row.get("perturbation_group")), 99),
                str(row.get("question_id")),
            )
        )

    wanted_ids = {str(row["question_id"]) for row in comparisons}
    units = _index_selected_rows(units_path, wanted_ids)
    missing_units = sorted(wanted_ids - set(units))
    if missing_units:
        preview = ", ".join(missing_units[:10])
        raise ValueError(f"missing {len(missing_units)} units, examples: {preview}")
    images = (
        _index_selected_rows(image_context_path, wanted_ids)
        if image_context_path
        else {}
    )
    labels = _load_labels(labels_path)
    review_rows = [
        _build_review_row(
            comparison,
            units[str(comparison["question_id"])],
            images.get(str(comparison["question_id"]), {}),
            labels,
        )
        for comparison in comparisons
    ]

    label_catalog = [
        {
            "label_id": label_id,
            "label_name": str(label.get("label_name") or ""),
            "label_path": str(label.get("label_path") or "").replace("->", "@"),
        }
        for label_id, label in sorted(labels.items())
    ]
    group_counts: dict[str, int] = {}
    for row in review_rows:
        group = row["perturbation_group"]
        group_counts[group] = group_counts.get(group, 0) + 1

    if output_jsonl_path:
        jsonl_path = Path(output_jsonl_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8", newline="\n") as output:
            for row in review_rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload = json.dumps(review_rows, ensure_ascii=False).replace("</", "<\\/")
    catalog_payload = json.dumps(label_catalog, ensure_ascii=False).replace(
        "</", "<\\/"
    )
    template_path = Path(__file__).with_name("volatility_review_batch_template.html")
    html = template_path.read_text(encoding="utf-8").replace(
        "__REVIEW_DATA__", payload
    ).replace(
        "__LABEL_CATALOG__", catalog_payload
    )
    output_path = Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    report = {
        "output_html": str(output_path),
        "review_questions": len(review_rows),
        "group_counts": group_counts,
        "questions_with_stem_image": sum(
            bool(row["stem_image_url"] or row["parent_stem_image_url"])
            for row in review_rows
        ),
        "questions_with_analysis_image": sum(
            bool(row["analysis_image_url"] or row["parent_analysis_image_url"])
            for row in review_rows
        ),
        "labels": len(label_catalog),
        "input_sha256": {
            "groups": group_hashes,
            "units": _file_sha256(units_path),
            "labels": _file_sha256(labels_path),
            **(
                {"image_context": _file_sha256(image_context_path)}
                if image_context_path
                else {}
            ),
        },
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>高中生物 Label 波动复核台</title>
  <style>
    :root{--ink:#19231f;--muted:#6c766f;--paper:#f4f0e7;--card:#fffdf7;--line:#d7d1c4;--green:#1f6952;--green2:#dcece4;--red:#a74736;--red2:#f7e1dc;--amber:#9a6728;--amber2:#f4e7cb;--blue:#315d7c;--blue2:#dce8ef;--shadow:0 18px 55px rgba(35,47,40,.12)}
    *{box-sizing:border-box} body{margin:0;color:var(--ink);background:radial-gradient(circle at 9% 4%,#fff8de 0,transparent 27%),linear-gradient(135deg,#ece9df,#f8f5ed 55%,#e6eee8);font-family:"Noto Serif SC","Songti SC",Georgia,serif;min-height:100vh}
    button,input,textarea,select{font:inherit} button{cursor:pointer}.shell{display:grid;grid-template-columns:300px minmax(0,1fr);min-height:100vh}.sidebar{position:sticky;top:0;height:100vh;padding:26px 22px;border-right:1px solid rgba(65,75,68,.18);background:rgba(249,247,239,.86);backdrop-filter:blur(18px);overflow:auto}.brand{font-size:27px;font-weight:900;line-height:1.1;letter-spacing:-1px}.brand small{display:block;margin-top:9px;color:var(--muted);font:600 12px/1.5 ui-monospace,monospace;letter-spacing:.08em}.progress-ring{margin:24px 0;padding:18px;border:1px solid var(--line);background:rgba(255,255,255,.6);border-radius:18px}.progress-big{font:800 31px/1 ui-monospace,monospace}.progress-caption{margin-top:7px;color:var(--muted);font-size:13px}.bar{height:7px;background:#dedbd2;border-radius:99px;margin-top:13px;overflow:hidden}.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--green),#62a47f);width:0;transition:.25s}.filter-title{font-size:12px;font-weight:800;color:var(--muted);letter-spacing:.12em;margin:20px 0 8px}.filter{display:flex;width:100%;align-items:center;justify-content:space-between;padding:10px 12px;margin:6px 0;border:1px solid transparent;border-radius:11px;background:transparent;color:var(--ink);text-align:left}.filter.active{background:var(--ink);color:#fff}.filter b{font:700 12px ui-monospace,monospace}.search{width:100%;border:1px solid var(--line);background:#fffdf8;padding:11px;border-radius:10px}.side-actions{display:grid;gap:8px;margin-top:20px}.side-actions button{border:1px solid var(--line);border-radius:10px;padding:10px;background:#fffdf8}.side-actions .primary{background:var(--green);color:white;border-color:var(--green)}
    main{padding:28px clamp(20px,4vw,64px) 64px;max-width:1500px;width:100%;margin:auto}.topbar{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:20px}.eyebrow{font:800 12px ui-monospace,monospace;color:var(--green);letter-spacing:.12em}.topbar h1{margin:4px 0 0;font-size:clamp(25px,3vw,42px);letter-spacing:-1.5px}.pager{display:flex;align-items:center;gap:8px}.pager button{border:1px solid var(--line);background:var(--card);border-radius:11px;padding:9px 14px}.pager input{width:76px;padding:9px;border:1px solid var(--line);border-radius:10px;text-align:center}.card{background:rgba(255,253,247,.94);border:1px solid rgba(117,116,104,.23);border-radius:24px;box-shadow:var(--shadow);overflow:hidden}.card-head{padding:18px 22px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}.qid{font:800 14px ui-monospace,monospace}.pill{padding:6px 10px;border-radius:99px;font:800 11px ui-monospace,monospace}.pill.A{background:var(--blue2);color:var(--blue)}.pill.B{background:var(--amber2);color:var(--amber)}.pill.C{background:var(--red2);color:var(--red)}.meta{color:var(--muted);font-size:12px}.content{padding:24px;display:grid;gap:22px}.section-title{font:900 13px ui-monospace,monospace;color:var(--green);letter-spacing:.08em;text-transform:uppercase;margin-bottom:9px}.text-box{white-space:pre-wrap;line-height:1.85;background:#faf8f1;border-left:4px solid #c9c1b0;padding:15px 17px;border-radius:0 12px 12px 0}.answer{border-left-color:var(--green);background:#eff7f1}.analysis{border-left-color:#7d6e55}.images{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}.image-box{border:1px solid var(--line);border-radius:15px;padding:10px;background:white}.image-box header{font-weight:800;font-size:12px;color:var(--muted);margin:2px 4px 9px}.image-box img{width:100%;max-height:520px;object-fit:contain;background:#f2f2ed;border-radius:10px;cursor:zoom-in}.compare{display:grid;grid-template-columns:1fr 1fr;gap:16px}.run{border:1px solid var(--line);border-radius:18px;padding:17px;background:#fff}.run.before{border-top:5px solid var(--blue)}.run.after{border-top:5px solid var(--amber)}.run h3{margin:0 0 4px;font-size:17px}.run-sub{font-size:12px;color:var(--muted);margin-bottom:13px}.label-chip{display:block;border:1px solid var(--line);border-radius:12px;padding:10px 11px;margin:8px 0;background:#fbfaf5}.label-chip.only{background:#fff3e9;border-color:#e7b893}.label-chip.shared{opacity:.72}.label-name{font-weight:900}.label-id{font:11px ui-monospace,monospace;color:var(--muted)}.label-path{font-size:11px;color:var(--muted);margin-top:4px;line-height:1.45}.empty{padding:16px;color:var(--muted);border:1px dashed var(--line);border-radius:12px}.added-candidates{padding:13px;border-radius:13px;background:var(--amber2);font-size:13px}.review{padding:22px;background:#1e2823;color:#f8f5ec}.review h2{margin:0 0 5px;font-size:21px}.review-note{color:#aebbb4;font-size:12px}.choice-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin:17px 0}.choice{border:1px solid #506058;background:#27342e;color:#eef4ef;padding:12px 8px;border-radius:12px}.choice.selected{background:#e9f3ec;color:#153e2f;border-color:#8ec2a6;box-shadow:0 0 0 2px #8ec2a633}.details-form{display:grid;gap:12px}.review textarea,.review input{width:100%;border:1px solid #58675f;border-radius:11px;background:#131b17;color:#fff;padding:12px}.label-picker{position:relative}.suggestions{position:absolute;left:0;right:0;top:100%;z-index:5;background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:12px;max-height:240px;overflow:auto;box-shadow:var(--shadow)}.suggestions button{display:block;width:100%;border:0;border-bottom:1px solid #eee7da;background:#fff;padding:10px;text-align:left}.selected-correct{display:flex;gap:7px;flex-wrap:wrap}.selected-correct span{background:#dcece4;color:#174b38;padding:7px 9px;border-radius:9px;font-size:12px}.selected-correct button{border:0;background:transparent;color:#a74736;margin-left:5px}.save-state{font:12px ui-monospace,monospace;color:#9fc6b1;margin-top:10px}.hidden{display:none!important}.empty-page{padding:80px;text-align:center;color:var(--muted)}
    .modal{position:fixed;inset:0;background:rgba(10,15,12,.88);display:flex;align-items:center;justify-content:center;z-index:20;padding:24px}.modal img{max-width:95vw;max-height:92vh;object-fit:contain}.modal button{position:absolute;right:22px;top:18px;border:0;background:white;border-radius:99px;width:42px;height:42px;font-size:24px}
    @media(max-width:900px){.shell{display:block}.sidebar{position:relative;height:auto;border-right:0;border-bottom:1px solid var(--line)}.compare{grid-template-columns:1fr}.choice-grid{grid-template-columns:1fr 1fr}.topbar{align-items:flex-start;flex-direction:column}}
  </style>
</head>
<body>
<div class="shell">
  <aside class="sidebar">
    <div class="brand">Label 波动复核台<small>BIOLOGY · PAIRED ADJUDICATION</small></div>
    <div class="progress-ring"><div class="progress-big" id="progressText">0 / 0</div><div class="progress-caption">已审核 / 当前筛选</div><div class="bar"><i id="progressBar"></i></div></div>
    <input class="search" id="searchInput" placeholder="搜索题号、题干、Label…">
    <div class="filter-title">波动分组</div>
    <button class="filter active" data-group="ALL">全部波动题 <b id="countALL">0</b></button>
    <button class="filter" data-group="A_same_candidate_set">A · 同候选 <b id="countA">0</b></button>
    <button class="filter" data-group="B_candidate_set_expanded_added_not_selected">B · 增候选未选中 <b id="countB">0</b></button>
    <button class="filter" data-group="C_added_legacy_selected">C · 新旧Label被选中 <b id="countC">0</b></button>
    <div class="filter-title">审核状态</div>
    <button class="filter active" data-status="ALL">全部</button>
    <button class="filter" data-status="UNREVIEWED">只看未审核</button>
    <button class="filter" data-status="REVIEWED">只看已审核</button>
    <div class="side-actions"><button class="primary" onclick="downloadJSONL()">导出审核 JSONL</button><button onclick="downloadProgress()">导出进度备份</button><button onclick="importProgress()">导入进度备份</button><button onclick="clearAll()">清空本页审核</button></div>
    <input id="importFile" type="file" accept="application/json" hidden>
  </aside>
  <main>
    <div class="topbar"><div><div class="eyebrow" id="positionText">QUESTION 0 / 0</div><h1>逐题判断两次 Label 是否合理</h1></div><div class="pager"><button onclick="move(-1)">← 上一题</button><input id="jumpInput" type="number" min="1" onchange="jumpTo(this.value)"><button onclick="move(1)">下一题 →</button></div></div>
    <div id="app"></div>
  </main>
</div>
<div id="modal" class="modal hidden" onclick="closeModal()"><button>×</button><img id="modalImg" alt="图片预览"></div>
<script id="review-data" type="application/json">__REVIEW_DATA__</script>
<script id="label-catalog" type="application/json">__LABEL_CATALOG__</script>
<script>
const DATA=JSON.parse(document.getElementById('review-data').textContent);const CATALOG=JSON.parse(document.getElementById('label-catalog').textContent);const STORAGE_KEY='bio-label-volatility-review-v1';let reviews=JSON.parse(localStorage.getItem(STORAGE_KEY)||'{}');let groupFilter='ALL',statusFilter='ALL',query='',filtered=[],cursor=0;
const GROUP_META={A_same_candidate_set:{short:'A',name:'候选集合未变化',desc:'两次Prompt候选完全相同；差异反映同输入重跑波动。'},B_candidate_set_expanded_added_not_selected:{short:'B',name:'增加候选，但新增项未被选中',desc:'旧ID扩大候选，但新增Label未进入最终结果。'},C_added_legacy_selected:{short:'C',name:'新增旧 Label 被选中',desc:'至少一个Top25之外的旧Label进入最终结果。'}};
const DECISIONS=[['BOTH_VALID','前后都合理'],['TOP25_VALID','仅纯Top25合理'],['LEGACY_VALID','仅Top25+旧ID合理'],['BOTH_WRONG','两边都有问题'],['UNSURE','无法判断']];
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}function save(){localStorage.setItem(STORAGE_KEY,JSON.stringify(reviews));renderProgress()}
function imageBox(url,title){if(!url)return'';return `<div class="image-box"><header>${esc(title)} · 点击放大</header><img loading="lazy" src="${esc(url)}" alt="${esc(title)}" onclick="openModal(this.src)" onerror="this.parentElement.innerHTML='<header>${esc(title)}</header><div class=empty>图片加载失败</div>'"></div>`}
function labelChip(x,kind){return `<div class="label-chip ${kind}"><div class="label-name">${esc(x.label_name)}</div><div class="label-id">${esc(x.label_id)}</div><div class="label-path">${esc(x.label_path)}</div></div>`}function labelList(xs,onlyIds){if(!xs.length)return'<div class="empty">未选择任何 Label</div>';const only=new Set(onlyIds);return xs.map(x=>labelChip(x,only.has(x.label_id)?'only':'shared')).join('')}
function currentReview(qid){return reviews[qid]||{decision:'',correct_labels:[],note:'',reviewed_at:''}}
function render(){const app=document.getElementById('app');if(!filtered.length){app.innerHTML='<div class="card empty-page">当前筛选没有题目</div>';document.getElementById('positionText').textContent='QUESTION 0 / 0';return}cursor=Math.max(0,Math.min(cursor,filtered.length-1));const q=filtered[cursor],r=currentReview(q.question_id),g=GROUP_META[q.perturbation_group];document.getElementById('positionText').textContent=`QUESTION ${cursor+1} / ${filtered.length}`;document.getElementById('jumpInput').value=cursor+1;const parent=q.parent_stem?`<section><div class=section-title>父题材料</div><div class=text-box>${esc(q.parent_stem)}</div></section>`:'';const imgs=[imageBox(q.parent_stem_image_url,'父题题干图'),imageBox(q.stem_image_url,'当前题干图'),imageBox(q.parent_analysis_image_url,'父题解析图'),imageBox(q.analysis_image_url,'当前题解析图')].join('');const added=q.added_candidate_labels.length?`<div class=added-candidates><b>由旧 knw_ids 新增的候选：</b>${q.added_candidate_labels.map(x=>esc(x.label_name)).join('、')}</div>`:'';app.innerHTML=`<article class=card><header class=card-head><span class=qid>ID ${esc(q.question_id)}</span><span class="pill ${g.short}">${g.short} · ${esc(g.name)}</span><span class=meta>${esc(q.unit_type)} · 候选 ${q.base_candidate_count} → ${q.augmented_candidate_count} · Jaccard ${q.selection_jaccard}</span></header><div class=content><div class=meta>${esc(g.desc)}</div>${parent}<section><div class=section-title>当前题干</div><div class=text-box>${esc(q.stem)||'无文本题干'}</div></section>${q.options?`<section><div class=section-title>选项</div><div class=text-box>${esc(q.options)}</div></section>`:''}<section><div class=section-title>答案</div><div class="text-box answer">${esc(q.answer_text)||'无'}</div></section><section><div class=section-title>解析</div><div class="text-box analysis">${esc(q.analysis)||'无'}</div></section>${imgs?`<section><div class=section-title>题干 / 解析图片</div><div class=images>${imgs}</div></section>`:''}${added}<section><div class=section-title>前后 Label 对照</div><div class=compare><div class="run before"><h3>纯 Hybrid Top25</h3><div class=run-sub>橙色为仅本次选中；灰色为两次共有</div>${labelList(q.top25_labels,q.top25_only_labels.map(x=>x.label_id))}</div><div class="run after"><h3>Top25 + 旧 knw_ids</h3><div class=run-sub>橙色为仅本次选中；灰色为两次共有</div>${labelList(q.legacy_labels,q.legacy_only_labels.map(x=>x.label_id))}</div></div></section></div><section class=review><h2>教师审核</h2><div class=review-note>判断两组 Label 集合是否都合理。合理多标允许，不要求最小集合。</div><div class=choice-grid>${DECISIONS.map(([v,l])=>`<button class="choice ${r.decision===v?'selected':''}" onclick="setDecision('${v}')">${l}</button>`).join('')}</div><div class=details-form><div id=correctLabelArea class="${r.decision==='BOTH_WRONG'?'':'hidden'}"><div class=review-note>两边都不对：搜索并添加合理 Label（可多选）</div><div class=label-picker><input id=labelSearch placeholder="输入Label名称、路径或ID" oninput="searchLabels(this.value)"><div id=suggestions class="suggestions hidden"></div></div><div id=selectedCorrect class=selected-correct>${renderCorrectLabels(r.correct_labels)}</div></div><textarea id=reviewNote rows=3 placeholder="可选：说明为什么对/错，或需要补充的Label…" oninput="setNote(this.value)">${esc(r.note)}</textarea><div class=save-state>${r.reviewed_at?'已自动保存：'+esc(r.reviewed_at):'选择结论后自动保存到浏览器'}</div></div></section></article>`;renderProgress()}
function renderCorrectLabels(ids){return ids.map(id=>{const x=CATALOG.find(v=>v.label_id===id)||{label_id:id,label_name:id};return `<span>${esc(x.label_name)}<button onclick="removeCorrect('${esc(id)}')">×</button></span>`}).join('')}
function setDecision(v){const q=filtered[cursor],r=currentReview(q.question_id);r.decision=v;r.reviewed_at=new Date().toISOString();if(v!=='BOTH_WRONG')r.correct_labels=[];reviews[q.question_id]=r;save();render()}function setNote(v){const q=filtered[cursor],r=currentReview(q.question_id);r.note=v;r.reviewed_at=new Date().toISOString();reviews[q.question_id]=r;save()}function searchLabels(v){const box=document.getElementById('suggestions'),needle=v.trim().toLowerCase();if(!needle){box.classList.add('hidden');return}const found=CATALOG.filter(x=>(x.label_name+' '+x.label_path+' '+x.label_id).toLowerCase().includes(needle)).slice(0,20);box.innerHTML=found.map(x=>`<button onclick="addCorrect('${esc(x.label_id)}')"><b>${esc(x.label_name)}</b><br><small>${esc(x.label_path)}</small></button>`).join('');box.classList.toggle('hidden',!found.length)}function addCorrect(id){const q=filtered[cursor],r=currentReview(q.question_id);if(!r.correct_labels.includes(id))r.correct_labels.push(id);r.reviewed_at=new Date().toISOString();reviews[q.question_id]=r;save();render()}function removeCorrect(id){const q=filtered[cursor],r=currentReview(q.question_id);r.correct_labels=r.correct_labels.filter(x=>x!==id);reviews[q.question_id]=r;save();render()}
function applyFilters(reset=true){filtered=DATA.filter(q=>{const reviewed=!!reviews[q.question_id]?.decision;if(groupFilter!=='ALL'&&q.perturbation_group!==groupFilter)return false;if(statusFilter==='UNREVIEWED'&&reviewed)return false;if(statusFilter==='REVIEWED'&&!reviewed)return false;if(query&&!JSON.stringify(q).toLowerCase().includes(query))return false;return true});if(reset)cursor=0;render();updateCounts()}function updateCounts(){const c={A:0,B:0,C:0};DATA.forEach(q=>c[GROUP_META[q.perturbation_group].short]++);document.getElementById('countALL').textContent=DATA.length;for(const k of ['A','B','C'])document.getElementById('count'+k).textContent=c[k]}
function renderProgress(){const reviewed=filtered.filter(q=>reviews[q.question_id]?.decision).length,total=filtered.length;document.getElementById('progressText').textContent=`${reviewed} / ${total}`;document.getElementById('progressBar').style.width=total?`${reviewed/total*100}%`:'0%'}function move(delta){if(!filtered.length)return;cursor=Math.max(0,Math.min(filtered.length-1,cursor+delta));render();scrollTo({top:0,behavior:'smooth'})}function jumpTo(v){cursor=Math.max(0,Math.min(filtered.length-1,Number(v)-1));render()}function openModal(src){document.getElementById('modalImg').src=src;document.getElementById('modal').classList.remove('hidden')}function closeModal(){document.getElementById('modal').classList.add('hidden')}
function download(name,text,type='application/json'){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();URL.revokeObjectURL(a.href)}function exportRows(){return DATA.filter(q=>reviews[q.question_id]?.decision).map(q=>({question_id:q.question_id,perturbation_group:q.perturbation_group,top25_label_ids:q.top25_labels.map(x=>x.label_id),legacy_label_ids:q.legacy_labels.map(x=>x.label_id),...reviews[q.question_id]}))}function downloadJSONL(){download('biology-label-volatility-reviews.jsonl',exportRows().map(x=>JSON.stringify(x)).join('\n')+'\n','application/x-ndjson')}function downloadProgress(){download('biology-label-volatility-progress.json',JSON.stringify({version:1,exported_at:new Date().toISOString(),reviews},null,2))}function importProgress(){document.getElementById('importFile').click()}document.getElementById('importFile').addEventListener('change',async e=>{const x=JSON.parse(await e.target.files[0].text());reviews=x.reviews||x;save();applyFilters(false)});function clearAll(){if(confirm('确定清空本页全部审核记录？此操作不可撤销。')){reviews={};save();applyFilters()}}
document.querySelectorAll('[data-group]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-group]').forEach(x=>x.classList.remove('active'));b.classList.add('active');groupFilter=b.dataset.group;applyFilters()});document.querySelectorAll('[data-status]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-status]').forEach(x=>x.classList.remove('active'));b.classList.add('active');statusFilter=b.dataset.status;applyFilters()});document.getElementById('searchInput').addEventListener('input',e=>{query=e.target.value.trim().toLowerCase();applyFilters()});document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA'].includes(document.activeElement.tagName))return;if(e.key==='ArrowRight')move(1);if(e.key==='ArrowLeft')move(-1)});updateCounts();applyFilters();
</script>
</body></html>'''
