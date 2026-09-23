# 10 万题：DS 精排与原 `knw_ids` 的逐 Label 对照

## 当前状态

本仓库已提供可复用的统计脚本；完整 DS 10 万题 `predictions.jsonl` 位于服务器，不在本机工作区，因此本文不填未经运行证实的比例。服务器执行下列命令后，`runtime/.../analysis.md` 会生成包含 458 个 Label 的实际结果文档。Qwen 跑完后用同一脚本和同一批 `pilot_units.jsonl` 即可横向对照。

## 数据与口径

- 单元：`runtime/20260918-095957-v91b-100k/sample/pilot_units.jsonl`，固定 100,000 道。两个 DS 策略使用同一单元文件。**这份盲评单元故意删除了 `legacy_knw_ids`**；必须先按题号从完整 `runtime/20260916-152856-label-units-with-updates/label_units.jsonl` 回填旧 ID 快照，核验 100,000 道全部找到。历史 `legacy_knw_ids` 去重并仅保留当前 `configs/labels.jsonl` 中的 458 个 ID；过时 ID 只记数量，不作对错依据。
- 纯 Top25：`runtime/20260918-095957-v91b-100k/adjudication-top25-no-legacy/predictions.jsonl`。
- Top25＋旧 ID：`runtime/20260918-095957-v91b-100k/adjudication-top25-plus-legacy-9204/predictions.jsonl`。
- 对每题，令 `H=当前有效旧ID集合`，`P=模型选中集合`。当 `H` 非空时：`P=H` 为完全一致；`H⊂P` 为预测包含旧集（只增）；`P⊂H` 为预测为旧集子集（只减，空选也在此类）；两边既有交集又各有独有项为有增有减；两边均非空且交集为空为完全不同。`H` 为空的题单列，不强行归为完全不同。
- 逐 Label 表以“旧 ID 中包含该 Label 且预测成功的题”为分母，给出该题五类关系的题数/比例，并附该 Label 被保留、漏掉、作为新标加入的题数。多旧标的题会在多个 Label 行出现；逐 Label 行不可简单相加当成独立题量。
- 未成功预测的题单列为缺预测，不把失败当空选。既有题目重复录入、旧标签可能有误，故这些指标是**与历史标签的一致性/差异性**，不是准确率、精确率或真实召回率。

## 服务器运行

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
git pull --ff-only origin main

BASE='runtime/20260918-095957-v91b-100k'
UNITS="$BASE/sample/pilot_units.jsonl"
TOP25="$BASE/adjudication-top25-no-legacy/predictions.jsonl"
LEGACY="$BASE/adjudication-top25-plus-legacy-9204/predictions.jsonl"
OUT="runtime/$(date +%Y%m%d-%H%M%S)-ds-legacy-label-comparison-100k"
FULL_UNITS='runtime/20260916-152856-label-units-with-updates/label_units.jsonl'

test -s "$UNITS" && test -s "$FULL_UNITS" && test -s "$TOP25" && test -s "$LEGACY"

PYTHONPATH=src python scripts/build_pilot_legacy_snapshot.py \
  --sample-units "$UNITS" --full-units "$FULL_UNITS" \
  --output "$OUT/legacy-ids-100k.jsonl" --expected-units 100000

PYTHONPATH=src python scripts/compare_predictions_with_legacy_labels.py \
  --units "$UNITS" --predictions "$TOP25" --labels configs/labels.jsonl \
  --legacy-snapshot "$OUT/legacy-ids-100k.jsonl" \
  --run-dir "$OUT/top25" --expected-units 100000

PYTHONPATH=src python scripts/compare_predictions_with_legacy_labels.py \
  --units "$UNITS" --predictions "$LEGACY" --labels configs/labels.jsonl \
  --legacy-snapshot "$OUT/legacy-ids-100k.jsonl" \
  --run-dir "$OUT/top25-plus-legacy" --expected-units 100000

PYTHONPATH=src python scripts/report_label_set_comparison.py \
  --run "DS纯Top25=$OUT/top25" \
  --run "DS Top25加旧ID=$OUT/top25-plus-legacy" \
  --output "$OUT/analysis.md"

printf '%s\n' "$OUT" > runtime/LATEST_DS_LEGACY_LABEL_COMPARISON_RUN
python -m json.tool "$OUT/top25/report.json"
python -m json.tool "$OUT/top25-plus-legacy/report.json"
sed -n '1,45p' "$OUT/analysis.md"
```

产物：旧 ID 快照 `legacy-ids-100k.jsonl` 与校验报告；两组各有 `report.json`、`per_label.jsonl`（458 行）、`per_question.jsonl`，以及合并的 `analysis.md`。要给别人只看简报，可以贴两个 `report.json` 和 `analysis.md` 前 45 行；要分析具体问题 Label，再贴相应的 `per_label.jsonl` 行或完整文档。Qwen 完成后复用同一个旧 ID 快照，换 `--predictions` 和输出目录即可。

## 解读时优先核查

1. 先检查 `units=100000`、目录 Label 数为 458、`compared + missing_prediction = 100000`，以及两组输入 units/labels SHA256 相同。
2. 比较两组“完全一致/包含/子集/增减/完全不同”时注意分母：如果成功预测题集合不同，微小比例差异可能仅由缺失题造成；需要精确比较策略效应时，另取共同成功题的交集。
3. 优先检查旧题数充足、漏掉率高或新增题数高的 Label。低支持量 Label 的百分比方差很大，不能自动下结论。
4. 复合题小题的原标签可能是父题知识点并集；应结合 `report.json` 的 `by_unit_type` 看是否主要由小题造成。
5. 对“完全不同”或“高新增”抽具体题时同时查看题目、释义及 DS 理由；旧标签本身可能错误，不能直接把这些题判成模型错标。
