# Label 逐条处理策略

`configs/label_strategies.jsonl` 是 458 个 Label 的逐条台账。每一行同时保留：

- 老师的四列释义（顶层字段）
- Stage 1 的名称自解释结果（`stage1`）
- Stage 2 Judge 的逐项判断（`stage2_judge`、`stage2_category`）
- 参考策略表原文（`reference_strategy`）
- 已知图谱风险（`taxonomy_issue`）
- 本项目最终采用的处理策略（`strategy`）

`configs/label_strategies.review2.jsonl` 是在上述台账上的第二遍独立复核结果。它不覆盖原始证据，而是在每行增加 `second_review` 和 `final_strategy`；`second_review.adjusted_from_previous=true` 表示第二遍认为上一版路由需要收紧。

## 策略模式

| `strategy.mode` | 处理方式 | 适用条件 |
|---|---|---|
| `name_only` | 只把 Label 名用于候选召回，再由 LLM 按当前设问裁决 | Stage 2=L1，且没有已知风险 |
| `name_plus_boundary` | Label 名 + 老师的一条易混淆边界 | 名称基本清楚，但参考策略建议保留边界 |
| `compact_definition` | Label 名 + 老师定义/核心概念/易混淆边界 | Stage 2=L2 或名称不足以稳定判定 |
| `strict_definition` | Label 名 + 完整老师释义 + 明确命中/排除规则 | Stage 2=L3 或已有 P1/P2 边界风险 |
| `taxonomy_hold` | 暂停自动最终打标，先修订图谱或硬路由 | P0 或名称与释义冲突 |
| `separate_dimension` | 作为信息载体、能力、学段、情境等独立维度存储 | 结构性风险或 KM 策略代码 |

`stage2_category` 只是名称与释义的对齐结论，不会单独覆盖策略。比如一个 L1 Label 如果已知和兄弟节点重叠，仍会被提升为 `strict_definition`；P0 默认进入 `taxonomy_hold`。目前对 123/124 两个有丝分裂条目采用了业务侧重点边界（知识点考查 vs 教材实验操作），因此保留 P0 作为风险证据，但实际路由改为 `strict_definition`，不再要求老师额外确认。第二遍共调整 23 个 Label：其中 21 个是原有边界收紧，另外 2 个是上述业务边界覆盖；其余 P0 的“蛋白质病毒的增殖”仍保持 `taxonomy_hold`。

## 重新生成

脚本会读取两个 evidence 文件并检查 458 个 Label 是否全部成功、Stage 2 记录中的 category 是否和规则重算结果一致。参考工作簿是可选的；不传时仍会生成策略，但不带 `reference_strategy` 和 `taxonomy_issue`。

```bash
python scripts/build_label_strategies.py \
  --stage1-evidence "$STAGE1/stage1.evidence.jsonl" \
  --stage2-evidence "$STAGE2/stage2.evidence.jsonl" \
  --output "$RUN/label_strategies.jsonl" \
  --report "$RUN/label_strategies.report.json"
```

本地有参考工作簿时再加：

```bash
  --reference-workbook '/path/to/高中生物_458个Label逐项打标策略_逐条复核版.xlsx'
```

报告中的 `manual_review_count` 是需要人工关注的行数，不等于失败数；`taxonomy_hold` 和 `separate_dimension` 要优先从自动知识标签流程中分流。

第二遍复核（基于已生成的第一版台账）可运行：

```bash
python scripts/review_label_strategies.py \
  --input configs/label_strategies.jsonl \
  --output configs/label_strategies.review2.jsonl \
  --report configs/label_strategies.review2.report.json
```
