# Bio-Know-Tag

高中生物知识点标签理解验证与题目数据清洗项目。

当前版本完成了：

- 将老师提供的 Excel 图谱规范化为 458 条 `configs/labels.jsonl`；
- 清洗原始题目 HTML，并将小题全局聚合到大题 JSONL；
- 阶段一：让 DS 只看 Label 名生成知识范围；
- 阶段二：让 DS Judge 对比老师原释义和生成释义，并生成 L1/L2/L3 建议及人工复核清单；
- 逐 Label 策略台账：合并老师释义、两阶段 DS 结果和图谱风险，给出每个 Label 的生产路由；
- 第二遍逐条复核：对边界误导项进行独立收紧，并输出 `configs/label_strategies.review2.jsonl`。
- 打标前派生处理：在不修改 2.8 GB 清洗基线的前提下，展开独立题/小题、清洗答案、精确去重、生成真实父题聚合计划并统计 R0/R1/R2 路由。
- 无旧标签Pilot抽样：按完整父题组、题型、难度、缺失状态和精确重复组确定性抽样，输出主动移除全部旧标签字段。
- Pilot候选召回：支持纯本地字符n-gram BM25及“全458名称+`@`路径”的DS粗召回基线，统一输出候选sidecar。
- Dense Pilot：支持可配置Transformers embedding精确余弦召回，并输出BM25/Dense重合度与最大分歧样本。
- 混合候选与精判：支持BM25主导的18+7配额融合及DS直接考查Label裁决，并统计第21～25名候选的实际命中情况。
- 人工审核样本：从2,500题中确定性抽取200题近似均匀指标集和100题覆盖导向压力集，保持题目与候选严格对齐。
- 精判对比：已完成同一300题的v4/v6/v7/v8.1/v8.2运行与重点回归复核；v8.3保留五道硬门槛和内部反证，但删除会干扰主任务的显式风险候选输出。
- 历史弱监督：支持用“旧 `knw_ids` 与当前458 Label的交集”大规模验证Recall@K、补充候选，并按Label抽题做DS T/F释义边界审核；废弃旧ID始终移除。
- 释义覆盖复核：完成179,568个独立题正样本与21,647个兄弟Label硬负样本审核，输出458个Label正负联合分档、source→target混淆对及具体题目证据。
- 伪负例校正：对硬负样本中第一阶段接受的题目做最小充分知识集二次Judge，区分合理共标、目标边界过宽、来源Label不足和无法判断，再计算校正边界误收率。
- 逐Label人工复核导出：支持将正样本按458个Label分别导出为可阅读JSON，完整保留题目信息、老师释义与DS判定，并按匹配分数从低到高排序。

DS 题目打标 Pilot 已完成v4、v6和v7；v7覆盖恢复但精度未过门槛，v8待在原300题上复测。仓库不会把未经人工审核的模型输出冒充金标。

## 目录

```text
source/     原始数据说明与清洗数据约定
configs/    标签定义、逐 Label 策略台账及质量报告
scripts/    数据清洗、阶段一/二实验和策略台账命令行脚本
src/        可复用且有单元测试的 Python 实现
tests/      自动化测试
runtime/    每次实验的时间戳输出（默认不提交）
docs/       实验决策、初审记录和运行手册
```

## 本地安装与测试

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

## 导出标签表

```bash
.venv/bin/python scripts/export_labels.py \
  --input '/Users/wishfine/Downloads/高中_生物_图谱_2026-08-20 09_42_27.xlsx'
```

期望输出：`processed=458`、`error=0`、`blank_fields=0`、重复数为 0。

## 运行

本地与服务器的 smoke/full 命令、PID 保存和验收步骤见 [实验运行手册](docs/experiment-runbook.md)。标签表初步检查见 [taxonomy 初审](docs/taxonomy-audit.md)。

从数据清洗、Label 理解到 BM25/Dense 召回和 v4/v6/v7 精判的完整进展、结论与下一步见 [实验进展总结](docs/experiment-progress-summary.md)。

逐 Label 策略的字段含义、优先级和重新生成命令见 [Label 策略台账](docs/label-strategy.md)。

## 逐 Label 复核文档

完整的 458 行表格（老师原释义、DS 释义、DS Judge、GPT Judge 二次复核和最终处理策略）见 [Label 复核表](docs/label-review-table.md)。如需从最新台账重新导出：

```bash
.venv/bin/python scripts/export_label_review_markdown.py \
  --input configs/label_strategies.review2.jsonl \
  --output docs/label-review-table.md
```

正式题目格式、无旧 `knw_ids` 的混合召回方案、DS裁决协议、父题聚合规则和Pilot口径见 [题库打标策略](docs/tagging-strategy.md)。

独立题正样本的458个Label逐项复核见 [正样本覆盖复核](docs/positive-coverage-audit/biology-458-label-positive-coverage-review.md)；加入兄弟Label硬负样本后的最终联合分析见 [正负样本联合复核](docs/positive-coverage-audit/biology-458-label-positive-negative-combined-review.md)，需学科重点确认的source→target证据见 [边界重点复核](docs/positive-coverage-audit/biology-label-boundary-focus-positive-negative-evidence.md)。
