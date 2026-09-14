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

DS 题目打标 Pilot 尚未开始。仓库不会用模型输出伪造金标；应先运行纯本地派生处理并核对真实路由规模。

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

逐 Label 策略的字段含义、优先级和重新生成命令见 [Label 策略台账](docs/label-strategy.md)。

## 逐 Label 复核文档

完整的 458 行表格（老师原释义、DS 释义、DS Judge、GPT Judge 二次复核和最终处理策略）见 [Label 复核表](docs/label-review-table.md)。如需从最新台账重新导出：

```bash
.venv/bin/python scripts/export_label_review_markdown.py \
  --input configs/label_strategies.review2.jsonl \
  --output docs/label-review-table.md
```
