# 双模型各单票打标

入口：`scripts/run_qwen_ds_strict_adaptive_full.py`。

通过 `--strategy` 控制打标策略：

| 参数 | 行为 |
| --- | --- |
| `adaptive`（默认） | 保留原策略：Qwen、DS 各首票；Label 集合不一致时各追加一票，可配置第三票诊断 |
| `single-pair` | Qwen、DS 各首票；不请求第二或第三票；逐题保留两个 Label 集合的交集 |

例如 Qwen 选 A、B，DS 选 A、C，最终保留 A，B、C 留在差异记录中。
两个模型都未选 Label 时保留空结果。请求失败属于未完成任务，不当作反对票。
共识仍是高精度候选，不会因此自动赋予训练数据合格标记；原质量过滤继续生效。

本次配置：两个服务各 64 个并发请求（合计最多 128），每个 HTTP 请求显式
`n=1`、`temperature=0`、`max_tokens=1024`、`stream=true`，关闭 thinking。
`n=1` 是每次请求生成一个回答，与业务上的票次数量不同。
HTTP 最多尝试 5 次，失败重试不是业务追加票。

## 原目录续跑

停止原控制器及其子进程后，在原目录添加 `--strategy single-pair`
并改为 `--workers-per-vote 64`。保持输入文件、服务、模型、其他请求参数、
`--seed-from-six-vote` 和旧证据授权参数与上次一致。

控制器允许变更并发和策略，归档上一份 manifest；输入、prompt、模型及请求配置
变化仍拒绝续跑。已有 `votes/qwen1`、`votes/ds1` 成功结果和 SQLite 断点复用，
已有第二、第三票文件保留，但 single-pair 不读取它们。
`consensus` 是当前策略的派生产物，完成后重新生成。
切换回 `--strategy adaptive` 时恢复原追加票逻辑。

所有首票完成后，`consensus/predictions.jsonl` 记录每题的交集、未保留 Label、
跨模型分歧和质量状态。`actual_vote_count=2`，`stage2_requested=false`。
