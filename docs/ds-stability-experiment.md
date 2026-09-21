# DS 波动性四组实验

## 目的

复用此前 Top25 与 Top25+旧 `knw_ids` 对比中已经出现输出差异的题目，尽量把“模型/服务自身波动”和“候选集合变化”分开观察。脚本完全复用生产判标 prompt（当前版本 `candidate-adjudication-v9.1b-compact-hard-boundaries`），不会修改生产打标流程。

默认实验组是：

| 组 | temperature | n | workers | 目的 |
|---|---:|---:|---:|---|
| `temp0-workers1` | 0 | 1 | 1 | 单线程基线 |
| `temp0-workers10` | 0 | 1 | 10 | 中等并发 |
| `temp0-workers30` | 0 | 1 | 30 | 高并发/动态批处理 |
| `temp0-seed42` | 0 | 1 | 1 | 固定 seed 的单线程对照 |

这里的 A 组题使用原来的纯 Top25 候选；B/C 组题使用原来的 Top25+旧 `knw_ids` 候选，尽量复现此前产生波动的输入条件。默认从 A/B/C 中 `same_output=false` 的题里用固定 seed 抽最多 5,391 道，实际不足时会全部使用。

每个条件都会保存：原始响应、每个 choice、解析后的短代码、映射后的 Label ID、请求 endpoint、重试、延迟、usage、prompt hash 和错误。实验可续跑：同一 run 目录再次执行时只补尚未成功的题；如果输入或条件不一致，会拒绝续跑并提示使用新目录。

## 运行

```bash
PYTHONPATH=src python scripts/run_ds_stability_experiment.py \
  --units runtime/20260918-095957-v91b-100k/sample/pilot_units.jsonl \
  --top25-candidates runtime/20260918-095957-v91b-100k/hybrid/candidates.jsonl \
  --legacy-candidates runtime/20260918-095957-v91b-100k/legacy-augmented/candidates.jsonl \
  --unstable-group runtime/groups/A_same_candidate_set.jsonl \
  --unstable-group runtime/groups/B_candidate_set_expanded_added_not_selected.jsonl \
  --unstable-group runtime/groups/C_added_legacy_selected.jsonl \
  --labels configs/labels.jsonl \
  --run-dir runtime/$(date +%Y%m%d-%H%M%S)-ds-stability-5391 \
  --endpoint 'http://172.22.0.35:9204/v1/chat/completions' \
  --model DeepSeek-V4-Flash \
  --limit 5391 \
  --timeout 300 \
  --retries 3 \
  --retry-delay 1 \
  --request-interval 0 \
  --max-tokens 512 \
  --progress-every 100
```

如果服务恢复后使用的是 9104，只替换 `--endpoint`；也可以重复传多个 `--endpoint`，脚本会轮询。

## 额外的 temperature/n 实验

导师提到的 `temperature=0.1,n=4/8` 可以用同一个脚本直接追加条件（传入任意 `--condition` 后会替换默认四组）：

```bash
  --condition temp0-baseline:0:1:1 \
  --condition temp01-n4:0.1:4:30 \
  --condition temp01-n8:0.1:8:30 \
  --condition temp0-seed42:0:1:1:42
```

`n=4/8` 的每个请求会保留全部 choice，并额外计算严格多数票结果；不会只读取 `choices[0]`。

## 结果

- `run_manifest.json`：输入文件 hash、prompt 版本、实验条件。
- `conditions/<condition>/responses.jsonl`：逐题原始请求结果，可人工追溯。
- `conditions/<condition>/report.json`：单组成功率、解析错误、n 路 choice 内部一致率。
- `report.json`：不同条件之间的 exact agreement、输出差异率、Label 集合 Jaccard，并按此前 A/B/C 组拆分。

解释时要区分：

1. 单线程温度 0 的重复差异，更接近模型/服务本身非确定性；
2. 并发组新增的差异，可能来自动态批处理和调度；
3. `seed` 组如果被服务端拒绝，应记录为接口不支持，不能把失败当成模型稳定；
4. `n=4/8` 的 choice 内部差异反映温度采样敏感性，不等于生产流程最终标签错误率。
