# Qwen 释义消融复现实验：旧实验核查与重复对照

## 旧实验到底做了什么

旧数据位于 `runtime/20260921-184755-definition-ablation/`。它从已打旧 `knw_ids` 的独立题中，按当前 458 个 Label 各抽 5 或 10 道题，共 3,257 个题目–Label 对。若 Label 在此前 DS 正样本实验中的匹配率低于 0.70，目标抽 10 道，否则抽 5 道；每个 Label 优先取前一轮 DS 精排双策略输出不同的题，再用该 Label 的旧 ID 正样本补齐。因此这是刻意加重困难样本的诊断集，不是题库总体的随机样本。

两组输入中的题目完全相同，包括题干、选项、答案和解析；一组只提供 Label 名，另一组提供 Label 名以及 `definition`、`core_concepts`、`common_assessments`、`distinctions` 四个字段。`label_path` 并没有进入判别提示词。旧结果为 3,257 对中 660 对状态改变（20.26%），其中仅名称 False→有说明 True 为 328，对向变化 332。

## 核查发现的限制

1. 旧实验没有“相同输入重复运行”的 A/A、B/B 控制，不能把 660 次翻转全部归因于释义。将这 3,257 对的旧全量有释义结果与消融时重新跑的有释义结果相比，544 对（16.70%）状态不同；两次批量大小与运行时段不同，因此这不是纯随机波动率，却足以说明原有因果解释过强。
2. 采样依赖 DS 先前的匹配率与波动结果；Qwen 在这批题上的统计可以与 DS 做定向对照，但不能外推成 Qwen 对全量题库的释义敏感率。每 Label 只有 5/10 道，也不能单凭 Label 百分比决定改释义。
3. 3,257 对中，1,052 对标有 `image_context_missing`，97 对解析为空；正文仍包含题干等信息，但这些质量层级要分开报告。样本中另有 6 组同 Label、题目文本完全重复而题号不同的记录，不宜将它们视作完全独立证据。
4. 旧“有释义”条件实际给的是四字段完整说明卡，不是单独的 `definition` 字段；报告应称“释义卡片效应”。仅名称条件的提示词仍写着“根据教研给出的知识点释义”，但说明字段为空，这种表述也可能影响模型。Qwen 复现保留原提示词以便与 DS 同口径比较；如需隔离 `definition` 字段本身，应另做三条件实验。

## Qwen 复现口径

保留相同 3,257 对和原提示词，分别独立运行四组：`name_1`、`name_2`、`definition_1`、`definition_2`。四组使用相同模型、接口池、温度（客户端固定为 0）、批量限制与输出预算，不能共享同一运行目录。所有结果按 `pair_id` 配对，不以输出先后顺序比较。

分析时输出：第一次和第二次 A/B 翻转率、名称 A/A 翻转率、完整卡片 B/B 翻转率、两次名称判定一致且两次完整卡片判定一致时的稳健 False→True/True→False 数；并按采样层、来源、图像上下文标记和解析是否为空分层。独立重复排除的是模型/服务/运行噪声，**不证明改变后的标签一定正确**，仍须人工核对代表题。

分析程序：`scripts/analyze_definition_ablation_repeats.py`。它校验四臂的 pair ID、题号、Label ID、输入哈希和关键运行配置，并生成 `report.json`、`per_pair.jsonl`、`per_label.jsonl`。

## 运行前条件

Qwen 服务必须先在服务器本机确认 `/v1/models` 可用。原服务预计为 9304 和 9305，实际端口和模型名以当前响应为准。旧样本和分析程序已打包于本机 `runtime/qwen-definition-ablation-input-v2.tar.gz`；这是固定实验输入，不应在复现中重新抽题。

服务器同步代码与输入后，从仓库根目录执行：

```bash
RUN_ROOT="runtime/$(date +%Y%m%d-%H%M%S)-qwen-definition-ablation-v2"
MODEL='Qwen3.8-27B'
ENDPOINTS=()
for port in {9304..9311}; do
  ENDPOINTS+=("http://127.0.0.1:$port/v1/chat/completions")
done

QWEN_ABLATION_TOTAL_WORKERS=240 QWEN_ABLATION_PER_ENDPOINT_LIMIT=30 \
  bash scripts/run_qwen_definition_ablation_arms.sh smoke "$RUN_ROOT" "$MODEL" "${ENDPOINTS[@]}"

nohup env QWEN_ABLATION_TOTAL_WORKERS=240 QWEN_ABLATION_PER_ENDPOINT_LIMIT=30 \
  bash scripts/run_qwen_definition_ablation_arms.sh full "$RUN_ROOT" "$MODEL" "${ENDPOINTS[@]}" \
  > "$RUN_ROOT/launcher.log" 2>&1 &
printf '%s\n' "$!" > "$RUN_ROOT/launcher.pid"
```

`smoke` 在独立子目录对每臂的前 30 对做同步解析检查。`full` 以 A→B→B→A 顺序逐臂运行，四臂共享同一批 9304–9311 服务；总 worker 数为 240，且每个端口同时在途请求不超过 30。这样避免不同实验臂同时争抢端口。高并发下应关注错误率和 vLLM 队列/显存；若服务出现错误，可停止并降低环境变量后在同一运行目录续跑。四臂全部报告 `processed=3257` 且 `error=0` 后分析：

```bash
PYTHONPATH=src python scripts/analyze_definition_ablation_repeats.py \
  --sample runtime/20260921-184755-definition-ablation/name_only_tasks.jsonl \
  --name-1 "$RUN_ROOT/full/name_1/results.jsonl" \
  --name-2 "$RUN_ROOT/full/name_2/results.jsonl" \
  --definition-1 "$RUN_ROOT/full/definition_1/results.jsonl" \
  --definition-2 "$RUN_ROOT/full/definition_2/results.jsonl" \
  --run-dir "$RUN_ROOT/analysis"
```

如果样本放在不同目录，可设置 `SAMPLE_ROOT=/path/to/sample` 运行 launcher，并将分析命令的 `--sample` 改成对应目录下的 `name_only_tasks.jsonl`。不要把 Qwen 输出写进旧 DS 实验目录。
