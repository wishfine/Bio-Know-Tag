# 全量 Qwen/DS：各两票，分歧题补第三票

策略版本 `adaptive-q2d2-disagreement-third-v1`。旧六票结果保留；新的自适应运行单独放在 `fine-full-six/adjudication-adaptive-2plus1-w35`，不得使用原来的 `adjudication-no-thinking` 目录。

## 行为与目录

1. Qwen1/2、DS1/2 同时全量跑，单票 35 并发，每服务最多 70、合计最多 140 请求。temperature=0、stream=true、thinking=false、max_tokens=1024、HTTP retries=5；超时沿用 Qwen 600 秒、DS 300 秒。前缀缓存仍须在服务器配置，不是客户端参数。
2. 四票全量成功后，比较每个模型自己的两份结果。Label 集合或 `context_insufficient` / `need_expand_recall` 不一致时给**该模型**补第三票；只顺序、reason或evidence不同不触发。
3. 两模型之间不同、但各自前两票相同，不补同模型第三票。此类留给后续边界/Judge审核。
4. 第三票阶段 Qwen和DS可并行，各35并发。它是阶段性补票，不是两票一落盘就即时补票。

```text
adjudication-no-thinking/                 # 旧六票目录，不修改
adjudication-adaptive-2plus1-w35/
  run_manifest.json, controller.lock
  seed_report.json                        # 旧前两票复制来源及SHA
  votes/{qwen1,qwen2,ds1,ds2}/             # 全量前两票，证据/SQLite/日志各自分离
  disagreements/{qwen,ds}/                # 冻结的第三票输入与分歧报告
  votes/{qwen3_disagreement,ds3_disagreement}/
  model_consensus/{qwen,ds}.jsonl          # 记录实际2/3票和每Label支持票数
  report.json
```

原六票前两票 evidence 与 manifest **复制**进新目录，不复制 SQLite，不使用软链接或硬链接；成功题由新执行器重建索引后跳过。旧第三票保留为对照，不混入新补票结果。迁移要求原控制器已停止，原输入SHA、模型、请求配置和prompt版本一致。续跑使用完全相同命令，不能换目录、模型、输入或参数。

## 精确停旧任务（仅Linux服务器）

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
git pull --ff-only origin main
RUN='runtime/20260924-104413-biology-s85-reuse-preprocess'
OLD="$RUN/fine-full-six/adjudication-no-thinking"

# 先显示精确匹配的控制器与其该目录下票进程；不触碰vLLM服务。
python scripts/stop_full_vote_run.py --run-dir "$OLD"
python scripts/stop_full_vote_run.py --run-dir "$OLD" --stop
```

停止器只向指定目录对应的Python投票任务发送SIGTERM，等待45秒；仍有进程时返回非零，不强杀。旧结果不删除。若返回失败，先核查未退出进程，不要启动迁移。

## 启动与续跑

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
RUN='runtime/20260924-104413-biology-s85-reuse-preprocess'
OLD="$RUN/fine-full-six/adjudication-no-thinking"
NEW="$RUN/fine-full-six/adjudication-adaptive-2plus1-w35"
mkdir -p "$NEW"
LOG="$NEW/controller-$(date +%Y%m%d-%H%M%S).log"

nohup env PYTHONPATH=src python -u scripts/run_qwen_ds_adaptive_votes_full.py \
  --units "$RUN/unified/retrieval_units.jsonl" \
  --candidates "$RUN/fine-full-six/legacy/candidates.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$NEW" \
  --seed-from-six-vote "$OLD" \
  --qwen-endpoint 'http://172.22.0.35:9204/v1/chat/completions' \
  --qwen-model 'qwen3.8-27b-fp8' \
  --ds-endpoint 'http://172.22.0.35:9205/v1/chat/completions' \
  --ds-model 'ds-v4-flash' \
  --workers-per-vote 35 \
  --max-tokens 1024 --retries 5 \
  > "$LOG" 2>&1 &
printf '%s\n' "$!" > "$NEW/controller.pid"
printf 'LOG=%s\n' "$LOG"
tail -n 40 "$LOG"
```

第一阶段各票的进度见 `votes/<vote>/runner.log` / `report.json`；旧成功证据复制和输入全量校验需要时间，启动时不立即发HTTP请求。同一目录有独占锁，防止重复控制器。SIGTERM会停止并回收子进程；证据残行在副本中保存为bin再移除，原目录不变。失败请求不是反对票；任何前两票未完成会停止进入第三票阶段，重跑同一命令先补齐它们。

最终每模型按实际票数做至少2票的多数集合，保留全部不确定状态和实际票数；`usable_for_training=false`，还需教师校准。旧六票16格分析器不能直接分析新目录：新第三票仅覆盖分歧题，未请求的票绝不能当作0票或复制票。父题最终标签仍须单独汇总“小题标签并集＋父题材料额外标签”。
