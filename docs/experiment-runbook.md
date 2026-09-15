# 实验运行手册

## 1. 本地验证

```bash
cd '/Users/wishfine/Desktop/xdf/ai题库/Bio-Know-Tag'
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src scripts
```

重新导出老师标签表：

```bash
.venv/bin/python scripts/export_labels.py \
  --input '/Users/wishfine/Downloads/高中_生物_图谱_2026-08-20 09_42_27.xlsx'
wc -l configs/labels.jsonl
python3 -m json.tool configs/labels.report.json
```

## 2. 同步服务器代码

当前 SSH 配置中 `xdf-35` 对应 `zhangyonglin@172.22.0.35`。首次部署：

```bash
ssh xdf-35
cd /local_data/zhangyonglin
git clone git@github.com:wishfine/Bio-Know-Tag.git Bio-Know-Tag
mkdir -p /local_data/zhangyonglin/data/bio-know-tag
cd /local_data/zhangyonglin/Bio-Know-Tag
git log -1 --oneline
```

后续只允许快进同步：

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
git pull --ff-only origin main
git log -1 --oneline
```

## 3. 准备数据与 Miniconda base 环境

使用 `cp -n` 避免覆盖已有原始数据：

```bash
mkdir -p /local_data/zhangyonglin/data/bio-know-tag
cp -n \
  '/home/share_ssd_data/nfs-data1/wangmeng148/data/tiku/high-geo-hist-pol/生物.jsonl' \
  '/local_data/zhangyonglin/data/bio-know-tag/biology.raw.jsonl'
wc -l '/local_data/zhangyonglin/data/bio-know-tag/biology.raw.jsonl'
sha256sum '/local_data/zhangyonglin/data/bio-know-tag/biology.raw.jsonl'

cd /local_data/zhangyonglin/Bio-Know-Tag
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
python --version  # 需要 Python 3.10+
python -m pip install -e '.[dev]'
python -m pytest -q
```

## 4. 题目清洗 smoke

所有输出进入本次时间戳目录：

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
RUN="runtime/$(date +%Y%m%d-%H%M%S)-smoke"
mkdir -p "$RUN"
python scripts/preprocess_questions.py \
  --input '/local_data/zhangyonglin/data/bio-know-tag/biology.raw.jsonl' \
  --limit 100 \
  --run-dir "$RUN/preprocess"
python -m json.tool "$RUN/preprocess/report.json"
wc -l "$RUN/preprocess/questions.jsonl"
```

必须确认 `processed == input == 100`、`error == 0`，并人工查看若干大题及其 `sub_questions`。

## 5. DS 连通性

在 35 服务器内部执行：

```bash
export DS1='http://172.22.0.35:9092/v1/chat/completions'
export DS2='http://172.22.0.35:9093/v1/chat/completions'
export MODEL='DeepSeek-V4-Flash'
curl -sS --max-time 30 \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4-Flash","messages":[{"role":"user","content":"只输出 JSON：{\"ok\":true}"}],"temperature":0,"max_tokens":64}' \
  "$DS1"
```

只有返回有效 `choices[0].message.content` 后才继续。

## 6. 三 Label 阶段一 smoke

```bash
SMOKE1="$RUN/stage1"
mkdir -p "$SMOKE1"
python scripts/run_label_self_explain.py \
  --run-dir "$SMOKE1" \
  --limit 3 \
  --endpoint "$DS1" \
  --endpoint "$DS2"
python -m json.tool "$SMOKE1/stage1.report.json"
wc -l "$SMOKE1/stage1.evidence.jsonl"
```

验收：`processed == input == success == 3`、`error == 0`、evidence 为 3 行，且每行 `parsed_response` 含 `core_meaning`、`included_content`、`excluded_content`。

## 7. 三 Label 阶段二 smoke

```bash
SMOKE2="$RUN/stage2"
mkdir -p "$SMOKE2"
python scripts/judge_label_alignment.py \
  --stage1-evidence "$SMOKE1/stage1.evidence.jsonl" \
  --run-dir "$SMOKE2" \
  --limit 3 \
  --endpoint "$DS1" \
  --endpoint "$DS2"
python -m json.tool "$SMOKE2/stage2.report.json"
wc -l "$SMOKE2/stage2.evidence.jsonl" "$SMOKE2/manual_review.jsonl"
```

验收：阶段二 `processed == input == success == 3`、`error == 0`，分层总数等于 3。

## 8. 后台全量阶段一

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
RUN="runtime/$(date +%Y%m%d-%H%M%S)-full"
mkdir -p "$RUN/stage1"
nohup python scripts/run_label_self_explain.py \
  --run-dir "$RUN/stage1" \
  --endpoint "$DS1" \
  --endpoint "$DS2" \
  > "$RUN/stage1/nohup.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$RUN/stage1/pid"
printf 'RUN=%s PID=%s\n' "$RUN" "$PID"
```

只通过持久化文件监控：

```bash
tail -n 50 "$RUN/stage1/nohup.log"
python -m json.tool "$RUN/stage1/stage1.report.json"
wc -l "$RUN/stage1/stage1.evidence.jsonl"
ps -p "$(cat "$RUN/stage1/pid")" -o pid,etime,stat,command
```

断连后使用同一个 `--run-dir` 重启会跳过已经成功的 Label。失败记录会保留为证据并在重启时重试。

## 9. 后台全量阶段二

阶段一必须先达到 `processed=458`、`success=458`、`error=0`：

```bash
mkdir -p "$RUN/stage2"
nohup python scripts/judge_label_alignment.py \
  --stage1-evidence "$RUN/stage1/stage1.evidence.jsonl" \
  --run-dir "$RUN/stage2" \
  --endpoint "$DS1" \
  --endpoint "$DS2" \
  > "$RUN/stage2/nohup.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$RUN/stage2/pid"
printf 'RUN=%s PID=%s\n' "$RUN" "$PID"
```

验收阶段二同样要求 `processed=458`、`success=458`、`error=0`、evidence 为 458 行。随后人工复核 `manual_review.jsonl` 中全部 L3 和固定种子的高分样本。

## 10. 题库全量清洗与物化

题目 smoke 通过后后台运行：

```bash
mkdir -p "$RUN/preprocess"
nohup python scripts/preprocess_questions.py \
  --input '/local_data/zhangyonglin/data/bio-know-tag/biology.raw.jsonl' \
  --run-dir "$RUN/preprocess" \
  > "$RUN/preprocess/nohup.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$RUN/preprocess/pid"
```

确认 `report.json` 中 `processed == input`、`error == 0`，抽样检查聚合结构后，再以不覆盖方式物化：

```bash
cp -n "$RUN/preprocess/questions.jsonl" \
  '/local_data/zhangyonglin/data/bio-know-tag/biology.cleaned.grouped.jsonl'
wc -l '/local_data/zhangyonglin/data/bio-know-tag/biology.cleaned.grouped.jsonl'
sha256sum '/local_data/zhangyonglin/data/bio-know-tag/biology.cleaned.grouped.jsonl'
```

## 11. 构建打标单元与 dry-run 路由（不调用 DS）

现有 `questions.jsonl` 作为不可变基线。本步骤只在新的时间戳目录中生成派生文件。

先运行 smoke；`--limit` 限制的是清洗文件的顶层记录数：

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
FULL_RUN="$(cat runtime/LATEST_PREPROCESS_RUN)"
AUDIT_RUN="$(cat runtime/LATEST_ORPHAN_AUDIT_RUN)"
UNIT_SMOKE="runtime/$(date +%Y%m%d-%H%M%S)-label-units-smoke"
mkdir -p "$UNIT_SMOKE"

python scripts/build_label_units.py \
  --input "$FULL_RUN/questions.jsonl" \
  --labels configs/label_strategies.review2.jsonl \
  --orphan-audit "$AUDIT_RUN/orphan_parents.jsonl" \
  --run-dir "$UNIT_SMOKE" \
  --limit 1000

python -m json.tool "$UNIT_SMOKE/build_report.json"
python -m json.tool "$UNIT_SMOKE/route_report.json"
wc -l \
  "$UNIT_SMOKE/label_units.jsonl" \
  "$UNIT_SMOKE/parent_aggregation.jsonl" \
  "$UNIT_SMOKE/duplicate_groups.jsonl"
```

抽查通过后后台运行全量，并立即保存 PID：

```bash
UNIT_RUN="runtime/$(date +%Y%m%d-%H%M%S)-label-units-full"
mkdir -p "$UNIT_RUN"
printf '%s\n' "$UNIT_RUN" > runtime/LATEST_LABEL_UNITS_RUN

nohup python scripts/build_label_units.py \
  --input "$FULL_RUN/questions.jsonl" \
  --labels configs/label_strategies.review2.jsonl \
  --orphan-audit "$AUDIT_RUN/orphan_parents.jsonl" \
  --run-dir "$UNIT_RUN" \
  > "$UNIT_RUN/nohup.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$UNIT_RUN/pid"
printf 'UNIT_RUN=%s PID=%s\n' "$UNIT_RUN" "$PID"
```

监控及验收：

```bash
tail -n 50 "$UNIT_RUN/nohup.log"
ps -p "$(cat "$UNIT_RUN/pid")" -o pid,etime,stat,command
python -m json.tool "$UNIT_RUN/build_report.json"
python -m json.tool "$UNIT_RUN/route_report.json"
wc -l \
  "$UNIT_RUN/label_units.jsonl" \
  "$UNIT_RUN/parent_aggregation.jsonl" \
  "$UNIT_RUN/duplicate_groups.jsonl"
```

全量关键验收口径：

```text
label_units                         = 1,857,591
standalone_units                    = 1,033,000
sub_question_units                  = 823,136
orphan_sub_question_units           = 1,455
real_compound_parents               = 219,741
synthetic_parent_containers_skipped = 388
error                               = 0
```

输出含义：

- `label_units.jsonl`：独立题、正常小题和缺父题小题；缺父题小题只打自身知识点。
- `parent_aggregation.jsonl`：仅 219,741 个真实组合题父题，供后续执行“小题 Label 并集 + 父题额外 Label”。
- `duplicate_groups.jsonl`：完全相同内容的题目 ID 组及旧标签冲突状态。
- `route_report.json`：R0/R1/R2、候选数量和未匹配旧 ID 的 dry-run 统计。
- `build_report.json`：构建数量、精确去重和错误统计。

## 12. 构建无旧标签依赖的Pilot样本（不调用DS）

正式策略见 `docs/tagging-strategy.md`。Pilot抽样不会读取旧 `knw_ids` 进行分层，输出也会删除全部 `legacy_*` 和旧路由字段。

先从全量派生文件头尾构造一个同时包含独立题和组合题的小型输入，验证命令和结构：

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
UNIT_RUN="$(cat runtime/LATEST_LABEL_UNITS_RUN)"
PILOT_SMOKE="runtime/$(date +%Y%m%d-%H%M%S)-pilot-sample-smoke"
mkdir -p "$PILOT_SMOKE/input" "$PILOT_SMOKE/output"

head -n 1000 "$UNIT_RUN/label_units.jsonl" \
  > "$PILOT_SMOKE/input/label_units.sample.jsonl"
tail -n 4000 "$UNIT_RUN/label_units.jsonl" \
  >> "$PILOT_SMOKE/input/label_units.sample.jsonl"
tail -n 1000 "$UNIT_RUN/parent_aggregation.jsonl" \
  > "$PILOT_SMOKE/input/parent_aggregation.sample.jsonl"
head -n 500 "$UNIT_RUN/duplicate_groups.jsonl" \
  > "$PILOT_SMOKE/input/duplicate_groups.sample.jsonl"

PYTHONPATH=src python scripts/build_pilot_sample.py \
  --label-units "$PILOT_SMOKE/input/label_units.sample.jsonl" \
  --parent-aggregation "$PILOT_SMOKE/input/parent_aggregation.sample.jsonl" \
  --duplicate-groups "$PILOT_SMOKE/input/duplicate_groups.sample.jsonl" \
  --run-dir "$PILOT_SMOKE/output" \
  --target-size 300 \
  --parent-groups 30 \
  --duplicate-samples 20 \
  --audit-sample-size 10 \
  --stratum-sample-size 1

python -m json.tool "$PILOT_SMOKE/output/pilot_report.json"
wc -l "$PILOT_SMOKE/output"/*.jsonl
```

Smoke通过后，全量派生文件上后台构建约2,500条Pilot：

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
UNIT_RUN="$(cat runtime/LATEST_LABEL_UNITS_RUN)"
PILOT_RUN="runtime/$(date +%Y%m%d-%H%M%S)-pilot-sample-full"
mkdir -p "$PILOT_RUN"
printf '%s\n' "$PILOT_RUN" > runtime/LATEST_PILOT_RUN

nohup env PYTHONPATH=src python scripts/build_pilot_sample.py \
  --label-units "$UNIT_RUN/label_units.jsonl" \
  --parent-aggregation "$UNIT_RUN/parent_aggregation.jsonl" \
  --duplicate-groups "$UNIT_RUN/duplicate_groups.jsonl" \
  --run-dir "$PILOT_RUN" \
  --target-size 2500 \
  --parent-groups 200 \
  --duplicate-samples 100 \
  --audit-sample-size 100 \
  --stratum-sample-size 3 \
  > "$PILOT_RUN/nohup.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$PILOT_RUN/pid"
printf 'PILOT_RUN=%s PID=%s\n' "$PILOT_RUN" "$PID"
```

监控和验收：

```bash
PILOT_RUN="$(cat runtime/LATEST_PILOT_RUN)"
tail -n 50 "$PILOT_RUN/nohup.log"
ps -p "$(cat "$PILOT_RUN/pid")" -o pid,etime,stat,%cpu,%mem,command
python -m json.tool "$PILOT_RUN/pilot_report.json"
wc -l "$PILOT_RUN"/*.jsonl

python -c 'import json,sys; bad=[]; f=open(sys.argv[1],encoding="utf-8"); exec("for line in f:\n r=json.loads(line); bad.extend(k for k in r if k.startswith(\"legacy_\") or k in {\"proposed_route\",\"route_reason\"})"); print({"legacy_fields_found":len(bad)}); raise SystemExit(bool(bad))' "$PILOT_RUN/pilot_units.jsonl"
```

必须确认：

- `legacy_fields_used` 为 `false`；
- `legacy_fields_found` 为0；
- `pilot_units` 约为2,500（完整题组和必选异常样本可能使结果略高）；
- `pilot_parent_groups` 为200或接近200；
- `duplicate_groups_sampled` 为100；
- 8个空题干全部进入 `empty_stem_units.jsonl`；
- 同一个父题被选中时，其全部小题都在 `pilot_units.jsonl` 中。

输出：

- `pilot_units.jsonl`：后续召回和DS实验的主要输入。
- `pilot_parents.jsonl`：Pilot中的完整真实父题组。
- `duplicate_samples.jsonl`：100个精确重复组，每组附两个成员。
- `image_context_samples.jsonl`：按题目类型抽取的图片风险题。
- `empty_stem_units.jsonl`：全部空题干记录。
- `pilot_report.json`：总体、题型、难度和抽样原因统计。

## 13. Pilot候选召回第一阶段

本阶段运行两个基线，均不读取旧 `knw_ids`：

- `char_ngram_bm25`：纯本地中文字符bigram/trigram BM25，使用老师Label Card。
- `ds_all_label_paths`：向DS发送全部458个Label的名称和`@`格式路径，不发送释义，粗召回Top 20。

### 13.1 BM25 smoke与全量Pilot

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
PILOT_RUN="$(cat runtime/LATEST_PILOT_RUN)"
SPARSE_SMOKE="runtime/$(date +%Y%m%d-%H%M%S)-sparse-recall-smoke"
mkdir -p "$SPARSE_SMOKE"

PYTHONPATH=src python scripts/run_sparse_retrieval.py \
  --units "$PILOT_RUN/pilot_units.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$SPARSE_SMOKE" \
  --top-k 20 \
  --limit 20

python -m json.tool "$SPARSE_SMOKE/report.json"
wc -l "$SPARSE_SMOKE/candidates.jsonl"
head -n 1 "$SPARSE_SMOKE/candidates.jsonl" | python -m json.tool
```

确认 `input=processed=20`、`error=0`，且所有 `label_path` 使用 `@` 后运行2,500条：

```bash
SPARSE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-sparse-recall-pilot"
mkdir -p "$SPARSE_RUN"
printf '%s\n' "$SPARSE_RUN" > runtime/LATEST_SPARSE_RECALL_RUN

PYTHONPATH=src python scripts/run_sparse_retrieval.py \
  --units "$PILOT_RUN/pilot_units.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$SPARSE_RUN" \
  --top-k 20

python -m json.tool "$SPARSE_RUN/report.json"
wc -l "$SPARSE_RUN/candidates.jsonl"
```

### 13.2 DS全Label名称/路径粗召回smoke

每个请求默认放5道题，Label目录只发送一次。先运行10道题：

```bash
export DS1='http://172.22.0.35:9092/v1/chat/completions'
export DS2='http://172.22.0.35:9093/v1/chat/completions'
export MODEL='DeepSeek-V4-Flash'

COARSE_SMOKE="runtime/$(date +%Y%m%d-%H%M%S)-ds-coarse-smoke"
mkdir -p "$COARSE_SMOKE"

PYTHONPATH=src python scripts/run_ds_coarse_recall.py \
  --units "$PILOT_RUN/pilot_units.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$COARSE_SMOKE" \
  --endpoint "$DS1" \
  --endpoint "$DS2" \
  --top-k 20 \
  --batch-size 5 \
  --limit 10

python -m json.tool "$COARSE_SMOKE/report.json"
wc -l "$COARSE_SMOKE/candidates.jsonl" "$COARSE_SMOKE/evidence.jsonl"
head -n 1 "$COARSE_SMOKE/candidates.jsonl" | python -m json.tool
```

验收要求：`input=processed=success=10`、`error=pending=0`、候选ID均属于458图谱、路径均使用`@`。相同 `--run-dir` 重跑时，已有成功题目会跳过，失败批次会继续重试。

DS粗召回v2在Prompt中使用 `B001` 至 `B458` 短代码，程序再映射为真实19位 `label_id`，避免模型抄错长ID。重复短代码会保持首次出现顺序去重，超过Top-K的尾部会截断；修复数量记录在evidence的 `parsed_response.normalization` 中。未知短代码、漏题或乱序仍按错误处理。v1与v2证据不能混用，升级后必须创建新的运行目录。

DS粗召回全2,500条需等10题smoke人工查看后再启动。该结果不是金标，只是与BM25、Dense和混合召回比较的候选基线。

项目当前决定不继续运行DS粗召回；代码和smoke仅作为历史基线保留。生产召回优先验证BM25与Dense。

## 14. Dense Pilot与BM25分歧实验

服务器的 `agentgym` 环境已有CUDA版PyTorch，但缺少Transformers。为避免污染既有环境，克隆为项目专用环境：

```bash
conda create \
  --prefix /local_data/zhangyonglin/conda_envs/bio-know-tag-dense \
  --clone /home/zhangyonglin/miniconda3/envs/agentgym \
  -y

DENSE_PY='/local_data/zhangyonglin/conda_envs/bio-know-tag-dense/bin/python'
"$DENSE_PY" -m pip install 'transformers>=4.41,<5' 'safetensors>=0.4'
```

模型使用 `BAAI/bge-small-zh-v1.5`，约24M参数（模型卡：https://huggingface.co/BAAI/bge-small-zh-v1.5）。模型缓存放在项目同根数据目录：

```bash
export HF_HOME='/local_data/zhangyonglin/data/bio-know-tag/huggingface'
mkdir -p "$HF_HOME"
```

先用GPU 0运行20题smoke：

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
PILOT_RUN="$(cat runtime/LATEST_PILOT_RUN)"
DENSE_SMOKE="runtime/$(date +%Y%m%d-%H%M%S)-dense-recall-smoke"
mkdir -p "$DENSE_SMOKE"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$DENSE_PY" scripts/run_dense_retrieval.py \
  --units "$PILOT_RUN/pilot_units.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$DENSE_SMOKE" \
  --model BAAI/bge-small-zh-v1.5 \
  --device cuda:0 \
  --top-k 20 \
  --batch-size 128 \
  --limit 20

python -m json.tool "$DENSE_SMOKE/report.json"
wc -l "$DENSE_SMOKE/candidates.jsonl"
head -n 1 "$DENSE_SMOKE/candidates.jsonl" | python -m json.tool
```

Smoke通过后运行2,500题：

```bash
DENSE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-dense-recall-pilot"
mkdir -p "$DENSE_RUN"
printf '%s\n' "$DENSE_RUN" > runtime/LATEST_DENSE_RECALL_RUN

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$DENSE_PY" scripts/run_dense_retrieval.py \
  --units "$PILOT_RUN/pilot_units.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$DENSE_RUN" \
  --model BAAI/bge-small-zh-v1.5 \
  --device cuda:0 \
  --top-k 20 \
  --batch-size 128

python -m json.tool "$DENSE_RUN/report.json"
wc -l "$DENSE_RUN/candidates.jsonl"
```

然后与已完成的BM25结果比较：

```bash
SPARSE_RUN="$(cat runtime/LATEST_SPARSE_RECALL_RUN)"
COMPARE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-bm25-dense-compare"
mkdir -p "$COMPARE_RUN"
printf '%s\n' "$COMPARE_RUN" > runtime/LATEST_RETRIEVAL_COMPARE_RUN

PYTHONPATH=src python scripts/compare_retrieval_runs.py \
  --units "$PILOT_RUN/pilot_units.jsonl" \
  --sparse-candidates "$SPARSE_RUN/candidates.jsonl" \
  --dense-candidates "$DENSE_RUN/candidates.jsonl" \
  --run-dir "$COMPARE_RUN" \
  --top-k 20 \
  --sample-size 200

python -m json.tool "$COMPARE_RUN/report.json"
wc -l "$COMPARE_RUN/disagreement_samples.jsonl"
```

`top1_agreement_rate`、`mean_overlap_at_k`和`mean_jaccard_at_k`只表示两个召回器的重合度，不是准确率。必须检查 `disagreement_samples.jsonl` 或建立人工金标后，才能决定Dense是否进入生产流程。

## 15. Top25配额融合与DS候选精判

人工检查BM25/Dense最大分歧样本后，当前Pilot采用BM25主导的候选集：先保留BM25前18项，再加入Dense中前7个尚未出现的新Label，最终得到25项。若某一路不足，才使用RRF顺序补足。该配置是待验证的Pilot参数，不是生产参数。

### 15.1 生成融合候选

如果BM25和Dense上游都只保留20项，二者高度重合时并集可能不足25。为了严格验证 `Recall@25`，先各自重跑Top30；耗时远低于DS精判：

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
PILOT_RUN="$(cat runtime/LATEST_PILOT_RUN)"
DENSE_PY='/local_data/zhangyonglin/conda_envs/bio-know-tag-dense/bin/python'
DENSE_MODEL='/local_data/zhangyonglin/data/bio-know-tag/models/bge-small-zh-v1.5'

SPARSE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-sparse-recall-top30"
mkdir -p "$SPARSE_RUN"
printf '%s\n' "$SPARSE_RUN" > runtime/LATEST_SPARSE_RECALL_30_RUN

PYTHONPATH=src python scripts/run_sparse_retrieval.py \
  --units "$PILOT_RUN/pilot_units.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$SPARSE_RUN" \
  --top-k 30

DENSE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-dense-recall-top30"
mkdir -p "$DENSE_RUN"
printf '%s\n' "$DENSE_RUN" > runtime/LATEST_DENSE_RECALL_30_RUN

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$DENSE_PY" scripts/run_dense_retrieval.py \
  --units "$PILOT_RUN/pilot_units.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$DENSE_RUN" \
  --model "$DENSE_MODEL" \
  --device cuda:0 \
  --top-k 30 \
  --batch-size 128

HYBRID_RUN="runtime/$(date +%Y%m%d-%H%M%S)-hybrid-s18-d7-k25"

mkdir -p "$HYBRID_RUN"
printf '%s\n' "$HYBRID_RUN" > runtime/LATEST_HYBRID_RECALL_RUN

PYTHONPATH=src python scripts/fuse_retrieval_candidates.py \
  --sparse-candidates "$SPARSE_RUN/candidates.jsonl" \
  --dense-candidates "$DENSE_RUN/candidates.jsonl" \
  --run-dir "$HYBRID_RUN" \
  --top-k 25 \
  --sparse-quota 18 \
  --dense-quota 7

python -m json.tool "$HYBRID_RUN/report.json"
wc -l "$HYBRID_RUN/candidates.jsonl"
head -n 1 "$HYBRID_RUN/candidates.jsonl" | python -m json.tool
```

验收要求：`input=processed=2500`、`error=0`、Top30上游生成的融合候选数量分布应全部为25，且 `retrieval_version=hybrid-v1-s18-d7-k25`。此前直接融合两个Top20运行时出现的21～24项不是数据错误，而是两个有限候选集合的并集不足25；该结果不用于严格比较 `Recall@20` 与 `Recall@25`。

### 15.2 运行10题DS精判smoke

每道题单独调用一次DS，同时发送题目和25张老师Label Card。这里只使用9102服务：

```bash
export DS1='http://172.22.0.35:9102/v1/chat/completions'
export MODEL='DeepSeek-V4-Flash'

JUDGE_SMOKE="runtime/$(date +%Y%m%d-%H%M%S)-candidate-judge-smoke"
mkdir -p "$JUDGE_SMOKE"
printf '%s\n' "$JUDGE_SMOKE" > runtime/LATEST_CANDIDATE_JUDGE_RUN

PYTHONPATH=src python scripts/run_candidate_adjudication.py \
  --units "$PILOT_RUN/pilot_units.jsonl" \
  --candidates "$HYBRID_RUN/candidates.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$JUDGE_SMOKE" \
  --endpoint "$DS1" \
  --limit 10 \
  --workers 1 \
  --max-tokens 1024

python -m json.tool "$JUDGE_SMOKE/report.json"
wc -l \
  "$JUDGE_SMOKE/predictions.jsonl" \
  "$JUDGE_SMOKE/evidence.jsonl" \
  "$JUDGE_SMOKE/tail_selected.jsonl"
```

验收要求：`input=processed=success=10`、`error=pending=0`。精判v3输出所有被设问直接考查的Label，允许语义重叠，但排除上位概念、背景和弱相关联想。若证据语义可用但不是题内逐字子串，程序保留标签，并输出 `evidence_verified=false`、`needs_review=true`；这类情况计入报告的 `unverified_evidence_items` 和 `questions_with_unverified_evidence`，不再把整题判为失败。`predictions.jsonl` 是结构化结果，`evidence.jsonl` 保存原始响应、usage和reasoning信息，`tail_selected.jsonl` 专门收集选中候选排名21～25的题目供人工复核。相同运行目录可安全续跑同一Prompt版本；不同Prompt版本必须使用新目录。

### 15.3 决定生产使用Top20还是Top25

不能根据DS是否选择尾部候选直接证明尾部正确，也不能根据10题smoke决定候选上限。Smoke只验证接口、Prompt和产物。

下一步从Pilot中分层运行至少200～300题，人工确认最终Label，并重点全审 `tail_selected.jsonl`：

- 若第21～25位补回了人工确认的正确Label，或某类知识点系统性依赖该尾部，生产保留Top25。
- 若人工确认尾部几乎只造成误选，且Top20没有明显漏标，生产降为Top20以缩短Prompt。
- 同时检查 `need_expand_recall` 和“人工真值不在25项内”的比例；它们反映召回上限，而不是DS精判能力。

最终应报告人工金标上的 `Recall@20`、`Recall@25`、两者增量、DS多打率、漏打率和每题平均标签数，再冻结生产参数。

精判器支持 `--workers` 并发请求；所有HTTP请求可并行，但 evidence 仍由主线程逐行安全落盘。`report.json` 会记录并发数、运行墙钟时间、请求吞吐及延迟的均值、P50、P95和最大值。先用20题、`--workers 4` 验证服务承载能力，再逐步增加到8；若出现超时或HTTP错误，应降低并发并在同一运行目录续跑失败项。

## 16. 300题DS精判与GPT人工审核集

审核集由两部分组成：200题使用固定seed做近似均匀抽样，用于估计整体指标；100题覆盖缺父题、图片风险、空答案/解析、题型难度、知识模块及BM25/Dense Top1分歧，用于发现失败模式。两部分必须分开报告，不能把压力集当作总体无偏样本。

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
PILOT_RUN="$(cat runtime/LATEST_PILOT_RUN)"
HYBRID_RUN="$(cat runtime/LATEST_HYBRID_RECALL_RUN)"
AUDIT_SAMPLE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-adjudication-audit-300"

mkdir -p "$AUDIT_SAMPLE_RUN"
printf '%s\n' "$AUDIT_SAMPLE_RUN" > runtime/LATEST_ADJUDICATION_AUDIT_SAMPLE_RUN

PYTHONPATH=src python scripts/build_adjudication_audit_sample.py \
  --units "$PILOT_RUN/pilot_units.jsonl" \
  --candidates "$HYBRID_RUN/candidates.jsonl" \
  --run-dir "$AUDIT_SAMPLE_RUN" \
  --sample-size 300 \
  --representative-size 200

python -m json.tool "$AUDIT_SAMPLE_RUN/report.json"
wc -l "$AUDIT_SAMPLE_RUN/audit_units.jsonl" \
  "$AUDIT_SAMPLE_RUN/audit_candidates.jsonl" \
  "$AUDIT_SAMPLE_RUN/audit_manifest.jsonl"
```

使用两个端点共同精判。客户端会在线程安全的轮询中把并发请求均匀分配给9102和9103；后台运行并保留PID：

```bash
export DS1='http://'172.22.0.35':9102/v1/chat/completions'
export DS2='http://'172.22.0.35':9103/v1/chat/completions'
export MODEL='DeepSeek-V4-Flash'

AUDIT_DS_RUN="runtime/$(date +%Y%m%d-%H%M%S)-adjudication-audit-300-ds"
mkdir -p "$AUDIT_DS_RUN"
printf '%s\n' "$AUDIT_DS_RUN" > runtime/LATEST_ADJUDICATION_AUDIT_DS_RUN

nohup env PYTHONPATH=src python scripts/run_candidate_adjudication.py \
  --units "$AUDIT_SAMPLE_RUN/audit_units.jsonl" \
  --candidates "$AUDIT_SAMPLE_RUN/audit_candidates.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$AUDIT_DS_RUN" \
  --endpoint "$DS1" \
  --endpoint "$DS2" \
  --workers 4 \
  --timeout 300 \
  --retries 3 \
  --max-tokens 1024 \
  > "$AUDIT_DS_RUN/nohup.log" 2>&1 &

PID=$!
printf '%s\n' "$PID" > "$AUDIT_DS_RUN/pid"
printf 'AUDIT_DS_RUN=%s PID=%s\n' "$AUDIT_DS_RUN" "$PID"
```

只通过日志和sidecar查看进度：

```bash
AUDIT_DS_RUN="$(cat runtime/LATEST_ADJUDICATION_AUDIT_DS_RUN)"
tail -n 30 "$AUDIT_DS_RUN/nohup.log"
python -m json.tool "$AUDIT_DS_RUN/report.json"
wc -l "$AUDIT_DS_RUN/evidence.jsonl" \
  "$AUDIT_DS_RUN/predictions.jsonl" \
  "$AUDIT_DS_RUN/tail_selected.jsonl"
```

若服务中途失败，使用完全相同的运行目录和参数重跑；程序跳过同Prompt版本的成功题，只请求未完成题。完成标准为 `processed=success=300`、`error=pending=0`、evidence行数不少于300。审核时提交 `audit_units.jsonl`、`audit_candidates.jsonl`、`predictions.jsonl`、`evidence.jsonl` 和 `report.json`。
