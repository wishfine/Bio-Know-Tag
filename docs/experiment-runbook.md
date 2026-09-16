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

验收要求：`input=processed=success=10`、`error=pending=0`。精判v4输出所有被设问直接考查的Label，允许有独立考查依据的语义重叠，但排除无独立依据的上位概念、背景和弱相关联想。错误选项只有在判断其正误确实需要该知识时才计入；“综合/应用”Label可以正常命中，但不能作为宽泛兜底。候选发送前按 `question_id` 确定性打乱，消除C01与召回第一名固定绑定造成的位置偏置，程序仍保留原始候选排名。题目同时发送 `parent_context_missing` 和 `image_context_missing`，输出用 `context_insufficient` 将上下文缺失与候选召回缺失分开。若证据语义可用但不是题内逐字子串，程序保留标签并输出 `evidence_verified=false`；这类情况仍计入报告的 `unverified_evidence_items` 和 `questions_with_unverified_evidence`，但该诊断字段不再单独触发 `needs_review=true`。`predictions.jsonl` 是结构化结果，`evidence.jsonl` 保存原始响应、usage和reasoning信息，`tail_selected.jsonl` 专门收集选中候选排名21～25的题目供人工复核。相同运行目录可安全续跑同一Prompt版本；不同Prompt版本必须使用新目录。

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

若服务中途失败，使用完全相同的运行目录和参数重跑；程序跳过同Prompt版本的成功题，只请求未完成题。完成标准为 `processed=success=300`、`error=pending=0`、evidence行数不少于300，且 `prompt_version=candidate-adjudication-v4`。审核时提交 `audit_units.jsonl`、`audit_candidates.jsonl`、`predictions.jsonl`、`evidence.jsonl` 和 `report.json`。

## 17. Top30 与 Cross-Encoder Reranker 实验

该实验只重跑300题的本地召回，不调用DS。比较两组新结果：

- B：BM25 Top20 + Dense独有Top10，直接得到30个候选。
- C：BM25 Top30和Dense Top30先取并集，再由`BAAI/bge-reranker-base`压到30个候选。

Dense v2 对长字段使用“头部+尾部”，避免只保留题目背景而截掉末尾设问。先同步代码并恢复变量：

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
git pull --ff-only origin main

AUDIT_SAMPLE_RUN="$(cat runtime/LATEST_ADJUDICATION_AUDIT_SAMPLE_RUN)"
DENSE_PY='/local_data/zhangyonglin/conda_envs/bio-know-tag-dense/bin/python'
DENSE_MODEL='/local_data/zhangyonglin/data/bio-know-tag/models/bge-small-zh-v1.5'
RERANK_MODEL_FILE='/local_data/zhangyonglin/data/bio-know-tag/models/bge-reranker-base.path'
```

若本机还没有reranker，只下载一次：

```bash
"$DENSE_PY" -c 'from huggingface_hub import snapshot_download; print(snapshot_download(repo_id="BAAI/bge-reranker-base", cache_dir="/local_data/zhangyonglin/data/bio-know-tag/models/hf-cache"))' > "$RERANK_MODEL_FILE"

RERANK_MODEL="$(cat "$RERANK_MODEL_FILE")"
printf 'RERANK_MODEL=%s\n' "$RERANK_MODEL"
```

在同一300题上重新产生两路Top30：

```bash
SPARSE30_RUN="runtime/$(date +%Y%m%d-%H%M%S)-audit-sparse30"
mkdir -p "$SPARSE30_RUN"

PYTHONPATH=src python scripts/run_sparse_retrieval.py \
  --units "$AUDIT_SAMPLE_RUN/audit_units.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$SPARSE30_RUN" \
  --top-k 30

DENSE30_RUN="runtime/$(date +%Y%m%d-%H%M%S)-audit-dense30-v2"
mkdir -p "$DENSE30_RUN"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$DENSE_PY" scripts/run_dense_retrieval.py \
  --units "$AUDIT_SAMPLE_RUN/audit_units.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$DENSE30_RUN" \
  --model "$DENSE_MODEL" \
  --device cuda:0 \
  --top-k 30 \
  --batch-size 128 \
  --local-files-only
```

生成直接Top30基线：

```bash
HYBRID30_RUN="runtime/$(date +%Y%m%d-%H%M%S)-audit-hybrid-s20-d10-k30"
mkdir -p "$HYBRID30_RUN"

PYTHONPATH=src python scripts/fuse_retrieval_candidates.py \
  --sparse-candidates "$SPARSE30_RUN/candidates.jsonl" \
  --dense-candidates "$DENSE30_RUN/candidates.jsonl" \
  --run-dir "$HYBRID30_RUN" \
  --top-k 30 \
  --sparse-quota 20 \
  --dense-quota 10
```

生成reranker Top30：

```bash
RERANK30_RUN="runtime/$(date +%Y%m%d-%H%M%S)-audit-rerank30"
mkdir -p "$RERANK30_RUN"
printf '%s\n' "$RERANK30_RUN" > runtime/LATEST_RERANK30_RUN

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$DENSE_PY" scripts/run_candidate_reranking.py \
  --units "$AUDIT_SAMPLE_RUN/audit_units.jsonl" \
  --labels configs/labels.jsonl \
  --sparse-candidates "$SPARSE30_RUN/candidates.jsonl" \
  --dense-candidates "$DENSE30_RUN/candidates.jsonl" \
  --run-dir "$RERANK30_RUN" \
  --model "$RERANK_MODEL" \
  --device cuda:0 \
  --sparse-pool 30 \
  --dense-pool 30 \
  --top-k 30 \
  --batch-size 64 \
  --local-files-only

python -m json.tool "$RERANK30_RUN/report.json"
wc -l "$RERANK30_RUN/candidates.jsonl"
```

分别检查三个目标题。第二题按当前458体系使用其最接近且已包含“外来物种入侵”的上位Label：

```bash
for RUN in "$HYBRID30_RUN" "$RERANK30_RUN"; do
  PYTHONPATH=src python scripts/check_recall_targets.py \
    --candidates "$RUN/candidates.jsonl" \
    --target '2590460368220094467=数据图像类' \
    --target '2327487096614821888=生物多样性丧失的原因及保护措施' \
    --target '2964930689699446802=扩散作用与渗透作用及相应实验' || true
done
```

只有reranker结果三项全部`hit=true`，并经人工抽查没有明显挤掉原来正确候选，才进入DS精判复测。`|| true`只用于让两个对比检查都能打印，不代表验收通过。

## 18. 紧凑精判v5与API重试审计

v4的300题审核运行保留为基线，不在运行中切换Prompt。v5用于后续实验，删除 `rejected_close_codes`、逐Label的 `necessity`、全局 `reason`，并由程序根据 `selected` 是否为空推导 `none_of_candidates`。模型只输出：

```json
{
  "selected": [{"code": "C01", "evidence": "不超过60字的题内原文"}],
  "need_expand_recall": false,
  "missing_knowledge": "",
  "context_insufficient": false
}
```

`need_expand_recall=true` 时 `missing_knowledge` 必须简述候选缺少的知识；否则必须为空。API层面对HTTP错误、连接重置和超时执行带随机抖动的指数退避，CLI用 `--retry-delay` 设置首次退避秒数。成功响应的 `retry_errors` 会写入evidence，报告汇总 `requests_retried` 与 `retry_error_types`。

v5必须使用新运行目录，推荐参数：

```bash
nohup env PYTHONPATH=src python scripts/run_candidate_adjudication.py \
  --units "$AUDIT_SAMPLE_RUN/audit_units.jsonl" \
  --candidates "$AUDIT_SAMPLE_RUN/audit_candidates.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$V5_RUN" \
  --endpoint "$DS1" \
  --workers 20 \
  --timeout 300 \
  --retries 5 \
  --retry-delay 1 \
  --max-tokens 1024 \
  > "$V5_RUN/nohup.log" 2>&1 &
```

预期 `prompt_version=candidate-adjudication-v5-compact`。输出字段减少后仍保留1024上限作为异常保护；正常响应应显著低于该上限。

## 19. 高精度优先精判v6（原Top25候选）

v6按训练数据目标调整为“允许漏标、允许空结果、严格避免错标”。它不要求每道小题覆盖完整知识点，也不因遗漏次要知识点强制扩大召回。每个小题独立精判；父题Label之后取已保留小题Label的并集，并另行补充父题公共材料直接考查的额外Label。

候选Label只发送`label_name`、`label_path`、`definition`、`core_concepts`和`distinctions`，不发送`common_assessments`、召回排名或分数。模型只输出：

```json
{
  "selected": [{"code": "C01", "evidence": "不超过40字的题内原文"}],
  "need_expand_recall": false,
  "context_insufficient": false
}
```

程序允许`selected=[]`，并生成：

```text
usable_for_training = selected非空
                      且need_expand_recall=false
                      且context_insufficient=false
```

在原300题、原Top25候选上重新运行，不使用Top30或reranker候选：

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
git pull --ff-only origin main

AUDIT_SAMPLE_RUN="$(cat runtime/LATEST_ADJUDICATION_AUDIT_SAMPLE_RUN)"
V6_RUN="runtime/$(date +%Y%m%d-%H%M%S)-candidate-judge-v6-precision"
mkdir -p "$V6_RUN"
printf '%s\n' "$V6_RUN" > runtime/LATEST_ADJUDICATION_V6_RUN

python - "$AUDIT_SAMPLE_RUN/audit_candidates.jsonl" <<'PY'
import json
import sys
from collections import Counter

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
versions = Counter(str(row.get("retrieval_version") or "") for row in rows)
counts = Counter(len(row.get("candidates") or []) for row in rows)
summary = {"rows": len(rows), "retrieval_versions": dict(versions), "candidate_counts": dict(counts)}
print(summary)
if len(rows) != 300 or versions != {"hybrid-v1-s18-d7-k25": 300} or counts != {25: 300}:
    raise SystemExit("原Top25候选预检失败，请勿启动v6")
PY

nohup env PYTHONPATH=src python scripts/run_candidate_adjudication.py \
  --units "$AUDIT_SAMPLE_RUN/audit_units.jsonl" \
  --candidates "$AUDIT_SAMPLE_RUN/audit_candidates.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$V6_RUN" \
  --endpoint 'http://172.22.0.35:9104/v1/chat/completions' \
  --model 'DeepSeek-V4-Flash' \
  --workers 20 \
  --timeout 300 \
  --retries 5 \
  --retry-delay 1 \
  --request-interval 2 \
  --max-tokens 512 \
  > "$V6_RUN/nohup.log" 2>&1 &

printf '%s\n' "$!" > "$V6_RUN/pid"
printf 'V6_RUN=%s PID=%s\n' "$V6_RUN" "$(cat "$V6_RUN/pid")"
```

监控与验收：

```bash
V6_RUN="$(cat runtime/LATEST_ADJUDICATION_V6_RUN)"
tail -n 30 "$V6_RUN/nohup.log"
python -m json.tool "$V6_RUN/report.json"
python -m json.tool "$V6_RUN/run_manifest.json"
wc -l "$V6_RUN/predictions.jsonl" "$V6_RUN/evidence.jsonl"
```

`--request-interval 2`表示所有worker共享一个请求启动节流器，任意两个HTTP尝试的启动时间至少间隔2秒，避免线程池启动和同步重试形成瞬时连接惊群；它不限制服务端同时处理的在途请求数。失败记录会保存真实的`attempts`、最后`endpoint`、总延迟及每次`retry_errors`。

完成标准：`input=processed=success=300`、`error=pending=0`、`prompt_version=candidate-adjudication-v6-precision-first`、`candidate_retrieval_versions=["hybrid-v1-s18-d7-k25"]`、`candidate_count_distribution={"25":300}`。`run_manifest.json`绑定题目、候选、Label文件哈希与模型参数；同目录恢复运行时若任何关键输入变化，程序会拒绝混跑。优先人工复核`usable_for_training=true`的非空结果是否存在错标；空结果、扩召和上下文不足结果直接过滤，不以漏标率作为本轮失败标准。

## 20. 合理多标与跨维度边界精判v7

v7根据v4/v6同一300题的逐题复核调整。它不再强求“最小Label集合”：同一题中只要多个Label均符合老师释义且能由设问、选项、答案或解析直接支持，就允许同时保留。同时增加“考查维度”硬边界，禁止把原理替代为实验、结论替代为发展史、跨膜运输替代为水的存在形式。

v7删除`evidence`输出，避免模型为已选Label反向寻找表面词证据。模型只输出：

```json
{
  "selected": ["C01", "C05"],
  "need_expand_recall": false,
  "context_insufficient": false
}
```

状态口径：

- 候选中只有相近但跨维度的Label：不硬选，`need_expand_recall=true`。
- 已能选出至少一个准确Label，仅可能漏掉次要项：`need_expand_recall=false`。
- 答案或解析已足够确定至少一个Label：即使缺图也保持`context_insufficient=false`。
- 非生物题或无有效设问：`selected=[]`，两个状态均为`false`，仍由空结果过滤。

为了只测Prompt变化，v7继续使用原300题和原Top25候选，不同时引入Top30或taxonomy兄弟扩展。因此“施肥烧苗”在本轮的正确行为是拒绝“观察质壁分离实验”并进入扩召，而不是直接产生未召回的渗透作用Label。

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
git pull --ff-only origin main

AUDIT_SAMPLE_RUN="$(cat runtime/LATEST_ADJUDICATION_AUDIT_SAMPLE_RUN)"
V7_RUN="runtime/$(date +%Y%m%d-%H%M%S)-candidate-judge-v7-balanced"
mkdir -p "$V7_RUN"
printf '%s\n' "$V7_RUN" > runtime/LATEST_ADJUDICATION_V7_RUN

nohup env PYTHONPATH=src python scripts/run_candidate_adjudication.py \
  --units "$AUDIT_SAMPLE_RUN/audit_units.jsonl" \
  --candidates "$AUDIT_SAMPLE_RUN/audit_candidates.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$V7_RUN" \
  --endpoint 'http://172.22.0.35:9104/v1/chat/completions' \
  --model 'DeepSeek-V4-Flash' \
  --workers 20 \
  --timeout 300 \
  --retries 5 \
  --retry-delay 1 \
  --request-interval 2 \
  --max-tokens 256 \
  > "$V7_RUN/nohup.log" 2>&1 &

printf '%s\n' "$!" > "$V7_RUN/pid"
printf 'V7_RUN=%s PID=%s\n' "$V7_RUN" "$(cat "$V7_RUN/pid")"
```

完成标准：`input=processed=success=300`、`error=pending=0`、`prompt_version=candidate-adjudication-v7-balanced-precision`。优先对照`docs/pilot-v4-v6-300-question-review.md`中的7道“v6精度改差”、7道“v6覆盖下降”和2道“路由状态改差”：前者应消除跨维度错标，后两类不应因过度保守继续丢失可判断样本。

## 22. 用有效旧 `knw_ids` 大规模验证候选召回

本实验只把“旧 `knw_ids` 与当前458 Label的交集”当作弱监督目标。旧树中有、但老师释义表中没有的ID是废弃Label：只统计数量，不进入Recall分母、候选或DS Prompt。

### 22.1 构建约5%的确定性大样本

5%预计从1,857,591个打标单元中抽取约9.3万题；只将至少有1个有效旧Label的题写入后续召回输入。

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
git pull --ff-only origin main

UNIT_RUN="$(cat runtime/LATEST_LABEL_UNITS_RUN)"
LEGACY_SAMPLE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-legacy-recall-sample"
mkdir -p "$LEGACY_SAMPLE_RUN"
printf '%s\n' "$LEGACY_SAMPLE_RUN" > runtime/LATEST_LEGACY_RECALL_SAMPLE_RUN

PYTHONPATH=src python scripts/build_legacy_recall_sample.py \
  --units "$UNIT_RUN/label_units.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$LEGACY_SAMPLE_RUN" \
  --sample-rate 0.05 \
  --seed legacy-recall-v1

python -m json.tool "$LEGACY_SAMPLE_RUN/report.json"
wc -l "$LEGACY_SAMPLE_RUN/evaluation_units.jsonl"
```

### 22.2 按原配方生成 Top25

```bash
DENSE_PY='/local_data/zhangyonglin/conda_envs/bio-know-tag-dense/bin/python'
DENSE_MODEL='/local_data/zhangyonglin/data/bio-know-tag/models/bge-small-zh-v1.5'

LEGACY_SPARSE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-legacy-sparse-top30"
LEGACY_DENSE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-legacy-dense-top30"
LEGACY_HYBRID_RUN="runtime/$(date +%Y%m%d-%H%M%S)-legacy-hybrid-s18-d7-k25"
mkdir -p "$LEGACY_SPARSE_RUN" "$LEGACY_DENSE_RUN" "$LEGACY_HYBRID_RUN"
printf '%s\n' "$LEGACY_SPARSE_RUN" > runtime/LATEST_LEGACY_SPARSE_RUN
printf '%s\n' "$LEGACY_DENSE_RUN" > runtime/LATEST_LEGACY_DENSE_RUN
printf '%s\n' "$LEGACY_HYBRID_RUN" > runtime/LATEST_LEGACY_HYBRID_RUN

PYTHONPATH=src python scripts/run_sparse_retrieval.py \
  --units "$LEGACY_SAMPLE_RUN/evaluation_units.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$LEGACY_SPARSE_RUN" \
  --top-k 30

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$DENSE_PY" scripts/run_dense_retrieval.py \
  --units "$LEGACY_SAMPLE_RUN/evaluation_units.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$LEGACY_DENSE_RUN" \
  --model "$DENSE_MODEL" \
  --device cuda:0 \
  --top-k 30 \
  --batch-size 128 \
  --local-files-only

PYTHONPATH=src python scripts/fuse_retrieval_candidates.py \
  --sparse-candidates "$LEGACY_SPARSE_RUN/candidates.jsonl" \
  --dense-candidates "$LEGACY_DENSE_RUN/candidates.jsonl" \
  --run-dir "$LEGACY_HYBRID_RUN" \
  --top-k 25 \
  --sparse-quota 18 \
  --dense-quota 7
```

### 22.3 计算Recall@5/10/20/25

```bash
LEGACY_EVAL_RUN="runtime/$(date +%Y%m%d-%H%M%S)-legacy-recall-eval"
mkdir -p "$LEGACY_EVAL_RUN"
printf '%s\n' "$LEGACY_EVAL_RUN" > runtime/LATEST_LEGACY_RECALL_EVAL_RUN

PYTHONPATH=src python scripts/evaluate_legacy_recall.py \
  --units "$LEGACY_SAMPLE_RUN/evaluation_units.jsonl" \
  --candidates "$LEGACY_HYBRID_RUN/candidates.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$LEGACY_EVAL_RUN"

python -m json.tool "$LEGACY_EVAL_RUN/report.json"
wc -l "$LEGACY_EVAL_RUN/misses.jsonl" "$LEGACY_EVAL_RUN/per_label.jsonl"
```

`report.json`同时给出any-hit、all-hit、micro recall和独立题/小题分层指标；`per_label.jsonl`用于找高频低召回Label，`misses.jsonl`用于人工抽查。这些是对历史弱标签的覆盖率，不是金标Recall。首先看`standalone`分层；组合题旧ID可能是父题并集，all-hit偏低不能直接判定召回失败。

### 22.4 验证通过后增加旧Label候选通道

这一步不直接继承旧Label，只将有效旧ID去重后追加到Top25，最终仍由DS拒绝或选择。每题最多补5个：

```bash
LEGACY_AUGMENTED_RUN="runtime/$(date +%Y%m%d-%H%M%S)-legacy-augmented-candidates"
mkdir -p "$LEGACY_AUGMENTED_RUN"
printf '%s\n' "$LEGACY_AUGMENTED_RUN" > runtime/LATEST_LEGACY_AUGMENTED_RUN

PYTHONPATH=src python scripts/augment_candidates_with_legacy.py \
  --units "$LEGACY_SAMPLE_RUN/evaluation_units.jsonl" \
  --candidates "$LEGACY_HYBRID_RUN/candidates.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$LEGACY_AUGMENTED_RUN" \
  --max-legacy-additions 5

python -m json.tool "$LEGACY_AUGMENTED_RUN/report.json"
```

## 23. 用旧 `knw_ids` 抽题验证老师释义边界

每个当前Label优先抽5道旧ID正例。第一轮只抽独立题，避免组合题的父题Label并集污染判断。DS只看题目、Label名和老师四个释义字段，不看旧ID，只输出`applicable: true/false`。

### 23.1 构建正例样本

```bash
UNIT_RUN="$(cat runtime/LATEST_LABEL_UNITS_RUN)"
BOUNDARY_SAMPLE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-label-boundary-positive-sample"
mkdir -p "$BOUNDARY_SAMPLE_RUN"
printf '%s\n' "$BOUNDARY_SAMPLE_RUN" > runtime/LATEST_LABEL_BOUNDARY_SAMPLE_RUN

PYTHONPATH=src python scripts/build_label_boundary_sample.py \
  --units "$UNIT_RUN/label_units.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$BOUNDARY_SAMPLE_RUN" \
  --positive-per-label 5 \
  --negative-per-label 0 \
  --unit-type standalone

python -m json.tool "$BOUNDARY_SAMPLE_RUN/report.json"
wc -l "$BOUNDARY_SAMPLE_RUN/boundary_samples.jsonl"
```

`labels_without_legacy_positive`表示在独立题中没有找到历史正例的当前Label，不代表该Label无效。后续可去掉`--unit-type standalone`再补抽，但组合题结果需要单独看待。

### 23.2 20对 smoke

```bash
BOUNDARY_SMOKE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-label-boundary-judge-smoke"
mkdir -p "$BOUNDARY_SMOKE_RUN"

nohup env PYTHONPATH=src python scripts/run_label_boundary_judge.py \
  --samples "$BOUNDARY_SAMPLE_RUN/boundary_samples.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$BOUNDARY_SMOKE_RUN" \
  --endpoint 'http://172.22.0.35:9104/v1/chat/completions' \
  --model 'DeepSeek-V4-Flash' \
  --limit 20 \
  --workers 4 \
  --timeout 300 \
  --retries 10 \
  --retry-delay 2 \
  --request-interval 2 \
  --max-tokens 32 \
  > "$BOUNDARY_SMOKE_RUN/nohup.log" 2>&1 &

printf '%s\n' "$!" > "$BOUNDARY_SMOKE_RUN/pid"
```

### 23.3 全部正例Judge

```bash
BOUNDARY_JUDGE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-label-boundary-positive-judge"
mkdir -p "$BOUNDARY_JUDGE_RUN"
printf '%s\n' "$BOUNDARY_JUDGE_RUN" > runtime/LATEST_LABEL_BOUNDARY_JUDGE_RUN

nohup env PYTHONPATH=src python scripts/run_label_boundary_judge.py \
  --samples "$BOUNDARY_SAMPLE_RUN/boundary_samples.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$BOUNDARY_JUDGE_RUN" \
  --endpoint 'http://172.22.0.35:9104/v1/chat/completions' \
  --model 'DeepSeek-V4-Flash' \
  --workers 4 \
  --timeout 300 \
  --retries 10 \
  --retry-delay 2 \
  --request-interval 2 \
  --max-tokens 32 \
  > "$BOUNDARY_JUDGE_RUN/nohup.log" 2>&1 &

PID=$!
printf '%s\n' "$PID" > "$BOUNDARY_JUDGE_RUN/pid"
printf 'BOUNDARY_JUDGE_RUN=%s PID=%s\n' "$BOUNDARY_JUDGE_RUN" "$PID"
```

同一运行目录重跑时会跳过已成功的题目-Label对，只补跑error/pending；样本、Label表、模型或参数变化时会被`run_manifest.json`拒绝混跑。验收：

```bash
python -m json.tool "$BOUNDARY_JUDGE_RUN/report.json"
wc -l "$BOUNDARY_JUDGE_RUN/predictions.jsonl" "$BOUNDARY_JUDGE_RUN/per_label.jsonl"
```

`legacy_positive_true_rate`表示DS根据老师释义接受历史正例的比例。false项必须人工抽查：可能是释义边界过窄，也可能是旧ID错标，不能自动当成老师释义有问题。正例完成后，再用`--negative-per-label 3`抽同父级难负例，验证释义是否过宽。

## 24. v8错标拒绝与反证复核

v7恢复了覆盖，但300题复核仍发现“渗透失水→质壁分离实验”、“连锁→分离定律”、“小分子跨膜→胞吞胞吐”和“载体蛋白→蛋白质变性”等可污染训练集的错标。v8以错标代价远高于漏标为核心：

- 候选必须同时通过直接考查、精确定义、考查维度、`distinctions`硬否决和必要性反问五道门槛。
- 模型对暂定`selected`再做一次反证复核；任一门槛不确定就删除。
- `rejected_risky`最多保留3个“一度想选但已被复核否决”的短代码，只用于审计，不进入训练Label。
- 若候选只有相近替代项，必须空标并扩召；不强制产出。
- Label Card增加`common_assessments`，用于识别原理/实验/应用等考查维度。
- Prompt不写入300题中已知错例，避免评测泄漏和针对个别题目过拟合；这些题只作回归验收集。
- v8.1的风险优先顺序使模型过度进入否决模式；299题中265题输出了`rejected_risky`，可训练题从v7的286降至184。v8.2只改回`selected → rejected_risky → context_insufficient → need_expand_recall`生成顺序，其他Prompt和程序逻辑不变，作为单变量顺序消融。
- v8.1明确`rejected_risky`不是所有未选候选的列表。若模型仍输出超过3项或与`selected`重叠，程序按“拒绝优先”保守归一化，设置`output_conflict=true`并禁止该题进入训练，不再无限重试。
- v8.2将可训练题恢复到239，证明顺序有影响；但265/299题仍输出显式风险候选，且仍有错标。v8.3因此彻底删除该输出字段及其归一化/过滤逻辑，只保留内部反证过程。

为了只比较Prompt，v8仍先使用原300题和原Top25，不同时接入旧ID增强候选：

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
git pull --ff-only origin main

AUDIT_SAMPLE_RUN="$(cat runtime/LATEST_ADJUDICATION_AUDIT_SAMPLE_RUN)"
V83_RUN="runtime/$(date +%Y%m%d-%H%M%S)-candidate-judge-v8-3-internal-reflection"
mkdir -p "$V83_RUN"
printf '%s\n' "$V83_RUN" > runtime/LATEST_ADJUDICATION_V83_RUN

nohup env PYTHONPATH=src python scripts/run_candidate_adjudication.py \
  --units "$AUDIT_SAMPLE_RUN/audit_units.jsonl" \
  --candidates "$AUDIT_SAMPLE_RUN/audit_candidates.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$V83_RUN" \
  --endpoint 'http://172.22.0.35:9204/v1/chat/completions' \
  --model 'DeepSeek-V4-Flash' \
  --workers 20 \
  --timeout 300 \
  --retries 5 \
  --retry-delay 1 \
  --request-interval 0 \
  --max-tokens 256 \
  > "$V83_RUN/nohup.log" 2>&1 &

PID=$!
printf '%s\n' "$PID" > "$V83_RUN/pid"
printf 'V83_RUN=%s PID=%s\n' "$V83_RUN" "$PID"
```

完成条件仍是`input=processed=success=300`、`error=pending=0`，Prompt版本应为`candidate-adjudication-v8.3-internal-reflection`。精度验收优先级高于`usable_for_training`数量：

1. 施肥烧苗不得选质壁分离实验；原Top25缺正确渗透Label时应空标+扩召。
2. 两基因三种表型应选连锁，不得用分离/自由组合替代。
3. 小分子跨膜不得选胞吞胞吐。
4. ABA-Cl⁻载体题不得选蛋白质变性/泛化功能。
5. 固定化脲酶题不得因父题的菌株分离背景选尿素分解菌；当前458中固定化酶归入“酶的保存及应用”。
6. 预测与报告中不应再出现`rejected_risky`或`output_conflict`字段；错标是否被排除只通过最终`selected_labels`和回归题人工审核判断。

### 24.1 v8.4当前小题优先消融

v8.3的300题结果中，固定化脲酶小题仍被父题“从土壤中分离胞外脲酶高产菌株”带偏，错选“分离以尿素为氮源的微生物”。v8.4仅改变题目语境的呈现顺序，不改候选、Label Card、判标门槛和输出结构：

```text
【唯一判标对象：当前小题】
当前小题 / 选项 / 答案 / 解析

【仅用于补全指代，不得作为独立判标依据】
父题公共材料
```

使用原300题和原Top25进行单变量复测：

```bash
AUDIT_SAMPLE_RUN="$(cat runtime/LATEST_ADJUDICATION_AUDIT_SAMPLE_RUN)"
V84_RUN="runtime/$(date +%Y%m%d-%H%M%S)-candidate-judge-v8-4-current-first"
mkdir -p "$V84_RUN"
printf '%s\n' "$V84_RUN" > runtime/LATEST_ADJUDICATION_V84_RUN

nohup env PYTHONPATH=src python scripts/run_candidate_adjudication.py \
  --units "$AUDIT_SAMPLE_RUN/audit_units.jsonl" \
  --candidates "$AUDIT_SAMPLE_RUN/audit_candidates.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$V84_RUN" \
  --endpoint 'http://172.22.0.35:9204/v1/chat/completions' \
  --model 'DeepSeek-V4-Flash' \
  --workers 20 \
  --timeout 300 \
  --retries 5 \
  --retry-delay 1 \
  --request-interval 0 \
  --max-tokens 256 \
  > "$V84_RUN/nohup.log" 2>&1 &

PID=$!
printf '%s\n' "$PID" > "$V84_RUN/pid"
printf 'V84_RUN=%s PID=%s\n' "$V84_RUN" "$PID"
```

验收时先检查固定化脲酶题`2326198965676548098`是否从“分离以尿素为氮源的微生物”改为“酶的保存及应用”；同时比较总空标数、`context_insufficient`、`need_expand_recall`和已知回归题，避免为修复一题造成系统性覆盖下降。

实测v8.4未通过：固定化脲酶仍选错，`context_insufficient`从57升至95，可训练题从218降至181；因此不作为后续基线。另发现候选短代码的确定性打乱曾绑定`PROMPT_VERSION`，修改版本名会同时改变候选位置，后续消融必须固定候选排列种子。

### 24.2 v8.5恢复v8.3并输出简短reason

v8.5完整恢复v8.3的题目JSON布局、判标规则和候选排列，候选排列固定使用`candidate-adjudication-v8.3-internal-reflection`作为种子。唯一新增行为是要求DS首先输出简短`reason`：1至2句话、不超过120字，只概括最终选择、置空、扩召或上下文不足的依据，不要求详细思考过程。`reason`同时写入`evidence.jsonl`的`parsed_response`和`predictions.jsonl`，仅用于审计，不参与训练过滤。

```bash
AUDIT_SAMPLE_RUN="$(cat runtime/LATEST_ADJUDICATION_AUDIT_SAMPLE_RUN)"
V85_RUN="runtime/$(date +%Y%m%d-%H%M%S)-candidate-judge-v8-5-reason-first"
mkdir -p "$V85_RUN"
printf '%s\n' "$V85_RUN" > runtime/LATEST_ADJUDICATION_V85_RUN

nohup env PYTHONPATH=src python scripts/run_candidate_adjudication.py \
  --units "$AUDIT_SAMPLE_RUN/audit_units.jsonl" \
  --candidates "$AUDIT_SAMPLE_RUN/audit_candidates.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$V85_RUN" \
  --endpoint 'http://172.22.0.35:9204/v1/chat/completions' \
  --model 'DeepSeek-V4-Flash' \
  --workers 20 \
  --timeout 300 \
  --retries 5 \
  --retry-delay 1 \
  --request-interval 0 \
  --max-tokens 256 \
  > "$V85_RUN/nohup.log" 2>&1 &

PID=$!
printf '%s\n' "$PID" > "$V85_RUN/pid"
printf 'V85_RUN=%s PID=%s\n' "$V85_RUN" "$PID"
```

实测v8.5将可训练题从v8.3的218提高到283，空标从67降至12，但重新引入“施肥烧苗→动物细胞吸水失水实验/质壁分离实验”等明确错标。`reason`显示模型会先形成宽泛的“相关即可选”叙述，再据此扩张`selected`。因此v8.5只作为诊断版本，不用于生产。

### 24.3 v8.6 selected-first、reason-last顺序消融

v8.6与v8.5使用完全相同的题目布局、候选、候选代码排列、判标规则和reason要求；唯一变化是输出顺序由`reason → selected → 状态`改为`selected → 状态 → reason`。用于区分“要求输出reason本身”与“reason先生成”对选标结果的影响。

```bash
AUDIT_SAMPLE_RUN="$(cat runtime/LATEST_ADJUDICATION_AUDIT_SAMPLE_RUN)"
V86_RUN="runtime/$(date +%Y%m%d-%H%M%S)-candidate-judge-v8-6-reason-last"
mkdir -p "$V86_RUN"
printf '%s\n' "$V86_RUN" > runtime/LATEST_ADJUDICATION_V86_RUN

nohup env PYTHONPATH=src python scripts/run_candidate_adjudication.py \
  --units "$AUDIT_SAMPLE_RUN/audit_units.jsonl" \
  --candidates "$AUDIT_SAMPLE_RUN/audit_candidates.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$V86_RUN" \
  --endpoint 'http://172.22.0.35:9204/v1/chat/completions' \
  --model 'DeepSeek-V4-Flash' \
  --workers 20 \
  --timeout 300 \
  --retries 5 \
  --retry-delay 1 \
  --request-interval 0 \
  --max-tokens 256 \
  > "$V86_RUN/nohup.log" 2>&1 &

PID=$!
printf '%s\n' "$PID" > "$V86_RUN/pid"
printf 'V86_RUN=%s PID=%s\n' "$V86_RUN" "$PID"
```

核心比较：v8.6与v8.3的Label集合一致率、v8.6与v8.5的一致率、可训练/空标/context/expand，以及施肥烧苗、固定化脲酶、ABA-Cl⁻、基因连锁、水跨膜、新冠疫苗等回归题。

## 25. Top25 + 当前458内旧knw_ids精判消融

目的：保持原300题、V8.6 Prompt和DS参数不变，只把每题历史`knw_ids`中仍属于当前458的ID追加到原Top25候选。旧ID不直接作为答案，仍交给DS逐个精判；释义表外的旧ID直接删除。由于审计样本为了盲测已移除旧ID，追加脚本必须通过`question_id`回连全量`label_units.jsonl`。

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag

UNIT_RUN="$(cat runtime/LATEST_LABEL_UNITS_RUN)"
AUDIT_SAMPLE_RUN="$(cat runtime/LATEST_ADJUDICATION_AUDIT_SAMPLE_RUN)"
LEGACY_AUG_RUN="runtime/$(date +%Y%m%d-%H%M%S)-audit-top25-plus-legacy"

mkdir -p "$LEGACY_AUG_RUN"
printf '%s\n' "$LEGACY_AUG_RUN" > runtime/LATEST_AUDIT_TOP25_PLUS_LEGACY_RUN

PYTHONPATH=src python scripts/augment_candidates_with_legacy.py \
  --units "$AUDIT_SAMPLE_RUN/audit_units.jsonl" \
  --candidates "$AUDIT_SAMPLE_RUN/audit_candidates.jsonl" \
  --legacy-units "$UNIT_RUN/label_units.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$LEGACY_AUG_RUN"

python -m json.tool "$LEGACY_AUG_RUN/report.json"
```

这里故意不传`--max-legacy-additions`：同一题所有仍属于当前458的旧ID都应进入候选，报告中的候选数因此可能大于25。

```bash
V86_LEGACY_RUN="runtime/$(date +%Y%m%d-%H%M%S)-candidate-judge-v8-6-top25-plus-legacy"
mkdir -p "$V86_LEGACY_RUN"
printf '%s\n' "$V86_LEGACY_RUN" > runtime/LATEST_ADJUDICATION_V86_LEGACY_RUN

nohup env PYTHONPATH=src python scripts/run_candidate_adjudication.py \
  --units "$AUDIT_SAMPLE_RUN/audit_units.jsonl" \
  --candidates "$LEGACY_AUG_RUN/candidates.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$V86_LEGACY_RUN" \
  --endpoint 'http://172.22.0.35:9204/v1/chat/completions' \
  --model 'DeepSeek-V4-Flash' \
  --workers 20 \
  --timeout 300 \
  --retries 5 \
  --retry-delay 1 \
  --request-interval 0 \
  --max-tokens 256 \
  > "$V86_LEGACY_RUN/nohup.log" 2>&1 &

PID=$!
printf '%s\n' "$PID" > "$V86_LEGACY_RUN/pid"
printf 'V86_LEGACY_RUN=%s PID=%s\n' "$V86_LEGACY_RUN" "$PID"
```

只比较以下变化：明确错标数、原Top25空标题是否被正确旧ID救回、是否因旧标签噪声新增错标、`usable_for_training`。这个实验不把旧ID当金标。

## 26. 生物Label释义覆盖实验（每Label最多500题）

### 26.1 合并长尾更新数据

更新文件同时包含地理、政治、历史和生物，程序只取`subject=生物`。按`question_id`用更新记录覆盖原始记录，并追加新题；原4.8GB文件保持不变。随后必须重新做父子题聚合和缺父题审计，不能直接把更新文件中的10个小题送入实验。

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag

RAW="$(find /local_data/zhangyonglin/data/bio-know-tag -maxdepth 1 -type f -name 'biology.raw*jsonl' -print -quit)"
UPDATE_ROOT='/home/share_ssd_data/nfs-data1/wangmeng148/data/tiku/high-geo-hist-pol/update-data'
MERGED_RAW='/local_data/zhangyonglin/data/bio-know-tag/biology.with-update-20260914.raw.jsonl'
MERGE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-merge-biology-updates"

test -n "$RAW" || { echo '找不到biology.raw*.jsonl' >&2; exit 1; }
printf 'RAW=%s\n' "$RAW"
mkdir -p "$MERGE_RUN"
PYTHONPATH=src python scripts/merge_question_updates.py \
  --base "$RAW" \
  --updates "$UPDATE_ROOT/all.jsonl" \
  --subject '生物' \
  --output "$MERGED_RAW" \
  --report "$MERGE_RUN/report.json"

python -m json.tool "$MERGE_RUN/report.json"
```

验收重点：`subject_update_rows=45`，且`output_rows = base_rows + new_rows_added`；`existing_rows_updated`与`new_rows_added`由实际数据决定。

### 26.2 重新预处理并构建打标单元

```bash
UPDATED_PREPROCESS_RUN="runtime/$(date +%Y%m%d-%H%M%S)-preprocess-with-updates"
mkdir -p "$UPDATED_PREPROCESS_RUN"

nohup env PYTHONPATH=src python scripts/preprocess_questions.py \
  --input "$MERGED_RAW" \
  --run-dir "$UPDATED_PREPROCESS_RUN" \
  > "$UPDATED_PREPROCESS_RUN/nohup.log" 2>&1 &

PID=$!
printf '%s\n' "$PID" > "$UPDATED_PREPROCESS_RUN/pid"
printf '%s\n' "$UPDATED_PREPROCESS_RUN" > runtime/LATEST_UPDATED_PREPROCESS_RUN
```

预处理结束后：

```bash
python -m json.tool "$UPDATED_PREPROCESS_RUN/report.json"

UPDATED_ORPHAN_RUN="runtime/$(date +%Y%m%d-%H%M%S)-orphan-audit-with-updates"
mkdir -p "$UPDATED_ORPHAN_RUN"
PYTHONPATH=src python scripts/audit_orphan_parents.py \
  --raw "$MERGED_RAW" \
  --processed "$UPDATED_PREPROCESS_RUN/questions.jsonl" \
  --run-dir "$UPDATED_ORPHAN_RUN"

UPDATED_UNIT_RUN="runtime/$(date +%Y%m%d-%H%M%S)-label-units-with-updates"
mkdir -p "$UPDATED_UNIT_RUN"
PYTHONPATH=src python scripts/build_label_units.py \
  --input "$UPDATED_PREPROCESS_RUN/questions.jsonl" \
  --labels configs/label_strategies.review2.jsonl \
  --orphan-audit "$UPDATED_ORPHAN_RUN/orphan_parents.jsonl" \
  --run-dir "$UPDATED_UNIT_RUN"

printf '%s\n' "$UPDATED_UNIT_RUN" > runtime/LATEST_UPDATED_LABEL_UNITS_RUN
```

### 26.3 关联题干/解析图片URL并审计文本可用性

图片映射文件约805MB，顶层为`question_id -> {stemImageUrl, analysisImageUrl}`。审计脚本流式读取该文件，不将整个JSON载入内存，也不下载图片或发送给文本版DS。输出与`label_units.jsonl`同序的sidecar，以及确定排除和待复核清单。

```bash
IMAGE_MAP='/home/share_ssd_data/nfs-data1/wangmeng148/data/tiku/high-geo-hist-pol/题干-解析图片url/four_subject_image_urls.json'
IMAGE_AUDIT_RUN="runtime/$(date +%Y%m%d-%H%M%S)-image-context-audit"
mkdir -p "$IMAGE_AUDIT_RUN"
printf '%s\n' "$IMAGE_AUDIT_RUN" > runtime/LATEST_IMAGE_CONTEXT_AUDIT_RUN

nohup env PYTHONPATH=src python scripts/audit_label_unit_images.py \
  --units "$UPDATED_UNIT_RUN/label_units.jsonl" \
  --image-map "$IMAGE_MAP" \
  --run-dir "$IMAGE_AUDIT_RUN" \
  > "$IMAGE_AUDIT_RUN/nohup.log" 2>&1 &

PID=$!
printf '%s\n' "$PID" > "$IMAGE_AUDIT_RUN/pid"
printf 'IMAGE_AUDIT_RUN=%s PID=%s\n' "$IMAGE_AUDIT_RUN" "$PID"
```

输出含义：

- `image_context.jsonl`：每个打标单元的当前题/父题题干图、解析图URL及文本可用状态；
- `excluded_units.jsonl`：当前设问缺失或题干上下文完全为空，不交给文本DS；
- `review_units.jsonl`：图片中才有题干，或只有一段stem但无选项、答案、解析，需要后续OCR/多模态或抽查；
- `report.json`：各状态数量。图片存在但未给文本DS，不再误记为原始图片丢失。

### 26.4 抽样与DS评分

每个Label最多抽500个仍带该当前Label历史ID的实际打标单元（独立题、小题、缺父题小题）。历史ID只是找题用的弱监督，不是金标。DS只看到Label名称、老师四字段释义和题目，不看到`knw_ids`。

```bash
COVERAGE_SAMPLE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-definition-coverage-sample500"
mkdir -p "$COVERAGE_SAMPLE_RUN"
printf '%s\n' "$COVERAGE_SAMPLE_RUN" > runtime/LATEST_DEFINITION_COVERAGE_SAMPLE_RUN

PYTHONPATH=src python scripts/build_label_boundary_sample.py \
  --units "$UPDATED_UNIT_RUN/label_units.jsonl" \
  --image-context "$IMAGE_AUDIT_RUN/image_context.jsonl" \
  --exclude-content-review \
  --labels configs/labels.jsonl \
  --run-dir "$COVERAGE_SAMPLE_RUN" \
  --positive-per-label 500 \
  --negative-per-label 0 \
  --seed definition-coverage-biology-v2

python -m json.tool "$COVERAGE_SAMPLE_RUN/report.json"
wc -l "$COVERAGE_SAMPLE_RUN/boundary_samples.jsonl"
```

先跑20条smoke：

```bash
COVERAGE_SMOKE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-definition-coverage-smoke"
mkdir -p "$COVERAGE_SMOKE_RUN"
PYTHONPATH=src python scripts/run_label_boundary_judge.py \
  --samples "$COVERAGE_SAMPLE_RUN/boundary_samples.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$COVERAGE_SMOKE_RUN" \
  --endpoint 'http://172.22.0.35:9204/v1/chat/completions' \
  --model 'DeepSeek-V4-Flash' \
  --workers 20 \
  --timeout 300 \
  --retries 5 \
  --retry-delay 1 \
  --request-interval 0 \
  --max-tokens 256 \
  --limit 20

python -m json.tool "$COVERAGE_SMOKE_RUN/report.json"
```

确认输出包含`score`、程序派生的`match`、`difference_type`和简短`reason`后再全量后台运行：

```bash
COVERAGE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-definition-coverage-full"
mkdir -p "$COVERAGE_RUN"
printf '%s\n' "$COVERAGE_RUN" > runtime/LATEST_DEFINITION_COVERAGE_RUN

nohup env PYTHONPATH=src python scripts/run_label_boundary_judge.py \
  --samples "$COVERAGE_SAMPLE_RUN/boundary_samples.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$COVERAGE_RUN" \
  --endpoint 'http://172.22.0.35:9204/v1/chat/completions' \
  --model 'DeepSeek-V4-Flash' \
  --workers 20 \
  --timeout 300 \
  --retries 5 \
  --retry-delay 1 \
  --request-interval 0 \
  --max-tokens 256 \
  > "$COVERAGE_RUN/nohup.log" 2>&1 &

PID=$!
printf '%s\n' "$PID" > "$COVERAGE_RUN/pid"
printf 'COVERAGE_RUN=%s PID=%s\n' "$COVERAGE_RUN" "$PID"
```

报告输出：`report.json`给总体分段、题型、差异类型和按题最高分的A/B/C/D级；`per_label.jsonl`给逐Label匹配率、均分、0分率、灰度区及差异类型；`predictions.jsonl`保留逐题逐Label的分数和理由。重点把`definition_too_narrow`与`legacy_label_wrong`分开，避免将旧题错标误诊为老师释义问题。

释义覆盖v2的20-Label smoke虽然20/20请求成功，但暴露三类一般性边界错误：只考Label内一个明确子主题时可能被过严拒绝；不同实验模块仅共享“消毒/无菌”等操作时可能被错误接纳；“综合”Label可能在未满足多方面联动门槛时被过宽接纳。v3据此仅调整一般规则，不加入具体题目或具体Label补丁：

1. 题目直接考查definition/core_concepts中的任一明确子主题即可匹配，无需覆盖释义全部内容；
2. 共享实验动作、工具或关键词不等于知识对象和任务目标相同；
3. 名称/释义明确为综合或整合的Label必须满足其多方面联动要求。

v3 Prompt版本为`label-definition-coverage-v3-boundary-calibrated`，必须使用新run目录与同一批smoke题复测，再决定是否启动全量。

### 26.5 Mentor-compatible同Label批处理主实验

为与高中历史释义覆盖实验保持可比较口径，正式主实验采用mentor-compatible批处理实现：同一个Label的释义只发送一次，每批最多40题、题目正文总计不超过55000字符；DS只输出`task_id/question_id/match/relevance_score`，不输出理由或差异类型；程序强制以0.70重新计算match。V3单题版保留为小规模诊断工具，不作为19.8万任务的主运行器。

该模式显式设置`chat_template_kwargs.enable_thinking=false`，使用非流式响应，默认8并发、600秒超时、3次重试。约198383个题目-Label对将由约5000至10000个批请求完成，实际批次数取决于55000字符预算。

```bash
COVERAGE_SAMPLE_RUN="$(cat runtime/LATEST_DEFINITION_COVERAGE_SAMPLE_RUN)"
SMOKE_INPUT="$COVERAGE_SAMPLE_RUN/smoke-20-labels.jsonl"
MENTOR_SMOKE_RUN="runtime/$(date +%Y%m%d-%H%M%S)-definition-coverage-mentor-smoke20"
mkdir -p "$MENTOR_SMOKE_RUN"

PYTHONPATH=src python scripts/run_definition_coverage_batches.py \
  --tasks "$SMOKE_INPUT" \
  --labels configs/labels.jsonl \
  --run-dir "$MENTOR_SMOKE_RUN" \
  --endpoint 'http://172.22.0.35:9204/v1/chat/completions' \
  --model 'DeepSeek-V4-Flash' \
  --workers 8 \
  --max-batch-size 40 \
  --char-budget 55000 \
  --max-tokens 7000 \
  --timeout 600 \
  --retries 3 \
  --retry-delay 1 \
  --request-interval 0
```

确认`report.json`中`input=processed=success=20`、`error=pending=0`、Prompt版本为`label-definition-coverage-mentor-batch-v1`后，再用`boundary_samples.jsonl`启动全量。全量必须新建目录并后台运行；失败批次保留在`evidence.jsonl`，使用完全相同命令和run目录即可只续跑未成功任务。
