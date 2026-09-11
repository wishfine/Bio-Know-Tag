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
