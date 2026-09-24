# 生物去重全量：预处理与统一粗召回

本阶段只生成数据与候选，不调用 Qwen 精排。唯一输入是 `生物-20260914-dedup-s85.jsonl`：这是已去重的原始题目 JSONL（1,681,314 行），按数据提供方的口径，其中不同 `question_id` 对应不同题目，不是重复簇映射。**不要再默认合并 `update-data/all.jsonl`**：追加独立来源的记录会使这份文件的去重保证不再适用于新集合。若以后要纳入未覆盖的更新题，应对合并后的全集重新去重并发布新版本，不能直接沿用本次 s85 版本名与结论。

父题材料、小题、独立题进入**同一个** `retrieval_units.jsonl`，一起通过 BM25、Dense、Hybrid。父题材料单元只负责判断额外知识点；父题最终标签在精排后汇总为“小题标签并集 + 父题材料额外标签”。精排执行器现支持混合输入，但百万级运行须先分片，不能把整份统一输入直接交给一次性读入内存的执行器。

s85 文件中已有 10,192 个被小题引用但不在文件内的父题 ID。来源审计显示，其中约 9,804 个可从原始库找回文本题干，388 个原始库也没有。**找得到的父题只作为上下文回连，不加回 s85 原始题目集合；真正缺来源的整组小题排除。**若源父题存在但只有图片或没有可用文本，其小题保留、标记缺上下文待复核，不因文本缺失而删除。若同 ID 原始记录实际是另一道小题而非父题，则不作为有效父题回连。该审计先前针对的是误合并 45 条更新的运行目录，正式 s85 运行须重新审计和校验，实际数量以新报告为准。

这里的“去重”以数据提供方已完成的 s85 文件为准：不再用清洗后文本哈希删除不同 `question_id` 的题。回连程序逐一核对处理前所有真实题目 ID 与 s85 原文件相等，拒绝混入旧的 `merged.raw.jsonl` 结果。

## 1. 直接预处理已去重文件

在服务器仓库中执行。各阶段成功后再进入下一阶段，不要并行启动依赖任务。

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
git pull --ff-only origin main

DEDUP='/home/share_ssd_data/nfs-data1/wangmeng148/data/tiku/high-geo-hist-pol/question-dedup-20260918/生物-20260914-dedup-s85.jsonl'
ORIGINAL='/local_data/zhangyonglin/data/bio-know-tag/biology.with-update-20260914.raw.jsonl'
RUN="runtime/$(date +%Y%m%d-%H%M%S)-biology-dedup-s85-full-coarse"
mkdir -p "$RUN"/{preprocess,orphan,orphan-source-audit,units,unified,sparse,dense,hybrid,image-audit}
printf '%s\n' "$RUN" > runtime/LATEST_DEDUP_S85_FULL_COARSE_RUN

test -s "$DEDUP" && test -s "$ORIGINAL"

PYTHONPATH=src python scripts/preprocess_questions.py \
  --input "$DEDUP" --run-dir "$RUN/preprocess"
python -m json.tool "$RUN/preprocess/report.json"

PYTHONPATH=src python scripts/audit_orphan_parents.py \
  --raw "$DEDUP" --processed "$RUN/preprocess/questions.jsonl" \
  --run-dir "$RUN/orphan"
python -m json.tool "$RUN/orphan/report.json"

PYTHONPATH=src python scripts/audit_orphan_parent_source.py \
  --orphan-audit "$RUN/orphan/orphan_parents.jsonl" \
  --original-raw "$ORIGINAL" \
  --run-dir "$RUN/orphan-source-audit"
python -m json.tool "$RUN/orphan-source-audit/report.json"

# 不预先 mkdir "$RUN/repaired"；修复器校验全量 ID 后原子发布该目录。
PYTHONPATH=src python scripts/repair_dedup_parent_context.py \
  --processed "$RUN/preprocess/questions.jsonl" \
  --parent-source-audit "$RUN/orphan-source-audit/per_parent.jsonl" \
  --dedup-raw "$DEDUP" \
  --run-dir "$RUN/repaired"
python -m json.tool "$RUN/repaired/report.json"

PYTHONPATH=src python scripts/build_label_units.py \
  --input "$RUN/repaired/questions.jsonl" \
  --labels configs/label_strategies.review2.jsonl \
  --orphan-audit "$RUN/repaired/orphan_parents.jsonl" \
  --run-dir "$RUN/units"
python -m json.tool "$RUN/units/build_report.json"

PYTHONPATH=src python scripts/build_unified_retrieval_units.py \
  --units "$RUN/units/label_units.jsonl" \
  --parent-aggregation "$RUN/units/parent_aggregation.jsonl" \
  --run-dir "$RUN/unified"
python -m json.tool "$RUN/unified/report.json"
```

验收：源文件 `wc -l` 为 1,681,314（提供方文件不变时）；预处理、父题审计、打标单元构建的 `error` 都必须为 0。`repaired.s85_question_ids` 必须等于 s85 行数，`output_s85_question_ids = s85_question_ids - dropped_orphan_children`。父题来源状态必须逐类对账：`recovered_context_parents + retained_source_parents_without_text + dropped_orphan_parent_groups = orphan_parent_ids`，相应三类小题数之和必须等于 `orphan_child_count`。`label_units.orphan_sub_question_units` 应为 0，但无文本父题的小题要检查 `parent_context_missing` / `image_context_missing` 标记。`unified.retrieval_units` 必须等于 `retrieval_label_units+retrieval_parent_extra_units`。`recovered_context_parents.jsonl` 记录回连父题；这些父题材料可参与额外知识点判断，但不是 s85 中新增的独立题。`dropped_orphan_groups.jsonl` 保留被排除小题的 ID，便于审计与回滚。

若已按旧命令生成了 `merged.raw.jsonl`，这份合并后文件及其下游结果不能**不经校验直接使用**。优先使用下面的增量裁剪路径；只有父子关系校验失败时，才从 `DEDUP` 原文件重新预处理。原始去重文件不受影响。

### 已完成旧预处理时的增量路径（优先用这个）

如果旧目录是 `runtime/20260923-190648-biology-dedup-s85-full-coarse`，可复用其中的 `preprocess/questions.jsonl`，不重跑 168 万题的 HTML 清洗。`filter_processed_to_s85.py` 只保留 s85 内的真实题号，剔除追加的 15 条新题，并将同 ID 的 30 条更新记录恢复成 s85 原文。若被剔除的记录原本是父题、其小题仍在 s85，则保留**空父题容器**待后续来源回连，绝不让更新题的题干冒充 s85 父题。它校验全部 s85 ID 恰好出现一次，且父子关系一致；如关系已被更新改变，会停止而不是静默改写，此时才需走上面的全量预处理路径。

```bash
OLD='runtime/20260923-190648-biology-dedup-s85-full-coarse'
UPDATES='/home/share_ssd_data/nfs-data1/wangmeng148/data/tiku/high-geo-hist-pol/update-data/all.jsonl'
RUN="runtime/$(date +%Y%m%d-%H%M%S)-biology-s85-reuse-preprocess"
mkdir -p "$RUN"/{orphan,orphan-source-audit,units,unified,sparse,dense,hybrid}

# 不预先创建 "$RUN/filtered" 或 "$RUN/repaired"；两步均原子发布。
PYTHONPATH=src python scripts/filter_processed_to_s85.py \
  --processed "$OLD/preprocess/questions.jsonl" \
  --dedup-raw "$DEDUP" --updates "$UPDATES" \
  --run-dir "$RUN/filtered"
python -m json.tool "$RUN/filtered/report.json"

PYTHONPATH=src python scripts/audit_orphan_parents.py \
  --raw "$DEDUP" --processed "$RUN/filtered/questions.jsonl" \
  --run-dir "$RUN/orphan"
PYTHONPATH=src python scripts/audit_orphan_parent_source.py \
  --orphan-audit "$RUN/orphan/orphan_parents.jsonl" \
  --original-raw "$ORIGINAL" --run-dir "$RUN/orphan-source-audit"
PYTHONPATH=src python scripts/repair_dedup_parent_context.py \
  --processed "$RUN/filtered/questions.jsonl" \
  --parent-source-audit "$RUN/orphan-source-audit/per_parent.jsonl" \
  --dedup-raw "$DEDUP" --run-dir "$RUN/repaired"
```

后续从第 1 节的 `build_label_units.py` 一行继续，使用这个新 `RUN`；第 2 节的粗召回命令不变。旧 `units`、`unified`、BM25、Dense、Hybrid 不能复用，因为父题背景和输入单元已改变。新的来源审计也要重跑：旧侧车只含题干，不包含此次父题额外打标要用的选项、解析和 `knw_ids`。

`unified/content_review.jsonl` 列出无当前题干的单元；其中有选项/解析者保留并标记，`no_current_question_text` 者进入 `text_ineligible.jsonl`，不参加文本召回。预处理会移除 `<img>`，空 `stem` 不等于原题没有内容；需结合图片审计核查，不能据此删除原始题。图片 URL 只供审核，不会让纯文本模型看到图片。

```bash
IMAGE_MAP='/home/share_ssd_data/nfs-data1/wangmeng148/data/tiku/high-geo-hist-pol/题干-解析图片url/four_subject_image_urls.json'
PYTHONPATH=src python scripts/audit_label_unit_images.py \
  --units "$RUN/units/label_units.jsonl" \
  --image-map "$IMAGE_MAP" --run-dir "$RUN/image-audit"
python -m json.tool "$RUN/image-audit/report.json"
```

## 2. 对统一输入做粗召回

```bash
DENSE_PY='/local_data/zhangyonglin/conda_envs/bio-know-tag-dense/bin/python'
DENSE_MODEL='/local_data/zhangyonglin/data/bio-know-tag/models/bge-small-zh-v1.5'
UNITS="$RUN/unified/retrieval_units.jsonl"
test -s "$UNITS" && test -x "$DENSE_PY" && test -d "$DENSE_MODEL"

PYTHONPATH=src python scripts/run_sparse_retrieval.py \
  --units "$UNITS" --labels configs/labels.jsonl \
  --run-dir "$RUN/sparse" --top-k 30 --progress-every 10000

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$DENSE_PY" scripts/run_dense_retrieval.py \
  --units "$UNITS" --labels configs/labels.jsonl \
  --run-dir "$RUN/dense" --model "$DENSE_MODEL" \
  --device cuda:0 --top-k 30 --batch-size 128 --local-files-only

PYTHONPATH=src python scripts/fuse_retrieval_candidates.py \
  --sparse-candidates "$RUN/sparse/candidates.jsonl" \
  --dense-candidates "$RUN/dense/candidates.jsonl" \
  --run-dir "$RUN/hybrid" --top-k 25 --sparse-quota 18 --dense-quota 7

python -m json.tool "$RUN/sparse/report.json"
python -m json.tool "$RUN/dense/report.json"
python -m json.tool "$RUN/hybrid/report.json"
wc -l "$UNITS" "$RUN/sparse/candidates.jsonl" \
  "$RUN/dense/candidates.jsonl" "$RUN/hybrid/candidates.jsonl"
```

四个行数应一致，粗召回各报告的 `error` 应为 0。Hybrid Top25 是粗排产物；已决定后续精排固定采用 Top25 加当前目录中有效的旧 `knw_ids`，需另生成候选文件，不覆盖纯 Top25 粗排。精排分片和混合父子题用法见 [全量三票方案](full-scale-qwen-three-vote-audit-plan.md)。
