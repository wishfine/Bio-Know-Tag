# 生物去重全量：预处理与统一粗召回

本阶段只生成数据与候选，不调用 Qwen 精排。输入是 `生物-20260914-dedup-s85.jsonl`：它已经是保留题目的原始 JSONL（1,681,314 行），不是重复簇映射。45 条补充生物题按 `question_id` 覆盖或追加；语义上可能仍与保留题重复，不能把它们当作经 s85 去重的记录。

父题材料、小题、独立题进入**同一个** `retrieval_units.jsonl`，一起通过 BM25、Dense、Hybrid。父题材料单元只负责将来判断额外知识点；父题最终标签应在精排后汇总为“小题标签并集 + 父题材料额外标签”。目前精排程序仍禁止父题材料与小题混合运行，部署 Qwen 前必须修改并验证，不能直接拿这份统一输入启动旧精排脚本。

## 1. 合并并预处理

在服务器仓库中执行。各阶段成功后再进入下一阶段，不要并行启动依赖任务。

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
git pull --ff-only origin main

DEDUP='/home/share_ssd_data/nfs-data1/wangmeng148/data/tiku/high-geo-hist-pol/question-dedup-20260918/生物-20260914-dedup-s85.jsonl'
UPDATES='/home/share_ssd_data/nfs-data1/wangmeng148/data/tiku/high-geo-hist-pol/update-data/all.jsonl'
MERGED='/local_data/zhangyonglin/data/bio-know-tag/biology.dedup-s85-with-updates.raw.jsonl'
RUN="runtime/$(date +%Y%m%d-%H%M%S)-biology-dedup-s85-full-coarse"
mkdir -p "$RUN"/{merge,preprocess,orphan,units,unified,sparse,dense,hybrid,image-audit}
printf '%s\n' "$RUN" > runtime/LATEST_DEDUP_S85_FULL_COARSE_RUN

test -s "$DEDUP" && test -s "$UPDATES"
PYTHONPATH=src python scripts/merge_question_updates.py \
  --base "$DEDUP" --updates "$UPDATES" --subject 生物 \
  --output "$MERGED" --report "$RUN/merge/report.json"
python -m json.tool "$RUN/merge/report.json"

PYTHONPATH=src python scripts/preprocess_questions.py \
  --input "$MERGED" --run-dir "$RUN/preprocess"
python -m json.tool "$RUN/preprocess/report.json"

PYTHONPATH=src python scripts/audit_orphan_parents.py \
  --raw "$MERGED" --processed "$RUN/preprocess/questions.jsonl" \
  --run-dir "$RUN/orphan"
python -m json.tool "$RUN/orphan/report.json"

PYTHONPATH=src python scripts/build_label_units.py \
  --input "$RUN/preprocess/questions.jsonl" \
  --labels configs/label_strategies.review2.jsonl \
  --orphan-audit "$RUN/orphan/orphan_parents.jsonl" \
  --run-dir "$RUN/units"
python -m json.tool "$RUN/units/build_report.json"

PYTHONPATH=src python scripts/build_unified_retrieval_units.py \
  --units "$RUN/units/label_units.jsonl" \
  --parent-aggregation "$RUN/units/parent_aggregation.jsonl" \
  --run-dir "$RUN/unified"
python -m json.tool "$RUN/unified/report.json"
```

验收：`merge.base_rows=1681314`（源文件不变时）；`merge.output_rows=base_rows+new_rows_added`；预处理、父题审计、打标单元构建的 `error` 都必须为 0。`unified.retrieval_units` 必须等于 `retrieval_label_units+retrieval_parent_extra_units`。去重后父题可能不在保留集，重点检查 `orphan/report.json`，不要将合成的缺父题容器判标。

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

四个行数应一致，粗召回各报告的 `error` 应为 0。这里先冻结纯 Hybrid Top25；是否额外补原 `knw_ids` 应等两组精排对比方案确定后单独生成，不能覆盖本次候选文件。
