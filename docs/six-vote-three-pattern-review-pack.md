# 六票三类重点票型：逐 Label 十题自查包

目标是从同一批六票均成功的题中，对每个 Label 分别抽取 `Q3/D3`、`Q3/D0`、`Q0/D3` 各最多 10 个题–Label 对。某 Label 在某票型不足 10 题则全取；没有该票型则为空。抽样使用固定随机种子和 reservoir sampling，**不是教师金标，也不能从这些诊断题直接估计准确率**。

本工具会重新读取当前仍在增长的六份证据，因此这是执行时的新快照，不保证与 2026-09-28 03:12 UTC 那份 142,047 道题快照完全相同。原先的报告每格每 Label 仅保留 3 道，无法从那份报告补出另外 7 道。运行不会修改六票证据或打标结果。

## 服务器运行

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
git pull --ff-only origin main

RUN='runtime/20260924-104413-biology-s85-reuse-preprocess'
REVIEW="$RUN/three-pattern-review-$(date +%Y%m%d-%H%M%S)"
IMAGE_MAP='/home/share_ssd_data/nfs-data1/wangmeng148/data/tiku/high-geo-hist-pol/题干-解析图片url/four_subject_image_urls.json'

test -s "$RUN/fine-full-six/adjudication-no-thinking/run_manifest.json"
test -s "$RUN/fine-full-six/legacy/candidates.jsonl"
test -s "$RUN/unified/retrieval_units.jsonl"
test -s "$IMAGE_MAP"

PYTHONPATH=src python -u scripts/analyze_vote_pattern_cube.py \
  --votes-root "$RUN/fine-full-six/adjudication-no-thinking" \
  --candidates "$RUN/fine-full-six/legacy/candidates.jsonl" \
  --units "$RUN/unified/retrieval_units.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$REVIEW" \
  --review-examples-per-label-pattern 10 \
  --image-map "$IMAGE_MAP"

jq '{labels,labels_with_examples,sample_counts,image_context_question_matches,common_successful_questions}' \
  "$REVIEW/review-three-patterns/manifest.json"
```

输出包括 `review-three-patterns/index.md`（458 Label 的三类总量/抽中题数索引）、`review-three-patterns/labels/<label_id>.json`（每 Label 一个文件）及 `review_samples.jsonl`（全部抽中题，便于后续制作审核页）。每条记录包含题号、父题号、题型、完整题干/选项/答案/解析、题干和解析图片 URL、候选 rank/来源、原有 `knw_ids`、六票分别选中的 Label ID，以及目标 Label 的释义。若图片映射没有该题的 URL，对应字段为空，不表示题目一定没有图片。

报告仍会生成十六格 `report.json` 和 `report.md`。正式教师抽审需要另行记录抽样概率、审核人和结论；这些自查题不能直接作为 5% 错标率的统计验证样本。当前分析器每票最多读取 50 万成功题，超过上限会明确报错，届时需切换磁盘式全量对齐实现。
