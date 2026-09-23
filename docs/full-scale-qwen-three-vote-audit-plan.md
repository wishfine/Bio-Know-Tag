# 高中生物全量精排：Qwen 三次投票与异构 Judge 复核方案

状态：实验设计稿。本文规定判定口径、留痕和放行条件；**不代表**已经完成全量精排、教师校准或最终数据发布。适用范围为去重后的生物题库、当前 458 个 Label，以及独立题、小题、真实复合题父题材料。对应的数据与粗排准备见 [去重全量粗排手册](full-dedup-coarse-runbook.md)。

## 1. 已确定的决策

1. 全量精排只使用一套候选：`Hybrid Top25 + 当前458目录中有效的旧 knw_ids`。旧 ID 只补充候选，不在提示词中注明“这是旧标签”，也不作为正确性依据。无效/过期 ID 不补入。
2. 每个判标单元由 **Qwen3.8-27B 独立请求三次**。三次使用同一题目、同一候选及顺序、同一 Label 卡、同一 prompt 和同一解析规则。保存三次原始响应，不能只保存多数票集合。
3. 后续可使用**不同模型家族**作 Judge。具体模型和版本在启动前固定、做小样本校准；Judge 不算 Qwen 的第四票，也不能看到旧标签、三次票数或先前 DS 判别结论。
4. 错标成本高于漏标。投票是稳定性证据，不是正确性证据；旧 `knw_ids` 是弱监督，不是金标；DS 对旧标签的 match 率是分层信号，不是逐题正确概率。
5. 自动放行不是仅凭规则立刻启用：先通过独立教师样本验证该层的错标与漏标风险，再启用该层的自动放行。

## 2. 输入冻结与运行前检查

为每次全量作业保存不可变 `run_manifest`：作为唯一题目源的 s85 去重文件 SHA-256、预处理版本、Label 文件 SHA-256、候选文件 SHA-256、候选构造版本、Qwen 模型权重/服务版本、prompt 版本、采样参数、temperature、seed（若服务支持）、端口、并发、输出预算、代码 commit。不能在该输入后直接追加 `update-data`，否则需重新去重并另起数据版本。三次运行的题目和候选行须逐题核对 ID、候选 ID **及顺序**，不一致者不能互投。

先用 300–1,000 道分层题做端到端 smoke：检查 JSON 解析、候选映射、三次请求是否真的独立、父子题逻辑、图片/空题干标记和失败续跑。再扩到固定的万题校准集，最后跑全量。三次运行使用不同 run 目录，故障重试只填补失败题，不覆盖已成功响应。不同服务实例应使用相同模型文件、推理参数和模板；记录端口，分析服务实例效应。`temperature=0` 也不能假设输出完全确定。

精排执行器现支持 `composite_parent_extra` 与独立题、小题混跑，并按单元类型使用对应提示词；每条结果写入自己的 `prompt_version`。执行器仍一次性读入输入，因此百万级作业先用 `shard_adjudication_inputs.py` 按题号同步切分题目与候选，再逐片运行。上线前仍须用真实 Qwen 服务做混合题小样本烟测与父题并集校验；本地单元测试不等于服务已验证。

**候选召回检查**：对每个单元记录 Top25、有效旧 ID、新增旧候选、最终候选序列、各 Label 来源和 rank。最终候选可能超过 25 个。三次 Qwen 必须使用完全相同的最终候选；若旧 ID 全在 Top25，最终候选应保持原 Top25 不变。增加候选与不增加候选的旧实验是敏感性证据，不构成本次投票的不同输入臂。

## 3. 判标单元与投票定义

独立题、小题、父题材料同处一个精排输入流，但各有自己的判标对象：

- 独立题：当前题干、选项、答案、解析。
- 小题：当前小题为唯一判标对象；父题公共材料仅用于补全指代。不能把父题或其他小题的考点自动迁入当前小题。
- 真实复合题父题材料：仅判断公共材料自身能够明确支持的**额外** Label；允许空集合。最终父题 Label 在精排后汇总为 `各小题最终 Label 并集 ∪ 父题材料额外 Label`，并保留来源。任一小题或父题材料的裁决尚未完成时，只能输出带缺失/争议标记的暂存并集，父题整体不得进入 `accepted` 或训练集。合成的缺父题容器不单独判标。

每次精排输出一个 Label 集合 `S1(q), S2(q), S3(q)`。对本题的**最终候选集合**（含补入的有效旧 ID）中的每个 Label `l` 计算 `votes(q,l) = Σ 1[l∈Si(q)]`，即 `0/3、1/3、2/3、3/3`；候选外 Label 不硬算为 `0/3`，另走召回审计。这样 `0/3` 旧 Label 确实是三次都见过却未选择，而不是根本没给模型看。另记录整题三个集合是否完全一致、两两 Jaccard、只有新增/只有删除/同数替换、空集合次数。**多标签题按每个 Label 投票**；即使整题集合三次都不同，某个核心 Label 仍可能是 `3/3`。若任何一次请求失败，该题为 `INCOMPLETE`，不能把失败当作反对票。

与旧标签的关系另算：`相等 / 新结果包含旧集 / 新结果是旧集子集 / 双向增减 / 完全不相交 / 无有效旧ID`。旧标签要先映射到当前 458 Label；过期 ID 单列。对每个旧 Label 再记录它的 Qwen 票数；对每个新选 Label 记录是否本来就在旧标签集，不能只用整题的“旧标一致/不一致”。

## 4. 外部证据：如何使用既有实验

给每个 Label 构造**风险卡**，只用于分层和审核优先级，不将多项 DS 结果相加伪装为概率。

| 证据 | 可用于 | 不可用于 |
|---|---|---|
| 正样本释义覆盖：179,568 个独立题–旧Label对，总 match 68.53%；135 个问题 Label 分档 | 识别旧标签/释义高冲突、`<0.10` 基本无关和 `0.10–0.69` 边界多的 Label | 直接断定某道题的旧 Label 错误、或释义太宽 |
| 122 个少于 300 道独立题的长尾 Label | 标记比例不稳，优先人工核对 | 用 1–29 道题的 match 率自动放行或封禁 |
| 21,647 个相邻 Label 硬负样本；V2 共标复核后的排除比例 | 定位可能共享名词、上下位/兄弟边界的 Label 对 | 当作人工金标误收率；“兄弟题”可能合理共标 |
| DS 五次稳定性、Qwen 重复运行及释义卡消融 | 找高波动 Label、候选/释义敏感题，形成压力测试集 | 外推困难样本的波动比例到全题库；把稳定等同正确 |
| Luna 逐题审核及修订、老师既有复核 | 典型错标模式和校准题候选 | 把未核准的模型审核结果称为教师金标 |

正负样本详情见 [458 Label 联合复核](positive-coverage-audit/biology-458-label-positive-negative-combined-review.md)；DS 重复请求在相同 prompt 下仍有集合波动，见 [五次稳定性实验](ds-stability-five-run-analysis.md)。Qwen 释义消融用的是刻意偏向难题的 3,257 对，且原实验缺 A/A、B/B 控制；复现实验的结论须按 [消融说明](qwen-definition-ablation-v2.md) 使用。Luna 记录曾出现明确修订，见 [逐题审核修订](luna-review-correction-20260923.md)。

风险卡至少含：有效正样本数和匹配/低分/边界比例、有效负样本数及边界争议、三次投票翻转率、历史审核结论、释义名实冲突、长尾等级。**高 match 仅说明历史正样本易被当前释义接受；低 match 可能是旧 ID 错、释义过窄、题目缺图或模型判断错，须看题。**

## 5. 异构 Judge 的任务与防锚定

异构 Judge 是逐题–Label 核验器，不是重跑同一 Qwen prompt 的“第四票”。优先以另一模型家族/服务完成，固定模型版本。输入包括当前判标单元的可见题目文本、必要的父题指代背景、目标 Label 名称/路径及四字段释义；可提供少量最相邻 Label 作辨析，但**不提供**旧 `knw_ids`、Qwen 三次票数、投票结论、候选来源标记和 DS 旧判断。Judge 任务池包括：全部 `2/3` 分歧 Label、高风险 `3/3` Label，以及按风险分层抽出的 `1/3` 和 `0/3` 漏标候选；低风险 `3/3` 另以教师随机抽样校准。只审当前候选集合会漏掉召回失败；另有候选外补标审计。

Judge 输出：`直接匹配 / 合理共标 / 不应打标 / 内容不足 / 释义或图谱冲突 / 候选外需补标`，附题目原文证据、当前设问、Label 定义边界、反证、置信度和机器可解析错误码。引用应可在原题文本中核验；引用失败不自动放行。Judge 认为“不应打标”且与高票 Qwen 冲突，或认为“需补标”却不在候选中，进入老师审核/召回修复，不让 Judge 单方面覆盖全部票数。

上线前用老师标注的校准集评估 Judge：错标拦截率、误杀合理共标率、漏标发现率、内容不足识别率。若 Judge 与 Qwen 高度同错，应降低其放行权重。Judge 也可能随 prompt 和服务波动；同一困难集至少重复一次评估。

## 6. 逐 Label 决策矩阵（初始规则）

表中 `AUTO_CANDIDATE` 仅为**待校准的自动放行候选层**，不是未经审核的金标。所有层都保留不可覆盖的原始三票；规则以后调整时从原始证据重算。

| Qwen票数/情况 | 旧标签与风险 | 下一步 | 初始状态 |
|---|---|---|---|
| `3/3` | 旧标含此 Label、正样本覆盖稳定、无边界/图像/父子风险 | 随机教师盲抽；该层达到放行门槛后批量采用 | `AUTO_CANDIDATE` |
| `3/3` | 新 Label 不在旧标，属于纯新增；或取代了低覆盖旧 Label | 异构 Judge 分层抽样；教师盲抽校准新标（有被删旧标时也校准旧标） | 低风险为 `AUTO_CANDIDATE`，高风险为 `REVIEW` |
| `3/3` | 新结果与高覆盖旧标冲突，或目标 Label 本身属高风险 | 异构 Judge 逐题核验；冲突送老师 | `REVIEW` |
| `3/3` | 旧标一致但目标 Label 属135问题集、长尾、边界风险或释义冲突 | 不因一致而放行；按风险卡抽样或逐题核验 | `REVIEW` |
| `2/3` | 任意旧标关系 | 对该**分歧 Label**逐题异构 Judge；Judge 无法确定或与投票冲突送老师 | `REVIEW` |
| `1/3` | 任意旧标关系 | 默认不进入训练正标；旧标高覆盖、长尾或候选高相关者抽样 Judge，检查漏标 | `REVIEW`/暂不选 |
| `0/3` | 旧标含此 Label 且其历史覆盖较高 | 抽样核查是否共同漏标、召回/释义/缺图问题；不能当负例 | `REVIEW`抽样 |
| 三次均空、任一次失败、候选不足、图片/文本不可判 | 任意 | 失败续跑或内容/召回修复；不自动给“无标签” | `BLOCKED` |
| 任何票数 | 已确认旧标签/新标签名实冲突、老师明确纠错 | 按经审核的规则处理，保留原始输出与人工证据 | `REVIEW`/人工裁决 |

状态迁移：`AUTO_CANDIDATE` 只有在所属风险层的教师校准与最终盲测均达到**事先批准**的门槛后，才能变成 `accepted`；未批准前仍为待审。`2/3` 的异构 Judge 判“直接匹配/合理共标”时，也只可进入经单独校准的“2/3+Judge同意”候选层；Judge 判“不应打标”时暂不选，但需抽检 Judge 误杀；Judge 无法判断或与既有高可信证据冲突时交老师。老师确认结论可直接改为 `accepted` 或排除，但必须留痕。

同一道题多个 Label 可以分别处于不同状态。整题用于训练须同时满足：已选 Label 均通过正标规则、关键冲突已解决、当前设问文本足够、无阻断性图片缺失、父题所需子题结果齐全。`selected=[]` 只表示模型未能给出可接受标签；不能自动生成全 458 Label 的负例。

## 7. 教师抽样与放行校准

审核单位为「题目–Label」；另抽整题检查是否漏掉其它合理 Label。老师在网页看题干与解析图片、可见文本及 Label 释义，**先不展示**旧标签、票数和 Judge 决定，避免锚定。记录 `直接匹配 / 合理共标 / 错标 / 漏标 / 题目不足 / 释义需改 / 无法判断` 与理由。老师分歧由第二位老师仲裁；仲裁前不算金标。

抽样采用互斥主层并加风险标签。主层至少为 `3/3旧标一致`、`3/3旧标冲突且旧标低覆盖`、`3/3旧标冲突且旧标高覆盖`、`2/3`、`1/3`、`0/3旧标高覆盖`、`空标/召回失败`；每层再覆盖独立题/小题/父题材料、135问题 Label、长尾、相邻边界风险、图片状态、候选仅由旧 ID 补入等。先做约 2,000–5,000 条分层校准，最终规模根据实测错误率和置信区间决定；这些不是全量比例的无权重估计。另从拟自动放行总体随机抽样，防止只审难题而看不到一致错标。

放行门槛必须在看教师结果**之前**定稿。建议初始目标：拟自动放行层的单 Label 错标率上界不超过 1%（95% 单侧区间），同时在各高风险子层没有明显失控；漏标率另报，不用高精度掩盖低召回。若某层样本 0 错，约 300 个独立样本只能将 95% 上界压到约 1%（经验“三倍法则”），不足样本不得宣称“错误率 <1%”。同一题/近重复题的多条 Label 判定并非独立样本，置信区间应按题或内容簇重采样。某层不达标则继续 Judge/老师审核，不调低门槛凑通过率。

教师样本分成**阈值开发集**与最终盲测集，按内容指纹分组切分，防止同题不同 ID 泄漏。每个 Label 仍应报告覆盖量和错误数；长尾 Label 不用全局平均遮蔽。已复核的错误案例进入错误库，用于改释义、候选或 prompt 后的回归测试，但不能再充当未见盲测集。

## 8. 全量报告与必要消融

全量汇总至少报告：题目数、判标单元数、独立/小题/父题材料数、有效旧标数及过期旧 ID 数、候选新增旧 Label 数、Qwen 三票完整率、整题集合一致率、逐 Label `3/3/2/3/1/3` 分布、三次均空率、Judge 分布、教师确认错标/漏标率及区间、自动放行率、阻断率、每 Label 指标、父题最终并集规模与来源。每项指标分题型、图片状态、长尾/135问题集、候选来源和 Label 风险分层。不要把“与旧标一致率”命名为 accuracy。

验证投票价值的最小消融：同一教师盲测集比较 `单次Qwen`、`2次一致才放行`、`3/3放行`、`2/3多数+异构Judge`、以及`无风险分层/有风险分层`，主要看**错标率上界、漏标率、可自动放行比例和每题成本**。若三票对错标无实质改善，只降低吞吐，就不应把三票当作长期默认。另独立分析「新增旧候选被选中」与「旧候选已在Top25」两类，不将候选扰动误归为模型波动。

## 9. 数据产物与审计字段

建议目录独立于任何 DS 历史运行：

```text
runtime/<full-qwen-run>/
  manifest.json
  input/units.jsonl
  input/candidates-top25-plus-legacy.jsonl
  votes/run1/evidence.jsonl, predictions.jsonl, report.json
  votes/run2/evidence.jsonl, predictions.jsonl, report.json
  votes/run3/evidence.jsonl, predictions.jsonl, report.json
  analysis/per_question.jsonl
  analysis/per_question_label.jsonl
  analysis/per_label.jsonl
  judge/tasks.jsonl, evidence.jsonl, results.jsonl, report.json
  teacher/sample.jsonl, decisions.jsonl, adjudications.jsonl
  release/accepted.jsonl, review.jsonl, blocked.jsonl, parent_aggregated.jsonl
  release/report.json
```

逐题–Label 行最少保留：`question_id`、`parent_id`、`unit_type`、`dedupe_hash`、`label_id`、三次布尔票与响应/错误引用、`vote_count`、旧标成员关系、候选来源/排名、Label 风险卡版本、图片/文本状态、Judge 结果与证据、教师结论、最终状态、状态理由码。父题汇总额外保留每个 Label 的来源小题 ID 或“父题材料”。所有汇总可由原始响应重算；任何人工覆盖都保存修改前后值与审核人/时间。

## 10. 执行阶段与停止条件

| 阶段 | 工作 | 进入下一阶段的条件 |
|---|---|---|
| 0. 固定输入 | 直接使用 s85 去重文件、统一粗排、Top25+旧ID、父子题/图片审计 | 输入 SHA、行数和候选映射一致；粗排无缺行 |
| 1. 小规模通路 | 300–1,000 题独立三票，失败续跑、父题并集、逐Label投票 | 三次输入逐题同哈希；输出解析/ID映射正确 |
| 2. 校准 | 万题运行，异构Judge小样本评估，教师分层盲审 | 决策规则和放行门槛冻结，主要高风险层有足够证据 |
| 3. 全量 | Qwen三票、逐Label投票、Judge分流、老师队列 | 三票成功率、漏标/错标抽检达到预设门槛；异常层不放行 |
| 4. 发布 | 输出 accepted/review/blocked、父题并集与报告 | 盲测集通过，版本/证据可追溯；未解决题不进入高置信训练集 |

**当前未定项**：异构 Judge 的具体模型、教师审核预算、可接受的错标率/漏标率目标、自动放行的业务范围（训练集还是线上标签）。这些需在阶段 2 结束前由项目方明确；本文的 1% 是建议的初始精度目标，不是已批准标准。

## 11. 混合精排的已实现接口与烟测顺序

以下命令是**服务就绪后的烟测模板**，不是已经执行的全量 Qwen 作业。确认 `RUN` 指向本次去重粗排目录，`MODEL` 必须采用服务 `/v1/models` 返回的准确名称。不要把全量统一输入直接交给 `run_candidate_adjudication.py`：该执行器按分片读入内存。

```bash
cd /local_data/zhangyonglin/Bio-Know-Tag
RUN="$(cat runtime/LATEST_DEDUP_S85_FULL_COARSE_RUN)"
FINE="$RUN/fine-prep"
mkdir -p "$FINE/legacy"

PYTHONPATH=src python scripts/augment_candidates_with_legacy.py \
  --units "$RUN/unified/retrieval_units.jsonl" \
  --candidates "$RUN/hybrid/candidates.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$FINE/legacy"

PYTHONPATH=src python scripts/shard_adjudication_inputs.py \
  --units "$RUN/unified/retrieval_units.jsonl" \
  --candidates "$FINE/legacy/candidates.jsonl" \
  --run-dir "$FINE/sharded" --shard-size 10000

python -m json.tool "$FINE/sharded/report.json" | head -n 40
```

切分器逐行核对题号，不允许题目和候选错位。先从分片中抽含父题材料、小题、独立题的少量样本，在**新的烟测目录**跑混合精排；不要把 `--limit` 用在全量输入上。烟测须核对三种单元的 prompt 内容、`prompt_version`、空父题额外标签、子题当前设问优先，以及同目录续跑不重发成功题。

分片内的命令形态如下；这是**单次** Qwen 投票，另外两次需用 `run2`、`run3` 的独立目录重复，同一分片的题目与候选文件不变：

```bash
SHARD="$FINE/sharded/shards/00001"
MODEL='请替换为 /v1/models 返回的模型名'
ENDPOINT='http://127.0.0.1:9304/v1/chat/completions'
PYTHONPATH=src python scripts/run_candidate_adjudication.py \
  --units "$SHARD/units.jsonl" \
  --candidates "$SHARD/candidates.jsonl" \
  --labels configs/labels.jsonl \
  --run-dir "$SHARD/votes/run1" \
  --endpoint "$ENDPOINT" --model "$MODEL" \
  --workers 30 --timeout 600 --retries 3 --max-tokens 1024 \
  --no-audited-exclusions
```

投票阶段不用已有的人工排除规则改写模型原始选择；已审定排除项在三票汇总后的决策层单独应用并留痕。这样分析时能区分“Qwen 原始选择”与“人工硬过滤”。

单票所有分片都完成、每片 `report.json` 均为 `success=input` 且 `error=0` 后，才允许合并。合并器逐行复核题号和顺序，缺一条即失败，不产生表面完整的结果：

```bash
PYTHONPATH=src python scripts/merge_sharded_predictions.py \
  --shard-root "$FINE/sharded" --vote-name run1 \
  --output "$FINE/run1.predictions.jsonl"

PYTHONPATH=src python scripts/aggregate_unified_parent_predictions.py \
  --parent-aggregation "$RUN/unified/parent_aggregation.jsonl" \
  --units "$RUN/unified/retrieval_units.jsonl" \
  --predictions "$FINE/run1.predictions.jsonl" \
  --output "$FINE/run1.parent_aggregated.jsonl"
```

父题汇总在**同一份精排结果**里读取小题和父题材料，不需要独立父题精排作业。被文本过滤掉的小题或未完成预测会出现在父题的 `missing_child_question_ids`，该父题 `needs_review=true`、`usable_for_training=false`。这只是每一票的暂存父题并集；最终放行仍须按第 3–7 节完成三票及复核。
