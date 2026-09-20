# Luna 逐 Label 逐题盲审 Prompt v2

下面内容可直接作为新 Goal 的完整提示词。

```text
你要对高中生物知识点打标结果执行逐 Label、逐题盲审。必须审核输入范围内的每一个 Label × 题目组合，不得跳过，不得只根据统计结果推断，也不得等全部结束后才统一写结果。

## 一、目标

判断当前题目打上当前目标 Label 是否合理。

本项目允许合理多标：

- Label 是题目最直接、最核心的考点时，判 DIRECT_MATCH；
- Label 不是最核心考点，但得出正确答案时确实需要实际调用该知识完成一个真实判断、计算、推理或选项辨析时，判 REASONABLE_CO_LABEL；
- 不得因为另一个 Label 更核心、更具体，就把本来合理的共标判成错标；
- 只有目标 Label 只是背景、材料描述、术语共现、同章节知识、可完全绕开的深层机制，或对象/任务明显不一致时，才判 WRONG_LABEL。

`生物学常识`、综合类、应用类等范围较宽的 Label 不因“范围宽”自动否决。只要当前题目真实落在其现有 definition、core_concepts、common_assessments 和 distinctions 所确定的范围内，就可以判 DIRECT_MATCH 或 REASONABLE_CO_LABEL；与更具体 Label 重叠也不是否决理由。

## 二、输入与盲审要求

输入根目录：

runtime/ds-partial-review-20260920-164855-extracted/

Label 定义来自：

configs/labels.jsonl

逐 Label 待审题目来自：

runtime/20260920-164702-ds-paired-partial-review/risk-review/labels/<label_id>.json

对每条题目，只允许使用：

1. 当前目标 Label 的：
   - label_name
   - label_path
   - definition
   - core_concepts
   - common_assessments
   - distinctions
2. 当前题目的：
   - parent_stem（仅在当前题有明确指代时使用）
   - stem
   - options
   - answer_text
   - analysis
   - 图片缺失或上下文缺失状态

审核时不得参考：

1. 原 knw_ids；
2. selection_status；
3. review_bucket；
4. top25_selection、legacy_selection；
5. top25_selected_label_ids、legacy_selected_label_ids；
6. DS 原判定、reason、evidence；
7. 历史正样本、负样本或风险分档；
8. 当前 Label 是由 Top25 还是旧 knw_ids 召回。

每审核新题前，只保留“当前 Label 定义 + 当前题目”的上下文。不得沿用上一道题的题干、对象、证据或理由。

## 三、判定标准

### 1. DIRECT_MATCH

满足以下任一情况：

- 当前设问明确询问该 Label 的概念、原理、规律、过程、结构、功能、方法、实验或应用；
- 正确答案中的核心判断直接依赖该 Label；
- 选择题的一个或多个实质性选项直接考查该 Label，判断其正误必须调用该知识；
- 非选择题的一个空或一个步骤直接要求该 Label；
- 题目明确要求使用该规律进行计算、概率推断、实验判断或结果解释。

一个题目可以同时直接考查多个 Label。不得因为还有其他知识点，就降低当前 Label 的匹配等级。

### 2. REASONABLE_CO_LABEL

满足以下条件：

- 当前 Label 不是题目的主标题或唯一核心；
- 但得出正确答案时，确实需要使用该 Label 完成一个真实且不可忽略的判断、计算、推理、实验辨析或选项排除；
- 它不是只在解析中顺带补充，也不是仅共享名词或底层机制。

特别注意：

- 伴性遗传、基因定位、染色体变异等题中，如果实际需要分析等位基因形成配子时的分离、后代比例或遗传概率，则“基因分离定律”至少可以是 REASONABLE_CO_LABEL；不能只因题目主轴是伴性遗传或基因定位就直接判错。
- 一个宽 Label 与具体原子 Label 同时成立时，可以合理共标。

### 3. WRONG_LABEL

只有满足以下任一条件时才使用：

- OBJECT_MISMATCH：物种、疾病、性状、材料、组织、器官、细胞类型、实验对象等存在明确限制且不一致；
- TASK_MISMATCH：题目考查的任务维度不同，例如实验操作与实验结论、基因定位与遗传规律、固定化酶与筛选微生物；
- LEVEL_MISMATCH：生命层级、结构层级或作用通道不一致；
- SHARED_TERM_ONLY：只共享名词或关键词；
- SHARED_MECHANISM_ONLY：只共享底层机制，但当前题不需要调用该 Label；
- BACKGROUND_ONLY：只作为材料背景或解析补充；
- DISTINCTION_CONFLICT：题目明确落入 distinctions 排除范围；
- OTHER：其他明确不匹配情况，必须说明。

不得仅因为存在一个更具体、更核心或更常用的 Label，就判 WRONG_LABEL。

### 4. INSUFFICIENT_CONTEXT

只有当缺图、缺父题、OCR 冲突或信息缺失导致无法可靠判断当前 Label 是否成立时使用。

- 如果题干较短，但答案和解析足以唯一恢复考点，不判上下文不足；
- 如果只是不确定 DIRECT_MATCH 还是 REASONABLE_CO_LABEL，应根据 Label 在解题中的作用选择其一，不使用 INSUFFICIENT_CONTEXT。

## 四、多选题、错误选项与多空题

- 正确选项不是唯一证据来源；如果判断一个错误选项为何错误必须调用当前 Label，且该选项构成实质性考查，也可以支持匹配。
- 单个孤立干扰项、无关背景句、解析中的扩展科普不能单独支持匹配。
- 多空题中，只要至少一个空直接考查当前 Label，即可判 DIRECT_MATCH。
- 一个 Label 只支撑题目的一部分，但该部分是实质性考查时，不得因题目还考查其他知识而判错。

## 五、证据要求：防止串题

每条审核必须输出当前题目的原文证据。

1. evidence_quote 必须从当前记录的 parent_stem、stem、options、answer_text 或 analysis 中逐字复制，不得改写、概括或补写；
2. evidence_quote 不超过 100 个汉字；必要时可以输出 2 条短引文；
3. evidence_source 必须准确标明来源；
4. evidence 不得包含当前题目没有出现的对象、术语、实验或结论；
5. 写入前重新读取当前 question_id，并检查 evidence_quote 是否能在该题对应字段中找到；找不到则不得落盘，必须重新生成；
6. 禁止使用上一道题的 evidence 或 reason。

允许的 evidence_source：

parent_stem | stem | options | answer_text | analysis

## 六、逐题输出格式

每审核一条，立即向 item_reviews.jsonl 追加一行，不得等待整个 Label 或全部任务完成后再批量写入。

每一行必须是合法 JSON：

{
  "label_id": "...",
  "question_id": "...",
  "decision": "DIRECT_MATCH | REASONABLE_CO_LABEL | WRONG_LABEL | INSUFFICIENT_CONTEXT",
  "error_type": null,
  "evidence": [
    {
      "source": "stem",
      "quote": "当前题目中的逐字原文"
    }
  ],
  "reason": "说明该 Label 在当前解题任务中的具体作用，不超过120字",
  "confidence": "HIGH | MEDIUM | LOW",
  "reviewed_at": "ISO-8601时间"
}

字段规则：

- decision=WRONG_LABEL 时，error_type 必须填写上述错标类型；
- decision 不是 WRONG_LABEL 时，error_type 必须为 null；
- decision=INSUFFICIENT_CONTEXT 时，evidence 可以为空数组，但 reason 必须说明缺失内容；
- 其他 decision 至少包含一条可以在当前题目中验证的逐字 evidence；
- 不输出 Markdown，不在 JSONL 中添加注释。

## 七、每完成一个 Label 后

生成：

labels/<label_id>.md

内容包括：

1. Label 名称、ID、路径和四个释义字段；
2. 本次审核样本数；
3. DIRECT_MATCH、REASONABLE_CO_LABEL、WRONG_LABEL、INSUFFICIENT_CONTEXT 数量；
4. 错标类型分布；
5. 每道题的题号、题干、答案、decision、原文 evidence、reason；
6. 标明这是抽样盲审，不代表该 Label 的全题库准确率。

Label 级建议允许：

FREEZE | KEEP_WITH_CAUTION | PROMPT_FIX | TEACHER_REVIEW | INSUFFICIENT_EVIDENCE

注意：

- Label 范围宽、属于综合类、常识类或与其他 Label 重叠，不能单独作为否决理由；
- 只要题目符合现有释义，宽 Label 可以保留；
- PROMPT_FIX 必须有至少两个语义同类的高置信错标作为证据，不能因单个边界题自动给出；
- 样本过少、图片缺失较多或错标口径不确定时，使用 INSUFFICIENT_EVIDENCE 或 TEACHER_REVIEW。

## 八、断点续跑与一致性检查

持续更新 progress.json，至少记录：

- status
- labels_total
- labels_completed
- records_total
- records_reviewed
- completed_labels
- last_key.label_id
- last_key.question_id

续跑时：

1. 以 label_id + question_id 为唯一键；
2. 已存在的合法记录不得重复写入；
3. 若发现同一键有两个不同判断，写入 conflicts.jsonl 并暂停该 Label；
4. 每完成一个 Label，校验 item_reviews.jsonl 中该 Label 的记录数与输入题数一致；
5. 校验所有非空 evidence_quote 均能在当前题目的指定 evidence_source 中找到；
6. 校验全部记录后再生成该 Label 的 Markdown。

## 九、最终汇总

全部完成后生成 final_summary.md，至少包括：

- 总 Label 数和完成数；
- 总任务数、成功数、冲突数；
- 四类 decision 总量和比例；
- 各 Label 的四类数量与比例；
- 各 error_type 的数量；
- 各 Label 的建议；
- 需要老师复核的 Label 和代表题号；
- 审核限制，包括图片未直接读取、抽样代表性和宽 Label 可与具体 Label 共存的口径。

开始全量前，先按本 Prompt 再审核 2–3 个边界容易混淆的 Label、合计约 50 条，并停止等待人工校准。人工确认后才继续全量。
```

