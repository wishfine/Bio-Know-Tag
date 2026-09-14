# 高中生物题库知识点打标策略

## 1. 文档结论

本项目采用“无旧标签依赖”的打标方案。原始 `knw_ids` 只作为源数据溯源信息保留，不参与候选召回、路由、Prompt、结果判断、Pilot 评价或后续训练。

生产主链路固定为：

```text
不可变清洗基线
→ 题目/小题检索单元
→ 稀疏召回 + 稠密召回 + taxonomy软路由
→ RRF融合得到Top 20候选
→ DS根据老师原释义联合裁决
→ 困难题扩大召回或二次验证
→ 父题聚合
→ 独立结果sidecar
```

不采用以下方案作为生产主方案：

- 每道题依次与458个Label做二分类，调用量不可接受。
- 每道题一次发送458个Label的完整释义，输入过长且无关信息会稀释边界判断。
- 直接继承旧 `knw_ids`，因为数据中存在旧体系ID、父题级标签并集和重复题标签冲突。

当前实现边界：题目清洗、打标单元、父题聚合计划、精确去重、无旧标签Pilot抽样、字符n-gram BM25，以及“全458名称+`@`路径”的DS粗召回基线已经实现。RRF核心融合函数已实现，但Dense embedding、taxonomy软路由、Top-20 DS精判和最终父题结果聚合尚未运行或实现。本文后续章节同时定义这些待实现阶段的数据契约和验收标准。

## 2. 数据规模与对象定义

不可变清洗基线为 `questions.jsonl`，共1,253,129条顶层记录、约2.8 GB。后续不覆盖该文件。

实际对象为：

| 对象 | 数量 | 处理方式 |
|---|---:|---|
| 独立题 | 1,033,000 | 根据自身题干、选项、答案和解析打标 |
| 正常组合题小题 | 823,136 | 结合父题公共材料，为每个小题分别打标 |
| 缺父题小题 | 1,455 | 视为独立打标单元，只根据自身内容打标 |
| 真实组合题父题 | 219,741 | 小题Label并集加父题额外Label |
| 虚拟父题容器 | 388 | 仅用于预处理结构，不打标、不进入最终结果 |

第一层题目打标单元共1,857,591个。最终输出包括1,857,591个题目/小题结果和219,741个真实父题结果，合计2,077,332条，与原始真实题目记录数一致。

## 3. 不可变基线与派生文件

现有文件继续保留：

```text
questions.jsonl              2.8 GB清洗聚合基线
label_units.jsonl            1,857,591个独立题/小题单元
parent_aggregation.jsonl     219,741个真实父题聚合计划
duplicate_groups.jsonl       35,484个精确重复组
```

当前 `label_units.jsonl` 中仍有 `legacy_*` 和旧 `proposed_route` 字段，这是早期dry-run留下的审计信息。后续打标程序禁止读取这些字段；Pilot导出时会主动移除它们，因此无需重写4.2 GB派生文件。

## 4. 题目检索单元格式

### 4.1 独立题

```json
{
  "question_id": "...",
  "parent_id": "...",
  "unit_type": "standalone",
  "parent_stem": "",
  "sibling_question_ids": [],
  "stem": "题干",
  "options": "A. ...\nB. ...",
  "answer": "原始答案或嵌套结构",
  "answer_text": "递归清洗后的可读答案",
  "analysis": "解析",
  "dedupe_hash": "sha256",
  "flags": {
    "parent_context_missing": false,
    "image_context_missing": false,
    "duplicate_of": null
  },
  "metadata": {
    "business_type": "...",
    "structure_type": "...",
    "difficulty": "..."
  }
}
```

### 4.2 正常组合题小题

```json
{
  "question_id": "小题ID",
  "parent_id": "真实父题ID",
  "unit_type": "sub_question",
  "parent_stem": "父题公共材料",
  "sibling_question_ids": ["同组其他小题ID"],
  "stem": "当前小题题干",
  "options": "当前小题选项",
  "answer": "原始答案",
  "answer_text": "规范化答案",
  "analysis": "当前小题解析",
  "dedupe_hash": "sha256",
  "flags": {
    "parent_context_missing": false,
    "image_context_missing": false,
    "duplicate_of": null
  },
  "metadata": {}
}
```

`parent_stem`只提供共同语境。模型必须依据当前小题的设问、答案和解析确定Label，不能把父题材料中出现的所有概念机械复制给每个小题。

### 4.3 缺父题小题

```json
{
  "question_id": "小题ID",
  "parent_id": "缺失父题ID",
  "unit_type": "orphan_sub_question",
  "parent_stem": "",
  "stem": "当前小题题干",
  "answer_text": "规范化答案",
  "analysis": "当前小题解析",
  "flags": {
    "parent_context_missing": true,
    "image_context_missing": false,
    "duplicate_of": null
  }
}
```

这1,455个小题只打自身知识点，不构造虚拟父题Label并集。

## 5. 真实父题聚合格式

```json
{
  "question_id": "父题ID",
  "unit_type": "composite_parent",
  "parent_stem": "公共材料",
  "child_question_ids": ["小题1", "小题2"],
  "flags": {
    "synthetic_parent": false
  }
}
```

父题最终知识点定义为：

```text
parent.knowledge_labels
= union(child.knowledge_labels)
∪ parent_extra_labels
```

`parent_extra_labels`只允许包含父题公共材料本身需要表达、且没有被任何小题Label覆盖的知识点。材料中仅作为背景出现的概念不补标。

## 6. Label Card格式

召回索引和DS裁决统一使用老师图谱字段：

```json
{
  "label_id": "...",
  "label_name": "...",
  "label_path": "知识点->一级模块->二级模块->...",
  "definition": "老师定义",
  "core_concepts": "老师核心概念",
  "common_assessments": "老师常见考查",
  "distinctions": "老师易混淆边界"
}
```

DS在Stage1生成的释义不进入生产Label Card。老师释义是当前项目的权威定义；`蛋白质病毒的增殖`在老师确认前作为单独的taxonomy风险项处理。

## 7. 检索查询构造

独立题检索文本包括题干、选项、答案和解析。组合题小题强调当前小题，并降低父题材料权重：

```text
当前小题题干：高权重
答案与解析：高权重
选项：中等权重
父题公共材料：低权重
```

推荐初始相对权重为：

```text
stem        2.0
answer      1.5
analysis    1.5
options     1.0
parent_stem 0.5
```

权重必须由Pilot评估确定，不能直接作为长期固定真值。图片上下文缺失时优先利用答案和解析；若图片、答案和解析同时不足，则进入困难路由。

## 8. 候选召回

### 8.1 稀疏召回

使用中文字符bigram/trigram BM25，对Label Card召回Top 15。它负责教材术语、实验操作名、专有名词和公式线索的精确匹配。

### 8.2 稠密召回

使用中文语义embedding，将458个Label Card离线编码。Label数量很少，不需要向量数据库；题目批量编码后直接与458个向量计算相似度，召回Top 15。

### 8.3 Taxonomy软路由

图谱包含6个一级模块和34个二级模块。先预测Top 1～3二级模块，并增加这些模块内的候选。模块结果只提供加分，不得硬过滤全局召回结果，以免漏掉跨模块综合题。

### 8.4 融合与候选上限

使用RRF融合稀疏、稠密和模块内排名：

```text
global_sparse_top15
∪ global_dense_top15
∪ taxonomy_module_candidates
→ RRF
→ final_top20
```

普通题最多20个候选，困难题扩展到30～50个。候选阶段首先保证召回率，再由DS控制精度。

## 9. 候选结果sidecar

```json
{
  "question_id": "...",
  "retrieval_query_version": "v1",
  "candidates": [
    {
      "label_id": "...",
      "sparse_rank": 2,
      "dense_rank": 1,
      "module_rank": 1,
      "rrf_score": 0.047,
      "candidate_rank": 1
    }
  ],
  "retrieval_flags": {
    "sparse_dense_top1_agree": true,
    "low_score": false,
    "cross_module": false
  }
}
```

该文件与题目基线分离，便于替换检索模型、调整Top-K并重新评估，而不重写题目数据。

候选输出的轻微格式错误按可审计原则归一化：同一道题内的重复候选ID保持首次出现顺序去重；未知Label ID、缺少题目、题目乱序或唯一候选数超过Top-K仍判为失败。原始响应和归一化计数必须写入evidence。

## 10. DS裁决策略

### 10.1 独立题

输入题目内容、Top 20候选及对应老师原释义。DS输出完成当前设问所需的最小充分Label集合。

### 10.2 组合题

同一真实父题下的所有小题原则上放在一次请求中，父题材料只发送一次。每个小题携带自己的候选集合，DS必须分别输出每个小题的Label，并额外输出 `parent_extra_labels`。超长题组按小题拆批，但父题材料保持一致。

### 10.3 核心判标规则

- 只有完成当前设问需要调用的知识点才打。
- 题干背景、实验工具、错误选项和干扰项不打。
- 输出最小充分知识点集合，不输出所有相关知识点。
- 允许所有候选均不适用，并设置 `need_expand_recall=true`。
- 综合Label只有在题目确实要求多个子模块联动时才添加。
- 每个命中Label必须提供来自题干、答案或解析的证据。

建议输出：

```json
{
  "question_id": "...",
  "selected_labels": [
    {
      "label_id": "...",
      "label_name": "...",
      "confidence": 0.95,
      "evidence": "解析中直接要求……"
    }
  ],
  "rejected_close_labels": [],
  "none_of_candidates": false,
  "need_expand_recall": false
}
```

## 11. 正式路由

旧R0/R1/R2仅保留为历史审计结果，不进入生产执行。新路由为：

### N0：检索高度一致

稀疏Top-1与稠密Top-1一致，双方分数和排名间隔均超过Pilot确定的阈值，且不是综合/taxonomy风险Label。Pilot证明可靠前，N0仍需DS确认；验证后才允许快速通过。

### N1：标准联合裁决

Top 20候选进入一次DS联合裁决，是默认主路由。

### N2：困难题升级

发生以下任一情况时扩大到相关Top 2～3模块或30～50个候选，再调用一次DS：

- 稀疏和稠密结果严重不一致；
- 首轮候选整体分数低；
- DS拒绝全部候选；
- 图片缺失且答案、解析不足；
- 综合题跨越多个模块；
- DS输出不满足JSON Schema或证据不足。

二次判断仍不确定时进入人工复核，不让模型强猜。

## 12. 精确去重

去重哈希由以下规范化内容计算：

```text
parent_stem + stem + options + answer_text + analysis
```

全量共有35,484个精确重复组、78,626个成员，理论可减少43,142个打标单元。由于旧 `knw_ids` 完全不参与打标，旧标签是否冲突不影响结果传播：同一哈希只判一次，新Label结果回填该组所有题目ID。

## 13. 最终结果sidecar

题目/小题结果：

```json
{
  "question_id": "...",
  "parent_id": "...",
  "unit_type": "standalone|sub_question|orphan_sub_question",
  "knowledge_labels": [
    {
      "label_id": "...",
      "label_name": "...",
      "confidence": 0.97,
      "evidence": "……"
    }
  ],
  "label_source": "hybrid_retrieval_ds|retrieval_consensus|human",
  "retrieval_meta": {
    "route": "N0|N1|N2",
    "candidate_count": 20,
    "retrieval_version": "v1",
    "prompt_version": "v1"
  },
  "flags": {
    "parent_context_missing": false,
    "image_context_missing": false,
    "duplicate_of": null,
    "needs_review": false
  }
}
```

真实父题结果额外保留来源：

```json
{
  "question_id": "父题ID",
  "unit_type": "composite_parent",
  "child_union_labels": [
    {
      "label_id": "...",
      "source_question_ids": ["小题1", "小题3"]
    }
  ],
  "parent_extra_labels": [],
  "knowledge_labels": []
}
```

## 14. Pilot抽样

第一版Pilot目标约2,500个题目/小题单元，不使用旧Label做分层。默认覆盖：

- 约55%独立题；
- 约40%正常组合题小题，并保留约200个完整父题组；
- 约5%缺父题小题；
- 100个精确重复组，每组取两个成员检查结果传播；
- 空题干全部纳入；
- 图片风险、空答案、空解析分别过采样；
- 按 `business_type`、`structure_type`、`difficulty` 和题目类型做确定性分层。

抽样使用固定seed和稳定哈希，相同输入必定得到相同结果。Pilot输出主动删除全部 `legacy_*`、旧路由和旧标签冲突字段。

抽样后再依次运行：

1. 全458名称与路径的DS粗召回基线，仅用于小规模Pilot。
2. 字符n-gram BM25。
3. Dense embedding。
4. BM25 + Dense。
5. BM25 + Dense + taxonomy软路由。

由人工确认的Pilot计算：

```text
Recall@5 / Recall@10 / Recall@20
平均候选数
零召回率
DS多打率 / 漏打率
综合Label误加率
重复运行一致率
每题或每题组耗时、输入token和输出token
```

候选召回的初始目标为 `Recall@20 >= 98%`，最终阈值以人工金标结果决定。约2,500条Pilot用于覆盖数据形态和发现问题；如果要对每个Label给出可靠的单独准确率，还需要后续追加按Label分层的人工金标。

## 15. 验收与版本化

所有实验输出进入 `runtime/<timestamp>/`，至少保存：

```text
pilot_units.jsonl
pilot_parents.jsonl
pilot_report.json
candidates.jsonl
predictions.jsonl
report.json
nohup.log
pid
```

每条候选和预测必须记录检索版本、Prompt版本、模型、endpoint、时间及失败重试次数。全量运行前必须满足：

- Pilot输入输出数量一致；
- 构建与请求错误为0；
- 候选Recall@20达到约定阈值；
- DS输出均通过JSON Schema校验；
- 父题Label等于小题并集加父题额外Label；
- 精确重复结果传播一致；
- 人工抽查确认多打和漏打在可接受范围内。
