# Round5 补充工作总结与 Round6 详细优化计划

Date: 2026-05-22

本文是 `docs/ROUND5_FINAL_DECISION_2026-05-22.md` 的补充说明，并给出下一轮
可直接执行的优化方案。核心结论先写在前面：

```text
当前最终模型仍是 Step7 ensemble。
Round5 通过了非 teacher-test 安全门槛，但 teacher-test 上没有触发任何 override。
Round5 是安全失败，不是破坏性失败。
Round6 不应继续简单放宽规则，而应专门学习 safe override 与 induced FP 的差异。
```

严格边界继续不变：

```text
data/raw/teacher_test.json 只能作为最终诊断。
不能用于训练、阈值选择、guard 校准、router 调参、stacker 训练或模型选择。
Round6 可以阅读 Round5 最终诊断中的聚合现象来形成假设，但不能把 teacher-test
文本、标签、样本 id 或逐行阈值条件放进训练/验证/选择流程。
```

## 0. 前五轮过程与结果总览

这一节补齐前几轮到 Round5 的连续路线，避免下一轮把已经验证失败的方向重新做一遍。

### 0.1 当前严格基线与目标差距

当前严格路线最终模型仍是 Step7 ensemble：

```text
DeBERTa:  outputs/models/deberta_lit_academic_poetry_step7_combined
TF-IDF:   outputs/models/tfidf_lit_academic_poetry
alpha:    0.5
threshold: 0.55
```

teacher-test 基线：

| System | Correct / 300 | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step7 ensemble | 274 | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 13 | 13 |

95% 的数学要求：

```text
target correct = 285 / 300
current correct = 274 / 300
needed net gain = +11
current errors = 26 = 13 FP + 13 FN
```

因此，任何下一轮候选都不能只用更多 LLM recall 换更多 human FP。要达到 95%，必须在
保护 high-style human precision 的同时净修复至少 11 个当前错误。

### 0.2 Round1: 全流程与 Step7 提升

Round1 建成了完整 pipeline：

```text
human seeds -> LLM rewrites -> pair-safe split -> TF-IDF -> DeBERTa ->
DeBERTa + TF-IDF ensemble -> teacher-test final diagnostic
```

最早 final ensemble：

| System | Correct / 300 | Accuracy | F1 | FP | FN | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| original final ensemble | 271 | 0.9033 | 0.9073 | 21 | 8 | [[129, 21], [8, 142]] |

之后尝试 hard-negative human、normalization、calibration、controlled hard-negative
quota、ChatGPT hard positives、poetry expansion，并在 Step7 neural retraining 取得当前最强结果：

| System | Correct / 300 | Accuracy | F1 | FP | FN | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Step7 ensemble | 274 | 0.9133 | 0.9133 | 13 | 13 | [[137, 13], [13, 137]] |

Round1 结论：

```text
有效提升来自 targeted train-side data + neural retraining。
单纯阈值、alpha 和 TF-IDF-only 调整只能做诊断，不是继续冲 95% 的主路线。
```

### 0.3 Round2: teacher-like dev、router、stacker、RoBERTa

Round2 从 95% 目标出发，先做 Phase 0 诊断：

| Diagnostic | Result |
| --- | ---: |
| Step7 teacher-test residual errors | 26 = 13 FP + 13 FN |
| best existing-family oracle threshold diagnostic | about 0.9267 |
| best simple-average oracle diagnostic | about 0.9333 |

这些 oracle 只是诊断，不是合法调参结果；但它们说明现有模型族距离 95% 仍然很远。

Round2 构建 teacher-like dev：

| Item | Result |
| --- | ---: |
| round2 dev rows | 1065 |
| train additions | 3027 |
| Step7 on round2 dev F1 | 0.6220 |
| Step7 round2 dev confusion | [[602, 24], [230, 209]] |

Round2 主要候选的 teacher-test 最终诊断：

| Candidate | Correct / 300 | Accuracy | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| step7 | 274 | 0.9133 | 0.9133 | 13 | 13 |
| bucket_routed | 270 | 0.9000 | 0.9020 | 18 | 12 |
| stacker_step7 | 273 | 0.9100 | 0.9109 | 15 | 12 |
| roberta_single | 248 | 0.8267 | 0.8219 | 22 | 30 |
| stacker_with_roberta | 266 | 0.8867 | 0.8917 | 24 | 10 |

Round2 结论：

```text
hard-positive recall 信号存在，但 teacher-test 上会诱发 human FP。
router/stacker/RoBERTa 都不能作为全局最终系统。
```

### 0.4 Round3: precision-guarded repair

Round3 不再做 aggressive global replacement，而是尝试只在局部覆盖 Step7：

```text
default prediction = Step7
only override Step7-human -> LLM under strong evidence
```

Round3 组件：

```text
round3_error_delta_audit.py
round3 precision-guard data
ELECTRA branch
OOF stacker
precision-guarded rule search
```

关键结果：

| Candidate | Accuracy | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| step7 | 0.9133 | 0.9133 | 13 | 13 |
| round3_electra | 0.8800 | 0.8759 | 13 | 23 |
| round3_oof_stacker | 0.9033 | 0.8997 | 9 | 20 |
| round3_precision_guard | 0.9100 | 0.9103 | 14 | 13 |

Round3 precision guard 在非 teacher guard-dev 上看起来有效：

```text
guard-dev fixed Step7 FN = 18
guard-dev induced FP = 0
```

但 teacher-test 上只触发 1 个 override：

```text
fixed Step7 FN = 0
induced FP = 1
```

Round3 结论：

```text
precision-guarded local override 是正确方向，但当时的 guard-dev 和 branch signal
仍未贴近 teacher-test residual distribution。
```

### 0.5 Round4: paired residual data + weighted DeBERTa + human-style guard

Round4 根据三轮复盘转向 residual repair：

```text
paired residual data -> weighted DeBERTa branch -> human-style guard ->
local override gate
```

Round4 residual dataset 验收：

| Item | Count |
| --- | ---: |
| hard human mirrors | 4727 |
| hard LLM positives | 3674 |
| hard human : hard LLM ratio | 1.2866 |
| old-prose human mirrors | 800 |
| poetry/freeverse human mirrors | 1500 |
| natural academic human mirrors | 1000 |
| hard-positive dev rows | 500 |
| hard-negative dev rows | 500 |
| teacher-test exact duplicates | 0 |
| round4 train rows | 17434 |

Round4 weighted DeBERTa 结果：

| Split | Step7 F1 / status | Round4 DeBERTa result | Interpretation |
| --- | ---: | ---: | --- |
| internal_test | 0.9564 | F1 0.9441, FP 59, FN 37 | global regression |
| hardpos dev | recall 0.4160 | recall 0.5660, F1 0.7229 | useful hard-positive signal |
| hardneg dev | FP 26 | FP 53 | unsafe human FP drift |

Round4 local override gate：

| Item | Result |
| --- | ---: |
| aligned non-teacher rows | 2731 |
| candidate rules searched | 3601 |
| feasible rules | 1 |
| feasible non-empty override rules | 0 |

Round4 结论：

```text
Round4 DeBERTa 可作为 hard-positive signal branch，但不能全局替代 Step7。
缺口不是更多 recall，而是专门识别 Round4-induced FP 的 guard。
```

### 0.6 Round5: FP-safe residual repair

Round5 接 Round4 的失败点，新增 Step7-vs-Round4 flip ledger 和 flip guard：

```text
Step7-human -> Round4-LLM, label LLM   = safe fixed-FN candidate
Step7-human -> Round4-LLM, label human = unsafe induced-FP candidate
```

Round5 non-teacher flip ledger：

| Split | Safe fixed-FN candidates | Unsafe induced-FP candidates | Total override candidates |
| --- | ---: | ---: | ---: |
| internal_test | 16 | 35 | 51 |
| hardpos | 88 | 0 | 88 |
| hardneg | 0 | 32 | 32 |

Round5 flip guard 通过非 teacher gate，并支持一个局部 override rule：

| Split | Step7 F1 | Round5 F1 | Step7 FP | Round5 FP | Step7 FN | Round5 FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| internal_test | 0.9564 | 0.9570 | 27 | 27 | 46 | 45 |
| hardpos | 0.5876 | 0.6928 | 0 | 0 | 292 | 235 |
| hardneg | 0.0000 | 0.0000 | 26 | 26 | 0 | 0 |

Round5 teacher-test 最终诊断：

```text
Step7 baseline = 274 / 300
Round5 override = 274 / 300
Round5 overrides = 0
```

Round5 结论：

```text
Round5 是安全失败：挡住了 14 个 unsafe teacher-test induced FP，但也没有释放
3 个 safe fixed-FN candidates。下一轮应该学习 safe override selector，而不是
简单放宽 p_unsafe_override 阈值。
```

### 0.7 跨轮总结

已经证明或基本耗尽的路线：

| Route | Status |
| --- | --- |
| global threshold / alpha sweep | oracle ceiling也不足 95%，不值得继续主攻 |
| bucket router as final classifier | hard-dev 好看，teacher-test 退化 |
| simple stacker | internal/dev 有收益，teacher-test 不稳 |
| RoBERTa/ELECTRA global branch | 信号不够安全 |
| Round4 DeBERTa global replacement | hardpos 变好但 hard-human FP 翻倍 |
| Round5 direct unsafe-veto rule | safe but no teacher-test gain |

仍然值得继续的路线：

| Route | Why |
| --- | --- |
| Step7 as default | 仍是唯一稳健最终基线 |
| residual/error-ledger diagnostics | 能清楚定位 fixed-FN 与 induced-FP |
| paired hard-positive / hard-negative data | Step7 的真实提升来自 targeted data |
| local Step7-human -> LLM override | 能限制 blast radius |
| candidate-level safe/unsafe selector | 正好针对 Round5 卡住的 3 safe vs 14 unsafe 矛盾 |

## 1. Round5 补充总结

### 1.1 已完成的新增组件

Round5 新增并验证了以下脚本：

```text
src/evaluation/build_round5_flip_ledger.py
src/models/train_round5_flip_guard.py
src/evaluation/predict_round5_flip_guard.py
src/evaluation/tune_round5_residual_override.py
src/evaluation/apply_round5_residual_override.py
src/evaluation/build_round5_inference_ledger.py
```

主要输出：

```text
outputs/evaluation/round5_baseline_frozen_report.md
outputs/evaluation/round5_flip_ledger_summary.md
outputs/evaluation/round5_flip_guard_report.md
outputs/evaluation/round5_residual_override_tuning_report.md
outputs/evaluation/round5_gate_report.md
outputs/evaluation/round5_teacher_test_comparison.md
docs/ROUND5_OPTIMIZATION_WORK_LOG_2026-05-22.md
docs/ROUND5_FINAL_DECISION_2026-05-22.md
```

Round4 DeBERTa 缺失模型也已重训并恢复：

```text
outputs/models/round4_deberta_weighted_residual/best_model
```

### 1.2 Round5 关键非 teacher-test 结果

Round5 的非 teacher-test gate 是成立的。

冻结规则：

```json
{
  "round5_threshold": 0.55,
  "min_delta": 0.0,
  "flip_guard_unsafe_max": 0.35,
  "human_style_veto_max": 0.8,
  "min_words": 0,
  "bucket_group": "old_short",
  "allowed_buckets": [
    "literary_old_prose",
    "literary_short_fragment"
  ],
  "disabled_baseline": false
}
```

非 teacher-test 表现：

| Split | Step7 F1 | Round5 F1 | Step7 FP | Round5 FP | Step7 FN | Round5 FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| internal_test | 0.9564 | 0.9570 | 27 | 27 | 46 | 45 |
| hardpos | 0.5876 | 0.6928 | 0 | 0 | 292 | 235 |
| hardneg | 0.0000 | 0.0000 | 26 | 26 | 0 | 0 |

非 teacher-test override delta：

| Split | Overrides | Fixed Step7 FN | Induced FP |
| --- | ---: | ---: | ---: |
| internal_test | 1 | 1 | 0 |
| hardpos | 57 | 57 | 0 |
| hardneg | 0 | 0 | 0 |

这说明 Round5 的局部门控思想是有效的：它确实能在非 teacher 数据里释放一部分
Round4 DeBERTa hard-positive signal，同时不增加 hard-human FP。

### 1.3 Round5 teacher-test 最终结果

最终 teacher-test 结果没有提升：

| Run | Correct / 300 | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step7 baseline | 274 | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 13 | 13 |
| Round5 override | 274 | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 13 | 13 |

Round4 DeBERTa 分支单独 teacher-test 表现：

| Branch | Correct / 300 | Accuracy | FP | FN | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Step7 | 274 | 0.9133 | 13 | 13 | current final |
| Round4 DeBERTa | 263 | 0.8767 | 23 | 14 | unsafe global branch |
| Round5 override | 274 | 0.9133 | 13 | 13 | safe no-op |

Round4-vs-Step7 teacher-test flip ledger：

| Flip type | Count |
| --- | ---: |
| stable_step7_correct | 256 |
| both_miss | 19 |
| induced_fp | 14 |
| round4_induced_fn | 4 |
| round4_fixed_fp | 4 |
| fixed_fn_candidate | 3 |

Round4 试图把 Step7-human 改成 LLM 的候选共有 17 个：

| Candidate type | Count |
| --- | ---: |
| unsafe induced FP | 14 |
| safe fixed-FN candidate | 3 |
| total | 17 |

Round5 最终没有触发 override 的直接原因：

1. 冻结规则只允许 `literary_old_prose` 和 `literary_short_fragment`。
2. 1 个 safe fixed-FN candidate 属于 `general_prose`，被 bucket 规则挡掉。
3. 2 个 safe `literary_short_fragment` candidate 的 `p_unsafe_override > 0.35`，
   被 flip guard 挡掉。
4. 14 个 unsafe induced-FP candidate 也被成功挡住。

因此 Round5 的失败类型是：

```text
precision-safe but recall-underpowered
```

它没有把 Step7 变差，但也没有修复 teacher-test 的 FN。

## 2. Round5 学到的东西

### 2.1 可复用资产

Round5 产出的以下资产值得继续复用：

| Asset | Use in Round6 |
| --- | --- |
| `round5_flip_ledger.jsonl` | 继续作为 override 安全性分析格式 |
| `round5_flip_guard_train.jsonl` | 初始 flip-guard 训练种子 |
| `round5_flip_guard_dev_hardpos.jsonl` | safe override 的非 teacher 代理 dev |
| `round5_flip_guard_dev_hardneg.jsonl` | unsafe override 的非 teacher 代理 dev |
| `round4_deberta_weighted_residual/best_model` | hard-positive signal branch |
| `tune_round5_residual_override.py` | Round6 rule search 的基础 |
| `apply_round5_residual_override.py` | 最终冻结规则应用入口 |

### 2.2 不应复用为最终模型的东西

以下组件不能直接晋级：

| Component | Reason |
| --- | --- |
| Round4 DeBERTa global classifier | teacher-test 只有 263/300，FP 风险过高 |
| Round5 override rule | teacher-test no-op，没有收益 |
| 当前 flip guard 阈值 | 非 teacher 安全，但过度 veto teacher-test safe candidates |

### 2.3 当前瓶颈

Round6 的真正瓶颈不是“模型不敢判 LLM”，而是：

```text
Round4 DeBERTa 能找到一小部分 Step7 FN，但它找到更多 Step7 正确的人类文本。
Flip guard 能挡住危险覆盖，但也把少量真正可修复 FN 挡掉。
```

所以下一轮要优化的是 override candidate 的二分类：

```text
safe_override: Step7 human -> Round4 LLM, label LLM
unsafe_override: Step7 human -> Round4 LLM, label human
```

## 3. Round6 总目标

建议下一轮命名为：

```text
Round6: Safe Override Distillation
```

目标不是训练一个更 aggressive 的全局模型，而是训练/校准一个更会挑选局部
override 的安全选择器。

### 3.1 非 teacher-test 硬门槛

| Gate | Required |
| --- | --- |
| default system | Step7 remains default |
| override direction | only Step7-human -> LLM |
| internal_test F1 | >= 0.9570 preferred, hard minimum >= 0.9564 |
| internal_test FP | <= 27, hard maximum <= 29 |
| internal induced FP | 0 preferred, hard maximum <= 1 |
| hardneg induced FP | 0 |
| hardpos fixed Step7 FN | >= 70 target, hard minimum >= 57 |
| hardpos safe override veto | <= 5% target, hard maximum <= 8% |
| candidate-level unsafe recall | >= 90% on held-out unsafe dev |
| teacher-test leakage | exact duplicate = 0; no teacher labels/text in tuning |

### 3.2 teacher-test 目标

| Result | Decision |
| --- | --- |
| <= 274 / 300 | reject, keep Step7 |
| 275-284 / 300 | partial success, not 95% |
| >= 285 / 300 | 95% achieved, promote if no leakage/tuning violation |

Round6 应先把目标定为：

```text
primary target: beat Step7, at least 275 / 300
stretch target: 285 / 300
```

## 4. Round6 详细执行计划

### Phase 0: 冻结 Round5 状态并清理评估入口

目的：确保下一轮比较的起点完全可复现。

需要固定：

```text
outputs/predictions/round5_teacher_test_predictions.jsonl
outputs/evaluation/round5_teacher_test_comparison.json
outputs/evaluation/round5_teacher_test_ledger_summary.json
outputs/models/round5_residual_override/rules.json
outputs/models/round5_flip_guard/flip_guard.pkl
outputs/models/round4_deberta_weighted_residual/best_model
```

建议新增：

```text
outputs/evaluation/round6_starting_point_report.md
```

报告必须包含：

1. Step7, Round4 DeBERTa, Round5 override 的 teacher-test 指标。
2. Round5 非 teacher gate 指标。
3. 当前可复用模型目录和预测文件。
4. 明确声明 Round6 不能用 teacher-test 做选择。

验收：

```text
Step7 teacher-test = 274 / 300
Round5 teacher-test = 274 / 300
Round4 DeBERTa teacher-test = 263 / 300
```

### Phase 1: 构建非 teacher safe/unsafe override 扩展集

目的：当前 flip guard 的训练样本太少，且 teacher-test 上 safe candidate 被挡掉。
Round6 需要更多非 teacher 的 safe/unsafe override 候选，而不是直接放宽阈值。

新增脚本：

```text
src/data/build_round6_safe_override_dataset.py
```

输入：

```text
outputs/evaluation/round5_flip_ledger.jsonl
data/processed/round4_residual_dev_hardpos.jsonl
data/processed/round4_residual_dev_hardneg.jsonl
data/processed/round4_residual_train.jsonl
data/processed/lit_academic_poetry_train.jsonl
data/processed/lit_academic_poetry_valid.jsonl
data/processed/lit_academic_poetry_internal_test.jsonl
```

输出：

```text
data/processed/round6_override_train.jsonl
data/processed/round6_override_dev_safe.jsonl
data/processed/round6_override_dev_unsafe.jsonl
data/processed/round6_override_probe_mixed.jsonl
data/processed/round6_override_dataset_report.json
```

构建原则：

1. 只用非 teacher 数据。
2. valid/internal_test 不改标签、不混入训练。
3. 样本按 `pair_id` 或 source id 分组切分，避免同源泄漏。
4. safe examples 来自非 teacher 的 `fixed_fn_candidate`。
5. unsafe examples 来自非 teacher 的 `induced_fp`。
6. 对 `general_prose` 和 `literary_short_fragment` 单独扩充，不再只依赖 old/short 合并规则。

目标规模：

| Type | Target |
| --- | ---: |
| safe_override train | >= 300 |
| unsafe_override train | >= 600 |
| safe dev | >= 100 |
| unsafe dev | >= 200 |
| general_prose safe dev | >= 40 |
| literary_short_fragment safe dev | >= 40 |

如果现有数据不足，优先做 train-only 数据补齐：

| Gap | Data action |
| --- | --- |
| general_prose safe LLM candidate 少 | 从 non-teacher human sources 生成 conservative LLM rewrites |
| short_fragment safe LLM candidate 少 | 生成 20-80 word literary/academic short rewrites |
| unsafe human short fragments 少 | 从 Gutenberg/ACL-OCL 抽取 high-style short human mirrors |
| general_prose unsafe 少 | 抽取 polished human prose, especially long fluent human paragraphs |

验收：

```text
teacher-test exact duplicate = 0
pair/source leakage across train/dev = 0
safe/unsafe label definition explicit
bucket distribution report present
```

### Phase 2: 训练候选级 safe-override selector

目的：Round5 flip guard 是第一版 veto 模型；Round6 应改成更直接的候选选择器：

```text
input = Step7-human -> Round4-LLM candidate
output = p_safe_override
```

新增脚本：

```text
src/models/train_round6_safe_override_selector.py
src/evaluation/predict_round6_safe_override_selector.py
```

推荐先用可解释模型：

```text
LogisticRegression
LinearSVC with calibration
HistGradientBoosting only as secondary candidate
```

候选特征：

| Feature group | Examples |
| --- | --- |
| probability | `step7_prob`, `round4_prob`, `prob_delta`, margins |
| guard | `p_unsafe_override`, `p_human_style`, current veto flags |
| bucket | `bucket`, `round4_bucket`, `domain`, `generator` |
| text shape | length, line count, punctuation, quote count, archaic markers |
| lexical | word/char ngram TF-IDF |
| source metadata | prompt type, generation type, source stage when available |

关键变化：

```text
Round5: learn p_unsafe_override and veto
Round6: learn p_safe_override and select only high-confidence safe cases
```

输出：

```text
outputs/models/round6_safe_override_selector/selector.pkl
outputs/models/round6_safe_override_selector/selector_report.json
outputs/evaluation/round6_safe_override_selector_report.md
```

验收：

| Metric | Required |
| --- | --- |
| unsafe dev recall as blocked | >= 0.90 |
| safe dev pass rate | >= 0.25, target >= 0.40 |
| hardneg induced FP after rule | 0 |
| internal induced FP after rule | <= 1 |

### Phase 3: 重写 override rule search

目的：不再让一个单一 `p_unsafe_override <= threshold` 决定是否覆盖，而是把
selector、bucket、margin 组合成候选级规则。

新增脚本：

```text
src/evaluation/tune_round6_safe_override.py
```

规则空间：

```text
default = Step7
allow only Step7-human -> LLM
require Round4 DeBERTa prediction = LLM
require p_safe_override >= threshold
require p_unsafe_override <= max threshold
require Round4 probability/margin lower bound
bucket-specific thresholds allowed
```

推荐搜索：

```text
p_safe_min = 0.55,0.60,0.65,0.70,0.75,0.80
p_unsafe_max = 0.20,0.30,0.35,0.40,0.45
round4_threshold = 0.50,0.55,0.60,0.65,0.70
min_delta = 0.00,0.05,0.10,0.15
bucket_policy = old_short, short_only, general_strict, old_short_plus_general_strict
```

Bucket-specific policy examples:

| Policy | Meaning |
| --- | --- |
| `old_short` | Round5 baseline policy |
| `short_only` | only `literary_short_fragment`, stricter than Round5 |
| `general_strict` | allow `general_prose` only with high `p_safe` and low `p_unsafe` |
| `old_short_plus_general_strict` | allow old/short normally, general only under stricter thresholds |

输出：

```text
outputs/models/round6_safe_override/rules.json
outputs/models/round6_safe_override/tuning_report.json
outputs/evaluation/round6_safe_override_tuning_report.md
outputs/predictions/round6_safe_override_internal_test_predictions.jsonl
outputs/predictions/round6_safe_override_hardpos_predictions.jsonl
outputs/predictions/round6_safe_override_hardneg_predictions.jsonl
```

晋级门槛：

```text
hardneg induced FP = 0
internal induced FP <= 1
internal F1 >= 0.9570 preferred
hardpos fixed Step7 FN >= 70 preferred
non-empty override required
```

### Phase 4: 可选的轻量 DeBERTa branch 修复

只有当 Phase 3 仍然 no-op 或 fixed-FN 不足时，才进入重训分支。

目标不是更高 recall，而是更少 induced FP：

| Candidate | Change | Purpose |
| --- | --- | --- |
| `round6_deberta_guardweighted` | unsafe-like human rows weight 3.0 | suppress induced FP |
| `round6_deberta_epoch1_lr5e6` | 1 epoch, lr 5e-6 | reduce overfitting |
| `round6_deberta_short_general_balanced` | add safe short/general positives with matched human negatives | improve candidate balance |

建议命令模板：

```powershell
.\.venv\Scripts\python.exe src\models\train_weighted_transformer.py `
  --train data\processed\round6_residual_train.jsonl `
  --valid data\processed\lit_academic_poetry_valid.jsonl `
  --test data\processed\lit_academic_poetry_internal_test.jsonl `
  --guard_dev data\processed\round6_override_probe_mixed.jsonl `
  --output_dir outputs\models\round6_deberta_guardweighted `
  --model_name microsoft/deberta-v3-base `
  --epochs 1 `
  --learning_rate 5e-6 `
  --batch_size 4 `
  --eval_batch_size 8 `
  --gradient_accumulation_steps 2 `
  --sample_weight_field sample_weight `
  --class_weight none `
  --fp16 `
  --seed 20260522
```

候选不应直接替代 Step7，只能作为新的 signal branch 进入 Phase 3 rule search。

### Phase 5: 生成 Round6 gate report

新增：

```text
outputs/evaluation/round6_gate_report.md
docs/ROUND6_OPTIMIZATION_WORK_LOG_2026-05-22.md
```

必须回答：

1. 修复多少 hardpos Step7 FN？
2. 是否新增 hardneg FP？
3. 是否新增 internal FP？
4. general_prose policy 是否安全？
5. literary_short_fragment safe pass 是否改善？
6. 是否满足 teacher-test 之前所有 gate？

显式决策：

```text
PROMOTE_TO_TEACHER_TEST = yes/no
FINAL_MODEL_CANDIDATE = yes/no
REASON = ...
```

### Phase 6: 只在通过 gate 后做 teacher-test final diagnostic

只有 Phase 5 写明：

```text
PROMOTE_TO_TEACHER_TEST = yes
```

才运行 teacher-test。

需要输出：

```text
outputs/predictions/round6_teacher_test_predictions.jsonl
outputs/evaluation/round6_teacher_test_comparison.md
outputs/evaluation/round6_teacher_test_ledger_summary.json
docs/ROUND6_FINAL_DECISION_2026-05-22.md
```

判断：

| Result | Decision |
| --- | --- |
| <= 274 / 300 | reject, keep Step7 |
| 275-284 / 300 | partial success |
| >= 285 / 300 | 95% achieved, promote if no leakage |

## 5. Round6 第一批立即任务

建议下一次直接从以下任务开始。

### Task 1: 写 Round6 starting-point report

新增：

```text
outputs/evaluation/round6_starting_point_report.md
```

内容：

```text
Step7 teacher-test = 274/300
Round5 teacher-test = 274/300
Round4 branch teacher-test = 263/300
Round5 teacher-test override candidates = 17
safe candidates = 3
unsafe candidates = 14
```

注意：只记录聚合诊断，不导出 teacher-test 文本到训练数据。

### Task 2: 实现 `build_round6_safe_override_dataset.py`

先不训练，先看数据是否够：

```powershell
.\.venv\Scripts\python.exe src\data\build_round6_safe_override_dataset.py
```

验收：

```text
safe train >= 300
unsafe train >= 600
safe dev >= 100
unsafe dev >= 200
teacher exact duplicate = 0
```

如果不够，先生成/抽取数据，不进入模型训练。

### Task 3: 训练 safe selector v1

```powershell
.\.venv\Scripts\python.exe src\models\train_round6_safe_override_selector.py
```

第一版只需要线性模型，先证明候选级 selection 有可分性。

### Task 4: 调 Round6 rule search

```powershell
.\.venv\Scripts\python.exe src\evaluation\tune_round6_safe_override.py
```

只有满足：

```text
hardneg induced FP = 0
internal induced FP <= 1
hardpos fixed Step7 FN >= 70
internal F1 >= 0.9570
```

才写 `PROMOTE_TO_TEACHER_TEST = yes`。

## 6. 不要做的事

Round6 不建议做以下事情：

1. 不要直接降低 `p_unsafe_override` 阈值去碰 teacher-test。
2. 不要把 Round4 DeBERTa 当全局模型。
3. 不要用 teacher-test 上那 3 个 safe candidate 的文本去训练。
4. 不要只看 hardpos recall，不看 hardneg/internal FP。
5. 不要因为非 teacher hardpos 修复多就自动晋级最终模型。

## 7. 最终建议

Round6 的核心路线应是：

```text
build more non-teacher candidate-level safe/unsafe override data
-> train a candidate-level safe selector
-> search bucket-specific safe override rules
-> gate on hardneg/internal FP
-> teacher-test only once after gate passes
```

如果 Round6 能把 teacher-test-like 的 safe override 从当前 0 个释放到至少 1-3 个，
且不引入 FP，它就能先完成 `>274/300` 的短期目标。95% 目标仍需要更大的净修复，
但不能靠牺牲 human precision 去换。

## 8. 下一次对话接力清单

本文件现在可以作为下一段 Round6 实作的起点。建议下一段对话先读：

```text
docs/ROUND5_SUPPLEMENT_AND_ROUND6_PLAN_2026-05-22.md
docs/ROUND5_FINAL_DECISION_2026-05-22.md
docs/ROUND5_OPTIMIZATION_WORK_LOG_2026-05-22.md
```

然后按以下顺序执行，不要跳步：

### 8.1 先冻结 starting point

产物：

```text
outputs/evaluation/round6_starting_point_report.md
```

必须记录：

| Item | Expected |
| --- | --- |
| Step7 teacher-test | 274 / 300 |
| Round5 teacher-test | 274 / 300 |
| Round4 DeBERTa teacher-test | 263 / 300 |
| teacher-test override candidates | 17 total = 3 safe + 14 unsafe |
| Round5 non-teacher gate | pass, but teacher-test no-op |

这一步只写聚合数字和文件路径，不导出 teacher-test 文本、逐行 label 或用于调参的 id 列表。

### 8.2 再构建 Round6 safe/unsafe override dataset

新增脚本：

```text
src/data/build_round6_safe_override_dataset.py
```

优先复用：

```text
outputs/evaluation/round5_flip_ledger.jsonl
data/processed/round5_flip_guard_train.jsonl
data/processed/round5_flip_guard_dev_hardpos.jsonl
data/processed/round5_flip_guard_dev_hardneg.jsonl
data/processed/round4_residual_train.jsonl
data/processed/round4_residual_dev_hardpos.jsonl
data/processed/round4_residual_dev_hardneg.jsonl
```

第一轮只做数据盘点，不训练模型。验收写进：

```text
data/processed/round6_override_dataset_report.json
```

如果目标规模达不到，就停在数据补齐，不进入 selector 训练。

### 8.3 selector 与 rule search 的晋级条件

只有数据集报告通过后，才进入：

```text
src/models/train_round6_safe_override_selector.py
src/evaluation/predict_round6_safe_override_selector.py
src/evaluation/tune_round6_safe_override.py
```

进入 teacher-test 前必须同时满足：

| Gate | Required |
| --- | --- |
| hardneg induced FP | 0 |
| internal induced FP | <= 1 |
| internal F1 | >= 0.9564 hard minimum, >= 0.9570 preferred |
| hardpos fixed Step7 FN | >= 57 hard minimum, >= 70 target |
| override rule | non-empty |
| teacher-test leakage | exact duplicate = 0, no teacher labels/text in tuning |

### 8.4 明确停止条件

如果 Round6 的非 teacher gate 没有同时压住 hardneg/internal FP，不要运行 teacher-test。

如果 rule search 只找到 no-op，结论应写成：

```text
PROMOTE_TO_TEACHER_TEST = no
KEEP_FINAL_MODEL = Step7 ensemble
NEXT_ACTION = data补齐或guard/selector重做
```

如果通过 gate 并运行 teacher-test，最终判断仍以 Step7 为下限：

| Result | Decision |
| --- | --- |
| <= 274 / 300 | reject, keep Step7 |
| 275-284 / 300 | partial success, not 95% |
| >= 285 / 300 | 95% achieved, promote only if no leakage |

### 8.5 本文档完成状态

```text
STATUS = complete handoff for Round6 start
FINAL_MODEL_BEFORE_ROUND6 = Step7 ensemble
ROUND5_PROMOTED = no
NEXT_IMPLEMENTATION_STEP = Round6 Phase 0 starting-point report
```
