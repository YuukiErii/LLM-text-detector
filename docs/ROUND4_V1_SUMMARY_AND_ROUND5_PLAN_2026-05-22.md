# Round4 v1 工作总结与 Round5 95% 优化执行方案

Date: 2026-05-22

本文件是在前三轮总结文件 `docs/THREE_ROUND_OPTIMIZATION_REVIEW_AND_95_ROUTE_2026-05-22.md` 之后，对本轮 Round4 v1 执行结果的整理，并给出下一轮可直接落地的优化方案。

最终目标不变：

```text
teacher-test 95% accuracy = at least 285 / 300 correct
current strict final Step7 = 274 / 300 correct
needed net gain = +11 correct samples
```

严格路线也不变：`data/raw/teacher_test.json` 不能用于训练、阈值选择、模型选择、stacker 训练或 router 调参；只能作为最终诊断。当前没有任何 Round4 v1 候选通过非 teacher-test 预检查，所以当前最终系统仍应保留 Step7 ensemble。

## 1. 本轮总决策

Round4 v1 完成了 residual data rebuild、human-style guard、weighted DeBERTa retrain、residual override gate 四个关键环节。

结论：

1. Round4 v1 的数据构建是成功的，可以继续复用。
2. Round4 DeBERTa 学到了 Step7 缺少的 hard-positive 信号。
3. 但 Round4 DeBERTa 作为全局模型会显著增加 human false positive。
4. 当前 human-style guard 保护力不够，不能让 Round4 分支安全覆盖 Step7。
5. residual override 搜索没有找到任何非空可行规则。
6. 因此 Round4 v1 不进入 teacher-test 诊断，不晋级最终模型。

一句话判断：Round4 v1 是有价值的训练和诊断资产，但还不是一个可提交模型。

## 2. 已完成工作总结

### 2.1 Residual dataset rebuild

新增脚本：

```text
src/data/build_round4_residual_dataset.py
```

主要输出：

```text
data/processed/round4_hard_human_mirror_seed.jsonl
data/processed/round4_hard_llm_positive_seed.jsonl
data/processed/round4_residual_train.jsonl
data/processed/round4_residual_dev_hardpos.jsonl
data/processed/round4_residual_dev_hardneg.jsonl
data/processed/round4_residual_spotcheck.jsonl
data/processed/round4_residual_report.json
```

关键结果：

| Item | Result |
| --- | ---: |
| hard human mirrors | 4727 |
| hard LLM positives | 3674 |
| human:LLM hard ratio | 1.2866 |
| round4 train rows | 17434 |
| hard-positive dev rows | 500 |
| hard-negative dev rows | 500 |
| old-prose human mirrors | 800 |
| poetry/freeverse human mirrors | 1500 |
| natural academic human mirrors | 1000 |
| teacher-test exact duplicates | 0 |

数据验收通过。尤其重要的是，本轮新增 `round4_bucket`，保留了 source-known residual bucket；原有 `bucket` 仍是文本特征分桶。这个设计避免 old-prose 样本因为文本长度或表面特征被误归到其他桶。

仍存在的数据短板：

| Bucket | Shortfall |
| --- | ---: |
| human literary_short_fragment | 273 |
| LLM poetry_classical | 197 |
| LLM academic_formal | 228 |
| LLM literary_short_fragment | 201 |

这些短板应成为下一轮的数据优先级。

### 2.2 Human-style guard

新增脚本：

```text
src/models/train_round4_human_style_guard.py
src/evaluation/predict_round4_human_style_guard.py
```

主要输出：

```text
outputs/models/round4_human_style_guard/human_style_guard.pkl
outputs/models/round4_human_style_guard/human_style_guard_report.json
outputs/evaluation/round4_human_style_guard_report.md
```

关键指标：

| Split | Meaning | Result |
| --- | --- | ---: |
| dev_hardpos_should_not_veto | LLM hard positives 被错误 veto 的比例 | 0.054 |
| dev_hardneg_should_veto | hard human 被保护的比例 | 0.214 |
| internal_test veto rate | internal-test 上触发保护比例 | 0.0116 |

解释：guard 很保守，误伤 LLM hard positives 很少，这是优点；但它只保护了 21.4% 的 hard-human dev，因此无法充分约束 Round4 DeBERTa 带来的 FP 风险。

### 2.3 Weighted DeBERTa retrain

训练输出：

```text
outputs/models/round4_deberta_weighted_residual/
outputs/models/round4_deberta_weighted_residual/best_model
```

训练配置要点：

```text
model_name = microsoft/deberta-v3-base
train = data/processed/round4_residual_train.jsonl
epochs = 3
learning_rate = 1e-5
sample_weight_field = sample_weight
class_weight = none
balanced_sampler = false
```

关键指标：

| Split | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 0.9543 | 0.9496 | 0.9574 | 0.9535 | 43 | 36 |
| internal_test | 0.9445 | 0.9321 | 0.9563 | 0.9441 | 59 | 37 |
| hardpos dev | 0.5660 | 1.0000 | 0.5660 | 0.7229 | 0 | 217 |

与 Step7 对比：

| Split | Step7 | Round4 DeBERTa | Interpretation |
| --- | ---: | ---: | --- |
| internal_test F1 | 0.9564 | 0.9441 | 全局退化 |
| hardpos recall | 0.4160 | 0.5660 | 明显捕获更多 hard LLM positives |
| hardneg FP | 26 | 53 | human FP 风险翻倍 |

解释：Round4 DeBERTa 的 hard-positive recall 改善是真信号，但它把大量高风格人类文本推向 LLM，不能直接替代 Step7。

### 2.4 Residual override gate

新增脚本：

```text
src/evaluation/tune_round4_residual_override.py
```

补丁：

```text
src/evaluation/predict_neural_model.py
src/evaluation/predict_ensemble.py
```

补丁目的：预测文件保留 `round4_bucket` / `round4_tag`，让后续 gate 能按 residual bucket 分析。

主要输出：

```text
outputs/models/round4_residual_override/rules.json
outputs/models/round4_residual_override/residual_override_tuning_report.json
outputs/evaluation/round4_residual_override_tuning_report.md
```

搜索结果：

| Item | Result |
| --- | ---: |
| aligned rows | 2731 |
| candidate rules | 3601 |
| feasible rules | 1 |
| feasible non-empty override rules | 0 |
| selected rule | disabled no-op baseline |

最终 selected rule 等于不覆盖 Step7：

| Split | F1 | FP | FN | Overrides | Fixed Step7 FN | Induced FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| internal_test | 0.9564 | 27 | 46 | 0 | 0 | 0 |
| hardpos | 0.5876 | 0 | 292 | 0 | 0 | 0 |
| hardneg | 0.0000 | 26 | 0 | 0 | 0 | 0 |

解释：如果要求 hardneg FP 不高于 Step7、internal FP 不增加、internal F1 不明显退化，就没有任何 Round4 override 可以安全启动。这个 gate 很严格，但方向是对的；95% 目标需要净修复 11 个 teacher-test 错误，不能靠牺牲 human precision 换来表面 recall。

## 3. Round4 v1 为什么没有晋级

Round4 v1 的失败点不是“没有学到东西”，而是“学到的东西还不能安全使用”。

主要问题：

1. Round4 DeBERTa 解决了部分 hard-positive FN，但同时制造更多 hard-negative FP。
2. 现有 human-style guard 是通用高风格人类保护器，不是专门保护 Round4-induced FP 的 flip guard。
3. 当前 override 规则只能看到概率、margin、bucket、guard score；这些信号不足以区分“Step7 漏掉的 LLM”与“Round4 误判的人类高风格文本”。
4. LLM old-prose、classical poetry、formal academic、short fragment 仍有生成端短板。
5. 三轮经验都显示，全局替换或简单阈值移动容易破坏 Step7 已经平衡的 FP/FN。

下一轮优化必须转向“先识别哪些 Round4 override 是危险的”，再谈修复更多 Step7 FN。

## 4. 下一轮优化目标

建议把下一轮命名为：

```text
Round5: FP-safe residual repair
```

它也可以理解为 Round4 v2：保留 Round4 v1 的数据资产，但不继续盲目扩大 aggressive LLM recall，而是先训练一个专门的 flip guard，再低风险重训 DeBERTa 分支。

### 4.1 非 teacher-test 目标

硬门槛：

| Gate | Required |
| --- | --- |
| internal_test F1 | >= 0.9544, preferably >= 0.9564 |
| internal_test FP | <= Step7 FP + 2, ideally <= Step7 |
| hardneg dev FP | <= 26 |
| hardpos dev recall | > 0.416, target >= 0.50 |
| non-empty override | required; no-op rule cannot晋级 |
| teacher-test leakage | exact duplicate = 0; no teacher labels in tuning |

推荐 stretch gate：

```text
hardpos fixed Step7 FN >= 40
hardneg induced FP = 0
internal induced FP <= 2
```

只有达到这些门槛，才值得进入 teacher-test 最终诊断。

### 4.2 Teacher-test 目标

晋级目标：

```text
beat Step7: > 274 / 300
95% target: >= 285 / 300
```

如果新候选只在 hardpos dev 上好看，但 non-teacher human FP 变差，就不应使用 teacher-test 诊断机会。

## 5. Round5 具体执行方案

### Phase 0: 冻结基线与预测文件

目的：确保下一轮所有比较都和同一套 Step7 / Round4 v1 预测对齐。

需要固定：

```text
Step7 predictions:
outputs/predictions/round4_step7_internal_test_predictions.jsonl
outputs/predictions/round4_step7_hardpos_predictions.jsonl
outputs/predictions/round4_step7_hardneg_predictions.jsonl

Round4 DeBERTa predictions:
outputs/predictions/round4_deberta_internal_test_predictions.jsonl
outputs/predictions/round4_deberta_hardpos_predictions.jsonl
outputs/predictions/round4_deberta_hardneg_predictions.jsonl

Round4 guard predictions:
outputs/predictions/round4_human_style_guard_internal_test_predictions.jsonl
outputs/predictions/round4_human_style_guard_hardpos_predictions.jsonl
outputs/predictions/round4_human_style_guard_hardneg_predictions.jsonl
```

新增输出：

```text
outputs/evaluation/round5_baseline_frozen_report.md
```

验收：

1. 三个 split 的 row id / pair id 完全对齐。
2. Step7 metrics 能复现 internal F1 0.9564、hardpos recall 0.416、hardneg FP 26。
3. Round4 metrics 能复现 internal F1 0.9441、hardpos recall 0.566、hardneg FP 53。

### Phase 1: 构建 Round4 flip ledger

目的：把 Round4 v1 的失败显式转成训练信号。

新增脚本：

```text
src/evaluation/build_round5_flip_ledger.py
```

输入：

```text
Step7 predictions
Round4 predictions
Round4 guard predictions
data/processed/round4_residual_dev_hardpos.jsonl
data/processed/round4_residual_dev_hardneg.jsonl
data/processed/lit_academic_poetry_internal_test.jsonl
```

输出：

```text
outputs/evaluation/round5_flip_ledger.jsonl
outputs/evaluation/round5_flip_ledger_summary.md
data/processed/round5_flip_guard_train.jsonl
data/processed/round5_flip_guard_dev_hardpos.jsonl
data/processed/round5_flip_guard_dev_hardneg.jsonl
```

ledger 每行至少包含：

```text
id / pair_id
label
text
split
bucket
round4_bucket
round4_tag
step7_prob
round4_prob
prob_delta
step7_pred
round4_pred
guard_p_human_style
flip_type
```

核心 `flip_type`：

| flip_type | Meaning | Use |
| --- | --- | --- |
| fixed_fn_candidate | Step7 human, Round4 LLM, label LLM | override 候选正例 |
| induced_fp | Step7 human, Round4 LLM, label human | override 危险负例 |
| stable_step7_correct | Step7 correct, Round4 unchanged | calibration anchor |
| both_miss | Step7 and Round4 both wrong | data generation target |

验收：

1. hardneg 中 `induced_fp` 应接近 27 个新增风险点，因为 Round4 hardneg FP 53、Step7 hardneg FP 26。
2. hardpos 中 `fixed_fn_candidate` 应能解释 Round4 recall 从 0.416 到 0.566 的新增修复空间。
3. 每个 flip group 输出 bucket breakdown。

### Phase 2: 补齐最缺 residual buckets

目的：Round4 v1 的数据缺口集中在 LLM side 的 old/high-style/specific-form buckets，以及 human short fragments。下一轮不能只继续堆泛化 general prose。

新增或复用脚本：

```text
src/data/build_round5_residual_augments.py
```

优先补齐：

| Bucket | Current issue | Round5 target |
| --- | --- | ---: |
| LLM literary_old_prose | only minimum 300 selected | 600-800 |
| LLM poetry_classical | shortfall 197 | +250 clean samples |
| LLM academic_formal | shortfall 228 | +300 clean samples |
| LLM literary_short_fragment | shortfall 201 | +250 clean samples |
| human literary_short_fragment | shortfall 273 | +300 clean samples |

生成原则：

1. 保留 train-only；`valid` 和 `internal_test` 不变。
2. 不使用 teacher-test 文本或标签。
3. LLM old-prose 不只用 ChatGPT，尽量加入 DeepSeek / Doubao / Gemini 风格差异。
4. 每个新样本写入 `source_stage`, `round5_bucket`, `round5_tag`, `sample_weight`。
5. 对 hard human mirror 继续给更高 `sample_weight`，但不要让全局模型被迫过度保守。

新增输出：

```text
data/processed/round5_residual_augments.jsonl
data/processed/round5_residual_train.jsonl
data/processed/round5_residual_report.json
```

验收：

1. `round5_residual_train.jsonl` 无 exact duplicate。
2. teacher-test exact duplicate = 0。
3. 新增 LLM old-prose 至少 600。
4. 新增 hard human mirrors 与 hard LLM positives 的比例保持在 1.1-1.5。
5. 人工 spotcheck 至少 100 条，记录 prompt leakage、乱码、过短、低质量比例。

### Phase 3: 训练专门的 flip guard

目的：当前 guard 只会识别一部分高风格人类文本。Round5 需要一个更针对性的模型：判断“Round4 想把 Step7-human 改成 LLM 时，这次 override 是否危险”。

新增脚本：

```text
src/models/train_round5_flip_guard.py
src/evaluation/predict_round5_flip_guard.py
```

训练数据：

```text
positive class = induced_fp / unsafe_override
negative class = fixed_fn_candidate / safe_override
```

可用特征：

```text
char ngram TF-IDF
word ngram TF-IDF
length / punctuation / quote / line-break features
bucket and round4_bucket one-hot
step7_prob
round4_prob
prob_delta
guard_p_human_style
```

建议先用 logistic regression 或 linear SVM，不要先上复杂神经模型。这个 guard 的任务是“拒绝危险 override”，可解释性比表达力更重要。

输出：

```text
outputs/models/round5_flip_guard/flip_guard.pkl
outputs/models/round5_flip_guard/flip_guard_report.json
outputs/evaluation/round5_flip_guard_report.md
```

验收：

| Metric | Required |
| --- | --- |
| hardneg induced_fp protection | >= 70% |
| hardpos safe_override veto | <= 10% |
| internal_test veto rate | <= 3% |
| bucket report | required |

如果这个 gate 失败，不继续训练更 aggressive 的 override；先回到 Phase 2 补数据。

### Phase 4: 低风险重训 Round5 DeBERTa branch

目的：Round4 DeBERTa 太 aggressive。下一轮不是简单重跑 3 epochs，而是训练一组更保守、以 FP-safe 为目标的小候选。

推荐候选：

| Candidate | Change | Purpose |
| --- | --- | --- |
| round5_deberta_epoch1_lr5e6 | 1 epoch, lr 5e-6 | 降低过拟合 residual buckets |
| round5_deberta_epoch2_lr5e6 | 2 epochs, lr 5e-6 | 保留更多 hard-positive recall |
| round5_deberta_guardweighted | induced_fp-like human rows weight 2.5 | 压制 Round4 FP |
| round5_deberta_no_general_boost | 限制 general_prose hard positives | 避免泛化到过多 polished human |

命令模板：

```powershell
.\.venv\Scripts\python.exe src\models\train_weighted_transformer.py `
  --train data\processed\round5_residual_train.jsonl `
  --valid data\processed\lit_academic_poetry_valid.jsonl `
  --test data\processed\lit_academic_poetry_internal_test.jsonl `
  --guard_dev data\processed\round4_residual_dev_hardpos.jsonl `
  --output_dir outputs\models\round5_deberta_epoch1_lr5e6 `
  --model_name microsoft/deberta-v3-base `
  --epochs 1 `
  --learning_rate 5e-6 `
  --threshold 0.5 `
  --sample_weight_field sample_weight `
  --class_weight none `
  --seed 20260522
```

每个候选必须输出：

```text
outputs/models/<candidate>/metrics.json
outputs/predictions/<candidate>_internal_test_predictions.jsonl
outputs/predictions/<candidate>_hardpos_predictions.jsonl
outputs/predictions/<candidate>_hardneg_predictions.jsonl
```

训练后先做候选表，不立刻融合：

| Candidate | internal F1 | internal FP | hardpos recall | hardneg FP | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Step7 | 0.9564 | 27 | 0.416 | 26 | baseline |
| Round4 v1 | 0.9441 | 59 | 0.566 | 53 | rejected as global |
| Round5 candidate | TBD | TBD | TBD | TBD | promote only if FP-safe |

### Phase 5: 重新调 residual override

目的：Step7 继续做默认系统，只允许在非常局部、证据很强时把 Step7-human 改成 LLM。

新增或扩展脚本：

```text
src/evaluation/tune_round5_residual_override.py
```

规则空间：

```text
default = Step7 prediction
allow only Step7 human -> LLM override
require high Round5 DeBERTa probability
require positive margin over Step7
require flip_guard unsafe probability below threshold
require human_style_guard below threshold
optionally require selected high-risk LLM buckets only
```

推荐搜索参数：

```text
round5_thresholds = 0.65,0.70,0.75,0.80,0.85,0.90,0.95
min_deltas = 0.05,0.10,0.15,0.20,0.25
flip_guard_unsafe_max = 0.20,0.30,0.40
human_style_veto_max = 0.65,0.70,0.75
min_words = 16,32,48
allowed_buckets = old_prose, poetry_classical, poetry_freeverse, academic_formal, short_fragment
```

输出：

```text
outputs/models/round5_residual_override/rules.json
outputs/models/round5_residual_override/residual_override_tuning_report.json
outputs/evaluation/round5_residual_override_tuning_report.md
```

晋级条件：

| Gate | Required |
| --- | --- |
| non-empty overrides | yes |
| hardpos fixed Step7 FN | >= 30, target >= 50 |
| hardneg induced FP | 0 |
| internal induced FP | <= 2 |
| internal F1 | >= 0.9544, target >= 0.9564 |

如果搜索结果再次只有 no-op 规则，说明当前信号仍不足，不跑 teacher-test。

### Phase 6: 生成 Round5 gate report

新增汇总文件：

```text
outputs/evaluation/round5_gate_report.md
docs/ROUND5_OPTIMIZATION_WORK_LOG_2026-05-22.md
```

报告必须回答：

1. 新模型修复了多少 Step7 hard-positive FN？
2. 新模型造成多少 hard-human FP？
3. 每个 bucket 的收益和风险是什么？
4. 与 Step7、Round4 v1 的错误重合如何？
5. 是否满足 teacher-test 之前的全部 gate？

必须显式给出 promotion decision：

```text
PROMOTE_TO_TEACHER_TEST = yes/no
FINAL_MODEL_CANDIDATE = yes/no
REASON = ...
```

### Phase 7: 只在通过 gate 后做 teacher-test final diagnostic

只有 Phase 6 写明 `PROMOTE_TO_TEACHER_TEST = yes` 后，才运行 teacher-test 诊断。

晋级判断：

| Result | Decision |
| --- | --- |
| <= 274 / 300 | reject, keep Step7 |
| 275-284 / 300 | improved but not 95%; report as partial success |
| >= 285 / 300 | reaches 95%; promote if no leakage/tuning violation |

无论结果如何，都要保留：

```text
outputs/predictions/round5_teacher_test_predictions.jsonl
outputs/evaluation/round5_teacher_test_comparison.md
docs/ROUND5_FINAL_DECISION_2026-05-22.md
```

## 6. 下一轮优先级排序

最推荐的执行顺序：

1. Phase 0 baseline freeze。
2. Phase 1 flip ledger。
3. Phase 3 flip guard。
4. Phase 5 用现有 Round4 DeBERTa + 新 flip guard 先尝试一次 override gate。
5. 如果仍无 non-empty safe rule，再做 Phase 2 数据补齐和 Phase 4 DeBERTa 重训。

原因：如果一个专门的 flip guard 能让现有 Round4 DeBERTa 的 hard-positive signal 安全释放，就可以少跑一次昂贵重训；如果 guard 本身区分不了 safe override 和 unsafe override，再扩大 DeBERTa 训练大概率仍会继续制造 FP。

## 7. 立即可开始的第一个任务

下一轮第一步建议直接实现：

```text
src/evaluation/build_round5_flip_ledger.py
```

并生成：

```text
outputs/evaluation/round5_flip_ledger_summary.md
data/processed/round5_flip_guard_train.jsonl
```

这一步成本低、信息量高，能马上告诉我们：

1. Round4 v1 新修复的 hard positives 到底集中在哪些 bucket。
2. Round4 v1 新制造的 hard human FP 是否有明显文本特征。
3. flip guard 是否有足够训练样本。
4. 下一轮应该先补数据，还是先训练 guard。

## 8. 当前结论

Round4 v1 不应被视作失败实验，而应视作把 95% 路线推进到了更清楚的位置：

```text
Step7 is still the final model.
Round4 residual data is reusable.
Round4 DeBERTa is useful only as a hard-positive signal branch.
The missing component is a strong FP-safe flip guard.
Round5 should optimize override safety before chasing more recall.
```

下一轮真正要解决的问题不是“让模型更敢判 LLM”，而是“只在 Step7 确实漏判 LLM、且不是高风格人类文本时才覆盖 Step7”。如果这个局部门控能稳定工作，才有现实机会从 274/300 向 285/300 推进。
