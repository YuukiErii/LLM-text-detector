# Round2 复盘与 Round3 优化计划

更新时间：2026-05-21

本文是给下一轮对话读取的交接文档。它总结三件事：

1. 第二轮优化已经完成了什么。
2. 为什么这一轮没有把 teacher-test 准确率推进到 95%。
3. 下一轮应当按什么顺序继续优化。

下一轮对话请先读本文，再按需查看：

```text
docs/SECOND_ROUND_95_OPTIMIZATION_PLAN.md
docs/ROUND2_PHASE0_DIAGNOSTICS.md
docs/ROUND2_RESULTS_SUMMARY.md
PROJECT_REPORT.md
README.md
```

## 1. 当前最终结论

当前严格泛化路线下，最终推荐系统仍是 Step7 DeBERTa + TF-IDF ensemble：

```text
DeBERTa:  outputs/models/deberta_lit_academic_poetry_step7_combined
TF-IDF:   outputs/models/tfidf_lit_academic_poetry
alpha:    0.5
threshold: 0.55
```

teacher-test 结果：

| 指标 | 数值 |
| --- | ---: |
| accuracy | 0.9133 |
| correct | 274 / 300 |
| precision | 0.9133 |
| recall | 0.9133 |
| F1 | 0.9133 |
| FP | 13 |
| FN | 13 |
| confusion | `[[137, 13], [13, 137]]` |

95% 目标对应：

```text
target correct = 285 / 300
current correct = 274 / 300
needed net gain = +11 correct examples
current errors = 26
maximum allowed errors at 95% = 15
```

第二轮没有达到 95%。当前最终提交文件为：

```text
outputs/predictions/round2_final_submission.json
```

## 2. 第二轮已经完成的成果

第二轮基本完成了原计划 Phase 0 到 Phase 5 的严格路线：没有把
teacher-test 标签用于训练、阈值选择、stacker 训练、router 调参或模型选择。
teacher-test 只作为最后诊断结果使用。

### 2.1 Phase 0：锁定基线和诊断上限

新增脚本：

```text
src/evaluation/export_error_ledger.py
src/evaluation/round2_threshold_family_diagnostics.py
```

生成的本地诊断产物：

```text
outputs/round2/error_ledger_teacher_step7.csv
outputs/round2/error_ledger_teacher_step7.jsonl
outputs/round2/existing_family_threshold_report.md
outputs/round2/existing_family_threshold_report.json
```

核心发现：

| 诊断项 | 结果 |
| --- | --- |
| Step7 teacher-test 剩余错误 | 26 个，13 FP + 13 FN |
| near-boundary 错误 | 9 / 26 |
| confidently wrong 错误 | 17 / 26 |
| 单个已有预测文件 oracle threshold 上限 | 约 0.9267 |
| 已有模型族简单平均 oracle 上限 | 约 0.9333 |

解释：剩余错误不是单纯的全局阈值问题。相当多错误属于模型很自信但判断错的情况，
因此需要新数据、新模型信号或更强的约束式融合。

### 2.2 Phase 1：构造 teacher-like development data

第二轮没有泛泛扩数据，而是围绕残余错误构造 teacher-like 数据。

主要数据产物：

```text
data/processed/round2_human_hardneg_source.jsonl
data/processed/rewrite_prompts_round2_chatgpt_hard_positive.jsonl
data/processed/round2_human_hardneg_seed.jsonl
data/processed/round2_llm_hardpos_seed.jsonl
data/processed/round2_teacher_like_train.jsonl
data/processed/round2_teacher_like_dev.jsonl
data/processed/round2_teacher_like_report.json
```

验收结果：

| 检查项 | 结果 |
| --- | ---: |
| hard buckets covered | 9 |
| round2 dev rows | 1065 |
| round2 train additions | 3027 |
| dev minimum class share | 41.2% |
| poetry represented | yes |
| academic represented | yes |

round2 dev 分布：

| Label | Rows |
| --- | ---: |
| human | 626 |
| LLM | 439 |

Step7 在这个刻意变难的 dev set 上表现：

```text
F1 = 0.6220
confusion = [[602, 24], [230, 209]]
```

解释：Phase 1 成功暴露了 Step7 的假阴性区域，尤其是保守 ChatGPT 改写、
保留诗歌结构的改写、old-fiction style 改写和自然学术 paraphrase。

### 2.3 Phase 2：domain router 和 bucket thresholds

新增脚本：

```text
src/evaluation/assign_text_bucket.py
src/evaluation/tune_bucket_thresholds.py
src/evaluation/predict_bucket_routed_ensemble.py
```

使用的可解释 buckets：

```text
poetry_classical
poetry_freeverse
literary_old_prose
literary_short_fragment
academic_formal
general_prose
```

结果：

| Split | Step7 F1 | Bucket-routed F1 | 主要效果 |
| --- | ---: | ---: | --- |
| internal_test | 0.9564 | 0.9526 | 小幅退化 |
| round2_dev | 0.6220 | 0.7392 | hard-positive recall 明显提升 |
| teacher_test | 0.9133 | 0.9020 | 退化 |

结论：router 有诊断价值，也可以作为 stacker 特征，但不能作为独立最终系统。

### 2.4 Phase 3：stacking fusion

新增脚本：

```text
src/models/train_stacking_fusion.py
src/evaluation/predict_stacking_fusion.py
src/evaluation/compare_round2_candidates.py
src/evaluation/merge_prediction_branches.py
```

重要修正：

初版 smoke test 使用了 `generator` 作为特征，导致 internal-test 结果不真实地接近完美。
这个特征被视为泄漏并移除。最终 stacker 只使用可部署特征：

```text
p_tfidf
p_deberta_step7
p_ensemble_step7
probability disagreement features
text length / line / punctuation / archaic / academic marker features
rule-based bucket
```

最佳 Step7-only stacker：

| Split | Step7 F1 | Stacker F1 |
| --- | ---: | ---: |
| internal_test | 0.9564 | 0.9604 |
| round2_dev | 0.6220 | 0.6999 |
| teacher_test | 0.9133 | 0.9109 |

结论：轻量 stacker 改善了 internal-test 和 hard-dev 诊断，但没有超过 Step7 的
teacher-test 表现。

### 2.5 Phase 4：第三模型分支 RoBERTa

训练了 RoBERTa：

```text
outputs/models/round2_roberta_base
model_name = roberta-base
train = data/processed/round2_teacher_like_train.jsonl
valid = data/processed/lit_academic_poetry_valid.jsonl
test = data/processed/lit_academic_poetry_internal_test.jsonl
```

RoBERTa standalone：

| Split | F1 |
| --- | ---: |
| validation | 0.9434 |
| internal_test | 0.9262 |
| round2_dev | 0.6920 |
| teacher_test | 0.8219 |

加入 RoBERTa 后的最佳 stacker：

| Split | F1 | Confusion |
| --- | ---: | --- |
| internal_test | 0.9610 | `[[851, 33], [33, 814]]` |
| round2_dev | 0.7662 | `[[561, 65], [126, 313]]` |
| teacher_test | 0.8917 | `[[126, 24], [10, 140]]` |

结论：RoBERTa 提供了 hard-positive 信号，但把最终边界过度推向 LLM recall，
在 teacher-test 上造成了过多 human false positives。

### 2.6 Phase 5：最终候选比较

teacher-test 最终比较：

| Candidate | Accuracy | Correct | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| step7 | 0.9133 | 274 / 300 | 0.9133 | 13 | 13 |
| bucket_routed | 0.9000 | 270 / 300 | 0.9020 | 18 | 12 |
| stacker_step7 | 0.9100 | 273 / 300 | 0.9109 | 15 | 12 |
| roberta_single | 0.8267 | 248 / 300 | 0.8219 | 22 | 30 |
| stacker_with_roberta | 0.8867 | 266 / 300 | 0.8917 | 24 | 10 |

最终决策：严格路线下继续保留 Step7 作为最终系统。

## 3. 这一轮为什么没有达到预期

一句话总结：

```text
Round2 的优化不是完全无效，而是优化目标和最终 teacher-test 分布存在偏差。
它有效提升了 hard-dev 和 hard-positive recall，但最终测试更需要在修复 FN 的同时
严格保护 high-style human text，避免新增 FP。
```

### 3.1 95% 目标是在强基线上的窄幅突破

Step7 已经有 274 / 300 正确。要到 95%，需要 285 / 300，也就是净增 11 个正确样本。

这意味着：

1. 不能只修 false negatives。
2. 修复 3 个 FN 但新增 5 个 FP 是失败。
3. internal-test 或 hard-dev F1 提升不等于 teacher-test 准确率提升。
4. 任何轻微过度激进都会被 300 条 teacher-test 放大成明显退化。

### 3.2 Phase 0 已经证明简单阈值调优不够

Phase 0 中，单个已有预测文件即使用 teacher-test oracle threshold，也只能到约 0.9267。
已有模型族简单平均即使用 oracle threshold，也只能到约 0.9333。

这说明当前概率信号本身不够，不能靠继续调 `threshold` 或 `alpha` 达到 95%。
需要的是更独立的信号和更精细的错误保护。

### 3.3 Round2 dev 是有价值的显微镜，但不是最终考试分布

round2 dev 刻意强调残余 FN：

1. conservative ChatGPT rewrites，
2. poetry-preserving rewrites，
3. old-fiction style rewrites，
4. natural academic paraphrases。

这让它非常适合发现 Step7 漏掉哪些 LLM 文本，但也让模型选择压力偏向“更积极地判 LLM”。

teacher-test 中存在不少看起来 polished、literary、poetic、formal、academic 的人类文本。
如果候选模型过度奖励 hard-positive recall，就会把这些高质量 human text 误杀为 LLM。

### 3.4 核心失败模式：修 FN 的代价是新增 FP

Step7 的 teacher-test 错误是平衡的：

```text
FP = 13
FN = 13
```

Round2 候选模型大多是在修 FN，但同时制造更多 FP：

| Candidate | FP | FN | 相对 Step7 的效果 |
| --- | ---: | ---: | --- |
| step7 | 13 | 13 | baseline |
| bucket_routed | 18 | 12 | 修 1 个 FN，新增 5 个 FP |
| stacker_step7 | 15 | 12 | 修 1 个 FN，新增 2 个 FP |
| stacker_with_roberta | 24 | 10 | 修 3 个 FN，新增 11 个 FP |

这就是第二轮失败的核心。新系统确实抓到了更多 LLM positive，但 human FP 代价更大。

### 3.5 RoBERTa 提供了新信号，但没有稳定转化为最终泛化

RoBERTa 在 round2_dev 和 stacker 特征中有价值，但 standalone internal-test F1 只有 0.9262。
它更像一个能发现 hard-positive 的诊断分支，而不是足够稳的最终分支。

问题包括：

1. 训练分布偏向 round2 hard-positive。
2. 对 high-style human text 的保护不足。
3. 和 stacker 结合后，把边界推得过于激进。
4. 其新增 recall 没有被相应的 precision guard 约束住。

### 3.6 stacker 协议还不够抗分布偏移

最终 stacker 移除了泄漏特征，但它仍是轻量版本：

1. 没有使用 out-of-fold base predictions。
2. meta-training 分布仍然偏向 round2 hard-positive。
3. 没有单独的 hard-negative mirror dev set 来保护 human FP。
4. 优化目标更像普通 F1，而不是带 FP 预算约束的目标。

所以它能提升 internal 和 hard-dev，但 teacher-test 不买账。

### 3.7 第二轮数据增强对 hard positives 更充分，对 hard negatives 仍不够

Phase 1 有 human hard negatives，但从结果看，最强的新学习信号仍是 LLM hard positives。
下一轮需要更明确的 mirror set：

```text
每一种 hard LLM positive 风格，都要配一组长度、领域、风格强度相近的人类 hard negative。
```

否则模型会学成“这种风格像 LLM”，而不是“这种风格本身模糊，需要更多证据”。

## 4. 下一轮核心假设

Round3 不应当是“更激进的 Round2”。它应当是 precision-guarded repair round。

核心假设：

```text
冲 95% 的关键不是最大化 hard-dev F1，而是在 hard-negative mirror set 和约束式融合下，
只修复那些有强证据的 false negatives，同时严格限制新增 false positives。
```

换句话说：

1. Step7 仍应作为默认决策。
2. 新模型只负责定向 override。
3. 每个 override 区域都必须证明不会明显增加 human FP。

## 5. Round3 总体路线

建议 Round3 分为六个阶段：

| Phase | 目标 | 主要输出 |
| --- | --- | --- |
| A. Error-delta audit | 分析 Round2 候选修复了哪些样本、又新错了哪些样本 | `outputs/round3/error_delta_audit.*` |
| B. Hard-negative mirror set | 构造和 hard LLM positives 对称的人类高风险负例 | `data/processed/round3_precision_guard_*.jsonl` |
| C. Balanced third branch | 训练 ELECTRA 或重新加权的第三分支 | `outputs/models/round3_electra_base/` |
| D. OOF stacking | 用 out-of-fold stacking 替代轻量 direct stacker | `outputs/models/round3_oof_stacker/` |
| E. Precision-guarded routing | 用约束阈值和 override 规则保护 human FP | `outputs/models/round3_precision_guard/` |
| F. Final comparison | 只有通过所有预检查才进行最终 teacher-test 比较 | `outputs/evaluation/round3_final_comparison.md` |

推荐执行顺序：

```text
Phase A -> Phase B -> Phase C
                 \-> Phase D -> Phase E -> Phase F
```

## 6. Phase A：Round2 error-delta audit

### 目标

先不要训练新模型。第一步要精确回答：

1. Round2 哪些候选真正修复了 Step7 的 FN？
2. 哪些候选把 Step7 原本正确的人类样本改错成 FP？
3. 新增 FP 是否集中在 poetry、high-style prose、academic 或 short fragments？
4. RoBERTa 是否修复了其他模型修不了的样本？
5. 是否存在安全的 override 区域？

### 建议新增脚本

```text
src/evaluation/round3_error_delta_audit.py
```

### 输入

使用已有候选预测：

```text
step7
bucket_routed
stacker_step7
roberta_single
stacker_with_roberta
```

teacher-test 标签只用于诊断，不用于后续调参。

### 输出

```text
outputs/round3/error_delta_audit.csv
outputs/round3/error_delta_audit.md
outputs/round3/error_delta_by_bucket.json
```

建议字段：

```text
id
label
text
rough_domain
bucket
step7_pred
step7_prob
candidate_preds
candidate_probs
fixed_by_candidates
broken_by_candidates
is_step7_fp
is_step7_fn
is_new_fp
is_new_fn
notes
```

### 验收标准

进入 Phase B/C 前，应至少明确：

1. 3 类可被候选修复的 FN 模式。
2. 3 类必须保护的新增 FP 模式。
3. 是否存在高置信、低风险的 override 规则雏形。

如果没有安全修复区域，下一轮重点应放在数据，而不是继续堆模型。

## 7. Phase B：hard-negative mirror set

### 目标

构造一个比 Round2 更平衡的 teacher-like dev/training set。Round2 强调 hard LLM positives，
Round3 必须同等强调 hard human negatives。

### 核心原则

每个 hard LLM positive bucket 都要有对应的人类 mirror bucket：

| Hard LLM positive | Human mirror |
| --- | --- |
| conservative literary rewrite | polished human literary prose |
| old-fiction style rewrite | public-domain old fiction original passages |
| archaic poetry rewrite | real archaic/classical poetry |
| free-verse rewrite | real modern free verse or lyrical fragments |
| natural academic paraphrase | human formal academic prose |
| short LLM fragment | human short reflective fragment |

### 新数据目标

建议至少新增：

| Bucket | Target rows | 目的 |
| --- | ---: | --- |
| human_poetry_classical_mirror | 400-600 | 保护 classical poetry 不被误判 LLM |
| human_poetry_freeverse_mirror | 400-600 | 保护 free verse / lyrical text |
| human_literary_old_prose_mirror | 400-600 | 保护 old-fiction human prose |
| human_ornate_literary_mirror | 400-600 | 保护 polished prose |
| human_formal_academic_mirror | 400-600 | 保护 academic human text |
| human_short_fragment_mirror | 200-400 | 保护短人类片段 |

同时可以继续增加 LLM hard positives，但比例要受控：

```text
new hard human negatives : new hard LLM positives >= 1 : 1
```

如果有生成器资源，LLM positives 不应只来自 ChatGPT：

```text
ChatGPT
DeepSeek
Gemini
Doubao
Claude, if available
```

### 建议新增或扩展脚本

```text
src/data/build_round3_precision_guard_set.py
```

可复用：

```text
src/data/build_round2_teacher_like_set.py
src/data/prepare_round2_hard_positive_prompts.py
src/data/generate_chatgpt_rewrites.py
```

### 输出

```text
data/processed/round3_hard_negative_mirror_source.jsonl
data/processed/round3_llm_hardpos_multi_generator_seed.jsonl
data/processed/round3_precision_guard_train.jsonl
data/processed/round3_precision_guard_dev.jsonl
data/processed/round3_precision_guard_report.json
```

### 验收标准

1. 新增 hard human negatives 至少 1800 条。
2. 如果生成器条件允许，新增 hard LLM positives 至少 1500 条。
3. `round3_precision_guard_dev` 中任一类别占比不低于 45%。
4. 每个高风险 domain 同时有人类样本和 LLM 样本。
5. 人工抽查至少 80 条，排除 prompt leakage、teacher-test 近重复和低质量文本。
6. Step7 在该集合上不能出现离谱的人类 FP 崩塌；如果崩塌，说明数据可能过于 adversarial 或标注不稳。

## 8. Phase C：balanced third branch

### 目标

训练一个真正互补的第三分支，修 hard positives，但不牺牲 high-style human precision。

### 推荐模型顺序

1. `google/electra-base-discriminator`
2. reweighted `roberta-base`
3. char n-gram / stylometry-only lightweight branch
4. 更大的 Transformer 只作为最后选择

优先 ELECTRA，因为它的 discriminative pretraining 可能带来不同于 DeBERTa 和 RoBERTa 的错误模式。

### 训练数据策略

不要只用 Round2 hard-positive-heavy 数据。建议混合：

```text
original train data
round2 teacher-like train
round3 precision-guard train
```

建议采样或加权：

| Sample type | Suggested weight |
| --- | ---: |
| original balanced data | 1.0 |
| round2 hard LLM positives | 1.0 |
| round3 hard human negatives | 1.5 |
| poetry/freeverse human mirrors | 1.5-2.0 |
| academic human mirrors | 1.2-1.5 |

如果当前训练脚本不支持权重，可新增：

```text
src/models/train_weighted_transformer.py
```

或扩展：

```text
src/models/train_deberta.py
```

建议支持参数：

```text
--sample_weight_field
--class_weight
--domain_weight_json
--balanced_sampler
```

### ELECTRA 命令模板

```powershell
.\.venv\Scripts\python.exe src\models\train_deberta.py `
  --train data\processed\round3_precision_guard_train.jsonl `
  --valid data\processed\lit_academic_poetry_valid.jsonl `
  --test data\processed\lit_academic_poetry_internal_test.jsonl `
  --output_dir outputs\models\round3_electra_base `
  --model_name google/electra-base-discriminator `
  --max_length 512 `
  --batch_size 4 `
  --eval_batch_size 8 `
  --gradient_accumulation_steps 2 `
  --learning_rate 1e-5 `
  --epochs 3
```

### 晋级标准

第三分支进入 fusion 前必须满足：

1. internal-test F1 至少 0.94，最好 0.95+。
2. round3 precision-dev 上 human FP 明显低于 Round2 RoBERTa。
3. 能修复一部分 Step7 修不了的 hard positives。
4. 与 Step7 的错误重合低于 DeBERTa-family 重合。
5. 不在 poetry / high-style prose 上制造明显 FP spike。

如果它只提升 hard-dev recall 但 human FP 很高，只能作为诊断特征，不能作为最终决策分支。

## 9. Phase D：out-of-fold stacking

### 目标

用 OOF stacking 替代 Round2 的 direct lightweight stacker。

### 原因

Round2 stacker 的问题不是实现完全无效，而是它抗分布偏移能力不够。
OOF stacking 能让 meta-model 看到更真实的 base-model 预测分布，减少在小 meta set 上过拟合。

### 建议新增脚本

```text
src/models/train_oof_base_predictions.py
src/models/train_round3_oof_stacker.py
src/evaluation/predict_round3_oof_stacker.py
```

### 推荐特征

使用：

```text
p_tfidf
p_deberta_step7
p_step7_ensemble
p_round3_electra
p_roberta_round2, optional
abs(p_tfidf - p_deberta_step7)
abs(p_step7_ensemble - p_round3_electra)
max probability
min probability
probability entropy
text length features
linebreak features
punctuation features
archaic marker features
academic marker features
bucket one-hot
```

不要使用：

```text
label-derived fields
generator
split name
source fields that directly identify class
teacher-test-derived fields
```

### Meta-model 顺序

1. regularized Logistic Regression。
2. shallow HistGradientBoostingClassifier。
3. calibrated Linear SVM / SGDClassifier。
4. 如果需要，再做小型 rule-guarded ensemble。

### 目标函数

不要只最大化普通 F1。建议使用约束目标：

```text
maximize recall on hard LLM positives
subject to:
  hard human negative FP <= Step7 FP + small tolerance
  poetry/freeverse human FP <= Step7 FP
  academic human FP <= Step7 FP
  internal-test F1 >= Step7 F1
```

### 验收标准

OOF stacker 只有在以下条件同时满足时才晋级：

1. internal-test F1 >= 0.960。
2. round2_dev 或 round3_dev 相对 Step7 明显改善。
3. hard-negative mirror set 上 FP 不高于 Step7，或只极小增加。
4. 新增 FP 少于 Round2 stacker/RoBERTa。
5. 能解释每类 override 为什么安全。

## 10. Phase E：precision-guarded routing

### 目标

不要让新模型全局接管 Step7，而是让它们作为定向修错器。

### 推荐决策设计

默认保留 Step7：

```text
final_pred = step7_pred
```

只有满足强证据时才允许 human -> LLM override：

```text
Step7 says human
AND new branch probability is high
AND OOF stacker probability is high
AND bucket is not in a high-risk human bucket
```

高风险 human buckets：

```text
poetry_classical
poetry_freeverse
literary_old_prose
ornate_literary_prose
academic_formal
short_fragment
```

这些 bucket 中需要更强约束：

```text
two or more branches agree strongly
probability disagreement is low
text is not extremely short
hard-negative guard threshold is satisfied
```

### 建议新增脚本

```text
src/evaluation/tune_precision_guard_rules.py
src/evaluation/predict_precision_guarded_ensemble.py
```

### 调参数据

只能使用：

```text
validation
round2_teacher_like_dev
round3_precision_guard_dev
internal-test diagnostics after fixed decisions
```

不要用 teacher-test 标签调参。

### 验收标准

precision-guarded candidate 需要满足：

1. hard LLM positives 上 FN 低于 Step7。
2. hard human negatives 上 FP 不增加或只极小增加。
3. round3 dev balanced accuracy 高于 Step7。
4. override 规则足够可解释，能写进报告。

如果规则越来越复杂但收益不稳，优先保留 Step7。

## 11. Phase F：最终比较

### 候选集合

至少比较：

| Candidate | 目的 |
| --- | --- |
| step7 | 当前 strict baseline |
| round2_bucket_routed | 上轮 router 参考 |
| round2_stacker_step7 | 上轮 stacker 参考 |
| round2_stacker_with_roberta | 上轮 aggressive recall 参考 |
| round3_electra_single | 新第三分支 |
| round3_oof_stacker | 更严格 fusion |
| round3_precision_guard | 最终定向 override candidate |

### 必需报告

```text
outputs/evaluation/round3_internal_comparison.md
outputs/evaluation/round3_round2_dev_comparison.md
outputs/evaluation/round3_precision_guard_dev_comparison.md
outputs/evaluation/round3_final_teacher_comparison.md
outputs/evaluation/round3_error_overlap_matrix.csv
docs/ROUND3_RESULTS_SUMMARY.md
```

### 晋级门槛

只有通过 teacher-test 之前的所有检查，才值得跑最终 teacher-test：

| Gate | Requirement |
| --- | --- |
| internal-test F1 | >= 0.960，最好 >= 0.963 |
| round2 hard-dev | 相对 Step7 改善 |
| round3 precision-dev | 相对 Step7 改善 |
| hard human FP | 不高于 Step7，或只极小增加且 FN 大幅下降 |
| poetry/freeverse human FP | 不退化 |
| academic human FP | 不退化 |
| error overlap | 与 Step7 的重合低于 Round2 候选 |
| interpretability | 能清楚解释修复了什么 |

最终 teacher-test 目标：

```text
teacher-test correct >= 285 / 300
teacher-test accuracy >= 0.9500
```

如果没有候选在预检查中击败 Step7，不要强推新模型，继续保留 Step7。

## 12. 实施 checklist

### Step 1：Round2 delta audit

交付：

```text
src/evaluation/round3_error_delta_audit.py
outputs/round3/error_delta_audit.md
outputs/round3/error_delta_audit.csv
```

验收：

```text
identified candidate-fixed FN IDs
identified candidate-induced FP IDs
identified high-risk buckets for guard rules
```

### Step 2：Round3 precision-guard data

交付：

```text
src/data/build_round3_precision_guard_set.py
data/processed/round3_precision_guard_train.jsonl
data/processed/round3_precision_guard_dev.jsonl
data/processed/round3_precision_guard_report.json
```

验收：

```text
hard human negatives >= 1800
hard LLM positives >= 1500 if possible
class minimum share >= 45%
manual spot check >= 80 rows
```

### Step 3：ELECTRA 或 balanced third branch

交付：

```text
outputs/models/round3_electra_base/
outputs/predictions/round3_electra_*_predictions.jsonl
outputs/evaluation/round3_electra_report.md
```

验收：

```text
internal-test F1 >= 0.94 minimum
hard-negative FP safer than Round2 RoBERTa
non-identical errors versus Step7
```

### Step 4：OOF stacker

交付：

```text
src/models/train_oof_base_predictions.py
src/models/train_round3_oof_stacker.py
src/evaluation/predict_round3_oof_stacker.py
outputs/models/round3_oof_stacker/
outputs/evaluation/round3_oof_stacker_report.md
```

验收：

```text
internal-test F1 >= 0.960
round3 precision-dev improves
hard human FP does not increase
```

### Step 5：precision-guarded final candidate

交付：

```text
src/evaluation/tune_precision_guard_rules.py
src/evaluation/predict_precision_guarded_ensemble.py
outputs/models/round3_precision_guard/
outputs/predictions/round3_precision_guard_submission.json
```

验收：

```text
fewer FN on hard LLM positives
no uncontrolled FP increase on hard human negatives
clear override rules
```

### Step 6：最终报告与交接

交付：

```text
docs/ROUND3_RESULTS_SUMMARY.md
PROJECT_REPORT.md update
README.md update
outputs/evaluation/round3_final_teacher_comparison.md
```

验收：

```text
if target reached: document final model and exact reproduction command
if target not reached: document remaining error buckets and next data/model need
```

## 13. 报告写法建议

后续项目报告中应诚实表达：

1. Round2 没有达到 95%。
2. Round2 的价值在于建立诊断基础设施，并暴露 teacher-like distribution shift。
3. Step7 仍是最终模型，因为它最好地平衡了 human precision 和 LLM recall。
4. Router、stacker、RoBERTa 有价值，但主要价值是告诉我们哪些方向危险。
5. 下一轮真正需要的是 hard-negative mirror data、OOF stacking 和 precision guard。

不要因为 internal-test 或 hard-dev 提升，就声称模型最终泛化更好。teacher-test 结果已经说明
Round2 候选没有真正超过 Step7。

## 14. 给下一轮对话的建议 prompt

```text
请先阅读 docs/ROUND2_POSTMORTEM_AND_ROUND3_PLAN.md，
再查看 docs/ROUND2_RESULTS_SUMMARY.md 和 docs/ROUND2_PHASE0_DIAGNOSTICS.md。

我们已经完成 Round2，但没有达到 95% teacher-test accuracy。
当前严格路线最终模型仍是 Step7，teacher-test 为 274/300。
请从 Round3 Phase A 开始：先做 Round2 error-delta audit，
然后构造 hard-negative mirror set，训练 balanced ELECTRA 或其他第三分支，
实现 OOF stacking，并做 precision-guarded routing。
不要用 teacher-test 标签调参。
```
