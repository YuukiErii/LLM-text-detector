# Round2 Postmortem And Round3 Optimization Plan

Updated: 2026-05-21

This handoff summarizes what Round2 completed, why it did not reach 95%
teacher-test accuracy, and how Round3 should proceed. Read this together with:

```text
docs/rounds/round2_95_optimization_plan_2026-05-21.md
docs/rounds/round2_phase0_diagnostics_2026-05-21.md
docs/rounds/round2_results_summary_2026-05-22.md
PROJECT_REPORT.md
README.md
```

## 1. Current Final Decision

Under the strict generalization route, the recommended final system is still:

```text
DeBERTa:   outputs/models/deberta_lit_academic_poetry_step7_combined
TF-IDF:    outputs/models/tfidf_lit_academic_poetry
alpha:     0.5
threshold: 0.55
```

Teacher-test result:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9133 |
| Correct | 274 / 300 |
| Precision | 0.9133 |
| Recall | 0.9133 |
| F1 | 0.9133 |
| FP | 13 |
| FN | 13 |
| Confusion | `[[137, 13], [13, 137]]` |

The 95% target means:

```text
target correct = 285 / 300
current correct = 274 / 300
needed net gain = +11 correct examples
current errors = 26
maximum errors at 95% = 15
```

Round2 did not reach 95%. The final Round2 submission alias was:

```text
outputs/predictions/round2_final_submission.json
```

## 2. What Round2 Completed

Round2 executed the planned Phase 0-5 route without using teacher-test labels
for training, threshold selection, stacker training, router tuning, or model
selection. Teacher-test remained a final diagnostic.

### 2.1 Phase 0: Baseline And Ceiling Diagnostics

New scripts:

```text
src/evaluation/export_error_ledger.py
src/evaluation/round2_threshold_family_diagnostics.py
```

Local diagnostic outputs:

```text
outputs/round2/error_ledger_teacher_step7.csv
outputs/round2/error_ledger_teacher_step7.jsonl
outputs/round2/existing_family_threshold_report.md
outputs/round2/existing_family_threshold_report.json
```

Key findings:

| Diagnostic | Result |
| --- | ---: |
| Step7 teacher-test residual errors | 26 = 13 FP + 13 FN |
| Near-boundary errors | 9 / 26 |
| Confidently wrong errors | 17 / 26 |
| Best single existing-run oracle threshold | about 0.9267 |
| Best simple average oracle | about 0.9333 |

Interpretation: the remaining errors are not primarily a global-threshold
problem. Many are confidently wrong, so the path to 95% requires new data,
new signal, or safer constrained fusion.

### 2.2 Phase 1: Teacher-Like Development Data

Round2 constructed a residual-focused teacher-like dataset rather than generic
augmentation.

Main data outputs:

```text
data/processed/round2_human_hardneg_source.jsonl
data/processed/rewrite_prompts_round2_chatgpt_hard_positive.jsonl
data/processed/round2_human_hardneg_seed.jsonl
data/processed/round2_llm_hardpos_seed.jsonl
data/processed/round2_teacher_like_train.jsonl
data/processed/round2_teacher_like_dev.jsonl
data/processed/round2_teacher_like_report.json
```

Acceptance summary:

| Check | Result |
| --- | ---: |
| Hard buckets covered | 9 |
| Round2 dev rows | 1065 |
| Round2 train additions | 3027 |
| Dev minimum class share | 41.2% |
| Poetry represented | yes |
| Academic represented | yes |

Round2 dev distribution:

| Label | Rows |
| --- | ---: |
| Human | 626 |
| LLM | 439 |

Step7 on this deliberately hard dev set:

```text
F1 = 0.6220
confusion = [[602, 24], [230, 209]]
```

The dataset successfully exposed Step7's false-negative region, especially
conservative ChatGPT rewrites, poetry-structure-preserving rewrites,
old-fiction-style rewrites, and natural academic paraphrases.

### 2.3 Phase 2: Domain Router And Bucket Thresholds

New scripts:

```text
src/evaluation/assign_text_bucket.py
src/evaluation/tune_bucket_thresholds.py
src/evaluation/predict_bucket_routed_ensemble.py
```

Interpretable buckets:

```text
poetry_classical
poetry_freeverse
literary_old_prose
literary_short_fragment
academic_formal
general_prose
```

Results:

| Split | Step7 F1 | Bucket-routed F1 | Main effect |
| --- | ---: | ---: | --- |
| internal_test | 0.9564 | 0.9526 | slight regression |
| round2_dev | 0.6220 | 0.7392 | much higher hard-positive recall |
| teacher_test | 0.9133 | 0.9020 | regression |

Conclusion: the router is diagnostically useful and may be a stacker feature,
but it is not a final system.

### 2.4 Phase 3: Stacking Fusion

New scripts:

```text
src/models/train_stacking_fusion.py
src/evaluation/predict_stacking_fusion.py
src/evaluation/compare_round2_candidates.py
src/evaluation/merge_prediction_branches.py
```

Important correction: the first smoke test used `generator` as a feature, which
made internal-test results unrealistically strong. It was treated as leakage
and removed. The final stacker used deployable features only:

```text
p_tfidf
p_deberta_step7
p_ensemble_step7
probability disagreement features
text length / line / punctuation / archaic / academic marker features
rule-based bucket
```

Best Step7-only stacker:

| Split | Step7 F1 | Stacker F1 |
| --- | ---: | ---: |
| internal_test | 0.9564 | 0.9604 |
| round2_dev | 0.6220 | 0.6999 |
| teacher_test | 0.9133 | 0.9109 |

Conclusion: the stacker improved internal-test and hard-dev diagnostics but did
not beat Step7 on teacher-test.

### 2.5 Phase 4: Third Model Branch, RoBERTa

RoBERTa training output:

```text
outputs/models/round2_roberta_base
model_name = roberta-base
train = data/processed/round2_teacher_like_train.jsonl
valid = data/processed/lit_academic_poetry_valid.jsonl
test = data/processed/lit_academic_poetry_internal_test.jsonl
```

RoBERTa standalone:

| Split | F1 |
| --- | ---: |
| validation | 0.9434 |
| internal_test | 0.9262 |
| round2_dev | 0.6920 |
| teacher_test | 0.8219 |

Best stacker with RoBERTa:

| Split | F1 | Confusion |
| --- | ---: | --- |
| internal_test | 0.9610 | `[[851, 33], [33, 814]]` |
| round2_dev | 0.7662 | `[[561, 65], [126, 313]]` |
| teacher_test | 0.8917 | `[[126, 24], [10, 140]]` |

Conclusion: RoBERTa contributed hard-positive signal but shifted the final
boundary too far toward LLM recall and introduced too many human false
positives.

### 2.6 Phase 5: Final Candidate Comparison

Teacher-test comparison:

| Candidate | Accuracy | Correct | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| step7 | 0.9133 | 274 / 300 | 0.9133 | 13 | 13 |
| bucket_routed | 0.9000 | 270 / 300 | 0.9020 | 18 | 12 |
| stacker_step7 | 0.9100 | 273 / 300 | 0.9109 | 15 | 12 |
| roberta_single | 0.8267 | 248 / 300 | 0.8219 | 22 | 30 |
| stacker_with_roberta | 0.8867 | 266 / 300 | 0.8917 | 24 | 10 |

Final decision: keep Step7 as the final strict system.

## 3. Why Round2 Missed The Target

Round2 was not useless; the mismatch was between the locally constructed hard
development surface and the final teacher-test distribution.

The key lessons are:

1. The teacher-like dev set was too aggressive toward hard LLM positives.
2. Router and stacker methods improved hard-positive recall but exposed human
   false-positive risk.
3. RoBERTa and related transformer branches made errors correlated with
   DeBERTa, rather than adding enough independent signal.
4. Threshold-only and alpha-only searches have already reached their practical
   ceiling.
5. The remaining teacher-test mistakes require more precise residual data and a
   safer local override design.

## 4. Round3 Plan

Round3 should stop searching for a global replacement. It should preserve
Step7 as the default prediction and only attempt a constrained local repair:

```text
default = Step7
only consider Step7-human -> LLM overrides
require strong non-teacher evidence before teacher-test
```

Recommended sequence:

1. Audit Round2 error deltas to separate fixed false negatives from induced
   false positives.
2. Build a precision-guard development set centered on unsafe human mirrors and
   safe hard positives.
3. Train one heterogeneous branch only if it is evaluated as a signal source,
   not a global classifier.
4. Train an OOF stacker without leakage features.
5. Search precision-guarded local override rules.
6. Reject any candidate that improves hard-positive recall by increasing
   hard-negative or internal-test false positives.
7. Run teacher-test only after the non-teacher gate is frozen.

## 5. Promotion Rule

A Round3 candidate may be promoted only if it satisfies all of:

```text
internal_test F1 >= Step7 minus a tiny tolerance
internal_test FP <= Step7 FP + small tolerance
hard-negative FP <= Step7
hard-positive fixed FN is meaningfully positive
teacher-test was not used for tuning
```

If no candidate clears the gate, Step7 remains the final baseline and Round3 is
reported as a diagnostic stage.
