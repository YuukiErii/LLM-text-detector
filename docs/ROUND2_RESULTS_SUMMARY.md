# Round2 Results Summary

Updated: 2026-05-21

This document summarizes the second-round optimization work that started after
Phase 0. The target was:

```text
95% teacher-test accuracy = at least 285 / 300 correct
```

The strict-route rule was preserved: teacher-test labels were not used to train
models, choose thresholds, choose stacker parameters, or select router settings.
Teacher-test results below are final diagnostics only.

## 1. Phase 1: Teacher-Like Development Data

Phase 1 built new residual-error-oriented data instead of generic expansion.

Generated / selected artifacts:

```text
data/processed/round2_human_hardneg_source.jsonl
data/processed/rewrite_prompts_round2_chatgpt_hard_positive.jsonl
data/processed/round2_human_hardneg_seed.jsonl
data/processed/round2_llm_hardpos_seed.jsonl
data/processed/round2_teacher_like_train.jsonl
data/processed/round2_teacher_like_dev.jsonl
data/processed/round2_teacher_like_report.json
```

Final Phase 1 acceptance:

| Check | Result |
| --- | ---: |
| hard buckets covered | 9 |
| round2 dev rows | 1065 |
| round2 train additions | 3027 |
| dev minimum class share | 41.2% |
| poetry represented | yes |
| academic represented | yes |

Round2 dev distribution:

| Label | Rows |
| --- | ---: |
| human | 626 |
| LLM | 439 |

The new dev set was intentionally hard. The Step7 ensemble dropped to:

```text
round2 teacher-like dev F1 = 0.6220
confusion = [[602, 24], [230, 209]]
```

This confirms that the new data targets the residual false-negative region:
conservative ChatGPT-style rewrites, poetry-preserving rewrites, old-fiction
style, and natural academic paraphrases.

## 2. Phase 2: Domain Router

Added scripts:

```text
src/evaluation/assign_text_bucket.py
src/evaluation/tune_bucket_thresholds.py
src/evaluation/predict_bucket_routed_ensemble.py
```

The router uses transparent text buckets:

```text
poetry_classical
poetry_freeverse
literary_old_prose
literary_short_fragment
academic_formal
general_prose
```

Tuning used validation + round2 teacher-like dev.

| Split | Step7 F1 | Bucket-routed F1 | Main effect |
| --- | ---: | ---: | --- |
| internal_test | 0.9564 | 0.9526 | slight regression |
| round2_dev | 0.6220 | 0.7392 | much better hard-positive recall |
| teacher_test | 0.9133 | 0.9020 | regression |

Conclusion: bucket routing is useful diagnostically and as a feature idea, but
it should not be promoted as a standalone final system.

## 3. Phase 3: Stacking Fusion

Added scripts:

```text
src/models/train_stacking_fusion.py
src/evaluation/predict_stacking_fusion.py
src/evaluation/compare_round2_candidates.py
src/evaluation/merge_prediction_branches.py
```

A first smoke test exposed a leakage bug: using `generator` as a feature made
internal-test performance unrealistically perfect. The stacker was corrected to
use only deployable features:

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

Conclusion: lightweight stacking improved internal-test and round2-dev
diagnostics, but did not beat Step7 on teacher-test final evaluation.

## 4. Phase 4: Third Branch

Trained RoBERTa:

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

RoBERTa was weak as a standalone model, but it was useful as a third probability
feature inside the stacker.

Best stacker with RoBERTa:

| Split | F1 | Confusion |
| --- | ---: | --- |
| internal_test | 0.9610 | `[[851, 33], [33, 814]]` |
| round2_dev | 0.7662 | `[[561, 65], [126, 313]]` |
| teacher_test | 0.8917 | `[[126, 24], [10, 140]]` |

Conclusion: the third branch improved hard-bucket development performance and
internal-test F1, but it over-shifted the teacher-test operating point toward
LLM recall and caused too many human false positives.

## 5. Final Candidate Comparison

Internal-test comparison:

| Candidate | Accuracy | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| step7 | 0.9578 | 0.9564 | 27 | 46 |
| bucket_routed | 0.9538 | 0.9526 | 37 | 43 |
| stacker_step7 | 0.9613 | 0.9604 | 32 | 35 |
| roberta_single | 0.9295 | 0.9262 | 40 | 82 |
| stacker_with_roberta | 0.9619 | 0.9610 | 33 | 33 |

Round2 teacher-like dev comparison:

| Candidate | Accuracy | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| step7 | 0.7615 | 0.6220 | 24 | 230 |
| bucket_routed | 0.7906 | 0.7392 | 100 | 123 |
| stacker_step7 | 0.7850 | 0.6999 | 57 | 172 |
| roberta_single | 0.7869 | 0.6920 | 43 | 184 |
| stacker_with_roberta | 0.8207 | 0.7662 | 65 | 126 |

Final teacher-test comparison:

| Candidate | Accuracy | Correct | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| step7 | 0.9133 | 274 / 300 | 0.9133 | 13 | 13 |
| bucket_routed | 0.9000 | 270 / 300 | 0.9020 | 18 | 12 |
| stacker_step7 | 0.9100 | 273 / 300 | 0.9109 | 15 | 12 |
| roberta_single | 0.8267 | 248 / 300 | 0.8219 | 22 | 30 |
| stacker_with_roberta | 0.8867 | 266 / 300 | 0.8917 | 24 | 10 |

## 6. Final Recommendation

The final recommended strict-route system remains the original Step7 ensemble:

```text
DeBERTa:  outputs/models/deberta_lit_academic_poetry_step7_combined
TF-IDF:   outputs/models/tfidf_lit_academic_poetry
alpha:    0.5
threshold: 0.55
```

Submission artifact:

```text
outputs/predictions/round2_final_submission.json
```

Teacher-test result:

```text
accuracy = 0.9133
correct = 274 / 300
f1 = 0.9133
confusion = [[137, 13], [13, 137]]
```

Round2 did not reach 95%. The best strict candidate improved internal-test and
hard-bucket development metrics, but teacher-test final evaluation showed that
the new hard-positive emphasis overcorrected the boundary and hurt human false
positives.

## 7. What To Do Next

The next attempt should not tune teacher-test thresholds. The best evidence
points to these next steps:

1. Add a held-out teacher-like set with more human high-style prose and poetry,
   not only LLM conservative rewrites.
2. Train the third branch with a more balanced objective or sample weights so it
   does not over-learn the round2 LLM hard-positive distribution.
3. Try ELECTRA as the next heterogeneous branch; RoBERTa helped dev but did not
   transfer well enough.
4. Use out-of-fold base predictions for stacking instead of training the stacker
   on direct validation/dev predictions.
5. Keep the Step7 ensemble as the final submission unless a future candidate
   improves both internal-test and teacher-test-like dev without increasing
   human false positives.
