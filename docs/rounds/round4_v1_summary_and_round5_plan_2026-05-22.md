# Round4 v1 Summary And Round5 95% Optimization Plan

Date: 2026-05-22

This document follows
`docs/rounds/round3_cross_round_review_and_95_route_2026-05-22.md`. It records
the Round4 v1 residual-repair result and the directly executable Round5 plan.

The overall target remained:

```text
teacher-test 95% accuracy = at least 285 / 300 correct
current strict final Step7 = 274 / 300 correct
needed net gain = +11 correct samples
```

The strict boundary also remained unchanged:

```text
data/raw/teacher_test.json must not be used for training, threshold selection,
model selection, stacker training, router tuning, or guard calibration.
```

No Round4 v1 candidate passed the non-teacher precheck, so Step7 remained the
final system.

## 1. Round4 Decision

Round4 v1 completed four major pieces:

1. residual data rebuild,
2. human-style guard,
3. weighted DeBERTa residual retraining,
4. residual override gate search.

Decision:

1. The residual data build was successful and reusable.
2. The Round4 DeBERTa branch learned genuine hard-positive signal missing from
   Step7.
3. As a global model, Round4 DeBERTa materially increased human false
   positives.
4. The human-style guard was too weak to make the branch safe.
5. The residual override search found no feasible non-empty rule.
6. Round4 v1 did not advance to teacher-test and was not promoted.

In short, Round4 v1 was a valuable training and diagnostic asset, not a
submission model.

## 2. Work Completed

### 2.1 Residual Dataset Rebuild

New script:

```text
src/data/build_round4_residual_dataset.py
```

Main outputs:

```text
data/processed/round4_hard_human_mirror_seed.jsonl
data/processed/round4_hard_llm_positive_seed.jsonl
data/processed/round4_residual_train.jsonl
data/processed/round4_residual_dev_hardpos.jsonl
data/processed/round4_residual_dev_hardneg.jsonl
data/processed/round4_residual_spotcheck.jsonl
data/processed/round4_residual_report.json
```

Key results:

| Item | Result |
| --- | ---: |
| Hard human mirrors | 4727 |
| Hard LLM positives | 3674 |
| Human:LLM hard ratio | 1.2866 |
| Round4 train rows | 17434 |
| Hard-positive dev rows | 500 |
| Hard-negative dev rows | 500 |
| Old-prose human mirrors | 800 |
| Poetry/freeverse human mirrors | 1500 |
| Natural academic human mirrors | 1000 |
| Teacher-test exact duplicates | 0 |

The data acceptance checks passed. The new `round4_bucket` field preserved the
source-known residual bucket while the existing `bucket` remained a text-feature
bucket. This avoided losing old-prose and rewrite-source information during
later analysis.

Remaining data shortfalls:

| Bucket | Shortfall |
| --- | ---: |
| human literary_short_fragment | 273 |
| LLM poetry_classical | 197 |
| LLM academic_formal | 228 |
| LLM literary_short_fragment | 201 |

These buckets became priority targets for the next round.

### 2.2 Human-Style Guard

New scripts:

```text
src/models/train_round4_human_style_guard.py
src/evaluation/predict_round4_human_style_guard.py
```

Main outputs:

```text
outputs/models/round4_human_style_guard/human_style_guard.pkl
outputs/models/round4_human_style_guard/human_style_guard_report.json
outputs/evaluation/round4_human_style_guard_report.md
```

Key metrics:

| Split | Meaning | Result |
| --- | --- | ---: |
| dev_hardpos_should_not_veto | LLM hard positives wrongly vetoed | 0.054 |
| dev_hardneg_should_veto | hard human protected | 0.214 |
| internal_test veto rate | internal-test protection rate | 0.0116 |

Interpretation: the guard was conservative and rarely vetoed LLM hard positives,
which is good. However, it protected only `21.4%` of hard-human dev examples,
so it was not strong enough to control Round4 DeBERTa's false-positive risk.

### 2.3 Weighted DeBERTa Retrain

Training output:

```text
outputs/models/round4_deberta_weighted_residual/
outputs/models/round4_deberta_weighted_residual/best_model
```

Training configuration:

```text
model_name = microsoft/deberta-v3-base
train = data/processed/round4_residual_train.jsonl
epochs = 3
learning_rate = 1e-5
sample_weight_field = sample_weight
class_weight = none
balanced_sampler = false
```

Metrics:

| Split | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 0.9543 | 0.9496 | 0.9574 | 0.9535 | 43 | 36 |
| internal_test | 0.9445 | 0.9321 | 0.9563 | 0.9441 | 59 | 37 |
| hardpos dev | 0.5660 | 1.0000 | 0.5660 | 0.7229 | 0 | 217 |

Compared with Step7:

| Split | Step7 | Round4 DeBERTa | Interpretation |
| --- | ---: | ---: | --- |
| internal_test F1 | 0.9564 | 0.9441 | global regression |
| hardpos recall | 0.4160 | 0.5660 | more hard LLM positives captured |
| hardneg FP | 26 | 53 | human FP risk doubled |

The branch found real hard-positive signal, but it pushed too many high-style
human texts toward the LLM class and could not replace Step7 globally.

### 2.4 Residual Override Gate

New script:

```text
src/evaluation/tune_round4_residual_override.py
```

Prediction scripts were patched to preserve `round4_bucket` and `round4_tag`:

```text
src/evaluation/predict_neural_model.py
src/evaluation/predict_ensemble.py
```

Main outputs:

```text
outputs/models/round4_residual_override/rules.json
outputs/models/round4_residual_override/residual_override_tuning_report.json
outputs/evaluation/round4_residual_override_tuning_report.md
```

Search result:

| Item | Result |
| --- | ---: |
| Aligned rows | 2731 |
| Candidate rules | 3601 |
| Feasible rules | 1 |
| Feasible non-empty override rules | 0 |
| Selected rule | disabled no-op baseline |

The selected rule made no overrides:

| Split | F1 | FP | FN | Overrides | Fixed Step7 FN | Induced FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| internal_test | 0.9564 | 27 | 46 | 0 | 0 | 0 |
| hardpos | 0.5876 | 0 | 292 | 0 | 0 | 0 |
| hardneg | 0.0000 | 26 | 0 | 0 | 0 | 0 |

The gate was strict, but correctly so: 95% requires net repair, not recall gains
purchased by extra human false positives.

## 3. Why Round4 v1 Was Not Promoted

Round4 did learn useful signal, but the signal could not be used safely.

Main blockers:

1. Round4 DeBERTa fixed some hard-positive false negatives while creating more
   hard-negative false positives.
2. The human-style guard was a generic high-style human protector, not a
   dedicated detector for Round4-induced false positives.
3. The override rule only saw probabilities, margins, buckets, and guard
   scores; these were insufficient to distinguish true Step7 false negatives
   from high-style human false positives.
4. The old-prose, classical-poetry, formal-academic, and short-fragment
   generation buckets remained under-covered.
5. Three rounds had already shown that global replacement and simple threshold
   moves damage Step7's FP/FN balance.

The next round therefore needed to learn which Round4 overrides are dangerous
before trying to repair more Step7 false negatives.

## 4. Round5 Goal

Recommended round name:

```text
Round5: FP-safe residual repair
```

Round5 should reuse Round4 data assets, but stop blindly expanding LLM recall.
It should train a flip guard and search for low-risk local overrides.

### 4.1 Non-Teacher Gates

Required gates:

| Gate | Required |
| --- | --- |
| internal_test F1 | >= 0.9544, preferably >= 0.9564 |
| internal_test FP | <= Step7 FP + 2, ideally <= Step7 |
| hardneg dev FP | <= 26 |
| hardpos dev recall | > 0.416, target >= 0.50 |
| non-empty override | required; no-op cannot promote |
| teacher-test leakage | exact duplicate = 0; no teacher labels in tuning |

Stretch target:

```text
hardpos fixed Step7 FN >= 40
hardneg induced FP = 0
internal induced FP <= 2
```

Only candidates meeting these gates should receive a teacher-test diagnostic.

### 4.2 Teacher-Test Goal

```text
minimum promotion: > 274 / 300
95% target:        >= 285 / 300
```

If a candidate only looks good on hardpos dev while worsening non-teacher human
precision, do not spend a teacher-test diagnostic on it.

## 5. Round5 Execution Plan

Round5 should proceed in this order:

1. Freeze Step7 and Round4 predictions.
2. Build a Step7-vs-Round4 flip ledger.
3. Label local override candidates as safe fixed-FN or unsafe induced-FP on
   non-teacher data only.
4. Train a flip guard on candidate-level features.
5. Search local override rules that preserve hard-negative and internal-test
   human precision.
6. Write a gate report.
7. Run teacher-test only if the gate report explicitly authorizes it.

Important data boundary:

```text
teacher-test may be read only for final diagnostic after freezing the rule.
It must not be used to choose thresholds, buckets, model weights, or guard
settings.
```

## 6. Final Round4 Recommendation

Keep these assets:

```text
round4 residual datasets
round4_bucket / round4_tag propagation
Round4 DeBERTa predictions as a signal branch
human-style guard diagnostics
residual override tuning report
```

Do not keep these as final-model components:

```text
Round4 DeBERTa global classifier
Round4 no-op residual override rule
human-style guard as the only safety mechanism
```

The public conclusion is conservative: Round4 improved diagnostic coverage but
did not produce a safer deployable model than Step7.
