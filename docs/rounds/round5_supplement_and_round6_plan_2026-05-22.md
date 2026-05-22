# Round5 Supplement And Round6 Optimization Plan

Date: 2026-05-22

This supplement expands `docs/rounds/round5_final_decision_2026-05-22.md` and
defines the next executable optimization plan. The headline conclusion is:

```text
The final model remains the Step7 ensemble.
Round5 passed non-teacher safety gates but made zero teacher-test overrides.
Round5 was a safe failure, not a damaging failure.
Round6 should learn the difference between safe overrides and induced false
positives instead of simply relaxing the guard.
```

Teacher-test remains diagnostic only:

```text
data/raw/teacher_test.json cannot be used for training, threshold selection,
guard calibration, router tuning, stacker training, or model selection.
Round6 may use aggregate observations from Round5 to form hypotheses, but it
must not use teacher-test text, labels, row ids, or row-level threshold
conditions in train/dev/selection.
```

## 0. Rounds 1-5 Overview

### 0.1 Strict Baseline And Target Gap

The strict final baseline is:

```text
DeBERTa:   outputs/models/deberta_lit_academic_poetry_step7_combined
TF-IDF:    outputs/models/tfidf_lit_academic_poetry
alpha:     0.5
threshold: 0.55
```

Teacher-test baseline:

| System | Correct / 300 | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step7 ensemble | 274 | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 13 | 13 |

95% requires:

```text
target correct = 285 / 300
current correct = 274 / 300
needed net gain = +11
current errors = 26 = 13 FP + 13 FN
```

No candidate should buy extra LLM recall by sacrificing high-style human
precision. The only defensible 95% route must net-fix errors while preserving
human precision.

### 0.2 Round1: Full Pipeline And Step7 Improvement

Round1 built the full project pipeline:

```text
human seeds -> LLM rewrites -> pair-safe split -> TF-IDF -> DeBERTa ->
DeBERTa + TF-IDF ensemble -> teacher-test final diagnostic
```

Original final ensemble:

| System | Correct / 300 | Accuracy | F1 | FP | FN | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| original final ensemble | 271 | 0.9033 | 0.9073 | 21 | 8 | `[[129, 21], [8, 142]]` |

Round1 then tested hard-negative humans, normalization, calibration,
controlled hard-negative quotas, ChatGPT hard positives, and poetry expansion.
The Step7 neural retrain became the strongest model:

| System | Correct / 300 | Accuracy | F1 | FP | FN | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Step7 ensemble | 274 | 0.9133 | 0.9133 | 13 | 13 | `[[137, 13], [13, 137]]` |

Round1 lesson:

```text
The real gain came from targeted train-side data plus neural retraining.
Threshold, alpha, and TF-IDF-only edits are useful diagnostics but not the main
route to 95%.
```

### 0.3 Round2: Teacher-Like Dev, Router, Stacker, RoBERTa

Round2 began with a Phase 0 diagnostic:

| Diagnostic | Result |
| --- | ---: |
| Step7 teacher-test residual errors | 26 = 13 FP + 13 FN |
| Best existing-family oracle threshold | about 0.9267 |
| Best simple-average oracle | about 0.9333 |

These oracle numbers are diagnostic only, not legal tuned results. They still
showed that the current model family was far from 95%.

Round2 teacher-like dev:

| Item | Result |
| --- | ---: |
| Round2 dev rows | 1065 |
| Train additions | 3027 |
| Step7 on round2 dev F1 | 0.6220 |
| Step7 round2 dev confusion | `[[602, 24], [230, 209]]` |

Final teacher-test comparison:

| Candidate | Correct / 300 | Accuracy | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| step7 | 274 | 0.9133 | 0.9133 | 13 | 13 |
| bucket_routed | 270 | 0.9000 | 0.9020 | 18 | 12 |
| stacker_step7 | 273 | 0.9100 | 0.9109 | 15 | 12 |
| roberta_single | 248 | 0.8267 | 0.8219 | 22 | 30 |
| stacker_with_roberta | 266 | 0.8867 | 0.8917 | 24 | 10 |

Round2 lesson:

```text
Hard-positive recall signal exists, but it induces human false positives on
teacher-test. Router, stacker, and RoBERTa cannot be used as global final
systems.
```

### 0.4 Round3: Precision-Guarded Repair

Round3 stopped searching for an aggressive global replacement and tried a local
repair:

```text
default prediction = Step7
override Step7-human -> LLM only under strong evidence
```

Round3 components:

```text
round3_error_delta_audit.py
precision-guard data
ELECTRA branch
OOF stacker
precision-guarded rule search
```

Key result:

| Candidate | Accuracy | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| step7 | 0.9133 | 0.9133 | 13 | 13 |
| round3_electra | 0.8800 | 0.8759 | 13 | 23 |
| round3_oof_stacker | 0.9033 | 0.8997 | 9 | 20 |
| round3_precision_guard | 0.9100 | 0.9103 | 14 | 13 |

Precision guard looked useful on non-teacher guard-dev:

```text
guard-dev fixed Step7 FN = 18
guard-dev induced FP = 0
```

But teacher-test produced:

```text
fixed Step7 FN = 0
induced FP = 1
```

Round3 lesson:

```text
Local guarded override is the right shape, but the guard-dev set and branch
signal were not close enough to the residual teacher-test distribution.
```

### 0.5 Round4: Paired Residual Data, Weighted DeBERTa, Human-Style Guard

Round4 moved to residual repair:

```text
paired residual data -> weighted DeBERTa branch -> human-style guard ->
local override gate
```

Residual dataset acceptance:

| Item | Count |
| --- | ---: |
| Hard human mirrors | 4727 |
| Hard LLM positives | 3674 |
| Hard human : hard LLM ratio | 1.2866 |
| Old-prose human mirrors | 800 |
| Poetry/freeverse human mirrors | 1500 |
| Natural academic human mirrors | 1000 |
| Hard-positive dev rows | 500 |
| Hard-negative dev rows | 500 |
| Teacher-test exact duplicates | 0 |
| Round4 train rows | 17434 |

Round4 weighted DeBERTa:

| Split | Step7 status | Round4 DeBERTa result | Interpretation |
| --- | ---: | ---: | --- |
| internal_test | F1 0.9564 | F1 0.9441, FP 59, FN 37 | global regression |
| hardpos dev | recall 0.4160 | recall 0.5660, F1 0.7229 | useful hard-positive signal |
| hardneg dev | FP 26 | FP 53 | unsafe human FP drift |

Round4 local override gate:

| Item | Result |
| --- | ---: |
| Aligned non-teacher rows | 2731 |
| Candidate rules searched | 3601 |
| Feasible rules | 1 |
| Feasible non-empty override rules | 0 |

Round4 lesson:

```text
Round4 DeBERTa can be a hard-positive signal branch, but it cannot globally
replace Step7. The missing piece is a guard that specifically identifies
Round4-induced false positives.
```

### 0.6 Round5: FP-Safe Residual Repair

Round5 added a Step7-vs-Round4 flip ledger and a flip guard:

```text
Step7-human -> Round4-LLM, label LLM   = safe fixed-FN candidate
Step7-human -> Round4-LLM, label human = unsafe induced-FP candidate
```

Non-teacher flip ledger:

| Split | Safe fixed-FN candidates | Unsafe induced-FP candidates | Total override candidates |
| --- | ---: | ---: | ---: |
| internal_test | 16 | 35 | 51 |
| hardpos | 88 | 0 | 88 |
| hardneg | 0 | 32 | 32 |

Round5 local rule passed the non-teacher gate:

| Split | Step7 F1 | Round5 F1 | Step7 FP | Round5 FP | Step7 FN | Round5 FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| internal_test | 0.9564 | 0.9570 | 27 | 27 | 46 | 45 |
| hardpos | 0.5876 | 0.6928 | 0 | 0 | 292 | 235 |
| hardneg | 0.0000 | 0.0000 | 26 | 26 | 0 | 0 |

Teacher-test diagnostic:

```text
Step7 baseline = 274 / 300
Round5 override = 274 / 300
Round5 overrides = 0
```

Round5 lesson:

```text
Round5 was a safe failure. It blocked 14 unsafe teacher-test induced-FP
candidates, but it also failed to release 3 safe fixed-FN candidates. Round6
should learn a safe override selector rather than lower the unsafe threshold.
```

## 1. Round5 Supplement

Round5 added these reusable assets:

```text
outputs/evaluation/round5_flip_ledger.jsonl
outputs/evaluation/round5_flip_ledger_summary.md
outputs/models/round5_flip_guard/
src/models/train_round5_flip_guard.py
src/evaluation/tune_round5_residual_override.py
src/evaluation/apply_round5_residual_override.py
```

Reusable concepts:

| Asset | Reuse |
| --- | --- |
| flip ledger | candidate-level safe/unsafe analysis |
| flip guard | first pass at induced-FP veto |
| hardpos/hardneg split | non-teacher safety gate |
| local override contract | safer than global classifier replacement |

Not promoted:

| Component | Reason |
| --- | --- |
| Round4 DeBERTa global classifier | teacher-test only 263 / 300; high FP risk |
| Round5 override rule | teacher-test no-op |
| current flip-guard threshold | safe locally but over-vetoed safe teacher-test candidates |

## 2. Round6 Objective

Round6 should be:

```text
Round6: safe override selector
```

The goal is not to make a more aggressive global classifier. It is to train and
calibrate a candidate-level selector that decides whether a local override is
safe.

### 2.1 Non-Teacher Gate

Required before any teacher-test diagnostic:

| Gate | Requirement |
| --- | --- |
| hard-positive fixed Step7 FN | materially positive |
| hard-negative induced FP | 0 |
| internal-test induced FP | 0 or tightly bounded |
| internal-test F1 | no meaningful regression |
| general-prose policy | explicitly safe |
| short-fragment policy | improved or explicitly bounded |

The practical minimum remained:

```text
fixed hard-positive FN >= 57
hard-negative induced FP = 0
internal induced FP <= 2
```

### 2.2 Teacher-Test Goal

Round6's short-term target:

```text
beat Step7: > 274 / 300
```

The 95% target still required:

```text
>= 285 / 300
```

## 3. Round6 Execution Plan

### Phase 0: Freeze Starting Point

Create a starting-point report with:

```text
Step7 teacher-test aggregate metrics
Round4 DeBERTa aggregate metrics
Round5 override aggregate metrics
non-teacher gate metrics
reusable model and prediction directories
explicit teacher-test-use restrictions
```

Do not export teacher-test text, row-level labels, or row-level threshold
conditions into training artifacts.

### Phase 1: Build Safe/Unsafe Override Dataset

Use non-teacher data only:

```text
safe examples   = Step7-human -> signal-LLM candidates with label LLM
unsafe examples = Step7-human -> signal-LLM candidates with label human
```

The split must be group-safe by pair or source id. The dataset should expand
general prose and short-fragment cases because Round5's rule was too narrow.

If there are not enough candidates, generate or extract more train-only data:

| Shortfall | Fix |
| --- | --- |
| general-prose safe LLM candidates | conservative LLM rewrites from non-teacher human sources |
| short-fragment safe LLM candidates | 20-80 word literary/academic rewrites |
| unsafe human short fragments | high-style Gutenberg/ACL-OCL human mirrors |
| unsafe general prose | polished human prose and long fluent human paragraphs |

### Phase 2: Train Candidate-Level Selector

Start with an interpretable model:

```text
logistic regression
linear SVM
small calibrated tree model
```

Candidate features:

```text
Step7 probability and margin
Round4 probability and margin
probability deltas
bucket / round4_bucket
text length and punctuation features
line and poetry markers
archaic and academic markers
flip-guard score
optional stylometry score
```

The selector should directly predict safe-vs-unsafe override, not just produce a
generic human-style veto.

### Phase 3: Rewrite Override Rule Search

Search combinations of:

```text
selector threshold
bucket allowlist
Step7 margin window
Round4 probability threshold
unsafe-score veto
```

Promote only if:

```text
fixed hard-positive Step7 FN is high enough
hard-negative induced FP = 0
internal induced FP <= small bound
internal F1 is not degraded
```

### Phase 4: Optional Lightweight DeBERTa Repair

Only enter this phase if the selector/rule route remains no-op or fixes too few
hard positives. The branch must be used as a signal branch, not as a global
replacement.

### Phase 5: Gate Report

The gate report must answer:

1. How many hard-positive Step7 false negatives are fixed?
2. Are any hard-negative false positives induced?
3. Are any internal-test false positives induced?
4. Is the general-prose policy safe?
5. Is the short-fragment policy improved?
6. Are all teacher-test restrictions satisfied?

Only a report that explicitly says `PROMOTE_TO_TEACHER_TEST = yes` may trigger a
teacher-test diagnostic.

### Phase 6: Final Diagnostic

Run teacher-test only after the full rule, selector, thresholds, model
checkpoints, and promotion gate are frozen. If the final candidate ties or loses
to Step7, keep Step7.

## 4. Stop Conditions

Do not:

```text
lower p_unsafe_override based on teacher-test behavior
use Round4 DeBERTa as a global final model
train on the 3 teacher-test safe candidates
optimize only hard-positive recall while ignoring human FP
promote a model solely because non-teacher hardpos recall improves
```

If rule search is no-op, record:

```text
PROMOTE_TO_TEACHER_TEST = no
NEXT_ACTION = data completion or guard/selector redesign
```

## 5. Final Round5/Round6 Recommendation

Step7 remains the final baseline. Round5 proved that local repair can be safe,
but it also showed that the guard was too conservative to release useful
teacher-test fixes. Round6 should learn a candidate-level safe selector on
non-teacher data; if it cannot do so without new false positives, the project
should stop and report Step7 as the validated final model.
