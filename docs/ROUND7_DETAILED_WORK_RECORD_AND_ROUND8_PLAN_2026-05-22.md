# Round7 Detailed Work Record And Round8 Optimization Plan

Date: 2026-05-22

This document is the durable handoff after Round7. It records what Round7
actually changed, why the non-teacher gate passed but teacher-test remained a
safe no-op, and the recommended Round8 route.

Core decision:

```text
KEEP_FINAL_MODEL = Step7 ensemble
ROUND7_PROMOTED = no
ROUND7_TEACHER_TEST = 274 / 300, tied Step7
NEXT_ROUTE = Round8 unsafe-guard bottleneck repair
```

The teacher-test boundary remains strict:

```text
data/raw/teacher_test.json is diagnostic-only.
It must not be used for training, threshold selection, selector calibration,
rule search, router tuning, stacker training, or model selection.
```

## 1. Starting Point

Round7 started from the Round6 postmortem:

| Item | Result |
| --- | ---: |
| Step7 teacher-test baseline | 274 / 300 |
| teacher-test Round4-vs-Step7 exact override candidates | 17 = 3 safe + 14 unsafe |
| Round6 internal exact unsafe probe block | 2 / 35 |
| Round6 hardpos fixed Step7 FN | 43, below the hard minimum 57 |

Round6 established that proxy safe/unsafe dev success was not enough. Round7
therefore narrowed the task to the exact local decision surface:

```text
Step7 predicts human
Round4 signal branch predicts LLM
selector decides whether that override is safe
```

## 2. Round7 Phase 0: Exact-Candidate Audit

New script:

```text
src/evaluation/audit_round7_exact_candidates.py
```

Output:

```text
outputs/evaluation/round7_exact_candidate_audit.md
outputs/evaluation/round7_exact_candidate_audit.json
```

The audit reproduced the handoff diagnosis:

| Split | Exact safe | Exact unsafe | Round6 exact unsafe blocked |
| --- | ---: | ---: | ---: |
| internal_test probe | 16 | 35 | 2 |

Decision:

```text
Round7 must train and gate on exact non-teacher disagreement candidates before
any new selector is trusted.
```

## 3. Round7 Phase 1: Exact-Candidate Dataset

New scripts:

```text
src/data/build_round7_exact_candidate_dataset.py
src/evaluation/mine_round7_train_exact_candidates.py
```

The first dataset pass used only Round5 exact `hardpos` / `hardneg` candidates:

```text
pool = 120 = 88 safe + 32 unsafe
PROMOTE_TO_ROUND7_SELECTOR_TRAINING = no
```

Round7 then mined non-teacher train-side disagreements from:

| Source | Use |
| --- | --- |
| `round4_residual_train.jsonl` | train-side safe and unsafe exact-like disagreement pool |
| `round4_old_prose_human_mirror_candidates.jsonl` | unused old-prose unsafe human disagreement pool |

Final exact-candidate dataset:

| Split | Rows | Safe | Unsafe |
| --- | ---: | ---: | ---: |
| train | 769 | 359 | 410 |
| dev | 202 | 82 | 120 |

Leakage checks:

| Check | Count |
| --- | ---: |
| train/dev group overlap | 0 |
| train/dev text overlap | 0 |
| teacher-test exact text duplicates | 0 |
| held-out probe/train text overlap | 0 |
| held-out probe/dev text overlap | 0 |

Important data finding:

```text
Exact-candidate counts can be met from non-teacher disagreement mining, but the
safe candidate coverage is still weaker in general_prose and has no
academic_formal safe examples in the first Round7 dataset.
```

## 4. Round7 Phase 2: Exact Selector Baseline

New scripts:

```text
src/models/train_round7_exact_candidate_selector.py
src/evaluation/predict_round7_exact_candidate_selector.py
```

Selected model:

```text
LogisticRegression
word TF-IDF + char TF-IDF + bucket / branch probability / shape features
positive label = safe_override
selected threshold = 0.6700
```

Selector metrics:

| Split | Safe pass | Unsafe block |
| --- | ---: | ---: |
| exact dev | 0.4634 | 0.9083 |
| held-out internal exact probe | 0.5625 | 0.8000 |

Probe comparison:

| Selector | Unsafe blocked |
| --- | ---: |
| Round6 v1b | 2 / 35 |
| Round7 exact selector | 28 / 35 |

This was a real improvement on the exact-candidate safety surface and justified
one non-teacher override search.

## 5. Round7 Phase 3: Two-Stage Rule Search

New script:

```text
src/evaluation/tune_round7_exact_safe_override.py
```

Rule shape:

```text
default = Step7
allow only Step7-human -> Round4-LLM
keep Round5 unsafe guard and human-style guard as stage-1 protection
require Round7 exact safe score as stage-2 allowance
keep stricter general_prose conditions
```

Search result:

| Item | Result |
| --- | ---: |
| candidate rules | 9217 |
| feasible non-teacher rules | 256 |
| selected non-teacher overrides | 72 |

Selected non-teacher metrics:

| Split | Step7 FP | Round7 FP | Step7 FN | Round7 FN | Fixed Step7 FN | Induced FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| internal_test | 27 | 27 | 46 | 36 | 10 | 0 |
| hardpos | 0 | 0 | 292 | 230 | 62 | 0 |
| hardneg | 26 | 26 | 0 | 0 | 0 | 0 |

The gate passed:

```text
PROMOTE_TO_TEACHER_TEST = yes
```

## 6. Round7 Frozen Teacher-Test Diagnostic

New script:

```text
src/evaluation/apply_round7_exact_safe_override.py
```

The teacher-test path scored the already-frozen exact selector on the existing
teacher-test flip ledger and applied the already-frozen rule once.

| Run | Correct / 300 | Accuracy | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Step7 baseline | 274 | 0.9133 | 0.9133 | 13 | 13 |
| Round4 signal branch | 263 | 0.8767 | 0.8803 | 23 | 14 |
| Round7 exact override | 274 | 0.9133 | 0.9133 | 13 | 13 |

Teacher-test override delta:

| Item | Count |
| --- | ---: |
| selected Round7 overrides | 0 |
| fixed Step7 FN | 0 |
| induced FP | 0 |

Teacher-test exact candidate surface:

| Candidate type | Count |
| --- | ---: |
| safe fixed-FN candidate | 3 |
| unsafe induced-FP candidate | 14 |

Why the final diagnostic stayed a no-op:

1. The one safe `general_prose` candidate fails the frozen strict general
   `p_unsafe_override <= 0.25` guard.
2. The two safe `literary_short_fragment` candidates fail the frozen base
   `p_unsafe_override <= 0.35` guard.
3. The same guard prevents unsafe Round4-induced FP candidates from becoming
   Round7 teacher-test false positives.

Final Round7 decision:

```text
PROMOTE_AS_FINAL = no
KEEP_FINAL_MODEL = Step7 ensemble
```

## 7. Round7 Assets To Keep

| Asset | Round8 use |
| --- | --- |
| exact-candidate audit | Shows the boundary Round8 must preserve |
| train-side exact disagreement miner | Builds non-teacher safe/unsafe pools without teacher-test |
| Round7 exact dataset builder | Reusable split and leakage scaffold |
| Round7 exact selector | Useful safe-signal feature and comparison baseline |
| Round7 frozen rule report | Shows that the remaining bottleneck is the unsafe hard veto |

Do not treat these as a final-model promotion:

| Asset | Why |
| --- | --- |
| Round7 teacher-test prediction | It ties Step7 and makes zero overrides |
| Round4 signal branch globally | It remains unsafe as a global classifier |
| Loosened teacher-test unsafe thresholds | That would tune on the diagnostic set |

## 8. Round8 Goal

Recommended name:

```text
Round8: Unsafe-Guard Bottleneck Repair
```

Round8 should not discard the FP-safe direction. It should answer a narrower
question:

```text
Can non-teacher evidence separate safe exact overrides that the Round5 unsafe
guard currently vetoes from the high-style human exact overrides it must still
block?
```

Primary target:

```text
Keep the Round7 exact-probe safety gain while releasing safe non-teacher
general_prose and literary_short_fragment candidates blocked by p_unsafe.
```

Promotion target:

```text
Pass a new non-teacher gate before any later teacher-test diagnostic.
```

## 9. Round8 Phase Plan

### Phase 0: Freeze Round7 Rejection And Audit The Unsafe Guard

Recommended new script:

```text
src/evaluation/audit_round8_unsafe_guard_bottleneck.py
```

Inputs:

```text
outputs/predictions/round7_exact_selector_internal_test_predictions.jsonl
outputs/predictions/round7_exact_selector_hardpos_predictions.jsonl
outputs/predictions/round7_exact_selector_hardneg_predictions.jsonl
data/processed/round7_exact_candidate_{train,dev}.jsonl
```

Outputs:

```text
outputs/evaluation/round8_unsafe_guard_bottleneck_audit.md
outputs/evaluation/round8_unsafe_guard_bottleneck_audit.json
```

Must report:

| Item | Required |
| --- | --- |
| safe candidates blocked only by `p_unsafe` | yes |
| unsafe candidates saved by `p_unsafe` | yes |
| `general_prose` and `literary_short_fragment` breakdown | yes |
| Round7 safe score versus Round5 unsafe score quadrants | yes |
| non-teacher examples kept text-free in the public report | yes |

### Phase 1: Build Unsafe-Guard Calibration Data

Recommended new script:

```text
src/data/build_round8_unsafe_guard_calibration_dataset.py
```

Start from non-teacher exact candidates only:

```text
Round7 exact train/dev data
Round7 train-side disagreement miner
unused train-side human mirror pools
new train-only exact-like safe positives when bucket coverage is missing
```

Data priorities:

| Bucket / class | Need |
| --- | --- |
| safe `general_prose` | substantial increase |
| safe `literary_short_fragment` | substantial increase |
| unsafe `general_prose` | keep strong |
| unsafe `literary_short_fragment` | keep strong |
| unsafe `academic_formal` | keep as FP guard |

Acceptance:

```text
teacher-test exact duplicate = 0
train/dev group overlap = 0
train/dev text overlap = 0
Round7 held-out internal probe remains untouched
```

### Phase 2: Compare Unsafe-Veto Designs

Do not immediately lower `p_unsafe_override`.

Compare these non-teacher designs:

| Candidate | Purpose |
| --- | --- |
| frozen Round7 hard veto | baseline safety |
| recalibrated unsafe guard | improve safety probability calibration on exact candidates |
| joint exact meta-selector | treat Round7 safe score, Round5 unsafe score, branch margins, and bucket features jointly |
| monotonic two-score rule | allow high safe-score overrides only when unsafe evidence is not extreme |

Preferred first model:

```text
transparent logistic meta-selector or calibrated linear model
```

The held-out probe is a gate, not threshold-training data.

### Phase 3: Round8 Rule Search

Rule shape:

```text
default = Step7
allow only Step7-human -> signal-LLM
protect hard human buckets first
allow exact safe candidates only when the Round8 unsafe bottleneck layer agrees
keep bucket-specific general_prose and short-fragment strictness
```

Hard non-teacher promotion gates:

| Gate | Requirement |
| --- | --- |
| hardneg induced FP | 0 |
| internal induced FP | <= 1 |
| internal F1 | >= 0.9570 |
| hardpos fixed Step7 FN | >= 62 Round7 result, target >= 70 |
| held-out exact probe unsafe block | do not regress materially from Round7 28 / 35 |
| safe exact dev pass in blocked buckets | improve over Round7 |

If the hard-human safety gates fail, stop before teacher-test.

### Phase 4: Optional Signal Branch Refresh

Enter only if Round8 repairs the unsafe bottleneck but the Round4 signal branch
still produces too few safe candidates.

Candidate direction:

```text
candidate-only signal branch with matched high-style human negatives and
general_prose / literary_short_fragment safe positives
```

Do not promote it as a global classifier.

### Phase 5: Round8 Gate And Final Diagnostic

Required durable outputs:

```text
outputs/evaluation/round8_gate_report.md
docs/ROUND8_OPTIMIZATION_WORK_LOG_2026-05-22.md
```

Only if the gate report says:

```text
PROMOTE_TO_TEACHER_TEST = yes
```

run one frozen teacher-test diagnostic and write:

```text
docs/ROUND8_FINAL_DECISION_2026-05-22.md
```

## 10. Round8 Immediate Task List

Start with these tasks:

1. Implement `src/evaluation/audit_round8_unsafe_guard_bottleneck.py`.
2. Quantify how many non-teacher Round7 safe candidates are blocked only by
   the Round5 unsafe score in `general_prose` and `literary_short_fragment`.
3. Build the Round8 unsafe-guard calibration dataset only after the audit shows
   the blocked-safe / protected-unsafe tradeoff clearly.
4. Keep the Round7 exact dataset, held-out probe, and teacher-test boundary
   unchanged while deciding whether `p_unsafe_override` stays a hard veto.

## 11. Stop Conditions

Round8 should stop before teacher-test if any of these is true:

```text
teacher-test exact duplicate > 0
train/dev leakage > 0
hardneg induced FP > 0
internal induced FP > 1
held-out exact unsafe probe regresses materially from Round7
safe blocked-bucket pass improves only by breaking high-style human precision
```

The discipline is unchanged:

```text
Teacher-test remains the final diagnostic, not the calibration set.
Step7 remains final until a later candidate beats it safely.
```
