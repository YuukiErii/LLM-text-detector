# Round6 Detailed Work Record And Round7 Optimization Plan

Date: 2026-05-22

This document is the durable handoff after Round6. It records what was actually
run, why Round6 stopped before teacher-test, and the recommended Round7 route.

Core decision:

```text
KEEP_FINAL_MODEL = Step7 ensemble
ROUND6_PROMOTED = no
PROMOTE_TO_TEACHER_TEST = no
NEXT_ROUTE = Round7 exact-candidate calibrated selector
```

The teacher-test boundary remains strict:

```text
data/raw/teacher_test.json is diagnostic-only.
It must not be used for training, threshold selection, selector calibration,
rule search, router tuning, stacker training, or model selection.
```

## 1. Starting Point

Round6 started from the Round5 final decision:

| Item | Result |
| --- | ---: |
| Step7 teacher-test | 274 / 300 |
| Round5 teacher-test | 274 / 300 |
| Round4 DeBERTa branch teacher-test | 263 / 300 |
| teacher-test Round4-vs-Step7 override candidates | 17 = 3 safe + 14 unsafe |

Round5 conclusion:

```text
Round5 passed non-teacher safety gates, but made zero teacher-test overrides.
It was precision-safe and recall-underpowered.
```

Round6 therefore did not try to lower thresholds blindly. It tried to learn a
candidate-level selector:

```text
input = Step7-human -> Round4-LLM candidate
output = p_safe_override
```

## 2. Round6 Phase 0: Starting Point Report

Script:

```text
src/evaluation/build_round6_starting_point_report.py
```

Command:

```powershell
.\.venv\Scripts\python.exe src\evaluation\build_round6_starting_point_report.py
```

Outputs:

```text
outputs/evaluation/round6_starting_point_report.md
outputs/evaluation/round6_starting_point_report.json
```

This phase froze the comparison baseline and confirmed that Round6 could move
only to non-teacher data construction.

## 3. Round6 Phase 1: Dataset Build

Script:

```text
src/data/build_round6_safe_override_dataset.py
```

Command:

```powershell
.\.venv\Scripts\python.exe src\data\build_round6_safe_override_dataset.py
```

Outputs:

```text
data/processed/round6_override_train.jsonl
data/processed/round6_override_dev_safe.jsonl
data/processed/round6_override_dev_unsafe.jsonl
data/processed/round6_override_probe_mixed.jsonl
data/processed/round6_override_dataset_report.json
data/processed/round6_override_dataset_report.md
```

Label definition:

| Field | Meaning |
| --- | --- |
| `label = 1` | safe override candidate |
| `label = 0` | unsafe override candidate |
| `original_detection_label` | original detector label, `1=LLM`, `0=human` |

Final v1b split:

| Split | Rows | Safe | Unsafe | Exact candidates |
| --- | ---: | ---: | ---: | ---: |
| train | 3411 | 1508 | 1903 | 60 |
| dev_safe | 160 | 160 | 0 | 44 |
| dev_unsafe | 240 | 0 | 240 | 16 |
| probe_mixed | 51 | 16 | 35 | 51 |

Leakage checks:

| Check | Count |
| --- | ---: |
| train/dev group overlap | 0 |
| train/dev text overlap | 0 |
| teacher-test exact text duplicates | 0 |
| probe/train text overlap | 0 |
| probe/dev text overlap | 0 |

Important design choice:

```text
The first split kept all exact candidates out of train. That made the selector
look good on proxy dev but weak on exact candidate probes.
The v1b split moved a limited set of non-teacher exact hardpos/hardneg
candidates into train while keeping a held-out exact probe.
```

## 4. Round6 Phase 2: Selector Training

Scripts:

```text
src/models/train_round6_safe_override_selector.py
src/evaluation/predict_round6_safe_override_selector.py
```

Training command:

```powershell
.\.venv\Scripts\python.exe src\models\train_round6_safe_override_selector.py
```

Model:

```text
LogisticRegression
word TF-IDF + char TF-IDF + structured metadata/text-shape features
positive label = safe_override
```

Outputs:

```text
outputs/models/round6_safe_override_selector/selector.pkl
outputs/models/round6_safe_override_selector/selector_report.json
outputs/evaluation/round6_safe_override_selector_report.md
outputs/predictions/round6_safe_selector_train_predictions.jsonl
outputs/predictions/round6_safe_selector_dev_safe_predictions.jsonl
outputs/predictions/round6_safe_selector_dev_unsafe_predictions.jsonl
outputs/predictions/round6_safe_selector_probe_mixed_predictions.jsonl
```

Selected threshold:

```text
p_safe_override >= 0.5100
```

Metrics:

| Split | n | Safe pass | Unsafe block | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| train | 3411 | 0.7798 | 0.9606 | 0.8807 |
| dev_safe | 160 | 0.6438 | NA | 0.6438 |
| dev_unsafe | 240 | NA | 0.9000 | 0.9000 |
| probe_mixed | 51 | 1.0000 | 0.0571 | 0.3529 |

The proxy dev gate technically passed:

```text
unsafe dev blocked = 0.9000
safe dev pass rate = 0.6438
```

But the exact-candidate probe exposed the central weakness:

```text
Only 2 / 35 internal exact unsafe candidates were blocked.
```

Interpretation:

```text
The selector learned proxy human-vs-LLM differences, but not enough of the
exact Step7-human -> Round4-LLM disagreement boundary.
```

## 5. Round6 Phase 3: Rule Search

Script:

```text
src/evaluation/tune_round6_safe_override.py
```

Prediction commands:

```powershell
.\.venv\Scripts\python.exe src\evaluation\predict_round6_safe_override_selector.py `
  --input outputs\predictions\round5_flip_guard_internal_test_predictions.jsonl `
  --text_source outputs\evaluation\round5_flip_ledger.jsonl `
  --output outputs\predictions\round6_safe_selector_internal_test_predictions.jsonl

.\.venv\Scripts\python.exe src\evaluation\predict_round6_safe_override_selector.py `
  --input outputs\predictions\round5_flip_guard_hardpos_predictions.jsonl `
  --text_source outputs\evaluation\round5_flip_ledger.jsonl `
  --output outputs\predictions\round6_safe_selector_hardpos_predictions.jsonl

.\.venv\Scripts\python.exe src\evaluation\predict_round6_safe_override_selector.py `
  --input outputs\predictions\round5_flip_guard_hardneg_predictions.jsonl `
  --text_source outputs\evaluation\round5_flip_ledger.jsonl `
  --output outputs\predictions\round6_safe_selector_hardneg_predictions.jsonl
```

Rule-search command:

```powershell
.\.venv\Scripts\python.exe src\evaluation\tune_round6_safe_override.py `
  --tune_set internal_test outputs\predictions\round6_safe_selector_internal_test_predictions.jsonl `
  --tune_set hardpos outputs\predictions\round6_safe_selector_hardpos_predictions.jsonl `
  --tune_set hardneg outputs\predictions\round6_safe_selector_hardneg_predictions.jsonl
```

Outputs:

```text
outputs/models/round6_safe_override/rules.json
outputs/models/round6_safe_override/tuning_report.json
outputs/evaluation/round6_safe_override_tuning_report.md
outputs/predictions/round6_safe_override_internal_test_predictions.jsonl
outputs/predictions/round6_safe_override_hardpos_predictions.jsonl
outputs/predictions/round6_safe_override_hardneg_predictions.jsonl
```

Search result:

| Item | Result |
| --- | ---: |
| candidate rules | 4321 |
| feasible rules | 0 |
| selected total overrides | 57 |

Best non-promoted rule:

```json
{
  "p_safe_min": 0.5,
  "p_unsafe_max": 0.35,
  "round4_threshold": 0.5,
  "min_delta": 0.0,
  "bucket_policy": "old_short_plus_general_strict",
  "general_p_safe_min": 0.6,
  "general_p_unsafe_max": 0.25,
  "general_round4_threshold": 0.55,
  "general_min_delta": 0.05,
  "disabled_baseline": false
}
```

Non-teacher metrics:

| Split | Step7 F1 | Round6 F1 | Step7 FP | Round6 FP | Step7 FN | Round6 FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| internal_test | 0.9564 | 0.9651 | 27 | 27 | 46 | 32 |
| hardpos | 0.5876 | 0.6684 | 0 | 0 | 292 | 249 |
| hardneg | 0.0000 | 0.0000 | 26 | 26 | 0 | 0 |

Override delta:

| Split | Overrides | Fixed Step7 FN | Induced FP |
| --- | ---: | ---: | ---: |
| internal_test | 14 | 14 | 0 |
| hardpos | 43 | 43 | 0 |
| hardneg | 0 | 0 | 0 |

Gate check:

| Gate | Required | Observed | Pass |
| --- | --- | ---: | --- |
| hardneg induced FP | 0 | 0 | yes |
| internal induced FP | <= 1 | 0 | yes |
| internal F1 | >= 0.9564 hard minimum, >= 0.9570 preferred | 0.9651 | yes |
| hardpos fixed Step7 FN | >= 57 hard minimum, >= 70 target | 43 | no |
| non-empty override | required | 57 total | yes |

Final Round6 v1b decision:

```text
PROMOTE_TO_TEACHER_TEST = no
FINAL_MODEL_CANDIDATE = no
KEEP_FINAL_MODEL = Step7 ensemble
```

## 6. Why Round6 Failed Safely

Round6 did not fail by creating false positives. It failed because the rule
became too selective on the hard-positive development set:

```text
Round5 hardpos fixed Step7 FN = 57
Round6 v1b hardpos fixed Step7 FN = 43
Round6 target = 70
```

The positive result:

```text
Round6 v1b found a stricter rule that improves internal_test F1 to 0.9651
without increasing internal or hardneg FP.
```

The blocking result:

```text
It does not release enough non-teacher hard-positive repair signal to justify a
teacher-test diagnostic.
```

The key diagnostic:

```text
Proxy dev success did not transfer to exact-candidate safety.
Round7 must train and validate directly on non-teacher exact disagreement
examples, not mostly on broad proxy residual data.
```

## 7. Round7 Goal

Recommended name:

```text
Round7: Exact-Candidate Calibrated Selector
```

Round7 should not be a broader model. It should be a narrower candidate
selector trained specifically on the same decision surface as the final
override:

```text
Step7 predicts human
Round4 / signal branch predicts LLM
Should this local override be allowed?
```

Primary target:

```text
Meet non-teacher gates and advance to one final teacher-test diagnostic.
```

Stretch target:

```text
Beat Step7 teacher-test, then continue toward 285 / 300.
```

## 8. Round7 Phase Plan

### Phase 0: Freeze Round6 v1b And Audit Exact Candidates

New script:

```text
src/evaluation/audit_round7_exact_candidates.py
```

Inputs:

```text
outputs/predictions/round6_safe_selector_internal_test_predictions.jsonl
outputs/predictions/round6_safe_selector_hardpos_predictions.jsonl
outputs/predictions/round6_safe_selector_hardneg_predictions.jsonl
outputs/evaluation/round5_flip_ledger.jsonl
```

Outputs:

```text
outputs/evaluation/round7_exact_candidate_audit.md
outputs/evaluation/round7_exact_candidate_audit.json
```

Must report:

| Item | Required |
| --- | --- |
| exact safe candidates by split/bucket | yes |
| exact unsafe candidates by split/bucket | yes |
| selector false-pass unsafe cases | yes |
| selector false-block safe cases | yes |
| p_safe / p_unsafe / round4_prob histograms | yes |

Acceptance:

```text
Audit explains why internal exact unsafe block is only 2 / 35.
No teacher-test row text or id enters the audit output used for training.
```

### Phase 1: Build Exact-Candidate Training Data

New script:

```text
src/data/build_round7_exact_candidate_dataset.py
```

Construct only rows that simulate the true override decision:

```text
step7_pred = 0
signal_branch_pred = 1
candidate label = safe if original label is LLM, unsafe if original label is human
```

Use non-teacher sources only:

```text
round5_flip_ledger hardpos/hardneg exact candidates
round4 residual dev hardpos/hardneg candidates
new train-only exact-like generated hard positives
new train-only exact-like high-style human negatives
```

Targets:

| Split | Safe | Unsafe |
| --- | ---: | ---: |
| train exact candidates | >= 250 | >= 350 |
| dev exact candidates | >= 80 | >= 120 |
| held-out internal-style probe | unchanged Round6 `probe_mixed` | unchanged Round6 `probe_mixed` |

Bucket targets:

| Bucket | Need |
| --- | --- |
| `general_prose` safe | increase substantially |
| `literary_short_fragment` safe | increase substantially |
| `general_prose` unsafe | increase substantially |
| `literary_old_prose` unsafe | keep strong |
| `academic_formal` unsafe | keep strong |

Acceptance:

```text
teacher-test exact duplicate = 0
train/dev pair overlap = 0
train/dev text overlap = 0
held-out exact probe untouched
```

### Phase 2: Train Calibrated Exact Selector

New script:

```text
src/models/train_round7_exact_candidate_selector.py
```

Model candidates:

| Candidate | Purpose |
| --- | --- |
| LogisticRegression | transparent baseline |
| Calibrated LinearSVC | sharper boundary |
| HistGradientBoosting | non-linear tabular/meta-feature candidate |

Feature groups:

| Group | Examples |
| --- | --- |
| branch probabilities | `step7_prob`, `round4_prob`, `prob_delta` |
| guard probabilities | `p_unsafe_override`, `p_human_style`, `p_safe_override_v1b` |
| exact-candidate flags | source split, exact/proxy origin |
| bucket metadata | `bucket`, `round4_bucket`, `round4_tag`, domain |
| stylometry | char n-grams, punctuation, lineation, archaic/academic markers |

Threshold selection must optimize:

```text
unsafe exact dev block >= 0.90
safe exact dev pass >= 0.35
held-out exact probe unsafe block improves materially over Round6 v1b
```

Round7 must not select a threshold only because proxy dev looks good.

### Phase 3: Two-Stage Rule Search

New script:

```text
src/evaluation/tune_round7_exact_safe_override.py
```

Rule shape:

```text
default = Step7
allow only Step7-human -> signal-LLM
stage 1: block obvious unsafe via Round5 p_unsafe and human-style guard
stage 2: allow only high-confidence exact-selector safe candidates
bucket-specific strictness for general_prose and literary_short_fragment
```

Promotion gates:

| Gate | Hard requirement |
| --- | --- |
| hardneg induced FP | 0 |
| internal induced FP | <= 1 |
| internal F1 | >= 0.9570 |
| hardpos fixed Step7 FN | >= 57 minimum, >= 70 target |
| exact probe unsafe block | materially better than Round6 v1b |
| non-empty override | yes |

If no rule meets `hardpos fixed Step7 FN >= 57`, stop before teacher-test.

### Phase 4: Optional Signal Branch Refresh

Only enter this if exact selector improves safety but recall remains below 57.

Candidate:

```text
round7_deberta_exact_candidate_guardweighted
```

Goal:

```text
Generate better Step7-human -> LLM candidates without increasing hard-human FP.
```

Do not promote it as a global classifier. Use it only as a signal branch in
the local override rule.

### Phase 5: Round7 Gate Report

Output:

```text
outputs/evaluation/round7_gate_report.md
docs/rounds/round7_optimization_work_log_2026-05-22.md
```

Required decision:

```text
PROMOTE_TO_TEACHER_TEST = yes/no
FINAL_MODEL_CANDIDATE = yes/no
KEEP_FINAL_MODEL = Step7 ensemble unless a candidate clears all non-teacher gates
```

### Phase 6: Final Teacher-Test Diagnostic Only If Gate Passes

Only run if Phase 5 says:

```text
PROMOTE_TO_TEACHER_TEST = yes
```

Outputs:

```text
outputs/predictions/round7_teacher_test_predictions.jsonl
outputs/evaluation/round7_teacher_test_comparison.md
outputs/evaluation/round7_teacher_test_ledger_summary.json
docs/rounds/round7_final_decision_2026-05-22.md
```

Decision:

| Teacher-test result | Decision |
| --- | --- |
| `<= 274 / 300` | reject, keep Step7 |
| `275-284 / 300` | partial success |
| `>= 285 / 300` | 95% achieved if no leakage/tuning violation |

## 9. Round7 Immediate Task List

Start with these exact tasks:

1. Implement `src/evaluation/audit_round7_exact_candidates.py`.
2. Build `outputs/evaluation/round7_exact_candidate_audit.md`.
3. Implement `src/data/build_round7_exact_candidate_dataset.py`.
4. Check whether exact candidate train/dev targets can be met without new generation.
5. If targets are not met, generate only train-side exact-like safe/unsafe data.
6. Do not train a new selector until the exact-candidate dataset passes leakage checks.

## 10. Stop Conditions

Round7 should stop before teacher-test if any of these is true:

```text
teacher-test exact duplicate > 0
train/dev group overlap > 0
hardneg induced FP > 0
internal induced FP > 1
hardpos fixed Step7 FN < 57
exact unsafe probe is not better than Round6 v1b
```

The key discipline remains:

```text
No teacher-test diagnostic until all non-teacher gates pass.
Step7 remains the final model unless a candidate beats it safely.
```
