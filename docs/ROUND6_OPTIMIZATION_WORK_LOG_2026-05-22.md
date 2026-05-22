# Round6 Optimization Work Log

Date: 2026-05-22

Round6 follows `docs/ROUND5_SUPPLEMENT_AND_ROUND6_PLAN_2026-05-22.md`.
The strict boundary remains unchanged:

```text
data/raw/teacher_test.json is diagnostic-only.
It is not used for training, threshold selection, selector calibration, rule search, or model selection.
```

For the fuller postmortem and next-round plan, read:

```text
docs/ROUND6_DETAILED_WORK_RECORD_AND_ROUND7_PLAN_2026-05-22.md
```

## 1. Phase 0: Starting Point Frozen

New script:

```text
src/evaluation/build_round6_starting_point_report.py
```

Command:

```powershell
.\.venv\Scripts\python.exe src\evaluation\build_round6_starting_point_report.py
```

Output:

```text
outputs/evaluation/round6_starting_point_report.md
outputs/evaluation/round6_starting_point_report.json
```

Frozen starting point:

| Item | Result |
| --- | ---: |
| Step7 teacher-test | 274 / 300 |
| Round5 teacher-test | 274 / 300 |
| Round4 DeBERTa teacher-test | 263 / 300 |
| teacher-test override candidates | 17 = 3 safe + 14 unsafe |

Decision:

```text
ROUND6_PHASE0_STATUS = complete
PROMOTE_TO_PHASE1_DATASET_BUILD = yes
TEACHER_TEST_SELECTION_ALLOWED = no
```

## 2. Phase 1: Safe/Unsafe Override Dataset

New script:

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

```text
label = 1 means safe_override
label = 0 means unsafe_override
original_detection_label preserves the detector label, where 1=LLM and 0=human
```

Final v1b data split:

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

Dataset decision:

```text
PROMOTE_TO_SELECTOR_TRAINING = yes
```

## 3. Phase 2: Safe Override Selector

New scripts:

```text
src/models/train_round6_safe_override_selector.py
src/evaluation/predict_round6_safe_override_selector.py
```

Training command:

```powershell
.\.venv\Scripts\python.exe src\models\train_round6_safe_override_selector.py
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

Selector metrics:

| Split | n | Safe pass | Unsafe block | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| train | 3411 | 0.7798 | 0.9606 | 0.8807 |
| dev_safe | 160 | 0.6438 | NA | 0.6438 |
| dev_unsafe | 240 | NA | 0.9000 | 0.9000 |
| probe_mixed | 51 | 1.0000 | 0.0571 | 0.3529 |

Selector decision:

```text
PROMOTE_TO_ROUND6_RULE_SEARCH = yes
```

Caveat:

```text
Proxy dev passes, but exact internal probe does not transfer well.
Only 2 / 35 internal exact unsafe candidates are blocked by the selector threshold.
Rule search must decide promotion; selector metrics alone are not enough.
```

## 4. Phase 3: Safe Override Rule Search

New script:

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

Rule search command:

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

Selected non-promoted rule:

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

Non-teacher result:

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

Gate decision:

```text
PROMOTE_TO_ROUND6_GATE_REPORT = no
```

Reason:

```text
The selected rule is FP-safe but recall-underpowered.
It fixes 43 hardpos Step7 FN, below the hard minimum 57 and target 70.
No teacher-test diagnostic should be run.
```

## 5. Phase 5 Gate Report

Output:

```text
outputs/evaluation/round6_gate_report.md
```

Final Round6 v1b decision:

```text
PROMOTE_TO_TEACHER_TEST = no
FINAL_MODEL_CANDIDATE = no
KEEP_FINAL_MODEL = Step7 ensemble
```

## 6. Recommended Next Move

Do not loosen thresholds or run teacher-test.

The next useful Round6 move is to improve the candidate-level selector:

1. Add more exact-candidate-like safe hard positives, especially `general_prose`
   and `literary_short_fragment`.
2. Add exact-candidate-like unsafe human examples that resemble internal-test
   induced FP, not only proxy hard negatives.
3. Keep a held-out exact-candidate probe and require unsafe block to improve
   before another rule search.
4. Re-run rule search only when hardpos fixed Step7 FN can reach at least 57
   with hardneg/internal induced FP still at zero or near-zero.
